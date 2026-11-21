"""
AUD/CAD signal bot (TEST / EVALUATION BUILD). Does NOT place trades.

IMPORTANT LIMITS
- Pocket Option OTC prices are the broker's own feed and are not publicly available.
  This bot analyses real AUD/CAD spot candles (Yahoo Finance) as a PROXY. OTC prices can differ.
- Forex spot is closed on weekends; the bot detects stale data and stays silent then.
- Signals are NOT verified. Use --backtest and --evaluate to measure results before trusting anything.
- Binary options are high risk and restricted in some countries. Not financial advice.

Setup:   pip install yfinance pandas numpy
Commands:
  python signal_bot.py --test-notify   send a test message via ntfy
  python signal_bot.py --telegram-setup  find your Telegram chat ID
  python signal_bot.py --test-telegram   send a test message via Telegram
  python signal_bot.py --backtest      replay the rules over ~30 days of history
  python signal_bot.py                 run live, alert on new signals (forward test)
  python signal_bot.py --once          one check (for cron / Task Scheduler)
  python signal_bot.py --evaluate      score logged live signals after their expiry
"""
import argparse, csv, json, math, os, time, urllib.error, urllib.request
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd


def _load_env_file(path=".env"):
    """Optional: load KEY=VALUE lines from a private .env file (never share or upload it)."""
    f = Path(path)
    if f.exists():
        for line in f.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env_file()


def tg_api(method, payload=None):
    """Calls the Telegram Bot API. Token comes from the TELEGRAM_BOT_TOKEN environment variable."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")
    req = urllib.request.Request(f"https://api.telegram.org/bot{token}/{method}",
                                 data=json.dumps(payload or {}).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:      # return Telegram's error text, never the URL/token
        return json.loads(e.read() or b"{}")


def tg_send(text):
    chat = os.getenv("TELEGRAM_CHAT_ID")
    if not chat:
        return False
    res = tg_api("sendMessage", {"chat_id": chat, "text": text[:4000]})
    if not res.get("ok"):
        print("Telegram error:", res.get("description", "unknown"))
    return bool(res.get("ok"))


def telegram_setup():
    """Finds your chat ID after you message the bot. Prints the ID, never the token."""
    if not os.getenv("TELEGRAM_BOT_TOKEN"):
        print("TELEGRAM_BOT_TOKEN is not set yet.")
        return
    res = tg_api("getUpdates")
    if not res.get("ok"):
        print("Telegram error:", res.get("description", "check your token"))
        return
    chats = {u["message"]["chat"]["id"]: u["message"]["chat"].get("first_name", "")
             for u in res.get("result", []) if "message" in u}
    if not chats:
        print("No messages found. Open your bot in Telegram, tap Start, send 'hi', then run this again.")
        return
    for cid, name in chats.items():
        print(f"Chat ID: {cid}  ({name})")
    print("Save that number as TELEGRAM_CHAT_ID, then run --test-telegram.")


# ================= SETTINGS =================
SYMBOL = "AUDCAD=X"
ASSET_NAME = "AUD/CAD OTC (analysed on AUDCAD spot proxy)"
CANDLE_MIN = 5                  # candle size in minutes
EXPIRY_CANDLES = 3              # suggested expiry = 3 candles = 15 min
NTFY_TOPIC = os.getenv("NTFY_TOPIC", "")   # your private ntfy topic (set as a secret; "" disables ntfy)

# Market-quality filters (signals are suppressed unless ALL pass)
ADX_MIN = 22                    # below this the market is ranging/unclear
MIN_ATR_PCT = 0.0002            # ATR/price; below this the market is too quiet (tune it)
STALE_MIN = 20                  # ignore data older than this (market closed / feed delay)

# Confirmation rules
MIN_CONFIRMATIONS = 4           # of 5 checks must agree
RSI_BUY_RANGE = (50, 68)
RSI_SELL_RANGE = (32, 50)
SR_LOOKBACK = 60                # candles used for support/resistance (60 x 5m = 5h)
SR_ROOM_ATR = 1.0               # need at least this many ATRs of room to S/R
COOLDOWN_CANDLES = 3            # no new signal within 3 candles of the last one

PAYOUT = 0.85                   # your Pocket Option payout, used for the break-even win rate
POLL_SEC = 30
LOG_FILE = Path("signals_log.csv")
# ============================================

CANDLE = pd.Timedelta(minutes=CANDLE_MIN)
LOG_COLS = ["signal_id", "detected_local", "candle_close_utc", "asset", "direction", "entry",
            "expiry_utc", "trend", "support", "resistance", "adx", "rsi", "macd_hist",
            "confirmations", "result", "exit_price"]


# ---------- data ----------
def fetch(period="30d"):
    import yfinance as yf
    df = yf.download(SYMBOL, period=period, interval=f"{CANDLE_MIN}m",
                     progress=False, auto_adjust=False)
    if df is None or df.empty:
        raise RuntimeError("No data returned (market closed or network issue)")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close"]].dropna()
    df.columns = ["open", "high", "low", "close"]
    idx = pd.DatetimeIndex(df.index)
    df.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    return df[~df.index.duplicated()]


def closed_only(df, now):
    """Drop the still-forming candle so signals use closed candles only."""
    return df[df.index + CANDLE <= now]


# ---------- indicators ----------
def indicators(df):
    d = df.copy()
    c, h, l = d["close"], d["high"], d["low"]
    d["ema_fast"] = c.ewm(span=20, adjust=False).mean()
    d["ema_slow"] = c.ewm(span=50, adjust=False).mean()

    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    d["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    d["macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()

    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    up, down = h.diff(), -l.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    d["plus_di"] = 100 * pd.Series(plus_dm, index=d.index).ewm(alpha=1 / 14, adjust=False).mean() / atr
    d["minus_di"] = 100 * pd.Series(minus_dm, index=d.index).ewm(alpha=1 / 14, adjust=False).mean() / atr
    dx = 100 * (d["plus_di"] - d["minus_di"]).abs() / (d["plus_di"] + d["minus_di"])
    d["adx"] = dx.ewm(alpha=1 / 14, adjust=False).mean()
    d["atr"] = atr

    d["resistance"] = h.rolling(SR_LOOKBACK).max().shift(1)   # excludes current candle
    d["support"] = l.rolling(SR_LOOKBACK).min().shift(1)
    d["below_ema_recent"] = (c < d["ema_fast"]).astype(int).rolling(4).max().shift(1)
    d["above_ema_recent"] = (c > d["ema_fast"]).astype(int).rolling(4).max().shift(1)
    return d


# ---------- signal logic ----------
def evaluate(ind, i):
    """Returns dict with 'signal' (BUY/SELL/None), 'reason', and analysis details."""
    r, p = ind.iloc[i], ind.iloc[i - 1]
    need = ["ema_slow", "rsi", "adx", "atr", "resistance", "support", "below_ema_recent"]
    if r[need].isna().any():
        return {"signal": None, "reason": "Not enough data yet"}
    out = {"signal": None, "adx": r.adx, "rsi": r.rsi, "macd_hist": r.macd_hist,
           "support": r.support, "resistance": r.resistance}

    if r.atr / r.close < MIN_ATR_PCT:
        return {**out, "reason": "Market too quiet (low volatility)"}
    if r.adx < ADX_MIN:
        return {**out, "reason": f"Unclear/ranging market (ADX {r.adx:.0f} < {ADX_MIN})"}

    if r.ema_fast > r.ema_slow and r.close > r.ema_slow:
        side, trend = "BUY", "UP"
    elif r.ema_fast < r.ema_slow and r.close < r.ema_slow:
        side, trend = "SELL", "DOWN"
    else:
        return {**out, "reason": "Trend unclear (EMAs and price disagree)"}

    if side == "BUY":
        room = r.resistance - r.close
        checks = {
            "+DI above -DI (buyers dominate)": r.plus_di > r.minus_di,
            "MACD histogram positive and rising": r.macd_hist > 0 and r.macd_hist > p.macd_hist,
            f"RSI {r.rsi:.0f} in {RSI_BUY_RANGE} (momentum, not overbought)": RSI_BUY_RANGE[0] <= r.rsi <= RSI_BUY_RANGE[1],
            "Closed back above EMA20 after a recent dip": bool(r.below_ema_recent == 1 and r.close > r.ema_fast),
            "Room to resistance (>= 1 ATR) or breakout": not (0 <= room < SR_ROOM_ATR * r.atr),
        }
    else:
        room = r.close - r.support
        checks = {
            "-DI above +DI (sellers dominate)": r.minus_di > r.plus_di,
            "MACD histogram negative and falling": r.macd_hist < 0 and r.macd_hist < p.macd_hist,
            f"RSI {r.rsi:.0f} in {RSI_SELL_RANGE} (momentum, not oversold)": RSI_SELL_RANGE[0] <= r.rsi <= RSI_SELL_RANGE[1],
            "Closed back below EMA20 after a recent bounce": bool(r.above_ema_recent == 1 and r.close < r.ema_fast),
            "Room to support (>= 1 ATR) or breakdown": not (0 <= room < SR_ROOM_ATR * r.atr),
        }
    passed = [k for k, v in checks.items() if v]
    out.update({"trend": trend, "checks": checks, "n_pass": len(passed)})
    if len(passed) < MIN_CONFIRMATIONS:
        return {**out, "reason": f"Only {len(passed)}/5 confirmations for {side}"}
    return {**out, "signal": side, "reason": f"{len(passed)}/5 confirmations"}


# ---------- alerts + logging ----------
def notify(title, body):
    print(f"\n>>> {title}\n{body}\n")
    if NTFY_TOPIC:
        try:
            req = urllib.request.Request(f"https://ntfy.sh/{NTFY_TOPIC}", data=body.encode("utf-8"),
                                         headers={"Title": title, "Tags": "chart_with_upwards_trend"})
            urllib.request.urlopen(req, timeout=10)
        except Exception as e:
            print("ntfy failed:", e)
    try:
        tg_send(f"{title}\n\n{body}")
    except Exception as e:
        print("Telegram failed:", type(e).__name__)


def local(ts):
    return ts.tz_convert(pd.Timestamp.now().astimezone().tzinfo)


def issue(res, ind, now):
    t = ind.index[-1]
    close_t = t + CANDLE
    expiry = close_t + EXPIRY_CANDLES * CANDLE
    entry = float(ind["close"].iloc[-1])
    sid = close_t.strftime("%Y%m%d%H%M")
    ticks = "\n".join(f"  {'[x]' if v else '[ ]'} {k}" for k, v in res["checks"].items())
    body = (f"Asset: {ASSET_NAME}\nSignal: {res['signal']}\n"
            f"Detected: {local(now).strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
            f"Entry price: {entry:.5f} (close of signal candle; check Pocket Option's own price)\n"
            f"Suggested expiry: {EXPIRY_CANDLES * CANDLE_MIN} min -> {local(expiry).strftime('%H:%M')} local\n"
            f"Trend: {res['trend']} (EMA20 vs EMA50, ADX {res['adx']:.0f})\n"
            f"Support: {res['support']:.5f}  Resistance: {res['resistance']:.5f}\n"
            f"Confirmations ({res['n_pass']}/5):\n{ticks}\n"
            f"TEST SIGNAL: unverified, may be wrong. Do not risk money you can't lose.")
    notify(f"{res['signal']} AUD/CAD OTC (test signal)", body)
    new = not LOG_FILE.exists()
    with open(LOG_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(LOG_COLS)
        w.writerow([sid, local(now).isoformat(timespec="seconds"), close_t.isoformat(), ASSET_NAME,
                    res["signal"], f"{entry:.5f}", expiry.isoformat(), res["trend"],
                    f"{res['support']:.5f}", f"{res['resistance']:.5f}", f"{res['adx']:.1f}",
                    f"{res['rsi']:.1f}", f"{res['macd_hist']:.6f}", res["n_pass"], "", ""])


def last_logged_candle():
    if not LOG_FILE.exists():
        return None
    rows = list(csv.DictReader(open(LOG_FILE)))
    return pd.Timestamp(rows[-1]["candle_close_utc"]) - CANDLE if rows else None


# ---------- statistics ----------
def wilson(w, n, z=1.96):
    if n == 0:
        return (0, 0)
    p = w / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def report(wins, losses, ties, label):
    n = wins + losses
    be = 1 / (1 + PAYOUT)
    print(f"\n=== {label} ===")
    print(f"Wins {wins} | Losses {losses} | Ties {ties}")
    if n == 0:
        print("No completed trades yet.")
        return
    lo, hi = wilson(wins, n)
    print(f"Win rate: {wins / n:.1%} (95% range {lo:.1%} to {hi:.1%}, n={n})")
    print(f"Break-even win rate at {PAYOUT:.0%} payout: {be:.1%}")
    print("Verdict:", "NOT proven profitable (range includes break-even or below)." if lo <= be
          else "Above break-even on this sample, still verify on more data and live.")
    if n < 100:
        print("Warning: fewer than 100 trades; results are mostly noise.")


# ---------- modes ----------
def run_backtest():
    df = closed_only(fetch("30d"), pd.Timestamp.now(tz="UTC"))
    ind = indicators(df)
    w = l = t = 0
    last, rejects = -999, Counter()
    for i in range(SR_LOOKBACK + 5, len(ind) - EXPIRY_CANDLES):
        res = evaluate(ind, i)
        if not res["signal"]:
            rejects[res["reason"].split(" (")[0].split(" for")[0]] += 1
            continue
        if i - last < COOLDOWN_CANDLES:
            continue
        last = i
        entry, exit_ = ind["close"].iloc[i], ind["close"].iloc[i + EXPIRY_CANDLES]
        if round(exit_, 5) == round(entry, 5):
            t += 1
        elif (res["signal"] == "BUY") == (exit_ > entry):
            w += 1
        else:
            l += 1
    print(f"Candles analysed: {len(ind)} | Why no signal (top reasons): {rejects.most_common(4)}")
    report(w, l, t, "BACKTEST (spot proxy data, in-sample, no spread)")
    print("Backtests flatter results: OTC feed, spreads and delays are not modelled.")


def run_evaluate():
    if not LOG_FILE.exists():
        print("No signals logged yet.")
        return
    rows = list(csv.DictReader(open(LOG_FILE)))
    ind = fetch("5d")
    now = pd.Timestamp.now(tz="UTC")
    for r in rows:
        exp = pd.Timestamp(r["expiry_utc"])
        if r["result"] or exp > now:
            continue
        k = exp - CANDLE
        if k not in ind.index:
            continue
        ex, en = float(ind.loc[k, "close"]), float(r["entry"])
        r["exit_price"] = f"{ex:.5f}"
        r["result"] = "TIE" if round(ex, 5) == round(en, 5) else \
            "WIN" if (r["direction"] == "BUY") == (ex > en) else "LOSS"
    with open(LOG_FILE, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=LOG_COLS)
        wr.writeheader()
        wr.writerows(rows)
    c = Counter(r["result"] for r in rows)
    report(c["WIN"], c["LOSS"], c["TIE"], "LIVE FORWARD TEST (spot proxy)")
    pend = sum(1 for r in rows if not r["result"])
    print(f"Pending: {pend}")
    n = c["WIN"] + c["LOSS"]
    rate = f"{c['WIN'] / n:.0%}" if n else "n/a"
    try:
        tg_send(f"Daily test report: {c['WIN']}W / {c['LOSS']}L / {c['TIE']}T, win rate {rate} "
                f"(break-even ~{1 / (1 + PAYOUT):.0%}), pending {pend}. Small samples are mostly noise.")
    except Exception as e:
        print("Telegram failed:", type(e).__name__)


def run_live(once=False):
    last_checked, last_sig = None, last_logged_candle()
    print(f"Watching {ASSET_NAME}. Ctrl+C to stop.")
    while True:
        try:
            now = pd.Timestamp.now(tz="UTC")
            ind = indicators(closed_only(fetch("5d"), now))
            t = ind.index[-1]
            if t != last_checked:
                last_checked = t
                stamp = local(now).strftime("%H:%M:%S")
                if now - (t + CANDLE) > pd.Timedelta(minutes=STALE_MIN):
                    print(f"[{stamp}] No signal: data is stale (market closed or feed delayed)")
                else:
                    res = evaluate(ind, len(ind) - 1)
                    cool = last_sig is not None and (t - last_sig) < COOLDOWN_CANDLES * CANDLE
                    if res["signal"] and not cool:
                        issue(res, ind, now)
                        last_sig = t
                    else:
                        why = "cooldown after last signal" if res["signal"] else res["reason"]
                        print(f"[{stamp}] No signal: {why}")
        except Exception as e:
            print("Error:", e)
        if once:
            return
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--test-notify", action="store_true")
    ap.add_argument("--telegram-setup", action="store_true")
    ap.add_argument("--test-telegram", action="store_true")
    a = ap.parse_args()
    if a.telegram_setup:
        telegram_setup()
    elif a.test_telegram:
        print('Sent!' if tg_send('Aloha168 Signals bot: test message OK') else 'Not sent. Check token and chat ID.')
    elif a.test_notify:
        notify("Signal bot test", "If you can read this, ntfy alerts work.")
    elif a.backtest:
        run_backtest()
    elif a.evaluate:
        run_evaluate()
    else:
        run_live(once=a.once)
