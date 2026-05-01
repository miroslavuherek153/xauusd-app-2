import os
import json
import requests
import pandas as pd
import ta
from datetime import datetime, timezone

GOLDAPI_KEY = os.environ["GOLDAPI_KEY"]
TWELVEDATA_KEY = os.environ["TWELVEDATA_KEY"]

GOLDAPI_LIVE = "https://www.goldapi.io/api/XAU/USD"
TWELVEDATA_URL = (
    "https://api.twelvedata.com/time_series"
    "?symbol=XAU/USD&interval=1h&outputsize=5000&apikey={key}"
)


def fetch_goldapi_live():
    headers = {"x-access-token": GOLDAPI_KEY}
    r = requests.get(GOLDAPI_LIVE, headers=headers, timeout=10)
    r.raise_for_status()
    d = r.json()
    return {
        "price": float(d["price"]),
        "high24h": float(d.get("high_price", d["price"])),
        "low24h": float(d.get("low_price", d["price"])),
        "change": float(d.get("ch", 0.0)),
        "changePct": float(d.get("chp", 0.0)),
    }


def fetch_history():
    url = TWELVEDATA_URL.format(key=TWELVEDATA_KEY)
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    data = r.json()

    if "values" not in data:
        raise RuntimeError(f"TwelveData error: {data}")

    df = pd.DataFrame(data["values"])

    # čas
    df["time"] = pd.to_datetime(df["datetime"])

    # přejmenování OHLC
    df = df.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
        }
    )

    # převod jen OHLC na float
    df["Open"] = df["Open"].astype(float)
    df["High"] = df["High"].astype(float)
    df["Low"] = df["Low"].astype(float)
    df["Close"] = df["Close"].astype(float)

    df = df[["time", "Open", "High", "Low", "Close"]]
    df = df.sort_values("time").reset_index(drop=True)
    return df


def compute_indicators(df):
    close = df["Close"]

    df["ema20"] = ta.trend.EMAIndicator(close, window=20).ema_indicator()
    df["ema50"] = ta.trend.EMAIndicator(close, window=50).ema_indicator()
    df["ema200"] = ta.trend.EMAIndicator(close, window=200).ema_indicator()

    df["rsi"] = ta.momentum.RSIIndicator(close, window=14).rsi()

    macd = ta.trend.MACD(close)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()

    atr = ta.volatility.AverageTrueRange(
        high=df["High"], low=df["Low"], close=close, window=14
    )
    df["atr"] = atr.average_true_range()

    return df


def detect_trend(row):
    if row["ema20"] > row["ema50"] > row["ema200"] and row["rsi"] > 55 and row["macd"] > 0:
        return "bullish"
    if row["ema20"] < row["ema50"] < row["ema200"] and row["rsi"] < 45 and row["macd"] < 0:
        return "bearish"
    return "neutral"


def build_tf_view(df):
    out = {}
    last = df.iloc[-1]

    out["1H"] = detect_trend(last)
    if len(df) >= 4:
        out["4H"] = detect_trend(df.iloc[-4:].mean(numeric_only=True))
    if len(df) >= 24:
        out["1D"] = detect_trend(df.iloc[-24:].mean(numeric_only=True))

    return out


def find_levels(series, n=3, mode="support"):
    vals = series.sort_values(ascending=(mode == "resistance"))
    uniq = vals.drop_duplicates().tolist()
    return uniq[:n]


def generate_analysis(trend, signal, rsi, atr, price):
    parts = []
    parts.append(f"Aktuální trend: {trend.upper()}.")
    parts.append(
        f"RSI je {rsi:.1f}, což naznačuje "
        f"{'přeprodanost' if rsi < 30 else 'překoupenost' if rsi > 70 else 'neutrální zónu'}."
    )

    if atr / price > 0.01:
        parts.append("Volatilita je zvýšená.")
    else:
        parts.append("Volatilita je nízká až střední.")

    if signal == "BUY":
        parts.append("Systém preferuje nákup.")
    elif signal == "SELL":
        parts.append("Systém preferuje prodej.")
    else:
        parts.append("Systém doporučuje vyčkat.")

    return " ".join(parts)


def main():
    live = fetch_goldapi_live()
    hist = fetch_history()
    hist = compute_indicators(hist)

    last = hist.iloc[-1]

    trend = detect_trend(last)
    price = live["price"]
    rsi = float(last["rsi"])
    atr = float(last["atr"])
    macd_val = float(last["macd"])

    signal = "WAIT"
    confidence = 50

    if trend == "bullish" and rsi > 50 and macd_val > 0:
        signal = "BUY"
        confidence = 75
    elif trend == "bearish" and rsi < 50 and macd_val < 0:
        signal = "SELL"
        confidence = 75

    atr_sl = 1.5
    atr_tp1 = 2.0
    atr_tp2 = 3.5

    if signal == "BUY":
        entry = price
        sl = price - atr_sl * atr
        tp1 = price + atr_tp1 * atr
        tp2 = price + atr_tp2 * atr
    elif signal == "SELL":
        entry = price
        sl = price + atr_sl * atr
        tp1 = price - atr_tp1 * atr
        tp2 = price - atr_tp2 * atr
    else:
        entry = price
        sl = None
        tp1 = None
        tp2 = None

    tf_view = build_tf_view(hist)
    support = find_levels(hist["Low"], n=3, mode="support")
    resistance = find_levels(hist["High"], n=3, mode="resistance")

    updated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    analysis = generate_analysis(trend, signal, rsi, atr, price)

    data = {
        "price": round(price, 2),
        "change": round(live["change"], 2),
        "changePct": round(live["changePct"], 2),
        "high24h": round(live["high24h"], 2),
        "low24h": round(live["low24h"], 2),
        "rsi": round(rsi, 2),
        "ema20": round(float(last["ema20"]), 2),
        "ema50": round(float(last["ema50"]), 2),
        "ema200": round(float(last["ema200"]), 2),
        "macd": round(macd_val, 4),
        "trend": trend,
        "signal": signal,
        "confidence": confidence,
        "entry": round(entry, 2),
        "sl": round(sl, 2) if sl else None,
        "tp1": round(tp1, 2) if tp1 else None,
        "tp2": round(tp2, 2) if tp2 else None,
        "tf": tf_view,
        "support": [round(x, 2) for x in support],
        "resistance": [round(x, 2) for x in resistance],
        "analysis": analysis,
        "updated": updated,
    }

    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
