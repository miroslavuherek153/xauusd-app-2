import os
import json
import requests
import pandas as pd
import yfinance as yf
import ta
from datetime import datetime, timezone

GOLDAPI_KEY = os.environ["GOLDAPI_KEY"]

SYMBOL_YF = "GOLD=X"  # Yahoo Finance symbol
GOLDAPI_URL = "https://www.goldapi.io/api/XAU/USD"


def fetch_goldapi():
    headers = {"x-access-token": GOLDAPI_KEY, "Content-Type": "application/json"}
    r = requests.get(GOLDAPI_URL, headers=headers, timeout=10)
    r.raise_for_status()
    d = r.json()
    return {
        "price": float(d["price"]),
        "open": float(d.get("open_price", d["price"])),
        "high24h": float(d.get("high_price", d["price"])),
        "low24h": float(d.get("low_price", d["price"])),
        "change": float(d.get("ch", 0.0)),
        "changePct": float(d.get("chp", 0.0)),
    }


def fetch_history():
    # 60 dní 1h svíčky
    data = yf.download(SYMBOL_YF, period="60d", interval="1h", auto_adjust=False)
    data = data.dropna()
    return data


def compute_indicators(df: pd.DataFrame):
    close = df["Close"]

    df["ema20"] = ta.trend.EMAIndicator(close, window=20).ema_indicator()
    df["ema50"] = ta.trend.EMAIndicator(close, window=50).ema_indicator()
    df["ema200"] = ta.trend.EMAIndicator(close, window=200).ema_indicator()

    df["rsi"] = ta.momentum.RSIIndicator(close, window=14).rsi()

    macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    atr = ta.volatility.AverageTrueRange(
        high=df["High"], low=df["Low"], close=close, window=14
    )
    df["atr"] = atr.average_true_range()

    return df


def detect_trend(row):
    ema20, ema50, ema200 = row["ema20"], row["ema50"], row["ema200"]
    rsi = row["rsi"]
    macd = row["macd"]

    if ema20 > ema50 > ema200 and rsi > 55 and macd > 0:
        return "bullish"
    if ema20 < ema50 < ema200 and rsi < 45 and macd < 0:
        return "bearish"
    return "neutral"


def build_tf_view(df: pd.DataFrame):
    # jednoduché MTF z posledních svíček
    out = {}
    last = df.iloc[-1]

    out["1H"] = detect_trend(last)

    if len(df) >= 4:
        out["4H"] = detect_trend(df.iloc[-4:].mean(numeric_only=True))
    if len(df) >= 24:
        out["1D"] = detect_trend(df.iloc[-24:].mean(numeric_only=True))
    if len(df) >= 96:
        out["4D"] = detect_trend(df.iloc[-96:].mean(numeric_only=True))

    return out


def find_levels(series: pd.Series, n=3, mode="support"):
    # velmi jednoduchý výběr extrémů
    vals = series.sort_values(ascending=(mode == "resistance"))
    uniq = vals.drop_duplicates().tolist()
    return uniq[:n]


def generate_analysis(trend, signal, rsi, atr, price):
    parts = []

    parts.append(f"Aktuální trend: {trend.upper()}.")
    parts.append(f"RSI je {rsi:.1f}, což naznačuje {'přeprodanost' if rsi < 30 else 'překoupenost' if rsi > 70 else 'neutrální zónu'}.")

    if atr / price > 0.01:
        parts.append("Volatilita je zvýšená (ATR je relativně vysoké vůči ceně).")
    else:
        parts.append("Volatilita je spíše nižší až střední.")

    if signal == "BUY":
        parts.append("Systém aktuálně preferuje nákupní scénář.")
    elif signal == "SELL":
        parts.append("Systém aktuálně preferuje prodejní scénář.")
    else:
        parts.append("Systém doporučuje vyčkat na jasnější signál.")

    return " ".join(parts)


def main():
    live = fetch_goldapi()
    hist = fetch_history()
    hist = compute_indicators(hist)

    last = hist.iloc[-1]

    trend = detect_trend(last)

    price = live["price"]
    atr = float(last["atr"])
    rsi = float(last["rsi"])
    ema20 = float(last["ema20"])
    ema50 = float(last["ema50"])
    ema200 = float(last["ema200"])
    macd_val = float(last["macd"])

    # jednoduchá logika signálu
    signal = "WAIT"
    confidence = 50

    if trend == "bullish" and rsi > 50 and macd_val > 0:
        signal = "BUY"
        confidence = 70
        if rsi > 60:
            confidence = 80
    elif trend == "bearish" and rsi < 50 and macd_val < 0:
        signal = "SELL"
        confidence = 70
        if rsi < 40:
            confidence = 80

    # ATR-based SL/TP
    atr_mult_sl = 1.5
    atr_mult_tp1 = 2.0
    atr_mult_tp2 = 3.5

    if signal == "BUY":
        entry = price
        sl = price - atr_mult_sl * atr
        tp1 = price + atr_mult_tp1 * atr
        tp2 = price + atr_mult_tp2 * atr
    elif signal == "SELL":
        entry = price
        sl = price + atr_mult_sl * atr
        tp1 = price - atr_mult_tp1 * atr
        tp2 = price - atr_mult_tp2 * atr
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
        "ema20": round(ema20, 2),
        "ema50": round(ema50, 2),
        "ema200": round(ema200, 2),
        "macd": round(macd_val, 4),
        "trend": trend,
        "signal": signal,
        "confidence": confidence,
        "entry": round(entry, 2) if entry is not None else None,
        "sl": round(sl, 2) if sl is not None else None,
        "tp1": round(tp1, 2) if tp1 is not None else None,
        "tp2": round(tp2, 2) if tp2 is not None else None,
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
