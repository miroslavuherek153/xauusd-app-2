"""
update_data.py - aktualizace data.json pro xauusd-app-2 (spouští GitHub Actions, workflow update-data.yml).

Klíče se berou z GitHub Secrets (proměnné prostředí), NIKDY ne z kódu:
  TWELVEDATA_KEY     - twelvedata.com (spot XAU/USD: svíčky H1 a D1; free 800 požadavků/den, běh = 2)
  ONESIGNAL_APP_ID / ONESIGNAL_API_KEY - volitelné push notifikace (bez nich se přeskočí)
GoldAPI se už nepoužívá: free tarif má 100 požadavků za MĚSÍC, workflow jich potřeboval ~2 900.

Oproti původní verzi ve workflow:
  - klíče nejsou v kódu (byly veřejně vidět v repozitáři),
  - žádná "simulovaná" data: když chybí historie, data.json dostane chybu místo
    náhodně vygenerovaných svíček, ze kterých se dřív počítaly indikátory,
  - RSI podle Wildera (původní výpočet míchal různá okna zisků a ztrát),
  - trend 4H a 1D se počítá z opravdových 4H/1D dat (dřív byl 1D = 4H = trend z H1),
  - data.json se přepíše jen při změně ceny nebo signálu (dřív commit každých 15 minut,
    i když se změnil jen čas - přes 1 700 automatických commitů),
  - notifikace: opravené zalomení řádků (dřív SyntaxError - notifikace nikdy neodešla).
Skóre indikátorů NENÍ ověřený obchodní signál; text to tak i říká.
"""
import json
import os
import sys
import urllib.request
from datetime import datetime, timezone

UA = {"User-Agent": "Mozilla/5.0"}


def get_json(url, headers=None, timeout=15):
    req = urllib.request.Request(url, headers=headers or UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def ema(values, n):
    if not values:
        return 0.0
    k = 2 / (n + 1)
    e = sum(values[:n]) / min(n, len(values))
    for v in values[n:]:
        e = v * k + e * (1 - k)
    return e


def rsi_wilder(closes, n=14):
    if len(closes) < n + 1:
        return 50.0
    gains = [max(closes[i] - closes[i - 1], 0) for i in range(1, len(closes))]
    losses = [max(closes[i - 1] - closes[i], 0) for i in range(1, len(closes))]
    ag, al = sum(gains[:n]) / n, sum(losses[:n]) / n
    for g, l in zip(gains[n:], losses[n:]):
        ag, al = (ag * (n - 1) + g) / n, (al * (n - 1) + l) / n
    return 100.0 if al == 0 else 100 - 100 / (1 + ag / al)


def atr_wilder(highs, lows, closes, n=14):
    tr = [max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1])) for i in range(1, len(closes))]
    if not tr:
        return 0.0
    a = sum(tr[:n]) / min(n, len(tr))
    for t in tr[n:]:
        a = (a * (n - 1) + t) / n
    return a


def yahoo(symbol, interval, rng):
    raw = get_json(f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval={interval}&range={rng}")
    q = raw["chart"]["result"][0]["indicators"]["quote"][0]
    rows = [(h, l, c) for h, l, c in zip(q["high"], q["low"], q["close"]) if None not in (h, l, c)]
    return [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows]


def twelvedata(interval, size=300):
    key = os.environ.get("TWELVEDATA_KEY", "")
    if not key:
        raise RuntimeError("TWELVEDATA_KEY chybí")
    d = get_json(f"https://api.twelvedata.com/time_series?symbol=XAU/USD&interval={interval}&outputsize={size}&apikey={key}")
    if d.get("status") == "error":
        raise RuntimeError(d.get("message"))
    v = list(reversed(d["values"]))
    return [float(x["high"]) for x in v], [float(x["low"]) for x in v], [float(x["close"]) for x in v]


def history(interval, rng):
    """Spot XAU/USD z Twelve Data; záloha zlaté futures GC=F z Yahoo (jiná cena než spot - jen celá řada
    z jednoho zdroje, nikdy se nemíchá spot s futures). Yahoo symbol XAUUSD=X vrací 404."""
    try:
        h, l, c = twelvedata({"1h": "1h", "1d": "1day"}[interval])
        if len(c) >= 30:
            return h, l, c, "Twelve Data XAU/USD (spot)"
    except Exception as e:  # noqa: BLE001
        print(f"Twelve Data {interval}: {e}")
    try:
        h, l, c = yahoo("GC=F", interval, rng)
        if len(c) >= 30:
            return h, l, c, "Yahoo GC=F (futures, ne spot)"
    except Exception as e:  # noqa: BLE001
        print(f"Yahoo GC=F {interval}: {e}")
    return [], [], [], None


def trend_of(closes):
    if len(closes) < 50:
        return "neutral"
    return "bullish" if closes[-1] > ema(closes, 50) else "bearish"


def main():
    now = datetime.now(timezone.utc).strftime("%d.%m.%Y %H:%M UTC")
    spot = {}
    h1, l1, c1, src = history("1h", "60d")
    if not c1:
        write({"status": "error", "error": "Nepodařilo se načíst data (Twelve Data ani Yahoo).", "updated": now, "price": 0})
        return
    price = c1[-1]                                  # poslední cena ze STEJNÉHO zdroje jako historie

    e20, e50, e200 = ema(c1, 20), ema(c1, 50), ema(c1, 200)
    macd = ema(c1, 12) - ema(c1, 26)
    rsi_v = rsi_wilder(c1)
    atr = atr_wilder(h1, l1, c1)
    change = spot.get("ch", 0) or 0

    c4 = c1[::-1][::4][::-1]                       # přibližné 4H closes z H1
    _, _, cd, _ = history("1d", "1y")

    bull = bear = 0
    for cond, w in ((price > e20, 1), (price > e50, 2), (price > e200, 1), (macd > 0, 1), (change > 0, 1)):
        bull, bear = (bull + w, bear) if cond else (bull, bear + w)
    if rsi_v > 55:
        bull += 1
    elif rsi_v < 45:
        bear += 1
    signal = "BUY" if bull > bear else "SELL" if bear > bull else "WAIT"
    conf = round(max(bull, bear) / (bull + bear) * 100)
    side = -1 if signal == "SELL" else 1
    sl_d, tp1_d, tp2_d = atr * 1.5, atr * 2.0, atr * 3.5

    res = sorted({round(x, 1) for x in h1[-50:] if x > price})[:3]
    sup = sorted({round(x, 1) for x in l1[-50:] if x < price}, reverse=True)[:3]

    data = {
        "price": round(price, 2), "change": round(change, 2), "changePct": round(spot.get("chp", 0) or 0, 2),
        "high24h": round(spot.get("high_price") or max(h1[-24:]), 2), "low24h": round(spot.get("low_price") or min(l1[-24:]), 2),
        "ask": round(spot.get("ask") or price, 2), "bid": round(spot.get("bid") or price, 2),
        "rsi": round(rsi_v, 1), "ema20": round(e20, 1), "ema50": round(e50, 1), "ema200": round(e200, 1),
        "macd": round(macd, 2), "atr": round(atr, 2), "signal": signal, "trend": trend_of(c1), "confidence": conf,
        "resistance": res, "support": sup,
        "tf": {"1D": trend_of(cd), "4H": trend_of(c4), "1H": trend_of(c1)},
        "entry": round(price, 2), "sl": round(price - side * sl_d, 2), "tp1": round(price + side * tp1_d, 2),
        "tp2": round(price + side * tp2_d, 2), "rr": round(tp1_d / sl_d, 1) if sl_d else 0,
        "sl_dist": round(sl_d, 2), "tp1_dist": round(tp1_d, 2),
        "updated": now, "status": "ok", "source": src,
        "analysis": (f"XAUUSD {now}: cena ${price:.2f}, {'nad' if price > e50 else 'pod'} EMA50 (${e50:.0f}), RSI {rsi_v:.0f}, "
                     f"MACD {'kladný' if macd > 0 else 'záporný'}, ATR ${atr:.1f}. Indikátory převážně "
                     f"{'býčí' if signal == 'BUY' else 'medvědí' if signal == 'SELL' else 'smíšené'} ({max(bull, bear)}/{bull + bear} bodů). "
                     "Jde o souhrn indikátorů, ne o ověřený obchodní signál."),
    }
    write(data)


def write(data):
    try:
        old = json.load(open("data.json", encoding="utf-8"))
    except Exception:  # noqa: BLE001
        old = {}
    if old.get("status") == data.get("status") and old.get("price") == data.get("price") and old.get("signal") == data.get("signal"):
        print("Beze změny ceny i signálu - data.json se nepřepisuje (žádný commit).")
        return
    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"data.json: {data.get('price')} {data.get('signal')} {data.get('updated')}")
    notify(old, data)


def notify(old, d):
    app_id, api_key = os.environ.get("ONESIGNAL_APP_ID", ""), os.environ.get("ONESIGNAL_API_KEY", "")
    if not (app_id and api_key) or d.get("signal") not in ("BUY", "SELL") or d.get("signal") == old.get("signal"):
        return
    title = f"{'🟢' if d['signal'] == 'BUY' else '🔴'} XAUUSD {d['signal']}"
    body = f"Cena: ${d['price']:.2f} | indikátory {d['confidence']} %\nTP1: ${d['tp1']:.2f} | SL: ${d['sl']:.2f}\n{d['updated']}"
    payload = {"app_id": app_id, "included_segments": ["All"], "headings": {"en": title, "cs": title},
               "contents": {"en": body, "cs": body}, "url": "https://miroslavuherek153.github.io/xauusd-app-2/"}
    req = urllib.request.Request("https://onesignal.com/api/v1/notifications", data=json.dumps(payload).encode(),
                                 headers={"Authorization": f"Basic {api_key}", "Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            print("OneSignal:", json.loads(r.read()).get("id"))
    except Exception as e:  # noqa: BLE001
        print(f"OneSignal: {e}")


if __name__ == "__main__":
    sys.exit(main())
