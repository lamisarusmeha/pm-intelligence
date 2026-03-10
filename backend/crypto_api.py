"""
Crypto Market Data — Kraken public REST API.
No API key required for public endpoints.
Fetches OHLCV, live prices, and order book depth for BTC, ETH, SOL, BNB, AVAX.

Switched from Binance (geo-restricted) to Kraken which is globally accessible.
"""

import httpx
import asyncio
from typing import Optional, List, Dict

KRAKEN_BASE = "https://api.kraken.com/0/public"
TIMEOUT      = 15

# Kraken pair names mapped from display symbols
KRAKEN_PAIRS = {
    "BTCUSDT":  "XBTUSD",
    "ETHUSDT":  "ETHUSD",
    "SOLUSDT":  "SOLUSD",
    "BNBUSDT":  "BNBUSD",
    "AVAXUSDT": "AVAXUSD",
}

DISPLAY_NAMES = {
    "BTCUSDT":  "BTC",
    "ETHUSDT":  "ETH",
    "SOLUSDT":  "SOL",
    "BNBUSDT":  "BNB",
    "AVAXUSDT": "AVAX",
}

TRACKED_SYMBOLS = list(KRAKEN_PAIRS.keys())


async def fetch_ticker(kraken_pair: str, client: httpx.AsyncClient) -> Optional[Dict]:
    """
    Returns current price, 24h high/low, volume, open via Kraken Ticker.
    Kraken ticker fields: c=last, h=high, l=low, v=volume, o=open
    """
    try:
        r = await client.get(
            f"{KRAKEN_BASE}/Ticker",
            params={"pair": kraken_pair},
            timeout=TIMEOUT
        )
        data = r.json()
        if data.get("error"):
            return None
        result = data.get("result", {})
        if not result:
            return None
        # Kraken may return a different key (e.g. XBTUSD → XXBTZUSD)
        info = list(result.values())[0]
        current_price = float(info["c"][0])
        open_price    = float(info["o"])
        high_24h      = float(info["h"][1])
        low_24h       = float(info["l"][1])
        volume_24h    = float(info["v"][1])
        price_change_pct = round(((current_price - open_price) / open_price) * 100, 4) if open_price else 0
        return {
            "price":            current_price,
            "price_change_pct": price_change_pct,
            "volume_24h":       volume_24h,
            "high_24h":         high_24h,
            "low_24h":          low_24h,
        }
    except Exception as e:
        print(f"[CRYPTO API] Ticker error for {kraken_pair}: {e}")
        return None


async def fetch_klines(kraken_pair: str, interval: int = 5, limit: int = 60,
                       client: Optional[httpx.AsyncClient] = None) -> List[Dict]:
    """
    Returns list of OHLCV dicts from Kraken OHLC endpoint.
    Kraken OHLC format: [time, open, high, low, close, vwap, volume, count]
    interval: minutes (1, 5, 15, 30, 60, 240, 1440, 10080, 21600)
    """
    async def _fetch(c: httpx.AsyncClient):
        r = await c.get(
            f"{KRAKEN_BASE}/OHLC",
            params={"pair": kraken_pair, "interval": interval},
            timeout=TIMEOUT
        )
        data = r.json()
        if data.get("error"):
            raise ValueError(f"Kraken OHLC error: {data['error']}")
        result = data.get("result", {})
        if not result:
            raise ValueError("Empty OHLC result")
        # Get the actual pair key (may differ from requested name)
        pair_key = next(k for k in result if k != "last")
        raw = result[pair_key]
        # Return last `limit` candles
        raw = raw[-limit:]
        return [
            {
                "ts":     k[0] * 1000,      # convert to ms
                "open":   float(k[1]),
                "high":   float(k[2]),
                "low":    float(k[3]),
                "close":  float(k[4]),
                "volume": float(k[6]),       # k[5]=vwap, k[6]=volume
            }
            for k in raw
        ]

    if client:
        return await _fetch(client)
    async with httpx.AsyncClient() as c:
        return await _fetch(c)


async def fetch_orderbook_imbalance(kraken_pair: str, client: httpx.AsyncClient) -> float:
    """
    Returns bid_vol / (bid_vol + ask_vol) for top 10 levels via Kraken Depth.
    > 0.55 = buy pressure, < 0.45 = sell pressure.
    """
    try:
        r = await client.get(
            f"{KRAKEN_BASE}/Depth",
            params={"pair": kraken_pair, "count": 10},
            timeout=TIMEOUT
        )
        data = r.json()
        if data.get("error"):
            return 0.5
        result = data.get("result", {})
        if not result:
            return 0.5
        book    = list(result.values())[0]
        bid_vol = sum(float(b[1]) for b in book["bids"])
        ask_vol = sum(float(a[1]) for a in book["asks"])
        total   = bid_vol + ask_vol
        return round(bid_vol / total, 4) if total > 0 else 0.5
    except Exception:
        return 0.5


async def fetch_all_crypto_data() -> List[Dict]:
    """
    Fetches full market snapshot for all tracked symbols:
    price, klines (5m × 60), order book imbalance, 24h stats.
    Returns list of market dicts ready for the feature engine.
    """
    results = []
    async with httpx.AsyncClient() as client:
        tasks = [_fetch_symbol(sym, client) for sym in TRACKED_SYMBOLS]
        snapshots = await asyncio.gather(*tasks, return_exceptions=True)
        for sym, snap in zip(TRACKED_SYMBOLS, snapshots):
            if isinstance(snap, Exception) or snap is None:
                print(f"[CRYPTO API] Skipping {sym}: {snap}")
                continue
            results.append(snap)
    return results


async def _fetch_symbol(symbol: str, client: httpx.AsyncClient) -> Optional[Dict]:
    kraken_pair = KRAKEN_PAIRS.get(symbol)
    if not kraken_pair:
        return None
    try:
        ticker_task = fetch_ticker(kraken_pair, client)
        klines_task = fetch_klines(kraken_pair, 5, 60, client)
        book_task   = fetch_orderbook_imbalance(kraken_pair, client)

        ticker, klines, book_imbalance = await asyncio.gather(
            ticker_task, klines_task, book_task
        )

        if not ticker or not klines:
            return None

        return {
            "symbol":           symbol,
            "display":          DISPLAY_NAMES.get(symbol, symbol),
            "price":            ticker["price"],
            "klines_5m":        klines,
            "book_imbalance":   book_imbalance,
            "price_change_pct": ticker["price_change_pct"],
            "volume_24h_usd":   ticker["volume_24h"] * ticker["price"],  # convert to USD
            "high_24h":         ticker["high_24h"],
            "low_24h":          ticker["low_24h"],
        }
    except Exception as e:
        print(f"[CRYPTO API] Error fetching {symbol}: {e}")
        return None
