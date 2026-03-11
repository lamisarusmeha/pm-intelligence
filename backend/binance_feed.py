"""
Binance WebSocket Feed â Real-time BTC, ETH, SOL prices.

Provides global `binance_prices` dict updated every ~1 second.
Used by Strategy 1 (Near-Certainty Grinder) and Strategy 3 (Binance Arb).

Auto-reconnects with exponential backoff. Non-fatal if disconnected.
"""

import asyncio
import json
import time
from collections import deque

# Global state â updated by WebSocket, read by strategies
binance_prices = {
    "BTC": {"price": 0.0, "timestamp": 0, "prices_5m": deque(maxlen=300), "prices_15m": deque(maxlen=900)},
    "ETH": {"price": 0.0, "timestamp": 0, "prices_5m": deque(maxlen=300), "prices_15m": deque(maxlen=900)},
    "SOL": {"price": 0.0, "timestamp": 0, "prices_5m": deque(maxlen=300), "prices_15m": deque(maxlen=900)},
}

# Binance WebSocket stream URL (combined streams)
BINANCE_WS_URL = "wss://stream.binance.com:9443/ws"
STREAMS = ["btcusdt@trade", "ethusdt@trade", "solusdt@trade"]
COMBINED_URL = f"wss://stream.binance.com:9443/stream?streams={'/'.join(STREAMS)}"

SYMBOL_MAP = {
    "BTCUSDT": "BTC",
    "btcusdt": "BTC",
    "ETHUSDT": "ETH",
    "ethusdt": "ETH",
    "SOLUSDT": "SOL",
    "solusdt": "SOL",
}


def get_price(symbol: str) -> float:
    """Get current price for BTC, ETH, or SOL."""
    return binance_prices.get(symbol, {}).get("price", 0.0)


def get_change(symbol: str, minutes: int = 5) -> float:
    """Get price change over last N minutes as a decimal (e.g., 0.02 = 2% up)."""
    data = binance_prices.get(symbol)
    if not data:
        return 0.0

    key = "prices_5m" if minutes <= 5 else "prices_15m"
    prices = data.get(key, deque())
    if len(prices) < 10:
        return 0.0  # Not enough data yet

    current = data["price"]
    # Get price from N minutes ago (approximately)
    target_idx = min(len(prices) - 1, minutes * 60)  # ~1 price per second
    old_price = prices[-target_idx] if target_idx < len(prices) else prices[0]

    if old_price <= 0:
        return 0.0
    return (current - old_price) / old_price


def get_status() -> dict:
    """Get feed status for debugging."""
    now = time.time()
    return {
        symbol: {
            "price": data["price"],
            "age_seconds": round(now - data["timestamp"], 1) if data["timestamp"] > 0 else -1,
            "samples_5m": len(data["prices_5m"]),
            "change_5m": round(get_change(symbol, 5) * 100, 2),
            "change_15m": round(get_change(symbol, 15) * 100, 2),
        }
        for symbol, data in binance_prices.items()
    }


async def binance_websocket_loop():
    """
    Maintain persistent WebSocket connection to Binance.
    Updates global binance_prices every trade tick (~1/second after throttling).
    Auto-reconnects with exponential backoff.
    """
    backoff = 1
    last_update = {"BTC": 0, "ETH": 0, "SOL": 0}
    throttle_ms = 500  # Only update every 500ms per symbol to avoid spam

    while True:
        try:
            import websockets
            print(f"[BINANCE] Connecting to {COMBINED_URL[:60]}...")

            async with websockets.connect(COMBINED_URL, ping_interval=20, ping_timeout=10) as ws:
                backoff = 1  # Reset on successful connection
                print("[BINANCE] Connected! Streaming BTC, ETH, SOL prices.")

                async for raw_msg in ws:
                    try:
                        msg = json.loads(raw_msg)
                        # Combined stream format: {"stream": "btcusdt@trade", "data": {...}}
                        data = msg.get("data", msg)
                        raw_symbol = data.get("s", "").upper()
                        symbol = SYMBOL_MAP.get(raw_symbol) or SYMBOL_MAP.get(raw_symbol.lower())

                        if not symbol:
                            continue

                        price = float(data.get("p", 0))
                        if price <= 0:
                            continue

                        now = time.time()
                        now_ms = int(now * 1000)

                        # Throttle: only update every 500ms per symbol
                        if now_ms - last_update.get(symbol, 0) < throttle_ms:
                            continue
                        last_update[symbol] = now_ms

                        # Update global state
                        entry = binance_prices[symbol]
                        entry["price"] = price
                        entry["timestamp"] = now
                        entry["prices_5m"].append(price)
                        entry["prices_15m"].append(price)

                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue

        except ImportError:
            print("[BINANCE] websockets package not installed â Binance feed disabled")
            print("[BINANCE] Install with: pip install websockets")
            return  # Don't retry if package missing

        except Exception as e:
            print(f"[BINANCE] Disconnected: {e}")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)
            print(f"[BINANCE] Reconnecting (backoff={backoff}s)...")
