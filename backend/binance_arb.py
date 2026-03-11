"""
Binance Momentum Arbitrage â exploits Polymarket's slow repricing on BTC 5-min markets.

Strategy: When Binance shows BTC moved >0.3% since a 5-min window opened,
but Polymarket odds haven't caught up, buy the correct side.
These markets resolve based on the same underlying price, so Binance
is essentially an oracle for Polymarket's binary outcome.

Called every loop from main.py (speed critical â these windows are short).
"""

import re
from datetime import datetime, timedelta
from typing import List, Optional

try:
    from binance_feed import get_price, get_status as get_binance_status
except ImportError:
    def get_price(symbol): return 0
    def get_binance_status(): return {}


# ââ Configuration ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

ARB_MOVE_THRESHOLD = 0.003    # 0.3% minimum Binance move to trigger entry
ARB_MAX_POLY_PRICE = 0.88     # Max Polymarket price for our side (12%+ edge required)
ARB_MIN_SECS_LEFT  = 30       # Don't enter with < 30 seconds remaining
ARB_MAX_SECS_LEFT  = 240      # Don't enter too early (wait for directional signal)

# ââ Module State âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

_arb_reference_prices = {}   # {market_id: {"ref_price": float, "window_start": datetime}}
_arb_entered_markets = set()  # Track entered markets to prevent double-entry


# ââ Helpers ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

def _is_btc_5min_market(market: dict) -> bool:
    """Detect BTC 5-minute up/down markets via slug or question text."""
    slug = market.get("slug", "")
    question = market.get("question", "").lower()

    # Slug-based detection (most reliable)
    if "btc" in slug.lower() and ("5m" in slug.lower() or "5min" in slug.lower()):
        return True

    # Question-based detection
    if "btc" in question and ("5 min" in question or "5-min" in question or "5min" in question):
        if "up" in question or "down" in question or "above" in question or "below" in question:
            return True

    return False


def _estimate_seconds_remaining(market: dict) -> float:
    """Estimate seconds until market closes based on end_date."""
    end_date_str = market.get("end_date", "")
    if not end_date_str:
        return 9999.0
    try:
        end_date_str = end_date_str.replace("Z", "+00:00")
        if "T" in end_date_str:
            end_dt = datetime.fromisoformat(end_date_str).replace(tzinfo=None)
        else:
            end_dt = datetime.strptime(end_date_str[:10], "%Y-%m-%d")
        return max(0, (end_dt - datetime.utcnow()).total_seconds())
    except Exception:
        return 9999.0


# ââ Main Signal Generator ââââââââââââââââââââââââââââââââââââââââââââââââââââ

def generate_arb_signals(markets: list) -> list:
    """
    Scan markets for BTC 5-min arbitrage opportunities.

    Called EVERY loop from main.py trading_loop() (speed critical).

    Returns list of signal dicts ready for maybe_enter_trade().
    """
    signals = []
    now = datetime.utcnow()

    # Safety: Binance feed must be live
    btc_price = get_price("BTC")
    if not btc_price or btc_price <= 0:
        return []

    # Check if price data is fresh (< 10 seconds old)
    binance_status = get_binance_status()
    btc_status = binance_status.get("BTC", {})
    last_update = btc_status.get("last_update")
    if last_update:
        try:
            if isinstance(last_update, str):
                last_dt = datetime.fromisoformat(last_update)
            else:
                last_dt = last_update
            if hasattr(last_dt, 'tzinfo') and last_dt.tzinfo:
                last_dt = last_dt.replace(tzinfo=None)
            age_seconds = (now - last_dt).total_seconds()
            if age_seconds > 10:
                return []  # Stale data
        except Exception:
            pass  # If we can't check freshness, proceed cautiously

    # Filter for BTC 5-min markets
    btc_5min_markets = [m for m in markets if _is_btc_5min_market(m)]

    for market in btc_5min_markets:
        market_id = market["id"]

        # Track new markets with reference price
        if market_id not in _arb_reference_prices:
            _arb_reference_prices[market_id] = {
                "ref_price": btc_price,
                "window_start": now,
            }
            continue  # Don't signal on first sight â need to observe movement

        ref_data = _arb_reference_prices[market_id]
        ref_price = ref_data["ref_price"]

        # Skip if already entered
        if market_id in _arb_entered_markets:
            continue

        # Calculate BTC move since window opened
        move = (btc_price / ref_price) - 1

        # Not enough signal
        if abs(move) < ARB_MOVE_THRESHOLD:
            continue

        # Determine direction: BTC up -> YES, BTC down -> NO
        direction = "YES" if move > 0 else "NO"
        yes_price = market.get("yes_price", 0.5)

        # Get Polymarket price for OUR side
        poly_price = yes_price if direction == "YES" else (1 - yes_price)

        # Check if there's still an edge (Polymarket hasn't caught up)
        if poly_price > ARB_MAX_POLY_PRICE:
            continue  # Market already repriced â no edge

        # Check timing window
        secs_left = _estimate_seconds_remaining(market)
        if secs_left < ARB_MIN_SECS_LEFT or secs_left > ARB_MAX_SECS_LEFT:
            continue

        # Calculate edge
        edge = (0.50 + abs(move) * 50) - poly_price
        if edge < 0.05:
            continue  # Need at least 5% edge

        # ââ Score the signal âââââââââââââââââââââââââââââââââââââââââââââ
        score = 80
        if abs(move) > 0.005:
            score += 10  # Strong move (0.5%+)
        if abs(move) > 0.008:
            score += 5   # Very strong (0.8%+)
        if edge > 0.20:
            score += 5   # Book very slow to reprice
        score = min(99, max(75, score))

        # ââ Build signal âââââââââââââââââââââââââââââââââââââââââââââââââ
        signal = {
            "market_id": market_id,
            "market_question": market.get("question", ""),
            "score": score,
            "confidence": min(0.95, 0.50 + abs(move) * 50),
            "direction": direction,
            "yes_price": yes_price,
            "market_type": "BINANCE_ARB",
            "can_enter": True,
            "entry_reason": (
                f"ARB: BTC {move:+.2%} from window open "
                f"(ref ${ref_price:,.0f} -> ${btc_price:,.0f}), "
                f"Poly {direction}@{poly_price:.2f}, edge={edge:.0%}"
            ),
            "factors_json": {
                "reference_price": ref_price,
                "current_price": btc_price,
                "move_pct": round(move, 6),
                "polymarket_price": round(poly_price, 4),
                "edge_pct": round(edge, 4),
                "seconds_remaining": round(secs_left, 0),
                "asset": "BTC",
                "timeframe_minutes": 5,
            },
            "created_at": now.isoformat(),
            "clob_token_ids": market.get("clob_token_ids", []),
            "condition_id": market.get("condition_id", ""),
            "liquidity": market.get("liquidity", 0),
        }
        signals.append(signal)

        print(
            f"[ARB] Signal: BTC {move:+.2%} | {direction}@{poly_price:.2f} "
            f"edge={edge:.0%} | secs_left={secs_left:.0f} | "
            f"'{market.get('question', '')[:40]}'"
        )

    # ââ Clean up expired markets (older than 6 minutes) ââââââââââââââââââ
    expired = []
    for mid, data in _arb_reference_prices.items():
        age = (now - data["window_start"]).total_seconds()
        if age > 360:  # 6 minutes
            expired.append(mid)
    for mid in expired:
        del _arb_reference_prices[mid]
        _arb_entered_markets.discard(mid)

    return signals
