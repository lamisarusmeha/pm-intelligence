"""
Strategy 4: Short-Duration 5m/15m Market Trading

Targets Polymarket rolling "Bitcoin/ETH/SOL Up or Down" markets that resolve
every 5 or 15 minutes. Enters when one side hits 80%+ near expiry.

These markets use slug patterns like:
  btc-updown-5m-{unix_timestamp}
  eth-updown-15m-{unix_timestamp}

This strategy should produce the highest trade volume â potentially dozens
of trades per hour instead of a few per day.
"""

import json
import math
import time
import re
from datetime import datetime, timedelta
from typing import Optional

try:
    from binance_feed import get_price, get_change
except ImportError:
    def get_price(s): return 0
    def get_change(s, m): return 0


# ââ Configuration ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

# Markets must have YES or NO price >= this to qualify
MIN_CONFIDENCE_PRICE = 0.75

# Maximum entry price (don't buy at 0.99 â no upside)
MAX_ENTRY_PRICE = 0.95

# Minimum liquidity to enter
MIN_LIQUIDITY = 500

# How close to expiry (in seconds) before we consider entering
# For 5m markets: enter in last 120s; for 15m: enter in last 300s
ENTRY_WINDOW_5M  = 180   # last 3 minutes of a 5-minute market
ENTRY_WINDOW_15M = 360   # last 6 minutes of a 15-minute market

# Binance price confirmation threshold
PRICE_CONFIRM_PCT = 0.001  # 0.1% â if Binance agrees with direction

# Slug patterns for short-duration markets
SHORT_DURATION_PATTERNS = [
    # Pattern: (regex, asset, timeframe_minutes)
    (re.compile(r"btc-updown-5m-(\d+)", re.IGNORECASE), "BTC", 5),
    (re.compile(r"btc-updown-15m-(\d+)", re.IGNORECASE), "BTC", 15),
    (re.compile(r"eth-updown-5m-(\d+)", re.IGNORECASE), "ETH", 5),
    (re.compile(r"eth-updown-15m-(\d+)", re.IGNORECASE), "ETH", 15),
    (re.compile(r"sol-updown-5m-(\d+)", re.IGNORECASE), "SOL", 5),
    (re.compile(r"sol-updown-15m-(\d+)", re.IGNORECASE), "SOL", 15),
]

# Also match by question text patterns
QUESTION_PATTERNS = [
    (re.compile(r"bitcoin\s+up\s+or\s+down.*?(\d+):(\d+)\s*(am|pm)", re.IGNORECASE), "BTC"),
    (re.compile(r"btc\s+up\s+or\s+down", re.IGNORECASE), "BTC"),
    (re.compile(r"ethereum\s+up\s+or\s+down", re.IGNORECASE), "ETH"),
    (re.compile(r"eth\s+up\s+or\s+down", re.IGNORECASE), "ETH"),
    (re.compile(r"solana\s+up\s+or\s+down", re.IGNORECASE), "SOL"),
    (re.compile(r"sol\s+up\s+or\s+down", re.IGNORECASE), "SOL"),
]


def _parse_short_duration_market(market: dict) -> Optional[dict]:
    """
    Check if a market is a short-duration up/down market.
    Returns parsed info or None.
    """
    slug = (market.get("slug") or "").lower()
    question = market.get("question", "")
    end_date = market.get("end_date", "")

    # Try slug-based detection first (most reliable)
    for pattern, asset, tf_minutes in SHORT_DURATION_PATTERNS:
        match = pattern.search(slug)
        if match:
            try:
                resolution_ts = int(match.group(1))
                return {
                    "asset": asset,
                    "timeframe_minutes": tf_minutes,
                    "resolution_timestamp": resolution_ts,
                    "source": "slug",
                }
            except (ValueError, IndexError):
                continue

    # Try question-based detection
    for pattern, asset in QUESTION_PATTERNS:
        if pattern.search(question):
            # Determine timeframe from end_date proximity
            if end_date:
                try:
                    ed = end_date.replace("Z", "+00:00")
                    if "T" in ed:
                        end_dt = datetime.fromisoformat(ed).replace(tzinfo=None)
                    else:
                        end_dt = datetime.strptime(ed[:10], "%Y-%m-%d")
                    minutes_left = (end_dt - datetime.utcnow()).total_seconds() / 60

                    # Only match if it resolves within 60 minutes
                    if minutes_left <= 60:
                        tf = 5 if minutes_left <= 10 else 15
                        return {
                            "asset": asset,
                            "timeframe_minutes": tf,
                            "resolution_timestamp": int(end_dt.timestamp()),
                            "source": "question",
                        }
                except Exception:
                    pass

    return None


def _seconds_until_resolution(parsed: dict) -> float:
    """How many seconds until this market resolves."""
    res_ts = parsed.get("resolution_timestamp", 0)
    if res_ts <= 0:
        return 9999
    return max(0, res_ts - time.time())


def _is_in_entry_window(parsed: dict) -> bool:
    """Check if we're in the entry window (close enough to expiry)."""
    secs_left = _seconds_until_resolution(parsed)
    tf = parsed.get("timeframe_minutes", 5)

    if tf <= 5:
        return secs_left <= ENTRY_WINDOW_5M and secs_left > 10  # Don't enter last 10s
    elif tf <= 15:
        return secs_left <= ENTRY_WINDOW_15M and secs_left > 15
    else:
        return secs_left <= 600 and secs_left > 20  # 60m markets: last 10 min


def _get_binance_direction(asset: str, tf_minutes: int) -> Optional[str]:
    """
    Check Binance price movement to confirm direction.
    Returns "UP" or "DOWN" or None if no clear signal.
    """
    price = get_price(asset)
    if price <= 0:
        return None

    change = get_change(asset, tf_minutes)
    if change is None:
        return None

    if change > PRICE_CONFIRM_PCT:
        return "UP"
    elif change < -PRICE_CONFIRM_PCT:
        return "DOWN"
    return None


def generate_short_duration_signals(markets: list) -> list:
    """
    Scan markets for short-duration up/down trading opportunities.

    Logic:
    1. Find 5m/15m up/down markets
    2. Check if we're in the entry window (near expiry)
    3. If one side is 75%+, that's a near-certainty at this timeframe
    4. Cross-check with Binance price direction
    5. Generate signal for paper_trader
    """
    signals = []

    for market in markets:
        try:
            parsed = _parse_short_duration_market(market)
            if not parsed:
                continue

            # Must be in entry window
            if not _is_in_entry_window(parsed):
                continue

            yes_price = market.get("yes_price", 0.5)
            no_price = 1 - yes_price
            liquidity = market.get("liquidity", 0) or 0
            secs_left = _seconds_until_resolution(parsed)

            # Minimum liquidity check
            if liquidity < MIN_LIQUIDITY:
                continue

            # Determine direction and entry price
            direction = None
            entry_price = 0

            if yes_price >= MIN_CONFIDENCE_PRICE:
                direction = "YES"
                entry_price = yes_price
            elif no_price >= MIN_CONFIDENCE_PRICE:
                direction = "NO"
                entry_price = no_price

            if not direction:
                continue

            # Max price guard
            if entry_price > MAX_ENTRY_PRICE or entry_price < 0.05:
                continue

            # Binance cross-check
            asset = parsed["asset"]
            tf = parsed["timeframe_minutes"]
            binance_dir = _get_binance_direction(asset, tf)

            # Determine if Binance confirms
            binance_confirms = False
            if binance_dir:
                question_lower = market.get("question", "").lower()
                is_up_market = "up" in question_lower and direction == "YES"
                is_down_market = "down" in question_lower and direction == "YES"

                if is_up_market and binance_dir == "UP":
                    binance_confirms = True
                elif is_down_market and binance_dir == "DOWN":
                    binance_confirms = True
                elif direction == "NO":
                    # Betting NO on up when Binance says DOWN (or vice versa)
                    if "up" in question_lower and binance_dir == "DOWN":
                        binance_confirms = True
                    elif "down" in question_lower and binance_dir == "UP":
                        binance_confirms = True

            # Score calculation
            # Base: entry_price strength (75% = 70pts, 95% = 90pts)
            score = int(70 + (entry_price - 0.75) * 100)

            # Binance confirmation bonus
            if binance_confirms:
                score += 8

            # Time pressure bonus (closer to expiry = more certain)
            if secs_left < 60:
                score += 5
            elif secs_left < 120:
                score += 3

            # Liquidity bonus
            if liquidity > 5000:
                score += 3
            elif liquidity > 2000:
                score += 1

            score = min(99, max(65, score))

            signal = {
                "market_id": market.get("id", ""),
                "market_question": market.get("question", ""),
                "score": score,
                "confidence": entry_price,
                "direction": direction,
                "yes_price": yes_price,
                "market_type": "SHORT_DURATION",
                "can_enter": True,
                "entry_reason": (
                    f"SHORT_{tf}m: {direction}@{entry_price:.2f}, "
                    f"{secs_left:.0f}s left, {asset}, "
                    f"binance={'YES' if binance_confirms else 'NO'}"
                ),
                "factors_json": json.dumps({
                    "asset": asset,
                    "timeframe_minutes": tf,
                    "entry_price": entry_price,
                    "seconds_left": round(secs_left, 1),
                    "binance_confirms": binance_confirms,
                    "binance_direction": binance_dir,
                    "liquidity": liquidity,
                }),
                "created_at": datetime.utcnow().isoformat(),
                "clob_token_ids": market.get("clob_token_ids", []),
                "condition_id": market.get("condition_id", ""),
                "liquidity": liquidity,
            }
            signals.append(signal)

            print(
                f"[SHORT] Signal: {asset} {tf}m {direction}@{entry_price:.2f} "
                f"({secs_left:.0f}s left) "
                f"binance={'confirmed' if binance_confirms else 'unconfirmed'} "
                f"score={score}"
            )

        except Exception as e:
            continue

    if signals:
        print(f"[SHORT] Generated {len(signals)} short-duration signals")
    return signals
