"""
Strategy 4: Short-Duration 5m/15m Market Trading (v4.0)

Targets Polymarket rolling "Bitcoin/ETH/SOL Up or Down" markets that resolve
every 5 or 15 minutes. Enters when one side hits 80%+ near expiry.

V4.0 CHANGES:
- Entry window tightened: 60s for 5m markets (was 180s), 120s for 15m (was 360s)
- Confidence threshold raised: 80% (was 70%)
- Binance confirmation REQUIRED â skip trade entirely if unconfirmed
- Range market blacklist ("between", "range")
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
# V4.0: Raised from 0.70 to 0.80 â only enter when 80%+ one side
MIN_CONFIDENCE_PRICE = 0.80

# Maximum entry price (don't buy at 0.99 â no upside)
MAX_ENTRY_PRICE = 0.93

# Minimum liquidity to enter
MIN_LIQUIDITY = 100

# V4.0: Tightened entry windows â enter much closer to resolution
# For 5m markets: enter in last 60s (was 180s)
# For 15m markets: enter in last 120s (was 360s)
ENTRY_WINDOW_5M  = 60    # last 60 seconds of a 5-minute market
ENTRY_WINDOW_15M = 120   # last 2 minutes of a 15-minute market

# Binance price confirmation threshold
PRICE_CONFIRM_PCT = 0.001  # 0.1% â if Binance agrees with direction

# V4.0: Range market blacklist â these caused catastrophic losses
RANGE_BLACKLIST_WORDS = ("between", "be between", "range")

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
    (re.compile(r"bitcoin\s+up\s+or\s+down", re.IGNORECASE), "BTC"),
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
            # Infer timeframe from question text
            time_range_match = re.search(
                r"(\d{1,2}):?(\d{2})?\s*(am|pm)\s*-\s*(\d{1,2}):?(\d{2})?\s*(am|pm)",
                question, re.IGNORECASE
            )
            hourly_match = re.search(
                r"(\d{1,2})\s*(am|pm)\s+et",
                question, re.IGNORECASE
            )

            tf = None
            if time_range_match:
                h1 = int(time_range_match.group(1))
                m1 = int(time_range_match.group(2) or 0)
                ap1 = time_range_match.group(3).upper()
                h2 = int(time_range_match.group(4))
                m2 = int(time_range_match.group(5) or 0)
                ap2 = time_range_match.group(6).upper()
                if ap1 == "PM" and h1 != 12: h1 += 12
                if ap1 == "AM" and h1 == 12: h1 = 0
                if ap2 == "PM" and h2 != 12: h2 += 12
                if ap2 == "AM" and h2 == 12: h2 = 0
                span_minutes = (h2 * 60 + m2) - (h1 * 60 + m1)
                if span_minutes < 0: span_minutes += 24 * 60
                if span_minutes <= 5:
                    tf = 5
                elif span_minutes <= 15:
                    tf = 15
                elif span_minutes <= 60:
                    tf = 60
            elif hourly_match:
                tf = 60

            if tf is None:
                continue

            if end_date:
                try:
                    ed = end_date.replace("Z", "+00:00")
                    if "T" in ed:
                        end_dt = datetime.fromisoformat(ed).replace(tzinfo=None)
                    else:
                        end_dt = datetime.strptime(ed[:10], "%Y-%m-%d")
                    minutes_left = (end_dt - datetime.utcnow()).total_seconds() / 60

                    if minutes_left <= 120 and minutes_left > -5:
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
        return secs_left <= ENTRY_WINDOW_5M and secs_left > 10
    elif tf <= 15:
        return secs_left <= ENTRY_WINDOW_15M and secs_left > 15
    elif tf <= 60:
        return secs_left <= 1200 and secs_left > 30  # Hourly: last 20 min
    else:
        return secs_left <= 1800 and secs_left > 60


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

    V4.0 Logic:
    1. Find 5m/15m up/down markets
    2. Check if we're in the entry window (last 60s for 5m, 120s for 15m)
    3. If one side is 80%+, that's a near-certainty at this timeframe
    4. REQUIRE Binance price direction confirmation (skip if unconfirmed)
    5. Blacklist range/between markets
    6. Generate signal for paper_trader
    """
    signals = []

    for market in markets:
        try:
            question = market.get("question", "")
            question_lower = question.lower()

            # V4.0: Range market blacklist
            if any(word in question_lower for word in RANGE_BLACKLIST_WORDS):
                continue

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
                is_up_market = "up" in question_lower and direction == "YES"
                is_down_market = "down" in question_lower and direction == "YES"

                if is_up_market and binance_dir == "UP":
                    binance_confirms = True
                elif is_down_market and binance_dir == "DOWN":
                    binance_confirms = True
                elif direction == "NO":
                    if "up" in question_lower and binance_dir == "DOWN":
                        binance_confirms = True
                    elif "down" in question_lower and binance_dir == "UP":
                        binance_confirms = True

            # V4.0: REQUIRE Binance confirmation â skip if unconfirmed
            # This eliminates wrong-direction entries that caused -$95 and -$97 losses
            if not binance_confirms:
                if binance_dir is not None:
                    # Binance actively disagrees â definitely skip
                    print(
                        f"[SHORT] SKIP: {asset} {tf}m {direction}@{entry_price:.2f} â "
                        f"Binance says {binance_dir}, trade says {direction}"
                    )
                continue

            # Score calculation
            score = int(70 + (entry_price - 0.75) * 100)

            # Binance confirmation bonus (always confirmed at this point)
            score += 8

            # Time pressure bonus (closer to expiry = more certain)
            if secs_left < 30:
                score += 7
            elif secs_left < 60:
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
                "market_question": question,
                "score": score,
                "confidence": entry_price,
                "direction": direction,
                "yes_price": yes_price,
                "market_type": "SHORT_DURATION",
                "can_enter": True,
                "entry_reason": (
                    f"SHORT_{tf}m: {direction}@{entry_price:.2f}, "
                    f"{secs_left:.0f}s left, {asset}, "
                    f"binance=CONFIRMED"
                ),
                "factors_json": json.dumps({
                    "asset": asset,
                    "timeframe_minutes": tf,
                    "entry_price": entry_price,
                    "seconds_left": round(secs_left, 1),
                    "binance_confirms": True,
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
                f"binance=CONFIRMED "
                f"score={score}"
            )

        except Exception as e:
            continue

    if signals:
        print(f"[SHORT] Generated {len(signals)} short-duration signals")
    return signals
