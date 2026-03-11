"""
Strategy 2: Volume Spike Trading

Detects when a market gets 3x its normal volume (sign that something is happening).
Uses Haiku to confirm direction. Widened price filter to capture more opportunities.
"""

import json
import os
from datetime import datetime
from typing import Optional

from volume_detector import detect_spike

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Widened from 0.15-0.85 to 0.08-0.95 to capture near-certainty spikes
MIN_SPIKE_PRICE = 0.08
MAX_SPIKE_PRICE = 0.95

MIN_SPIKE_LIQUIDITY = 5000  # Reduced from 10K for more signals
MAX_MARKETS_TO_SCAN = 150   # Increased from 100


async def _infer_direction_with_haiku(market: dict) -> Optional[str]:
    """Use Haiku to determine trade direction on a spiking market."""
    if not HAS_ANTHROPIC or not ANTHROPIC_API_KEY:
        # Fallback: if price > 0.6, bet YES (momentum); else NO
        yes_price = market.get("yes_price", 0.5)
        return "YES" if yes_price > 0.6 else "NO"

    try:
        client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
        question = market.get("question", "")
        yes_price = market.get("yes_price", 0.5)
        volume = market.get("volume24hr", 0)

        response = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=10,
            temperature=0.1,
            messages=[{"role": "user", "content": (
                f'Market: "{question}"\n'
                f"YES price: {yes_price:.0%}, Volume spike detected (3x+ normal).\n"
                f"24h volume: ${volume:,.0f}\n\n"
                f"The volume spike suggests informed activity. "
                f"Which side is the smart money on? Answer YES or NO only."
            )}],
        )
        answer = response.content[0].text.strip().upper()
        if "YES" in answer:
            return "YES"
        elif "NO" in answer:
            return "NO"
        return None

    except Exception as e:
        print(f"[SPIKE] Haiku direction error: {e}")
        # Fallback on error
        yes_price = market.get("yes_price", 0.5)
        return "YES" if yes_price > 0.6 else "NO"


async def generate_spike_signals(markets: list) -> list:
    """
    Scan markets for volume spike trading opportunities.

    1. detect_spike() finds 3x+ volume anomalies
    2. Price filter: 0.08-0.95 (widened to capture near-certainty spikes)
    3. Liquidity > $5K
    4. Haiku confirms direction (with fallback)
    5. Score based on spike magnitude
    """
    signals = []

    for market in markets[:MAX_MARKETS_TO_SCAN]:
        try:
            market_id = market.get("id", "")
            vol24 = market.get("volume24hr", 0) or 0
            yes_price = market.get("yes_price", 0.5)
            total_vol = market.get("volume", 0) or 0
            liquidity = market.get("liquidity", 0) or 0

            # Run spike detection (also records snapshot)
            spike = await detect_spike(
                market_id, vol24, yes_price, total_vol, liquidity
            )

            if not spike:
                continue

            # Price filter â widened range
            if yes_price < MIN_SPIKE_PRICE or yes_price > MAX_SPIKE_PRICE:
                continue

            # Liquidity check
            if liquidity < MIN_SPIKE_LIQUIDITY:
                continue

            # Get direction from Haiku (or fallback)
            direction = await _infer_direction_with_haiku(market)
            if not direction:
                continue

            # Calculate entry price
            entry_price = yes_price if direction == "YES" else (1 - yes_price)

            # Entry price sanity check
            if entry_price > 0.95 or entry_price < 0.05:
                continue

            # Score: based on spike multiplier
            multiplier = spike.get("spike_multiplier", 3.0)
            alert_type = spike.get("alert_type", "VOLUME_SURGE")

            base_score = 65
            if multiplier >= 10:
                base_score = 95
            elif multiplier >= 7:
                base_score = 90
            elif multiplier >= 5:
                base_score = 85
            elif multiplier >= 4:
                base_score = 80
            elif multiplier >= 3:
                base_score = 70

            # Bonus for specific alert types
            if alert_type == "WHALE_MOVE":
                base_score += 5
            elif alert_type == "ACCUMULATION":
                base_score += 3

            score = min(99, max(65, base_score))

            signal = {
                "market_id": market_id,
                "market_question": market.get("question", ""),
                "score": score,
                "confidence": min(0.95, multiplier / 10),
                "direction": direction,
                "yes_price": yes_price,
                "market_type": "VOLUME_SPIKE",
                "can_enter": True,
                "entry_reason": (
                    f"SPIKE: {alert_type} {multiplier:.1f}x vol, "
                    f"{direction}@{entry_price:.2f}, "
                    f"liq=${liquidity:,.0f}"
                ),
                "factors_json": json.dumps({
                    "spike_multiplier": multiplier,
                    "alert_type": alert_type,
                    "volume_24h": vol24,
                    "liquidity": liquidity,
                    "entry_price": entry_price,
                }),
                "created_at": datetime.utcnow().isoformat(),
                "clob_token_ids": market.get("clob_token_ids", []),
                "condition_id": market.get("condition_id", ""),
                "liquidity": liquidity,
            }
            signals.append(signal)

            print(
                f"[SPIKE] Signal: {alert_type} {multiplier:.1f}x "
                f"{direction}@{entry_price:.2f} "
                f"'{market.get('question', '')[:45]}' "
                f"liq=${liquidity:,.0f} score={score}"
            )

        except Exception as e:
            continue

    if signals:
        print(f"[SPIKE] Generated {len(signals)} spike signals")
    return signals
