"""
Strategy 2: Volume Spike Trading

Detects abnormal volume surges (3x+ baseline) and trades in the direction
of smart money movement.

Uses existing volume_detector.py for spike detection.
Uses Haiku for quick direction inference when spike detected.
"""

import json
import os
from datetime import datetime
from typing import Optional

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")


async def _infer_direction_with_haiku(market: dict) -> Optional[str]:
    """
    Quick Haiku call to determine trade direction on a volume spike.
    Costs ~$0.001 per call. Only called when spike is confirmed.
    """
    if not HAS_ANTHROPIC or not ANTHROPIC_API_KEY:
        return None

    try:
        client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
        question = market.get("question", "")
        yes_price = market.get("yes_price", 0.5)
        volume = market.get("volume24hr", 0)

        response = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=5,
            messages=[{"role": "user", "content": (
                f'Prediction market: "{question}"\n'
                f"YES price: {yes_price:.0%}\n"
                f"24h volume: ${volume:,.0f}\n"
                f"Abnormal volume spike detected (smart money moving).\n\n"
                f"Which outcome is more likely? Answer ONLY 'YES' or 'NO'."
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
        return None


async def generate_spike_signals(markets: list) -> list:
    """
    Scan markets for volume spike trading opportunities.

    Uses volume_detector.detect_spike() for spike detection,
    then Haiku for direction inference.

    Returns list of signal dicts compatible with paper_trader.
    """
    from volume_detector import detect_spike

    signals = []
    spike_count = 0

    for market in markets[:100]:  # Check top 100 by volume
        try:
            spike = await detect_spike(
                market.get("id", ""),
                market.get("volume24hr", 0) or 0,
                market.get("yes_price", 0.5),
                market.get("volume", 0) or 0,
                market.get("liquidity", 0) or 0,
            )

            if not spike:
                continue

            spike_count += 1
            alert_type = spike.get("alert_type", "UNKNOWN")

            # Skip if price is already extreme (no room to trade)
            yes_price = market.get("yes_price", 0.5)
            if yes_price < 0.15 or yes_price > 0.85:
                continue

            # Skip low liquidity
            liquidity = market.get("liquidity", 0) or 0
            if liquidity < 10000:
                continue

            # Get direction from Haiku
            direction = await _infer_direction_with_haiku(market)
            if not direction:
                continue

            entry_price = yes_price if direction == "YES" else (1 - yes_price)

            # Skip extreme entry prices
            if entry_price > 0.85 or entry_price < 0.15:
                continue

            # Score based on spike severity
            magnitude = spike.get("magnitude", 3.0)
            score = int(65 + min(25, (magnitude - 3) * 8))
            score = min(95, max(65, score))

            signal = {
                "market_id": market.get("id", ""),
                "market_question": market.get("question", ""),
                "score": score,
                "confidence": 0.70,
                "direction": direction,
                "yes_price": yes_price,
                "market_type": "VOLUME_SPIKE",
                "can_enter": True,
                "entry_reason": f"SPIKE: {alert_type} {magnitude:.1f}x, {direction}@{entry_price:.2f}",
                "factors_json": json.dumps({
                    "alert_type": alert_type,
                    "magnitude": magnitude,
                    "volume24hr": market.get("volume24hr", 0),
                    "liquidity": liquidity,
                }),
                "created_at": datetime.utcnow().isoformat(),
                "clob_token_ids": market.get("clob_token_ids", []),
                "condition_id": market.get("condition_id", ""),
                "liquidity": liquidity,
            }
            signals.append(signal)
            print(f"[SPIKE] {alert_type} {magnitude:.1f}x: {direction}@{entry_price:.2f} "
                  f"'{market['question'][:50]}'")

        except Exception as e:
            # Silently skip errors (volume detector needs 3+ snapshots)
            continue

    if spike_count > 0:
        print(f"[SPIKE] Found {spike_count} spikes, generated {len(signals)} signals")
    return signals
