"""
Strategy 3: Binance Price Lag Arbitrage

Exploits the 10-30 second lag between Binance exchange prices and
Polymarket 5-min/15-min crypto prediction markets.

Based on real wallet pattern: 0x8dxd ($2.04M profit, 90%+ win rate)

Runs EVERY 3-second loop â speed is critical for this strategy.
"""

import json
import re
from datetime import datetime
from typing import Optional

from binance_feed import binance_prices, get_price, get_change

# Crypto keywords for market matching
CRYPTO_KEYWORDS = {
    "BTC": ["bitcoin", "btc"],
    "ETH": ["ethereum", "eth"],
    "SOL": ["solana", "sol"],
}

# Timeframe detection patterns
TIMEFRAME_PATTERNS = [
    (r"5[\s-]?min", 5),
    (r"5M", 5),
    (r"15[\s-]?min", 15),
    (r"15M", 15),
    (r"1[\s-]?hour", 60),
    (r"1H", 60),
]

# Direction detection
UP_KEYWORDS = ["up", "increase", "rise", "above", "higher", "gain", "green"]
DOWN_KEYWORDS = ["down", "decrease", "fall", "below", "lower", "drop", "red", "dip"]


def _parse_crypto_market(question: str) -> Optional[dict]:
    """
    Parse a crypto prediction market question.

    Returns dict with: symbol, timeframe_minutes, is_up_market
    Example: "Will BTC be up in the next 5 minutes?" -> {symbol: "BTC", timeframe: 5, is_up: True}
    """
    q = question.lower()

    # Find crypto symbol
    symbol = None
    for sym, keywords in CRYPTO_KEYWORDS.items():
        for kw in keywords:
            if kw in q:
                symbol = sym
                break
        if symbol:
            break
    if not symbol:
        return None

    # Find timeframe
    timeframe = None
    for pattern, minutes in TIMEFRAME_PATTERNS:
        if re.search(pattern, question, re.IGNORECASE):
            timeframe = minutes
            break

    # Must be a short-duration market for arb to work
    if timeframe is None or timeframe > 60:
        return None

    # Determine if it's an UP or DOWN market
    is_up = None
    for kw in UP_KEYWORDS:
        if kw in q:
            is_up = True
            break
    if is_up is None:
        for kw in DOWN_KEYWORDS:
            if kw in q:
                is_up = False
                break

    if is_up is None:
        return None  # Can't determine direction type

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "is_up": is_up,
    }


def _calculate_divergence(market: dict, parsed: dict) -> Optional[dict]:
    """
    Calculate divergence between Binance price movement and Polymarket odds.

    Returns dict with: direction, divergence, exchange_change
    Or None if no significant divergence.
    """
    symbol = parsed["symbol"]
    timeframe = parsed["timeframe"]
    is_up_market = parsed["is_up"]

    # Get exchange price change over the timeframe
    exchange_change = get_change(symbol, timeframe)
    exchange_price = get_price(symbol)

    if exchange_price <= 0:
        return None  # No Binance data

    yes_price = market.get("yes_price", 0.5)

    # For an "UP" market:
    # - If exchange is moving UP strongly, YES should be HIGH
    # - If YES is still LOW, there's an arb (buy YES)
    # For a "DOWN" market:
    # - If exchange is moving DOWN strongly, YES should be HIGH
    # - If YES is still LOW, there's an arb (buy YES)

    if is_up_market:
        if exchange_change > 0.005:  # Exchange moved up >0.5%
            # YES should be high â the asset IS going up
            implied_yes = min(0.95, 0.50 + exchange_change * 10)  # Rough mapping
            if yes_price < implied_yes - 0.05:  # Polymarket lags by >5%
                return {
                    "direction": "YES",
                    "divergence": implied_yes - yes_price,
                    "exchange_change": exchange_change,
                    "implied_yes": implied_yes,
                }
        elif exchange_change < -0.005:  # Exchange moved down >0.5%
            # NO should be high â the asset is NOT going up
            implied_yes = max(0.05, 0.50 + exchange_change * 10)
            if yes_price > implied_yes + 0.05:  # Polymarket lags
                return {
                    "direction": "NO",
                    "divergence": yes_price - implied_yes,
                    "exchange_change": exchange_change,
                    "implied_yes": implied_yes,
                }
    else:  # DOWN market
        if exchange_change < -0.005:  # Exchange moved down
            # YES should be high â the asset IS going down
            implied_yes = min(0.95, 0.50 + abs(exchange_change) * 10)
            if yes_price < implied_yes - 0.05:
                return {
                    "direction": "YES",
                    "divergence": implied_yes - yes_price,
                    "exchange_change": exchange_change,
                    "implied_yes": implied_yes,
                }
        elif exchange_change > 0.005:  # Exchange moved up
            # NO should be high â the asset is NOT going down
            implied_yes = max(0.05, 0.50 - exchange_change * 10)
            if yes_price > implied_yes + 0.05:
                return {
                    "direction": "NO",
                    "divergence": yes_price - implied_yes,
                    "exchange_change": exchange_change,
                    "implied_yes": implied_yes,
                }

    return None  # No significant divergence


def generate_arb_signals(markets: list) -> list:
    """
    Scan markets for Binance price lag arbitrage opportunities.

    This is SYNCHRONOUS (no async) for maximum speed â runs every 3s loop.
    No LLM calls needed â pure price comparison.

    Returns list of signal dicts compatible with paper_trader.
    """
    signals = []

    for market in markets:
        try:
            question = market.get("question", "")

            # Parse: is this a short-duration crypto market?
            parsed = _parse_crypto_market(question)
            if not parsed:
                continue

            # Calculate divergence
            div = _calculate_divergence(market, parsed)
            if not div:
                continue

            direction = div["direction"]
            divergence = div["divergence"]
            exchange_change = div["exchange_change"]
            yes_price = market.get("yes_price", 0.5)
            entry_price = yes_price if direction == "YES" else (1 - yes_price)

            # Skip extreme entry prices
            if entry_price > 0.90 or entry_price < 0.10:
                continue

            # Score based on divergence size
            score = int(70 + min(25, divergence * 200))
            score = min(95, max(70, score))

            signal = {
                "market_id": market.get("id", ""),
                "market_question": question,
                "score": score,
                "confidence": 0.75,
                "direction": direction,
                "yes_price": yes_price,
                "market_type": "BINANCE_ARB",
                "can_enter": True,
                "entry_reason": (
                    f"ARB: {parsed['symbol']} {exchange_change*100:+.1f}% on Binance, "
                    f"Poly={yes_price:.0%}, div={divergence:.0%}, {direction}@{entry_price:.2f}"
                ),
                "factors_json": json.dumps({
                    "symbol": parsed["symbol"],
                    "timeframe": parsed["timeframe"],
                    "exchange_change": round(exchange_change, 4),
                    "divergence": round(divergence, 4),
                    "exchange_price": get_price(parsed["symbol"]),
                    "implied_yes": round(div["implied_yes"], 4),
                }),
                "created_at": datetime.utcnow().isoformat(),
                "clob_token_ids": market.get("clob_token_ids", []),
                "condition_id": market.get("condition_id", ""),
                "liquidity": market.get("liquidity", 0),
            }
            signals.append(signal)
            print(f"[ARB] {parsed['symbol']} {exchange_change*100:+.1f}%: "
                  f"{direction}@{entry_price:.2f} div={divergence:.0%} "
                  f"'{question[:45]}'")

        except Exception as e:
            continue

    if signals:
        print(f"[ARB] Generated {len(signals)} arb signals")
    return signals
