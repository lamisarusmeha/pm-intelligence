"""
Strategy 4: Short-Duration Crypto Markets (5-min / 15-min)

Targets the "Bitcoin Up or Down" (and ETH, SOL) rolling markets on Polymarket.
These markets resolve every 5 or 15 minutes - new ones are created continuously.

Strategy:
  - Fetch current and near-expiry 5m/15m markets by computed slug
  - When one side is 80%+, enter on that side (near-certainty before resolution)
  - Hold to resolution (~minutes, not days)
  - Also detect Binance price momentum to predict direction on fresh markets

Slug patterns:
  btc-updown-5m-{unix_ts}   where unix_ts = floor(now / 300) * 300
  btc-updown-15m-{unix_ts}  where unix_ts = floor(now / 900) * 900
  eth-updown-5m-{unix_ts}
  sol-updown-5m-{unix_ts}
"""

import json
import time
import urllib.request
from datetime import datetime
from typing import Optional

GAMMA_API = "https://gamma-api.polymarket.com"

# All slug patterns to scan
SLUG_PATTERNS = [
    ("btc-updown-5m", 300, "BTC_5M"),
    ("btc-updown-15m", 900, "BTC_15M"),
    ("eth-updown-5m", 300, "ETH_5M"),
    ("sol-updown-5m", 300, "SOL_5M"),
]

# Entry thresholds
MIN_NEAR_CERTAINTY = 0.80
MIN_MOMENTUM_ENTRY = 0.55
MAX_SECONDS_TO_EXPIRY = 240
MIN_SECONDS_TO_EXPIRY = 30


def _fetch_event(slug):
    """Fetch a single event from Gamma API by slug."""
    url = f"{GAMMA_API}/events?slug={slug}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "PMIntelligence/3.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            if data and isinstance(data, list) and len(data) > 0:
                return data[0]
            elif data and isinstance(data, dict):
                return data
    except Exception:
        pass
    return None


def _parse_5m_market(event, label):
    """Parse a 5m/15m event into a standardized market dict."""
    markets = event.get("markets", [])
    if not markets:
        return None

    m = markets[0]
    market_id = str(m.get("id", ""))
    question = m.get("question", "")
    end_date = m.get("endDate", "")

    outcome_prices = m.get("outcomePrices", ["0.5", "0.5"])
    if isinstance(outcome_prices, str):
        outcome_prices = json.loads(outcome_prices)

    yes_price = float(outcome_prices[0]) if outcome_prices else 0.5
    no_price = 1 - yes_price

    seconds_left = 9999
    if end_date:
        try:
            end_str = end_date.replace("Z", "+00:00")
            if "T" in end_str:
                end_dt = datetime.fromisoformat(end_str).replace(tzinfo=None)
            else:
                end_dt = datetime.strptime(end_str[:10], "%Y-%m-%d")
            seconds_left = (end_dt - datetime.utcnow()).total_seconds()
        except Exception:
            pass

    if seconds_left <= 0:
        return None

    if m.get("closed", False):
        return None

    volume = float(m.get("volume", 0) or 0)
    liquidity = float(m.get("liquidity", 0) or 0)

    return {
        "id": market_id,
        "question": question,
        "slug": event.get("slug", ""),
        "yes_price": round(yes_price, 4),
        "no_price": round(no_price, 4),
        "volume": volume,
        "volume24hr": volume,
        "liquidity": liquidity,
        "active": 1,
        "closed": 0,
        "end_date": end_date,
        "last_updated": datetime.utcnow().isoformat(),
        "condition_id": str(m.get("conditionId", "")),
        "clob_token_ids": m.get("clobTokenIds", []),
        "seconds_left": round(seconds_left),
        "label": label,
        "category": "crypto",
    }


def fetch_short_duration_markets():
    """Fetch all current 5m/15m crypto markets."""
    now_unix = int(time.time())
    markets = []

    for slug_prefix, interval, label in SLUG_PATTERNS:
        current_ts = (now_unix // interval) * interval

        for ts in [current_ts, current_ts - interval]:
            slug = f"{slug_prefix}-{ts}"
            event = _fetch_event(slug)
            if event:
                parsed = _parse_5m_market(event, label)
                if parsed and parsed["seconds_left"] > 0:
                    markets.append(parsed)

    return markets


def generate_short_duration_signals(markets, binance_prices=None):
    """
    Generate trading signals from short-duration markets.
    Mode 1: NEAR_CERTAINTY - One side 80%+ with <4 min to expiry
    Mode 2: MOMENTUM - Fresh market + Binance >0.1% move
    """
    signals = []
    if binance_prices is None:
        binance_prices = {}

    for market in markets:
        yes_price = market["yes_price"]
        no_price = market["no_price"]
        seconds_left = market["seconds_left"]
        label = market["label"]

        if seconds_left < MIN_SECONDS_TO_EXPIRY:
            continue

        # MODE 1: Near-certainty (one side 80%+)
        if yes_price >= MIN_NEAR_CERTAINTY and seconds_left <= MAX_SECONDS_TO_EXPIRY:
            direction = "YES"
            entry_price = yes_price
            score = int(75 + (entry_price - 0.80) * 120)
            score = min(99, max(75, score))
            signals.append({
                "market_id": market["id"],
                "market_question": market["question"],
                "direction": direction,
                "yes_price": yes_price,
                "score": score,
                "can_enter": True,
                "market_type": "SHORT_DURATION",
                "entry_reason": f"5m near-certainty: {label} YES@{entry_price:.2f} {seconds_left}s left",
                "factors_json": json.dumps({"near_certainty": entry_price, "seconds_left": seconds_left, "market_label": label}),
                "seconds_left": seconds_left,
                "end_date": market["end_date"],
            })

        elif no_price >= MIN_NEAR_CERTAINTY and seconds_left <= MAX_SECONDS_TO_EXPIRY:
            direction = "NO"
            entry_price = no_price
            score = int(75 + (entry_price - 0.80) * 120)
            score = min(99, max(75, score))
            signals.append({
                "market_id": market["id"],
                "market_question": market["question"],
                "direction": direction,
                "yes_price": yes_price,
                "score": score,
                "can_enter": True,
                "market_type": "SHORT_DURATION",
                "entry_reason": f"5m near-certainty: {label} NO@{entry_price:.2f} {seconds_left}s left",
                "factors_json": json.dumps({"near_certainty": entry_price, "seconds_left": seconds_left, "market_label": label}),
                "seconds_left": seconds_left,
                "end_date": market["end_date"],
            })

        # MODE 2: Momentum entry on fresh markets
        elif seconds_left > 120:
            coin = "BTC"
            if "ETH" in label:
                coin = "ETH"
            elif "SOL" in label:
                coin = "SOL"

            coin_data = binance_prices.get(coin, {})
            change_5m = coin_data.get("change_5m", 0) or 0

            if abs(change_5m) >= 0.1:
                if change_5m > 0:
                    direction = "YES"
                    entry_price = yes_price
                else:
                    direction = "NO"
                    entry_price = no_price

                if 0.30 <= entry_price <= 0.70:
                    score = int(50 + abs(change_5m) * 200)
                    score = min(80, max(50, score))
                    signals.append({
                        "market_id": market["id"],
                        "market_question": market["question"],
                        "direction": direction,
                        "yes_price": yes_price,
                        "score": score,
                        "can_enter": True,
                        "market_type": "SHORT_DURATION",
                        "entry_reason": f"5m momentum: {coin} {change_5m:+.2f}% -> {direction} {label}",
                        "factors_json": json.dumps({"momentum": change_5m, "seconds_left": seconds_left, "market_label": label, "coin": coin}),
                        "seconds_left": seconds_left,
                        "end_date": market["end_date"],
                    })

    return signals
