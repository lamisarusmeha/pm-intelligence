"""
Polymarket API client.
Primary:  Polymarket Gamma API  (https://gamma-api.polymarket.com)
Fallback: Demo mode with realistic simulated data when API is unreachable.
"""

import httpx
import json
import random
import asyncio
from datetime import datetime, timedelta
from typing import Optional, Union, Tuple, List

GAMMA_BASE = "https://gamma-api.polymarket.com"
TIMEOUT = 20  # seconds

# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_outcome_prices(raw: Union[str, list, None]) -> Tuple[float, float]:
    """Return (yes_price, no_price) from Polymarket's outcomePrices field."""
    try:
        if raw is None:
            return 0.5, 0.5
        if isinstance(raw, list):
            prices = [float(p) for p in raw]
        else:
            prices = json.loads(raw)
        yes = float(prices[0])
        no  = float(prices[1]) if len(prices) > 1 else round(1 - yes, 4)
        return yes, no
    except Exception:
        return 0.5, 0.5


def _normalize_market(raw: dict) -> Optional[dict]:
    """Convert a raw Gamma API market dict into our internal format."""
    try:
        yes_price, no_price = _parse_outcome_prices(raw.get("outcomePrices"))
        return {
            "id":          raw.get("id", ""),
            "question":    raw.get("question", "Unknown market"),
            "slug":        raw.get("slug", ""),
            "category":    raw.get("category") or raw.get("groupItemTitle") or "General",
            "yes_price":   yes_price,
            "no_price":    no_price,
            "volume":      float(raw.get("volume") or 0),
            "volume24hr":  float(raw.get("volume24hr") or 0),
            "liquidity":   float(raw.get("liquidity") or 0),
            "active":      1 if raw.get("active") else 0,
            "closed":      1 if raw.get("closed") else 0,
            "end_date":    raw.get("endDate") or raw.get("endDateIso") or "",
            "last_updated": datetime.utcnow().isoformat(),
        }
    except Exception:
        return None


# ── Real API ──────────────────────────────────────────────────────────────────

async def fetch_active_markets(limit: int = 100) -> list[dict]:
    """
    Fetch top active markets from Polymarket Gamma API, sorted by volume.
    Falls back to demo data if the API is unreachable.
    """
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.get(
                f"{GAMMA_BASE}/markets",
                params={
                    "active":     "true",
                    "closed":     "false",
                    "limit":      limit,
                    "order":      "volume24hr",
                    "ascending":  "false",
                }
            )
            resp.raise_for_status()
            raw_list = resp.json()
            markets = [_normalize_market(m) for m in raw_list]
            markets = [m for m in markets if m and m["id"]]
            if markets:
                print(f"[API] Fetched {len(markets)} live markets from Polymarket")
                return markets
    except Exception as e:
        print(f"[API] Polymarket unreachable ({e}), using demo mode")

    return _generate_demo_markets(limit)


async def fetch_markets_by_ids(market_ids: list[str]) -> list[dict]:
    """Fetch specific markets by ID."""
    results = []
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            for mid in market_ids:
                try:
                    resp = await client.get(f"{GAMMA_BASE}/markets/{mid}")
                    if resp.status_code == 200:
                        m = _normalize_market(resp.json())
                        if m:
                            results.append(m)
                    await asyncio.sleep(0.1)  # light rate-limiting
                except Exception:
                    pass
    except Exception:
        pass
    return results


# ── Demo / Simulation mode ────────────────────────────────────────────────────

DEMO_QUESTIONS = [
    ("Will the Fed cut rates in Q2 2026?",            "Economics",   0.62),
    ("Will SpaceX land on Mars before 2030?",          "Science",     0.18),
    ("Will Bitcoin exceed $120k by end of 2026?",      "Crypto",      0.44),
    ("Will there be a US recession in 2026?",          "Economics",   0.31),
    ("Will Apple release AR glasses in 2026?",         "Technology",  0.55),
    ("Will the Lakers win the 2026 NBA Championship?", "Sports",      0.09),
    ("Will Elon Musk acquire another major company?",  "Business",    0.27),
    ("Will the US pass major AI regulation by 2027?",  "Politics",    0.67),
    ("Will Netflix hit 400M subscribers by 2027?",     "Business",    0.73),
    ("Will a major bank collapse in 2026?",            "Economics",   0.14),
    ("Will OpenAI release GPT-5 in 2026?",             "Technology",  0.81),
    ("Will Ukraine-Russia war end by end of 2026?",    "Politics",    0.38),
    ("Will Nvidia remain the top AI chip company?",    "Technology",  0.77),
    ("Will Kamala Harris run in 2028 primaries?",      "Politics",    0.52),
    ("Will a new COVID variant cause lockdowns?",       "Health",      0.08),
    ("Will gold exceed $3500/oz in 2026?",             "Commodities", 0.41),
    ("Will the DOW hit 50,000 by end of 2026?",        "Economics",   0.59),
    ("Will Amazon surpass Apple in market cap?",       "Business",    0.23),
    ("Will any country adopt Bitcoin as reserve?",     "Crypto",      0.19),
    ("Will the EU break up before 2030?",              "Politics",    0.06),
]

_demo_state: dict[str, dict] = {}


def _generate_demo_markets(limit: int = 20) -> list[dict]:
    """Generate realistic simulated market data for demo/offline mode."""
    global _demo_state
    now = datetime.utcnow()
    markets = []

    questions = DEMO_QUESTIONS[:min(limit, len(DEMO_QUESTIONS))]

    for i, (question, category, base_price) in enumerate(questions):
        mid = f"demo_{i:04d}"

        # Initialise or drift the price
        if mid not in _demo_state:
            _demo_state[mid] = {
                "yes_price":  base_price,
                "volume":     random.uniform(50_000, 2_000_000),
                "volume24hr": random.uniform(1_000, 80_000),
                "liquidity":  random.uniform(2_000, 50_000),
                "tick":       0,
            }

        state = _demo_state[mid]
        state["tick"] += 1

        # Simulate price walk (±1-3% per refresh)
        drift = random.gauss(0, 0.015)
        state["yes_price"] = max(0.02, min(0.98, state["yes_price"] + drift))

        # Occasionally simulate a volume spike (insider signal)
        if random.random() < 0.12:
            spike_mult = random.uniform(3, 12)
            state["volume24hr"] = state["volume24hr"] * spike_mult
            state["volume"]    += state["volume24hr"]
        else:
            # Normal daily volume accrual
            daily_accrual = random.uniform(200, 3_000)
            state["volume"]    += daily_accrual
            state["volume24hr"] = random.uniform(
                state["volume24hr"] * 0.7,
                state["volume24hr"] * 1.3
            )

        yes = round(state["yes_price"], 4)
        markets.append({
            "id":          mid,
            "question":    question,
            "slug":        question.lower().replace(" ", "-")[:40],
            "category":    category,
            "yes_price":   yes,
            "no_price":    round(1 - yes, 4),
            "volume":      round(state["volume"], 2),
            "volume24hr":  round(state["volume24hr"], 2),
            "liquidity":   round(state["liquidity"], 2),
            "active":      1,
            "closed":      0,
            "end_date":    (now + timedelta(days=random.randint(30, 365))).strftime("%Y-%m-%d"),
            "last_updated": now.isoformat(),
        })

    return markets
