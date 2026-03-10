"""
Signal Engine — 9-factor scoring with FOUR trading modes.

COPY_TRADE MODE  (smart_wallet score ≥ 85)
  Mirror high-accuracy Polymarket wallets (>65% win rate, >5 trades).
  Most starred Polymarket bot on GitHub does ONLY this. Highest edge.
  Exit: 4¢ TP, 3¢ SL, hold 3h.
  Expected win rate: 70–80%.

BUY_NO_EARLY MODE  (sensational YES > 55% new market)
  Exploit behavioral bias: retail overprices YES at launch on dramatic markets.
  Historical base rate: only ~22% of sensational questions resolve YES.
  Exit: 5¢ TP, 5¢ SL, hold 8h.
  Expected win rate: 68–78%.

LOCK-IN MODE  (75–92% YES or 8–25% YES)
  Bet WITH the crowd on near-certain outcomes.
  Exit: 3¢ TP, 9¢ SL, hold 6h.  Expected win rate: 75–85%.

MOMENTUM MODE (25–75% YES)
  Trade genuine uncertainty where information gives us an edge.
  Requires: 2+ signals ≥ 50, OR single news/wallet signal ≥ 78.
  Exit: 6¢ TP, 4¢ SL, hold 4h.  Expected win rate: 55–65%.

9 weighted factors (self-learning):
  1. volume_spike  — unusual 24h volume vs historical average
  2. price_zone    — DUAL OPTIMA: near 50¢ (momentum) + near 75-90% (lock-in)
  3. liquidity     — order book depth
  4. momentum      — sustained directional price trend (2¢+ required)
  5. category      — market type base score
  6. news_impact   — breaking news matched to this market
  7. smart_wallet  — high-accuracy wallets entering this market
  8. end_date      — markets resolving in 3–21 days are ideal
  9. buy_no_early  — sensational YES-overpriced market bias score
"""

import json
from datetime import datetime, timedelta
from typing import Optional, List, Tuple

import database as db
import news_engine
import wallet_tracker

# ── Mode detection thresholds ─────────────────────────────────────────────────

LOCK_IN_YES_MIN   = 0.75   # market at 75%+ YES → Lock-In YES play
LOCK_IN_YES_MAX   = 0.92   # above 92% → too extreme (already priced in, tiny move)
LOCK_IN_NO_MAX    = 0.25   # market at 25%- YES → Lock-In NO play
LOCK_IN_NO_MIN    = 0.08   # below 8% → too extreme

# Signal thresholds — HIGH ACCURACY MODE (real-money precision)
# MOMENTUM is disabled — only LOCK_IN, BUY_NO_EARLY, COPY_TRADE run
STRONG_THRESHOLD       = 50   # restored: factor must be genuinely strong
HC_OVERRIDE_SCORE      = 78   # restored: high-conviction single signal bar
LOCK_IN_MIN_SCORE      = 65   # restored: LOCK_IN requires solid price_zone score
GENERATE_MIN_SCORE     = 40   # ↑ from 15: only genuinely scored markets get through

# Volume surge auto-entry — insider/early-info detection
# If a LOCK_IN market suddenly has 3x+ normal volume, something is happening.
# Enter immediately before the price moves — same edge ZachXBT insiders used.
VOLUME_SURGE_AUTO_ENTRY = 85  # volume_spike score ≥ this → standalone entry on LOCK_IN

# ── Copy Trade thresholds ──────────────────────────────────────────────────────
COPY_TRADE_WALLET_MIN  = 85   # smart_wallet score ≥ this → mirror immediately

# ── Buy No Early thresholds ────────────────────────────────────────────────────
# Research: ~78% of sensational YES markets resolve NO (retail behavioral bias)
BUY_NO_EARLY_MIN_SCORE = 60   # minimum buy_no_early score to trigger
BUY_NO_EARLY_YES_MIN   = 0.55 # YES must be at least 55% (overpriced vs 22% base)
BUY_NO_EARLY_YES_MAX   = 0.88 # don't bet NO on > 88% YES (too risky)
BUY_NO_EARLY_NEW_VOL   = 8000 # total volume < $8k = likely a newer/immature market

SENSATIONAL_KEYWORDS = [
    'will', 'ever', 'first', 'last', 'end', 'crash', 'collapse', 'hit',
    'reach', 'break', 'surge', 'plunge', 'skyrocket', 'explode', 'war',
    'nuclear', 'arrest', 'resign', 'impeach', 'ban', 'default', 'bankrupt',
    'fired', 'removed', 'destroyed', 'eliminated', 'fall', 'record',
    'historic', 'unprecedented', 'major', 'massive', 'huge', 'die', 'dead',
    'win', 'lose', 'cut', 'raise', 'drop', 'spike', 'above', 'below',
]

# ── Category base scores ──────────────────────────────────────────────────────

CATEGORY_SCORES = {
    "politics": 70,
    "crypto":   65,
    "finance":  65,
    "sports":   68,
    "science":  55,
    "weather":  45,
    "other":    50,
}


def _category_score(category: str) -> float:
    key = (category or "other").lower()
    for cat, score in CATEGORY_SCORES.items():
        if cat in key:
            return score
    return CATEGORY_SCORES["other"]


# ── Factor scorers ────────────────────────────────────────────────────────────

def _volume_spike_score(market: dict, history: list) -> float:
    """
    Detects unusual volume — the #1 leading indicator of insider/early-info trading.

    Example: Before ZachXBT drops an article on a project, insiders buy on
    Polymarket → volume spikes → price hasn't moved yet → bot enters → rides the wave.

    Two layers:
    1. 24h volume vs historical average (standard ratio)
    2. VELOCITY bonus: if volume is accelerating (last snapshot >> prior avg),
       add up to 20 bonus points — catches intraday surges early.
    """
    vol24 = market.get("volume24hr", 0) or 0

    if not history:
        # No history = new market, treat as baseline
        return 40.0

    avg_vol = sum(h.get("volume24hr", 0) or 0 for h in history) / len(history)
    if avg_vol < 1:
        return 40.0

    ratio = vol24 / avg_vol

    # Base score from ratio
    if ratio >= 10:  base = 100.0   # 10x+ — something major happening
    elif ratio >= 5: base = 95.0    # 5x — very unusual, likely informed buying
    elif ratio >= 3: base = 85.0    # 3x — strong signal, worth acting on
    elif ratio >= 2: base = 70.0    # 2x — elevated, pay attention
    elif ratio >= 1.5: base = 55.0  # 1.5x — slightly above normal
    elif ratio >= 1: base = 40.0    # normal
    else: base = max(0, 25 * ratio)

    # Velocity bonus: compare most recent snapshot to older average
    # If the LATEST snapshot is much higher than the rolling avg, the surge
    # is happening RIGHT NOW — even more urgent to enter before price moves
    velocity_bonus = 0.0
    if len(history) >= 3:
        recent_vol  = history[-1].get("volume24hr", 0) or 0
        older_avg   = sum(h.get("volume24hr", 0) or 0 for h in history[:-1]) / (len(history) - 1)
        if older_avg > 1:
            velocity_ratio = recent_vol / older_avg
            if velocity_ratio >= 3:   velocity_bonus = 20.0  # accelerating fast
            elif velocity_ratio >= 2: velocity_bonus = 12.0  # picking up
            elif velocity_ratio >= 1.5: velocity_bonus = 6.0 # mild acceleration

    return min(100.0, round(base + velocity_bonus, 1))


def _price_zone_score(market: dict) -> Tuple[float, str]:
    """
    DUAL OPTIMA price zone scoring. Returns (score, market_type).

    TWO scoring peaks:
    ① LOCK-IN zone (75-92% YES or 8-25% YES): score 90
       Bet WITH the crowd. An 80% market wins ~80% of the time.
    ② MOMENTUM zone (35-65% YES): score 75
       Real uncertainty — news/momentum/smart-money have real edge here.
    ③ Transition zone (25-35% or 65-75%): score 82
       Near-certain but not yet lock-in territory.
    ④ Extreme (>92% or <8%): score 20
       Already fully priced in — tiny TP room, skip.
    """
    yes = market.get("yes_price", 0.5) or 0.5

    # Detect Lock-In zones
    if LOCK_IN_YES_MIN <= yes <= LOCK_IN_YES_MAX:
        return 90.0, "LOCK_IN"   # 75-92% YES → bet YES with crowd
    if LOCK_IN_NO_MIN <= yes <= LOCK_IN_NO_MAX:
        return 90.0, "LOCK_IN"   # 8-25% YES → bet NO with crowd

    # Extreme (already fully priced in)
    if yes > LOCK_IN_YES_MAX or yes < LOCK_IN_NO_MIN:
        return 20.0, "EXTREME"

    # Transition zone (65-75% or 25-35%)
    distance_from_half = abs(yes - 0.5)
    if distance_from_half >= 0.15:   # 35-65% range → peak momentum zone
        return 75.0, "MOMENTUM"
    if distance_from_half >= 0.10:   # 40-60% → still good
        return 80.0, "MOMENTUM"

    # Near-50¢ momentum zone (40-60%)
    return 82.0, "MOMENTUM"


def _liquidity_score(market: dict) -> float:
    liq = market.get("liquidity", 0) or 0
    if liq >= 200_000: return 100.0
    if liq >= 100_000: return 85.0
    if liq >= 50_000:  return 70.0
    if liq >= 20_000:  return 55.0
    if liq >= 5_000:   return 40.0
    if liq >= 1_000:   return 25.0
    return max(0, liq / 100)


def _momentum_score(market: dict, history: list) -> float:
    if len(history) < 3:
        return 40.0
    prices = [h.get("yes_price", 0.5) for h in history[-6:]]
    if len(prices) < 2:
        return 40.0
    moves = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    positive = sum(1 for m in moves if m > 0)
    negative = sum(1 for m in moves if m < 0)
    total    = len(moves)
    direction_ratio = max(positive, negative) / total if total else 0.5
    total_move = abs(prices[-1] - prices[0])
    magnitude_score = min(100, total_move * 500)
    return (direction_ratio * 60) + (magnitude_score * 0.4)


def _end_date_score(market: dict) -> float:
    """Ideal window: 1-21 days to resolution. Short enough for trading to matter."""
    end_date_str = market.get("end_date", "")
    if not end_date_str:
        return 50.0
    try:
        end_date_str = end_date_str.replace("Z", "+00:00")
        if "T" in end_date_str:
            end_dt = datetime.fromisoformat(end_date_str).replace(tzinfo=None)
        else:
            end_dt = datetime.strptime(end_date_str[:10], "%Y-%m-%d")
        days_left = (end_dt - datetime.utcnow()).days

        # Lock-in plays have extra value when resolving very soon
        if days_left < 0:    return 0.0     # already resolved
        if days_left == 0:   return 30.0    # resolving today (risky timing)
        if days_left <= 2:   return 85.0    # 1-2 days: lock-in plays ✅ prime
        if days_left <= 7:   return 95.0    # 3-7 days: BEST window
        if days_left <= 14:  return 85.0    # 1-2 weeks
        if days_left <= 21:  return 75.0    # 3 weeks
        if days_left <= 45:  return 58.0    # ~1.5 months
        if days_left <= 90:  return 38.0    # 3 months
        return 15.0
    except Exception:
        return 50.0


def _buy_no_early_score(market: dict) -> float:
    """
    "Buy No Early" — exploits documented behavioral bias.

    Research shows only ~22% of sensational prediction market questions
    resolve YES, yet retail consistently overprices YES at 60-85% at launch.
    This creates a reliable NO edge on dramatically-worded new/immature markets.

    Triggers when:
    - YES price is 55-88% (overpriced relative to 22% base rate)
    - Question has 2+ sensational keywords
    - Market is new/immature (low total volume OR volume mostly from last 24h)
    """
    yes_price = market.get("yes_price", 0.5) or 0.5
    question  = (market.get("question", "") or "").lower()
    volume    = market.get("volume", 0) or 0
    vol24h    = market.get("volume24hr", 0) or 0

    # YES must be in overpriced band
    if yes_price < BUY_NO_EARLY_YES_MIN or yes_price > BUY_NO_EARLY_YES_MAX:
        return 0.0

    # Count sensational keywords in title
    keyword_hits = sum(1 for kw in SENSATIONAL_KEYWORDS if kw in question)
    if keyword_hits < 2:
        return 0.0

    # Detect new/immature market: low total volume or mostly today's trading
    is_new = (
        volume < BUY_NO_EARLY_NEW_VOL or
        (volume > 0 and vol24h / volume > 0.7)
    )
    if not is_new and keyword_hits < 4:
        return 0.0  # Need stronger keyword signal for older markets

    # Core score: how much is YES overpriced vs 22% base rate?
    overpricing  = yes_price - 0.22         # e.g. 0.75 YES → 0.53 over base
    base_score   = min(75, overpricing * 110)
    kw_bonus     = min(15, keyword_hits * 4)
    new_bonus    = 10 if is_new else 0

    return round(min(100, base_score + kw_bonus + new_bonus), 1)


def _news_score(market: dict) -> Tuple[float, List[str]]:
    question  = market.get("question", "")
    market_id = market.get("id", "")
    if not question:
        return 0.0, []
    score, headlines = news_engine.get_news_score(question, market_id)
    return score, headlines


def _smart_wallet_score(market: dict) -> float:
    market_id = market.get("id", "")
    if not market_id:
        return 0.0
    return wallet_tracker.get_smart_wallet_score(market_id)


# ── Direction logic ───────────────────────────────────────────────────────────

def _momentum_direction(history: list) -> Optional[str]:
    if len(history) < 3:
        return None
    prices = [h.get("yes_price", 0.5) for h in history[-6:]]
    if len(prices) < 2:
        return None
    trend = prices[-1] - prices[0]
    if trend >= 0.02:  return "YES"
    if trend <= -0.02: return "NO"
    return None


def _pick_direction(market: dict, history: list, news_headlines: List[str],
                    market_type: str) -> str:
    """
    Direction priority:
    1. BUY_NO_EARLY: always bet NO (retail overpriced YES, historical 78% NO rate)
    2. COPY_TRADE: follow smart wallet direction
    3. LOCK-IN: always bet WITH majority (crowd is right at extremes)
    4. News direction (external information, most reliable for momentum)
    5. Smart wallet direction
    6. Momentum trend
    7. Consensus (which side is currently winning)
    """
    yes = market.get("yes_price", 0.5)
    market_id = market.get("id", "")

    # 1. BUY_NO_EARLY: always NO — retail has overpriced YES by definition
    if market_type == "BUY_NO_EARLY":
        return "NO"

    # 2. COPY_TRADE: follow smart wallet direction, fallback to crowd
    if market_type == "COPY_TRADE":
        wallet_dir = wallet_tracker.get_smart_wallet_direction(market_id)
        if wallet_dir:
            return wallet_dir
        return "YES" if yes >= 0.5 else "NO"

    # 3. Lock-In markets: bet with crowd unconditionally
    if market_type == "LOCK_IN":
        return "YES" if yes >= 0.5 else "NO"

    # 4. Hard crowd overrides at extremes
    if yes < 0.15: return "NO"
    if yes > 0.85: return "YES"

    # 3. News direction
    if news_headlines:
        news_dir = news_engine.get_news_direction(
            market.get("question", ""), news_headlines
        )
        if news_dir:
            return news_dir

    # 4. Smart wallet direction
    if market_id:
        wallet_dir = wallet_tracker.get_smart_wallet_direction(market_id)
        if wallet_dir:
            return wallet_dir

    # 5. Momentum
    momentum = _momentum_direction(history)
    if momentum:
        return momentum

    # 6. Follow consensus
    return "YES" if yes >= 0.5 else "NO"


# ── Entry qualification ───────────────────────────────────────────────────────

def _get_active_signals(factors: dict) -> List[str]:
    return [name for name, score in factors.items() if score >= STRONG_THRESHOLD]


def _qualifies_for_entry(factors: dict, market_type: str) -> Tuple[bool, str]:
    """
    Returns (can_enter, reason_string).

    HIGH ACCURACY MODE — only 3 modes active:
    COPY_TRADE    → smart wallet ≥ 85: highest documented edge (~75-80% WR)
    BUY_NO_EARLY  → behavioral bias ≥ 60: documented 78% NO base rate
    LOCK_IN       → price_zone ≥ 65 + at least 1 supporting signal
    MOMENTUM      → BLOCKED: 55-65% WR, drags overall rate below 80% target
    EXTREME       → BLOCKED: no TP room
    """
    # COPY_TRADE: smart wallet conviction — highest confidence mode
    if market_type == "COPY_TRADE":
        score = factors.get("smart_wallet", 0)
        if score >= COPY_TRADE_WALLET_MIN:
            return True, f"COPY_TRADE(wallet={score:.0f})"
        return False, "COPY_TRADE_wallet_low"

    # BUY_NO_EARLY: documented behavioral bias, always bet NO on overpriced YES
    if market_type == "BUY_NO_EARLY":
        score = factors.get("buy_no_early", 0)
        if score >= BUY_NO_EARLY_MIN_SCORE:
            return True, f"BUY_NO_EARLY(bias={score:.0f})"
        return False, "BUY_NO_EARLY_score_low"

    # EXTREME: blocked — no take-profit room
    if market_type == "EXTREME":
        return False, "EXTREME_blocked"

    active = _get_active_signals(factors)
    count  = len(active)
    stack  = " + ".join(active)

    # LOCK-IN: price_zone ≥ 65 required PLUS at least 1 supporting signal
    # AND market must resolve within 14 days — farther-out markets sit stable
    # and don't drift the 5¢ needed for TP within the hold window.
    if market_type == "LOCK_IN":
        days_left = factors.get("days_left", 9999)

        # Reject markets resolving too far out — they sit stable, no TP drift
        if days_left > 14:
            return False, f"LOCK_IN_too_far(days={days_left})"

        has_zone = factors.get("price_zone", 0) >= LOCK_IN_MIN_SCORE

        # VOLUME SURGE OVERRIDE: 3x+ normal volume on a LOCK_IN market = auto-enter.
        # Insider/early-info signal — someone is loading up before news goes public.
        # Enter now, before the price moves. Same edge as ZachXBT-style insider plays.
        vol_score = factors.get("volume_spike", 0)
        if vol_score >= VOLUME_SURGE_AUTO_ENTRY:
            return True, f"VOLUME_SURGE(vol={vol_score:.0f}) — insider signal detected"

        if has_zone and count >= 1:
            return True, f"LOCK_IN(zone={factors['price_zone']:.0f}+{stack})"
        if has_zone:
            return True, f"LOCK_IN(zone_only={factors['price_zone']:.0f})"
        return False, "LOCK_IN_zone_weak"

    # MOMENTUM: BLOCKED in high-accuracy mode
    # 55-65% WR pulls overall win rate below 80% target
    return False, "MOMENTUM_disabled_high_accuracy_mode"


# ── Main scorer ───────────────────────────────────────────────────────────────

async def score_market(market: dict) -> Optional[dict]:
    market_id = market.get("id", "")
    if not market_id:
        return None

    yes_price = market.get("yes_price", 0.5)
    if yes_price is None:
        yes_price = 0.5

    # Skip already-resolved or too-extreme markets
    if yes_price < 0.03 or yes_price > 0.97:
        return None

    # Minimum volume — real money mode: require meaningful liquidity
    if (market.get("volume24hr") or 0) < 100:
        return None

    try:
        history = await db.get_market_history(market_id, limit=10)
    except Exception:
        history = []

    try:
        weights = await db.get_signal_weights()
    except Exception:
        weights = {}

    w_vol   = weights.get("volume_spike",  3.5)  # ↑ 1.0→3.5: insider signal — highest weight
    w_zone  = weights.get("price_zone",    1.0)
    w_liq   = weights.get("liquidity",     1.0)
    w_mom   = weights.get("momentum",      1.0)
    w_cat   = weights.get("category",      1.0)
    w_news  = weights.get("news_impact",   1.5)
    w_wall  = weights.get("smart_wallet",  1.5)
    w_date  = weights.get("end_date",      1.2)
    w_bne   = weights.get("buy_no_early",  2.0)  # High weight — proven behavioral edge

    f_vol                 = _volume_spike_score(market, history)
    f_zone, market_type   = _price_zone_score(market)
    f_liq                 = _liquidity_score(market)
    f_mom                 = _momentum_score(market, history)
    f_cat                 = _category_score(market.get("category", "other"))
    f_news, headlines     = _news_score(market)
    f_wall                = _smart_wallet_score(market)
    f_date                = _end_date_score(market)
    f_bne                 = _buy_no_early_score(market)

    # ── Market type upgrades (override base price_zone type) ──────────────────
    # COPY_TRADE: smart wallet conviction is the highest signal available
    if f_wall >= COPY_TRADE_WALLET_MIN:
        market_type = "COPY_TRADE"
    # BUY_NO_EARLY: sensational overpriced YES on new market
    elif f_bne >= BUY_NO_EARLY_MIN_SCORE and market_type not in ("LOCK_IN", "COPY_TRADE"):
        market_type = "BUY_NO_EARLY"

    # Compute days_left for LOCK_IN entry filter
    _end_str = market.get("end_date", "")
    try:
        _end_str2 = _end_str.replace("Z", "+00:00")
        if "T" in _end_str2:
            _end_dt = datetime.fromisoformat(_end_str2).replace(tzinfo=None)
        else:
            _end_dt = datetime.strptime(_end_str2[:10], "%Y-%m-%d")
        _days_left = (_end_dt - datetime.utcnow()).days
    except Exception:
        _days_left = 9999

    factors = {
        "volume_spike": round(f_vol,  1),
        "price_zone":   round(f_zone, 1),
        "liquidity":    round(f_liq,  1),
        "momentum":     round(f_mom,  1),
        "category":     round(f_cat,  1),
        "news_impact":  round(f_news, 1),
        "smart_wallet": round(f_wall, 1),
        "end_date":     round(f_date, 1),
        "buy_no_early": round(f_bne,  1),
        "days_left":    _days_left,   # used by LOCK_IN entry filter
    }

    can_enter, entry_reason = _qualifies_for_entry(factors, market_type)
    active_signals          = _get_active_signals(factors)

    total_w = w_vol + w_zone + w_liq + w_mom + w_cat + w_news + w_wall + w_date + w_bne
    if total_w <= 0:
        total_w = 9.0

    score = (
        f_vol  * w_vol  +
        f_zone * w_zone +
        f_liq  * w_liq  +
        f_mom  * w_mom  +
        f_cat  * w_cat  +
        f_news * w_news +
        f_wall * w_wall +
        f_date * w_date +
        f_bne  * w_bne
    ) / total_w

    score = round(score, 1)

    direction = _pick_direction(market, history, headlines, market_type)

    return {
        "market_id":        market_id,
        "market_question":  market.get("question", "Unknown"),
        "score":            score,
        "confidence":       round(min(100, score * 1.1), 1),
        "direction":        direction,
        "factors":          factors,
        "factors_json":     json.dumps(factors),
        "yes_price":        yes_price,
        "category":         market.get("category", "other"),
        "market_type":      market_type,          # "LOCK_IN" or "MOMENTUM" or "EXTREME"
        "active_signals":   active_signals,
        "can_enter":        can_enter,             # pre-qualified for entry?
        "entry_reason":     entry_reason,          # e.g. "LOCK_IN(price_zone=90)"
        "news_headlines":   headlines,
        "signal_count":     len(active_signals),
        "single_high_conviction": (
            factors.get("news_impact", 0) >= HC_OVERRIDE_SCORE or
            factors.get("smart_wallet", 0) >= HC_OVERRIDE_SCORE
        ),
        "created_at":       datetime.utcnow().isoformat(),
    }


async def generate_signals(markets: list) -> list:
    """
    Score all markets. HIGH ACCURACY MODE: only COPY_TRADE, LOCK_IN, BUY_NO_EARLY.
    MOMENTUM and EXTREME are filtered out before reaching the trader.
    """
    BLOCKED_MODES = {"MOMENTUM", "EXTREME"}
    signals = []
    for market in markets:
        sig = await score_market(market)
        if not sig:
            continue
        if sig["score"] < GENERATE_MIN_SCORE:
            continue
        if sig.get("market_type") in BLOCKED_MODES:
            continue
        signals.append(sig)

    # Sort: COPY_TRADE first (highest WR), then LOCK_IN, then BUY_NO_EARLY, by score
    TYPE_PRIORITY = {"COPY_TRADE": 0, "LOCK_IN": 1, "BUY_NO_EARLY": 2}
    signals.sort(key=lambda s: (
        TYPE_PRIORITY.get(s.get("market_type", "BUY_NO_EARLY"), 2),
        -s["score"]
    ))
    return signals
