"""
Crypto Leverage Trading Agent — Hyperliquid Blueprint + Self-Learning (Paper Mode)

Architecture based on the production Hyperliquid blueprint:
  - Regime classifier (trend / chop / breakout / event / cascade)
  - 8-factor setup scorecard (0–100) with LEARNED factor weights
  - 4 strategy families (trend pullback, range sweep, breakout, failed breakout)
  - Hard risk engine (0.25–0.5% per-trade, 2% daily drawdown kill, 3-loss cool-off)
  - Kill switches for system degradation
  - Self-learning: weights adjusted every 5 closed trades based on outcome analysis

This runs in paper/simulation mode against live Kraken prices.
Graduate to real Hyperliquid trading after shadow-mode validation.
"""

import json
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
import math

from database import (
    save_crypto_trade, get_open_crypto_trades, close_crypto_trade,
    get_crypto_portfolio, update_crypto_portfolio, get_all_crypto_trades,
    set_crypto_leverage, save_crypto_trade_meta,
)
from crypto_learner import (
    get_factor_weights, get_dynamic_threshold,
    apply_weights_to_breakdown, maybe_learn, get_learning_stats,
)

# ── Risk parameters (Hyperliquid blueprint §7) ────────────────────────────────
PER_TRADE_RISK_PCT     = 0.004   # 0.4% of equity per trade
MAX_OPEN_RISK_PCT      = 0.015   # 1.5% total open risk at once
MAX_DAILY_DRAWDOWN_PCT = 0.02    # 2% daily drawdown → kill switch
MAX_CONSECUTIVE_LOSSES = 3       # cool-off after 3 in a row
MAX_OPEN_TRADES        = 4       # max simultaneous positions
# MIN_SETUP_SCORE is now dynamic — see get_dynamic_threshold() from crypto_learner
REDUCED_SIZE_SCORE     = 75      # 75–84 → reduced size
FULL_SIZE_SCORE        = 85      # 85+ → full allowed size

# Stop-loss / take-profit in % move against/for the position
STOP_LOSS_PCT          = 0.03    # 3% adverse move → stop out
TAKE_PROFIT_PCT_1R     = 0.04    # 4% gain → take partial (1R)
TAKE_PROFIT_PCT_2R     = 0.08    # 8% gain → close runner

# Per-asset max leverage tier (blueprint: leverage policy must be asset-aware)
ASSET_MAX_LEVERAGE = {
    "BTCUSDT":  10,
    "ETHUSDT":  10,
    "SOLUSDT":  5,
    "BNBUSDT":  5,
    "AVAXUSDT": 3,
}

# In-memory state
_cool_off_until:    Optional[datetime] = None
_consecutive_losses: int               = 0
_daily_start_balance: float            = 10000.0
_daily_date:         Optional[str]     = None
_price_history:      Dict[str, List]   = {}   # symbol → list of close prices


# ── Feature engine ────────────────────────────────────────────────────────────

def _compute_features(market: Dict) -> Dict:
    """
    Derive technical features from klines and book data.
    Returns a feature dict used by regime classifier and setup detector.
    """
    klines  = market.get("klines_5m", [])
    if len(klines) < 20:
        return {}

    closes  = [k["close"]  for k in klines]
    highs   = [k["high"]   for k in klines]
    lows    = [k["low"]    for k in klines]
    volumes = [k["volume"] for k in klines]

    # Short-term momentum: % change over last 3 candles
    mom3  = (closes[-1] - closes[-4]) / closes[-4] * 100 if closes[-4] > 0 else 0

    # Medium momentum: % change over last 12 candles (1h on 5m chart)
    mom12 = (closes[-1] - closes[-13]) / closes[-13] * 100 if len(closes) >= 13 and closes[-13] > 0 else 0

    # Volatility: std dev of last 20 close % changes
    pct_changes = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
    recent_changes = pct_changes[-19:]
    mean_chg = sum(recent_changes) / len(recent_changes) if recent_changes else 0
    variance = sum((x - mean_chg) ** 2 for x in recent_changes) / max(len(recent_changes), 1)
    volatility = math.sqrt(variance) * 100  # as %

    # RSI (14-period)
    rsi = _compute_rsi(closes, period=14)

    # Volume spike: current vs 20-bar average
    avg_vol = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else volumes[-1]
    vol_spike = volumes[-1] / avg_vol if avg_vol > 0 else 1.0

    # Structure: range of last 20 bars
    range_high = max(highs[-20:])
    range_low  = min(lows[-20:])
    range_size = (range_high - range_low) / range_low * 100 if range_low > 0 else 0

    # Distance from range extremes (0 = at low, 1 = at high)
    price_position = (closes[-1] - range_low) / (range_high - range_low) if range_high > range_low else 0.5

    # EMA 9 and EMA 21 for trend
    ema9  = _ema(closes, 9)
    ema21 = _ema(closes, 21)
    ema_trend = (ema9 - ema21) / ema21 * 100 if ema21 > 0 else 0  # + = uptrend, - = downtrend

    # Compression: very tight range in last 5 bars vs last 20
    recent_range = (max(highs[-5:]) - min(lows[-5:])) / closes[-1] * 100
    is_compressed = recent_range < (range_size * 0.25)

    # Book imbalance
    book_imbalance = market.get("book_imbalance", 0.5)

    return {
        "price":          closes[-1],
        "mom3":           mom3,
        "mom12":          mom12,
        "volatility":     volatility,
        "rsi":            rsi,
        "vol_spike":      vol_spike,
        "range_high":     range_high,
        "range_low":      range_low,
        "range_size":     range_size,
        "price_position": price_position,
        "ema_trend":      ema_trend,
        "is_compressed":  is_compressed,
        "book_imbalance": book_imbalance,
        "price_change_24h": market.get("price_change_pct", 0),
    }


def _compute_rsi(closes: List[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    gains, losses = [], []
    for i in range(1, period + 1):
        diff = closes[-period + i - 1] - closes[-period + i - 2] if i > 1 else closes[-period] - closes[-period - 1]
        (gains if diff >= 0 else losses).append(abs(diff))
    avg_gain = sum(gains) / period if gains else 0
    avg_loss = sum(losses) / period if losses else 0.001
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 1)


def _ema(values: List[float], period: int) -> float:
    if len(values) < period:
        return values[-1] if values else 0
    k = 2 / (period + 1)
    ema = sum(values[:period]) / period
    for v in values[period:]:
        ema = v * k + ema * (1 - k)
    return ema


# ── Regime classifier (blueprint §3) ─────────────────────────────────────────

def classify_regime(features: Dict) -> str:
    """
    Labels the market as: trend | chop | breakout | event | cascade
    """
    if not features:
        return "unknown"

    vol   = features.get("volatility", 0)
    mom12 = features.get("mom12", 0)
    rsi   = features.get("rsi", 50)
    ema_t = features.get("ema_trend", 0)
    spike = features.get("vol_spike", 1)
    comp  = features.get("is_compressed", False)

    # Cascade / liquidation event: extreme volatility + extreme RSI
    if vol > 3.0 and (rsi < 15 or rsi > 85):
        return "cascade"

    # Event volatility: sharp move + huge volume spike
    if abs(mom12) > 5 and spike > 3.0:
        return "event"

    # Breakout after compression
    if comp and spike > 2.0 and abs(mom12) > 1.5:
        return "breakout"

    # Clear trend: EMA aligned + sustained momentum
    if abs(ema_t) > 0.5 and abs(mom12) > 2.0:
        return "trend"

    # Default: choppy / range-bound
    return "chop"


# ── Setup scorecard (blueprint §6) ────────────────────────────────────────────

def score_setup(strategy: str, features: Dict, regime: str) -> Tuple[float, Dict]:
    """
    Scores a potential trade 0–100 across 8 factors.
    Applies learned factor weights from crypto_learner to re-weight the scorecard.
    Returns (weighted_score, raw_breakdown_dict).
    """
    raw = {}

    # 1. Higher timeframe alignment (15 pts) — EMA trend direction
    ema_t = features.get("ema_trend", 0)
    if strategy == "LONG":
        raw["htf_alignment"] = min(15, max(0, int((ema_t + 1) * 7.5)))
    else:
        raw["htf_alignment"] = min(15, max(0, int((-ema_t + 1) * 7.5)))

    # 2. Regime clarity (15 pts) — does regime match strategy?
    regime_match = {
        "trend_pullback": {"trend": 15, "breakout": 8, "chop": 3, "event": 0, "cascade": 0},
        "range_sweep":    {"chop": 15, "trend": 5, "breakout": 3, "event": 0, "cascade": 0},
        "breakout":       {"breakout": 15, "trend": 8, "chop": 2, "event": 5, "cascade": 0},
        "failed_breakout":{"chop": 15, "trend": 5, "breakout": 10, "event": 0, "cascade": 0},
    }
    raw["regime_clarity"] = regime_match.get(strategy, {}).get(regime, 0)

    # 3. Local structure quality (15 pts) — range size and position
    pp = features.get("price_position", 0.5)
    rs = features.get("range_size", 0)
    if strategy in ("trend_pullback", "breakout") and "LONG" in str(features):
        raw["structure"] = 15 if (pp < 0.35 and rs > 1.5) else 8 if pp < 0.5 else 3
    elif strategy == "range_sweep":
        raw["structure"] = 13 if (pp < 0.2 or pp > 0.8) else 5
    else:
        raw["structure"] = 10 if rs > 1.0 else 5

    # 4. Volume confirmation (10 pts)
    vs = features.get("vol_spike", 1.0)
    raw["volume"] = min(10, int(vs * 4))

    # 5. Book/flow confirmation (10 pts)
    bi = features.get("book_imbalance", 0.5)
    if strategy in ("LONG", "trend_pullback", "breakout"):
        raw["book_flow"] = min(10, int((bi - 0.4) * 50)) if bi > 0.5 else 0
    else:
        raw["book_flow"] = min(10, int((0.6 - bi) * 50)) if bi < 0.5 else 0

    # 6. Reward-to-risk quality (15 pts) — RSI gives contrarian edge
    rsi = features.get("rsi", 50)
    if strategy == "LONG":
        raw["r2r"] = min(15, max(0, int((50 - rsi) / 50 * 15 + 5)))
    else:
        raw["r2r"] = min(15, max(0, int((rsi - 50) / 50 * 15 + 5)))

    # 7. Liquidity quality (10 pts) — vol spike not too extreme (slippage risk)
    if 1.2 < vs < 4.0:
        raw["liquidity"] = 10
    elif vs <= 1.2:
        raw["liquidity"] = 5
    else:
        raw["liquidity"] = 3

    # 8. Event/news risk (10 pts) — penalise cascade/event regime
    if regime in ("cascade", "event"):
        raw["event_risk"] = 0
    elif regime == "breakout":
        raw["event_risk"] = 7
    else:
        raw["event_risk"] = 10

    # ── Apply learned factor weights (renormalized to 0–100) ─────────────────
    weighted_score = apply_weights_to_breakdown(raw)
    return weighted_score, raw


# ── Strategy families (blueprint §5) ─────────────────────────────────────────

def detect_setups(symbol: str, features: Dict, regime: str) -> List[Dict]:
    """
    Returns a list of candidate trade setups for this asset.
    Uses the dynamic score threshold from the self-learning engine.
    Each setup has: strategy, direction, score, score_breakdown, entry_reason.
    """
    setups     = []
    threshold  = get_dynamic_threshold()   # ← adaptive, learned from outcomes

    mom3   = features.get("mom3", 0)
    mom12  = features.get("mom12", 0)
    rsi    = features.get("rsi", 50)
    pp     = features.get("price_position", 0.5)
    ema_t  = features.get("ema_trend", 0)
    vs     = features.get("vol_spike", 1.0)
    comp   = features.get("is_compressed", False)
    bi     = features.get("book_imbalance", 0.5)

    # A. Trend pullback continuation
    if regime == "trend":
        if ema_t > 0.3 and mom12 > 1.5 and rsi < 60 and pp < 0.5 and mom3 > -0.5:
            score, breakdown = score_setup("trend_pullback", {**features, "LONG": True}, regime)
            if score >= threshold:
                setups.append({
                    "strategy":  "trend_pullback",
                    "direction": "LONG",
                    "score":     score,
                    "breakdown": breakdown,
                    "regime":    regime,
                    "reason":    f"Uptrend pullback | EMA trend {ema_t:.2f}% | RSI {rsi}",
                })
        if ema_t < -0.3 and mom12 < -1.5 and rsi > 40 and pp > 0.5 and mom3 < 0.5:
            score, breakdown = score_setup("trend_pullback", {**features, "SHORT": True}, regime)
            if score >= threshold:
                setups.append({
                    "strategy":  "trend_pullback",
                    "direction": "SHORT",
                    "score":     score,
                    "breakdown": breakdown,
                    "regime":    regime,
                    "reason":    f"Downtrend pullback | EMA trend {ema_t:.2f}% | RSI {rsi}",
                })

    # B. Range sweep and reclaim
    if regime in ("chop", "trend"):
        if pp < 0.15 and rsi < 35 and bi > 0.52:
            score, breakdown = score_setup("range_sweep", features, regime)
            if score >= threshold:
                setups.append({
                    "strategy":  "range_sweep",
                    "direction": "LONG",
                    "score":     score,
                    "breakdown": breakdown,
                    "regime":    regime,
                    "reason":    f"Low sweep reclaim | PP {pp:.2f} | RSI {rsi} | Book {bi:.2f}",
                })
        if pp > 0.85 and rsi > 65 and bi < 0.48:
            score, breakdown = score_setup("range_sweep", features, regime)
            if score >= threshold:
                setups.append({
                    "strategy":  "range_sweep",
                    "direction": "SHORT",
                    "score":     score,
                    "breakdown": breakdown,
                    "regime":    regime,
                    "reason":    f"High sweep rejection | PP {pp:.2f} | RSI {rsi} | Book {bi:.2f}",
                })

    # C. Breakout after compression
    if regime == "breakout" and comp:
        if mom3 > 1.0 and vs > 2.0 and bi > 0.55:
            score, breakdown = score_setup("breakout", features, regime)
            if score >= threshold:
                setups.append({
                    "strategy":  "breakout",
                    "direction": "LONG",
                    "score":     score,
                    "breakdown": breakdown,
                    "regime":    regime,
                    "reason":    f"Breakout from compression | Vol spike {vs:.1f}x | Mom {mom3:.2f}%",
                })
        if mom3 < -1.0 and vs > 2.0 and bi < 0.45:
            score, breakdown = score_setup("breakout", features, regime)
            if score >= threshold:
                setups.append({
                    "strategy":  "breakout",
                    "direction": "SHORT",
                    "score":     score,
                    "breakdown": breakdown,
                    "regime":    regime,
                    "reason":    f"Breakdown from compression | Vol spike {vs:.1f}x | Mom {mom3:.2f}%",
                })

    # D. Failed breakout / failed breakdown
    if regime in ("chop", "breakout"):
        if pp > 0.9 and mom3 < -0.8 and vs > 1.5 and bi < 0.47:
            score, breakdown = score_setup("failed_breakout", features, regime)
            if score >= threshold:
                setups.append({
                    "strategy":  "failed_breakout",
                    "direction": "SHORT",
                    "score":     score,
                    "breakdown": breakdown,
                    "regime":    regime,
                    "reason":    f"Failed breakout snapback | PP {pp:.2f} | Mom reversal {mom3:.2f}%",
                })
        if pp < 0.1 and mom3 > 0.8 and vs > 1.5 and bi > 0.53:
            score, breakdown = score_setup("failed_breakout", features, regime)
            if score >= threshold:
                setups.append({
                    "strategy":  "failed_breakout",
                    "direction": "LONG",
                    "score":     score,
                    "breakdown": breakdown,
                    "regime":    regime,
                    "reason":    f"Failed breakdown reclaim | PP {pp:.2f} | Mom reversal {mom3:.2f}%",
                })

    return setups


# ── Risk engine (blueprint §7) ────────────────────────────────────────────────

async def _risk_check(equity: float, setup_score: int) -> Tuple[bool, str]:
    global _cool_off_until, _daily_start_balance, _daily_date

    now     = datetime.utcnow()
    today   = now.strftime("%Y-%m-%d")

    # Reset daily tracker at new day
    if _daily_date != today:
        _daily_date          = today
        _daily_start_balance = equity

    # Kill switch: daily drawdown
    daily_drawdown = (_daily_start_balance - equity) / _daily_start_balance
    if daily_drawdown >= MAX_DAILY_DRAWDOWN_PCT:
        return False, f"Daily drawdown kill switch ({daily_drawdown*100:.1f}% ≥ 2%)"

    # Cool-off after consecutive losses
    if _cool_off_until and now < _cool_off_until:
        remaining = int((_cool_off_until - now).total_seconds() / 60)
        return False, f"Cool-off period active ({remaining}m remaining after {MAX_CONSECUTIVE_LOSSES} losses)"

    # Score threshold — adaptive (learned from trade outcomes)
    threshold = get_dynamic_threshold()
    if setup_score < threshold:
        return False, f"Setup score too low ({setup_score} < {threshold} adaptive threshold)"

    return True, "OK"


def _position_size(equity: float, entry_price: float, stop_pct: float,
                   setup_score: int, multiplier: int) -> Tuple[float, float]:
    """
    Calculate position size based on risk budget and score.
    Returns (cost_in_usd, quantity_of_asset).
    Blueprint: size from stop distance + confidence + volatility.
    """
    # Score-based size scaling
    if setup_score >= FULL_SIZE_SCORE:
        risk_pct = PER_TRADE_RISK_PCT          # full size
    elif setup_score >= REDUCED_SIZE_SCORE:
        risk_pct = PER_TRADE_RISK_PCT * 0.6    # reduced
    else:
        risk_pct = PER_TRADE_RISK_PCT * 0.3    # probe only

    risk_dollars     = equity * risk_pct
    # Cost = risk / stop_pct (risk-based sizing)
    cost             = min(risk_dollars / stop_pct, equity * 0.05)  # cap at 5% of equity
    cost             = round(max(cost, 5.0), 2)
    leveraged_exposure = round(cost * multiplier, 2)
    quantity         = round(leveraged_exposure / entry_price, 8)
    return cost, quantity


# ── Main entry point ──────────────────────────────────────────────────────────

async def run_crypto_cycle(markets: List[Dict]) -> List[Dict]:
    """
    Called every cycle. Processes each asset, detects setups, enters trades.
    Returns list of new trades entered this cycle.
    """
    global _consecutive_losses

    portfolio  = await get_crypto_portfolio()
    equity     = portfolio.get("cash_balance", 10000)
    multiplier = min(
        portfolio.get("leverage_multiplier", 2),
        10  # hard safety cap
    )

    new_trades = []

    for market in markets:
        symbol   = market.get("symbol")
        features = _compute_features(market)
        if not features:
            continue

        regime = classify_regime(features)

        # Skip cascade and event regimes (blueprint §7: no trading during extremes)
        if regime in ("cascade", "event", "unknown"):
            continue

        setups = detect_setups(symbol, features, regime)
        if not setups:
            continue

        # Take the highest-scoring setup
        best = max(setups, key=lambda s: s["score"])

        # Risk check
        open_trades = await get_open_crypto_trades()
        if len(open_trades) >= MAX_OPEN_TRADES:
            continue

        # Don't duplicate — no two trades on same symbol
        if any(t["symbol"] == symbol for t in open_trades):
            continue

        allowed, reason = await _risk_check(equity, best["score"])
        if not allowed:
            print(f"[CRYPTO RISK] Blocked {symbol}: {reason}")
            continue

        # Asset-aware leverage cap
        asset_max_lev = ASSET_MAX_LEVERAGE.get(symbol, 3)
        effective_lev = min(multiplier, asset_max_lev)

        entry_price = features["price"]
        cost, quantity = _position_size(
            equity, entry_price, STOP_LOSS_PCT, best["score"], effective_lev
        )

        if cost > equity * 0.5:  # safety: never more than 50% cash in one trade
            continue

        trade = {
            "symbol":              symbol,
            "direction":           best["direction"],
            "entry_price":         entry_price,
            "quantity":            quantity,
            "cost":                cost,
            "leveraged_exposure":  round(cost * effective_lev, 2),
            "leverage_multiplier": effective_lev,
            "signal_reason":       f"[{best['strategy'].upper()}] {best['reason']} | Score {best['score']}",
            "status":              "OPEN",
            "created_at":          datetime.utcnow().isoformat(),
        }

        trade_id    = await save_crypto_trade(trade)
        trade["id"] = trade_id
        await update_crypto_portfolio(cash_delta=-cost, invested_delta=cost)

        # ── Save entry snapshot for post-hoc learning ─────────────────────────
        snapshot = {
            "strategy":  best["strategy"],
            "direction": best["direction"],
            "regime":    best.get("regime", "unknown"),
            "score":     best["score"],
            "breakdown": best["breakdown"],
            "features": {
                k: round(v, 5) if isinstance(v, float) else v
                for k, v in features.items()
                if isinstance(v, (int, float, bool))
            },
        }
        await save_crypto_trade_meta(trade_id, json.dumps(snapshot))

        print(f"[CRYPTO {effective_lev}x] {best['direction']} {symbol} @ ${entry_price:,.2f} "
              f"| {best['strategy']} score {best['score']:.1f} | threshold {get_dynamic_threshold()} | cost ${cost:.2f}")
        new_trades.append(trade)

    return new_trades


async def update_open_crypto_trades(markets: List[Dict]) -> None:
    """
    Check all open crypto trades for stop-loss / take-profit / timeout.
    Blueprint: every live trade has three exit layers.
    """
    global _consecutive_losses, _cool_off_until

    open_trades = await get_open_crypto_trades()
    if not open_trades:
        return

    market_map = {m["symbol"]: m for m in markets}

    for trade in open_trades:
        symbol  = trade["symbol"]
        market  = market_map.get(symbol)
        if not market:
            continue

        features   = _compute_features(market)
        cur_price  = features.get("price", trade["entry_price"])
        entry      = trade["entry_price"]
        direction  = trade["direction"]
        created    = datetime.fromisoformat(trade["created_at"])
        age_hours  = (datetime.utcnow() - created).total_seconds() / 3600

        # Calculate current P&L
        if direction == "LONG":
            pct_move = (cur_price - entry) / entry
        else:
            pct_move = (entry - cur_price) / entry

        lev = trade.get("leverage_multiplier", 2)
        lev_pct = pct_move * lev  # leveraged % gain/loss

        reason = None
        exit_price = cur_price

        # Layer 1: Stop loss (tighter due to leverage)
        if lev_pct <= -STOP_LOSS_PCT:
            reason = "STOP_LOSS"

        # Layer 2: Take profit 2R
        elif lev_pct >= TAKE_PROFIT_PCT_2R:
            reason = "TAKE_PROFIT"

        # Layer 3: Time-based exit (8 hours max hold for crypto)
        elif age_hours >= 8:
            reason = "TIMEOUT"

        if reason:
            cost    = trade["cost"]
            qty     = trade["quantity"]
            # Actual PnL = quantity * (exit - entry) * direction_sign
            sign    = 1 if direction == "LONG" else -1
            raw_pnl = qty * (exit_price - entry) * sign
            pnl     = round(raw_pnl, 2)
            won     = pnl > 0
            outcome = "WIN" if won else "STOP_LOSS" if reason == "STOP_LOSS" else "LOSS"

            await close_crypto_trade(trade["id"], exit_price, pnl, outcome)
            cash_back = round(cost + pnl, 2)
            await update_crypto_portfolio(
                cash_delta=cash_back,
                pnl_delta=pnl,
                invested_delta=-cost,
                win=won
            )

            # Consecutive loss tracking (blueprint §7)
            if won:
                _consecutive_losses = 0
            else:
                _consecutive_losses += 1
                if _consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
                    _cool_off_until = datetime.utcnow() + timedelta(minutes=30)
                    print(f"[CRYPTO RISK] {MAX_CONSECUTIVE_LOSSES} consecutive losses — "
                          f"30-minute cool-off activated")

            print(f"[CRYPTO] Closed ({reason}): {outcome} {symbol} | "
                  f"PnL ${pnl:+.2f} ({lev_pct*100:+.1f}% lev)")

            # ── Trigger self-learning after every batch of closed trades ──────
            await maybe_learn()


async def get_crypto_portfolio_summary() -> Dict:
    """Full portfolio summary + learning stats for WebSocket broadcast."""
    port      = await get_crypto_portfolio()
    open_t    = await get_open_crypto_trades()
    all_t     = await get_all_crypto_trades(200)

    closed    = [t for t in all_t if t["status"] != "OPEN"]
    wins      = [t for t in closed if (t.get("pnl") or 0) > 0]
    losses    = [t for t in closed if (t.get("pnl") or 0) <= 0]
    stops     = [t for t in closed if t.get("status") == "STOP_LOSS"]

    total_pnl = sum(t.get("pnl", 0) for t in closed)
    win_rate  = round(len(wins) / max(len(closed), 1) * 100, 1)
    roi_pct   = round(total_pnl / 10000 * 100, 2)

    return {
        "cash_balance":        round(port.get("cash_balance", 10000), 2),
        "total_pnl":           round(total_pnl, 2),
        "roi_pct":             roi_pct,
        "win_rate":            win_rate,
        "total_trades":        len(all_t),
        "open_trades":         len(open_t),
        "wins":                len(wins),
        "losses":              len(losses),
        "liquidations":        len(stops),
        "leverage_multiplier": port.get("leverage_multiplier", 2),
        "consecutive_losses":  _consecutive_losses,
        "cool_off_active":     _cool_off_until is not None and datetime.utcnow() < _cool_off_until,
        "daily_drawdown_pct":  round((_daily_start_balance - port.get("cash_balance", 10000))
                                     / max(_daily_start_balance, 1) * 100, 2),
        "learning":            get_learning_stats(),
    }
