"""
Paper Trader â 3-STRATEGY MODE

Strategy 1: NEAR_CERTAINTY (Near-Certainty Grinder)
  Entry: 80-95% probability, resolves <7d, verified via Binance or Haiku
  Exit:  Hold to resolution â NO TP/SL (the whole point is to hold to $1.00)
  Max hold: 48h timeout (market should resolve by then)

Strategy 2: VOLUME_SPIKE (Volume Spike Trading)
  Entry: 3x+ volume surge detected, Haiku confirms direction
  Exit:  +4Â¢ TP / -3Â¢ SL / 2h timeout

Strategy 3: BINANCE_ARB (Binance Price Lag Arbitrage)
  Entry: >5% divergence between Binance price and Polymarket odds
  Exit:  +5Â¢ TP / -4Â¢ SL / 30min timeout

Legacy modes (LOCK_IN, BUY_NO_EARLY, MOMENTUM, COPY_TRADE, LLM_ANALYSIS)
kept for exit management of any remaining open trades, but NO NEW entries.

40 SIMULTANEOUS POSITIONS | Kelly Criterion sizing | $100k balance
"""

import json
from datetime import datetime, timedelta
from typing import Optional, Tuple

import database as db
from trade_explainer import explain_entry, explain_exit, generate_lesson
import self_improvement_engine as sie


def _market_days_left(market: Optional[dict]) -> float:
    """Return days until market resolves. Returns 9999 if market/end_date unknown."""
    if not market:
        return 9999.0
    end_date_str = market.get("end_date", "")
    if not end_date_str:
        return 9999.0
    try:
        end_date_str = end_date_str.replace("Z", "+00:00")
        if "T" in end_date_str:
            end_dt = datetime.fromisoformat(end_date_str).replace(tzinfo=None)
        else:
            end_dt = datetime.strptime(end_date_str[:10], "%Y-%m-%d")
        return max(0, (end_dt - datetime.utcnow()).total_seconds() / 86400)
    except Exception:
        return 9999.0


def _lock_in_exit_params(market: Optional[dict]) -> Tuple[float, float, float]:
    """Legacy LOCK_IN exit params â only used for draining old open trades."""
    days = _market_days_left(market)
    if days <= 0.5:
        return 0.12, 0.09, 36.0
    if days <= 2:
        return 0.07, 0.09, 24.0
    return 0.05, 0.09, 12.0


# ââ NEW STRATEGY EXIT CONSTANTS âââââââââââââââââââââââââââââââââââââââââââââââ

# Strategy 1: NEAR_CERTAINTY â hold to resolution
NEAR_CERTAINTY_HOLD_HOURS = 168.0  # Max 7 days â market should resolve by then

# Strategy 2: VOLUME_SPIKE
VOLUME_SPIKE_TP         = 0.04   # +4Â¢ take-profit
VOLUME_SPIKE_SL         = 0.03   # -3Â¢ stop-loss
VOLUME_SPIKE_HOLD_HOURS = 2.0    # 2h timeout

# Strategy 3: BINANCE_ARB
BINANCE_ARB_TP         = 0.05   # +5Â¢ take-profit
BINANCE_ARB_SL         = 0.04   # -4Â¢ stop-loss
BINANCE_ARB_HOLD_HOURS = 0.5    # 30min timeout (speed matters)


# ââ Legacy mode constants (for draining old open trades) ââââââââââââââââââââââ
COPY_TRADE_TP         = 0.04
COPY_TRADE_SL         = 0.03
COPY_TRADE_HOLD_HOURS = 2
BUY_NO_EARLY_TP         = 0.06
BUY_NO_EARLY_SL         = 0.05
BUY_NO_EARLY_HOLD_HOURS = 6
LOCK_IN_TP         = 0.05
LOCK_IN_SL         = 0.09
LOCK_IN_HOLD_HOURS = 12
MOMENTUM_TP        = 0.06
MOMENTUM_SL        = 0.04
MOMENTUM_HOLD_HOURS = 2


# ââ Shared constants âââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
ENTRY_THRESHOLD        = 40
MAX_OPEN_TRADES        = 40
BASE_RISK_PCT          = 0.005
LEARN_RATE             = 0.20

# Win probability estimates per mode (for Kelly Criterion)
KELLY_WIN_PROBS = {
    "NEAR_CERTAINTY": 0.85,   # High â that's the whole point
    "VOLUME_SPIKE":   0.65,   # Moderate â spike direction is uncertain
    "BINANCE_ARB":    0.72,   # Good â exchange price is leading indicator
    # Legacy (for sizing any remaining open trades)
    "COPY_TRADE":     0.75,
    "BUY_NO_EARLY":   0.70,
    "LOCK_IN":        0.78,
    "MOMENTUM":       0.55,
    "LLM_ANALYSIS":   0.65,
}

# Market types that are allowed to take NO direction
NO_ALLOWED_TYPES = {
    "BUY_NO_EARLY", "LOCK_IN", "LLM_ANALYSIS",
    "NEAR_CERTAINTY", "VOLUME_SPIKE", "BINANCE_ARB",
}


# ââ Kelly Criterion position sizing âââââââââââââââââââââââââââââââââââââââââââ

def _kelly_position_size(portfolio: dict, signal: dict) -> float:
    """
    Kelly Criterion: f* = (b*p - q) / b
    Uses 25% fractional Kelly. Bounds: min 0.1%, max 0.2% of capital.
    """
    cash         = portfolio.get("cash_balance", 10000)
    market_type  = signal.get("market_type", "MOMENTUM")
    direction    = signal.get("direction", "YES")
    yes_price    = signal.get("yes_price", 0.5)
    entry_price  = yes_price if direction == "YES" else (1 - yes_price)

    if entry_price <= 0 or entry_price >= 1:
        return round(cash * BASE_RISK_PCT, 2)

    p = KELLY_WIN_PROBS.get(market_type, 0.58)
    q = 1 - p
    b = (1 - entry_price) / entry_price

    if b <= 0:
        return round(cash * BASE_RISK_PCT, 2)

    kelly = (b * p - q) / b
    if kelly <= 0:
        return round(cash * 0.001, 2)

    kelly_frac = kelly * 0.25

    # Score bonus: higher confidence = up to 20% more
    score_bonus = min(0.2, (signal.get("score", 50) - 50) / 250)
    kelly_frac  = kelly_frac * (1 + score_bonus)

    bet = cash * kelly_frac
    # Clamp: minimum 0.2%, maximum 0.5% per trade ($200-$500 on $100k)
    return round(max(cash * 0.002, min(cash * 0.005, bet)), 2)


# ââ Entry âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

async def maybe_enter_trade(signal: dict) -> Optional[dict]:
    """
    Enter a trade if it passes qualification gates.
    Works for all 3 new strategy types + legacy types (for backward compat).
    """
    score = signal.get("score", 0)
    if score < ENTRY_THRESHOLD:
        return None

    if not signal.get("can_enter", False):
        return None

    # Check open trade count
    open_trades = await db.get_open_paper_trades()
    if len(open_trades) >= MAX_OPEN_TRADES:
        return None

    # No double positions in same market
    if signal["market_id"] in {t["market_id"] for t in open_trades}:
        return None

    # Portfolio check â Kelly Criterion sizing
    portfolio = await db.get_portfolio()
    cost = _kelly_position_size(portfolio, signal)
    if cost < 1.0:
        return None
    if cost > portfolio.get("cash_balance", 0):
        return None

    direction   = signal.get("direction", "YES")
    market_type = signal.get("market_type", "MOMENTUM")

    # Block NO direction unless market type explicitly allows it
    if direction == "NO" and market_type not in NO_ALLOWED_TYPES:
        return None

    yes_price   = signal.get("yes_price", 0.5)
    entry_price = yes_price if direction == "YES" else (1 - yes_price)
    if entry_price <= 0:
        return None

    # SANITY CHECK: Block extreme prices
    # NEAR_CERTAINTY is allowed up to 0.92 (that's its sweet spot)
    max_price = 0.95
    if entry_price > max_price or entry_price < 0.05:
        print(f"[GATE] EXTREME price {entry_price:.4f} â skip '{signal['market_question'][:40]}'")
        return None

    shares      = round(cost / entry_price, 4)
    signal_id   = await db.save_signal(signal)
    now         = datetime.utcnow().isoformat()

    trade = {
        "signal_id":       signal_id,
        "market_id":       signal["market_id"],
        "market_question": signal["market_question"],
        "direction":       direction,
        "entry_price":     entry_price,
        "shares":          shares,
        "cost":            cost,
        "market_type":     market_type,
        "status":          "OPEN",
        "created_at":      now,
    }
    trade_id = await db.save_paper_trade(trade)
    trade["id"] = trade_id

    await db.update_portfolio(cash_delta=-cost, invested_delta=cost)

    try:
        entry_expl = explain_entry(signal, trade)
        await db.save_trade_explanation({
            "trade_id":          trade_id,
            "market_question":   signal["market_question"],
            "direction":         direction,
            "entry_explanation": entry_expl,
            "factors_json":      signal.get("factors_json", "{}"),
            "score":             score,
            "created_at":        now,
        })
    except Exception as e:
        print(f"[EXPLAINER] Entry failed: {e}")

    print(f"[TRADE] {market_type} {direction} "
          f"'{signal['market_question'][:48]}' "
          f"@ {entry_price:.3f} | ${cost:.2f} | score={score:.0f} | "
          f"{signal.get('entry_reason','')}")
    return trade


# ââ Exit helpers ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

async def _close_at_price(trade: dict, exit_price: float, reason: str):
    """Close a trade, update portfolio, trigger self-learning."""
    trade_id  = trade["id"]
    shares    = trade["shares"]
    cost      = trade["cost"]
    direction = trade.get("direction", "YES")

    payout = shares * exit_price
    pnl    = round(payout - cost, 2)
    won    = pnl > 0

    outcome = "WIN" if won else "LOSS"
    if reason in ("STOP_LOSS", "TIMEOUT") and not won:
        outcome = reason

    await db.close_paper_trade(trade_id, exit_price, pnl, outcome)
    await db.update_portfolio(cash_delta=payout, pnl_delta=pnl,
                               invested_delta=-cost, win=won)

    pnl_pct = round((pnl / cost) * 100, 1) if cost else 0
    try:
        await db.resolve_signal(trade["signal_id"], outcome, pnl_pct)
    except Exception:
        pass

    # Self-improvement
    try:
        factors = {}
        try:
            expls = await db.get_trade_explanations(100)
            for ex in expls:
                if ex.get("trade_id") == trade_id:
                    factors = ex.get("factors", {})
                    break
        except Exception:
            pass

        await sie.record_trade_result(
            trade_id    = trade_id,
            market_type = trade.get("market_type", "UNKNOWN"),
            direction   = direction,
            entry_price = trade.get("entry_price", 0),
            exit_price  = exit_price,
            pnl         = pnl,
            won         = won,
            signal_factors = factors,
        )
    except Exception as e:
        print(f"[SELF-IMPROVE] Failed: {e}")

    try:
        weights = await db.get_signal_weights()
        trade["exit_price"] = exit_price
        trade["pnl"]        = pnl

        factors = {}
        try:
            expls = await db.get_trade_explanations(100)
            for ex in expls:
                if ex.get("trade_id") == trade_id:
                    factors = ex.get("factors", {})
                    break
        except Exception:
            pass

        exit_expl = explain_exit(trade, reason, pnl)
        lesson    = generate_lesson(factors, pnl, weights, reason)
        await db.update_trade_explanation_exit(trade_id, exit_expl, lesson, outcome, pnl)

        if factors:
            await _adjust_weights(factors, won, weights)
    except Exception as e:
        print(f"[EXPLAINER] Exit failed: {e}")

    pnl_sign = "+" if pnl >= 0 else ""
    emoji    = "+" if won else "X"
    mode     = trade.get("market_type", "?")
    print(f"[TRADE] {emoji} CLOSE [{mode}] {direction} "
          f"'{trade['market_question'][:38]}' "
          f"@ {exit_price:.3f} | PNL={pnl_sign}{pnl:.2f} | {outcome}")


async def _adjust_weights(factors: dict, won: bool, current_weights: dict):
    """20% weight adjustment per trade â self-learning."""
    tops    = sorted(factors, key=lambda k: factors.get(k, 0), reverse=True)[:2]
    bottoms = sorted(factors, key=lambda k: factors.get(k, 0))[:2]

    if won:
        for f in tops:
            w = current_weights.get(f, 1.0)
            await db.update_signal_weight(f, round(w * (1 + LEARN_RATE), 4))
    else:
        for f in tops:
            w = current_weights.get(f, 1.0)
            await db.update_signal_weight(f, round(w * (1 - LEARN_RATE), 4))
        for f in bottoms:
            w = current_weights.get(f, 1.0)
            await db.update_signal_weight(f, round(w * (1 + LEARN_RATE * 0.5), 4))


# ââ LEVERAGE MODE (kept for backward compat) âââââââââââââââââââââââââââââââââ

async def maybe_enter_leverage_trade(signal: dict) -> Optional[dict]:
    """Disabled â no new leverage trades in 3-strategy mode."""
    return None

async def check_leverage_exits(markets_by_id: dict):
    """Drain any remaining open leverage trades."""
    open_lev = await db.get_open_leverage_trades()
    if not open_lev:
        return
    now = datetime.utcnow()
    for trade in open_lev:
        try:
            created = datetime.fromisoformat(trade["created_at"])
            age_hours = (now - created).total_seconds() / 3600
            if age_hours > 4:  # Force close after 4h
                market = markets_by_id.get(trade["market_id"])
                entry_px = trade.get("entry_price", 0)
                direction = trade.get("direction", "YES")
                cost = trade.get("cost", 0)
                yes_price = market.get("yes_price") if market else None
                cur_price = (yes_price if direction == "YES" else (1 - yes_price)) if yes_price is not None else entry_px
                pnl = round((cur_price - entry_px) * trade["shares"], 2)
                won = pnl > 0
                outcome = "WIN" if won else "LOSS"
                await db.close_leverage_trade(trade["id"], cur_price, pnl, outcome)
                payout = cost + pnl
                await db.update_leverage_portfolio(cash_delta=payout, invested_delta=-cost)
                print(f"[LEV] DRAIN {direction} '{trade['market_question'][:35]}' PNL=${pnl:+.2f}")
        except Exception:
            continue


# ââ Exit scan âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

async def check_exits(markets_by_id: dict):
    """
    Check all open trades for TP / SL / timeout exits.
    Handles both new 3-strategy types and legacy types.
    """
    open_trades = await db.get_open_paper_trades()
    if not open_trades:
        return

    now = datetime.utcnow()
    for trade in open_trades:
        market_id   = trade["market_id"]
        entry_px    = trade.get("entry_price", 0)
        direction   = trade.get("direction", "YES")
        market_type = trade.get("market_type", "MOMENTUM")

        try:
            created = datetime.fromisoformat(trade["created_at"])
        except Exception:
            continue

        market = markets_by_id.get(market_id)

        # ââ Determine exit params by market type ââââââââââââââââââââââââââ
        if market_type == "NEAR_CERTAINTY":
            # Hold to resolution â no TP/SL, just timeout
            take_profit_delta = None  # No TP â hold for full $1.00
            stop_loss_delta   = None  # No SL â trust the verification
            max_hold_hours    = NEAR_CERTAINTY_HOLD_HOURS
        elif market_type == "VOLUME_SPIKE":
            take_profit_delta = VOLUME_SPIKE_TP
            stop_loss_delta   = VOLUME_SPIKE_SL
            max_hold_hours    = VOLUME_SPIKE_HOLD_HOURS
        elif market_type == "BINANCE_ARB":
            take_profit_delta = BINANCE_ARB_TP
            stop_loss_delta   = BINANCE_ARB_SL
            max_hold_hours    = BINANCE_ARB_HOLD_HOURS
        # Legacy types (for draining old trades)
        elif market_type == "COPY_TRADE":
            take_profit_delta = COPY_TRADE_TP
            stop_loss_delta   = COPY_TRADE_SL
            max_hold_hours    = COPY_TRADE_HOLD_HOURS
        elif market_type == "BUY_NO_EARLY":
            take_profit_delta = BUY_NO_EARLY_TP
            stop_loss_delta   = BUY_NO_EARLY_SL
            max_hold_hours    = BUY_NO_EARLY_HOLD_HOURS
        elif market_type == "LOCK_IN":
            take_profit_delta, stop_loss_delta, max_hold_hours = _lock_in_exit_params(market)
        elif market_type == "LLM_ANALYSIS":
            take_profit_delta = 0.06
            stop_loss_delta   = 0.05
            max_hold_hours    = 8.0
        else:
            take_profit_delta = MOMENTUM_TP
            stop_loss_delta   = MOMENTUM_SL
            max_hold_hours    = MOMENTUM_HOLD_HOURS

        # Get current price
        yes_price = None
        if market:
            yes_price = market.get("yes_price")
        cur_price = (yes_price if direction == "YES" else (1 - yes_price)) if yes_price is not None else entry_px

        # Timeout check
        age_hours = (now - created).total_seconds() / 3600
        if age_hours > max_hold_hours:
            await _close_at_price(trade, cur_price, "TIMEOUT")
            continue

        if yes_price is None:
            continue

        # Early resolution guard â YES trade on a market going to 0
        if direction == "YES" and yes_price < 0.04:
            await _close_at_price(trade, cur_price, "STOP_LOSS")
            continue

        # For NEAR_CERTAINTY: check if market resolved (price near 1.0 or 0.0)
        if market_type == "NEAR_CERTAINTY":
            if direction == "YES" and yes_price >= 0.98:
                await _close_at_price(trade, cur_price, "TAKE_PROFIT")
            elif direction == "NO" and yes_price <= 0.02:
                await _close_at_price(trade, cur_price, "TAKE_PROFIT")
            # Otherwise just hold â no TP/SL for near-certainty
            continue

        # Standard TP/SL for other types
        if take_profit_delta is None or stop_loss_delta is None:
            continue

        move = cur_price - entry_px
        if move >= take_profit_delta:
            await _close_at_price(trade, cur_price, "TAKE_PROFIT")
        elif move <= -stop_loss_delta:
            await _close_at_price(trade, cur_price, "STOP_LOSS")
