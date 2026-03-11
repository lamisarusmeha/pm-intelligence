"""
Paper Trading Engine â Kelly Criterion position sizing + self-learning weight adjustment.
Supports all 4 strategies + legacy modes for backward compatibility.
"""

import json
from datetime import datetime, timedelta
from typing import Optional, Tuple
import traceback

import database as db
from trade_explainer import explain_entry, explain_exit, generate_lesson
import self_improvement_engine as sie

# Learning error tracking (max 20, FIFO) â exposed via /api/llm/debug
_learning_errors = []


def _market_days_left(market: Optional[dict]) -> float:
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
    days = _market_days_left(market)
    if days <= 0.5:
        return 0.12, 0.09, 36.0
    if days <= 2:
        return 0.07, 0.09, 24.0
    return 0.05, 0.09, 12.0


# ââ Strategy Exit Constants ââââââââââââââââââââââââââââââââââââââââââââââââââ

# Strategy 1: NEAR_CERTAINTY â hold to resolution
NEAR_CERTAINTY_HOLD_HOURS = 720.0

# Strategy 2: VOLUME_SPIKE
VOLUME_SPIKE_TP         = 0.04
VOLUME_SPIKE_SL         = 0.03
VOLUME_SPIKE_HOLD_HOURS = 2.0

# Strategy 3: BINANCE_ARB â hold to resolution (5-min binary markets)
BINANCE_ARB_HOLD_HOURS = 0.15

# Strategy 4: SHORT_DURATION â hold to resolution (5-15 min markets)
SHORT_DURATION_HOLD_HOURS = 0.5  # 30 min max â these resolve in 5-15 min

# Legacy mode constants
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

# LLM_ANALYSIS type
LLM_ANALYSIS_TP         = 0.06
LLM_ANALYSIS_SL         = 0.05
LLM_ANALYSIS_HOLD_HOURS = 8.0


# Shared constants
ENTRY_THRESHOLD        = 40
MAX_OPEN_TRADES        = 40
BASE_RISK_PCT          = 0.005
LEARN_RATE             = 0.20

KELLY_WIN_PROBS = {
    "NEAR_CERTAINTY": 0.85,
    "VOLUME_SPIKE":   0.65,
    "BINANCE_ARB":    0.72,
    "SHORT_DURATION": 0.80,
    "COPY_TRADE":     0.75,
    "BUY_NO_EARLY":   0.70,
    "LOCK_IN":        0.78,
    "MOMENTUM":       0.55,
    "LLM_ANALYSIS":   0.65,
}

NO_ALLOWED_TYPES = {
    "BUY_NO_EARLY", "LOCK_IN", "LLM_ANALYSIS",
    "NEAR_CERTAINTY", "VOLUME_SPIKE", "BINANCE_ARB",
    "SHORT_DURATION",
}

MAX_POS_NEAR_CERT_HIGH  = 500
MAX_POS_NEAR_CERT_MED   = 200
MAX_POS_BINANCE_ARB     = 75
MAX_POS_SHORT_DURATION  = 100
MAX_POS_DEFAULT         = 150


# ââ Position Cap âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

def _get_position_cap(signal: dict) -> float:
    market_type = signal.get("market_type", "")
    direction = signal.get("direction", "YES")
    yes_price = signal.get("yes_price", 0.5)
    entry_price = yes_price if direction == "YES" else (1 - yes_price)
    if market_type == "BINANCE_ARB":
        return MAX_POS_BINANCE_ARB
    if market_type == "SHORT_DURATION":
        return MAX_POS_SHORT_DURATION
    if market_type == "NEAR_CERTAINTY":
        if entry_price >= 0.90:
            return MAX_POS_NEAR_CERT_HIGH
        return MAX_POS_NEAR_CERT_MED
    return MAX_POS_DEFAULT


# ââ Kelly Criterion ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

def _kelly_position_size(portfolio: dict, signal: dict) -> float:
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

    score_bonus = min(0.2, (signal.get("score", 50) - 50) / 250)
    kelly_frac  = kelly_frac * (1 + score_bonus)

    bet = cash * kelly_frac

    cap = _get_position_cap(signal)

    # SHORT_DURATION: smaller position sizes since these resolve fast
    if market_type == "SHORT_DURATION":
        bet = max(cash * 0.001, min(cash * 0.003, bet))
        return round(min(bet, cap), 2)

    # Standard: 0.2%-0.5% per trade
    bet = max(cash * 0.002, min(cash * 0.005, bet))
    return round(min(bet, cap), 2)


# ââ Entry âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

async def maybe_enter_trade(signal: dict) -> Optional[dict]:
    score = signal.get("score", 0)
    if score < ENTRY_THRESHOLD:
        return None

    if not signal.get("can_enter", False):
        return None

    open_trades = await db.get_open_paper_trades()
    if len(open_trades) >= MAX_OPEN_TRADES:
        return None

    if signal["market_id"] in {t["market_id"] for t in open_trades}:
        return None

    portfolio = await db.get_portfolio()
    cost = _kelly_position_size(portfolio, signal)
    if cost < 1.0:
        return None
    if cost > portfolio.get("cash_balance", 0):
        return None

    direction   = signal.get("direction", "YES")
    market_type = signal.get("market_type", "MOMENTUM")

    if direction == "NO" and market_type not in NO_ALLOWED_TYPES:
        return None

    yes_price   = signal.get("yes_price", 0.5)
    entry_price = yes_price if direction == "YES" else (1 - yes_price)
    if entry_price <= 0:
        return None

    # SANITY CHECK: Block extreme prices
    max_price = 0.97 if market_type == "NEAR_CERTAINTY" else 0.95
    if entry_price > max_price or entry_price < 0.05:
        print(f"[GATE] EXTREME price {entry_price:.4f} â skip '{signal['market_question'][:40]}'")
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


# ââ Exit Helpers ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

async def _close_at_price(trade: dict, exit_price: float, reason: str):
    """Close a trade, update portfolio, trigger self-learning on EVERY trade."""
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

    # ââ SELF-LEARNING: Record EVERY trade result âââââââââââââââââââââââââ
    # Split into two blocks so factor extraction failure doesn't prevent recording
    factors = {}
    try:
        factors = _extract_factors(trade_id, await db.get_trade_explanations(200))
    except Exception as e:
        err = {"ts": datetime.utcnow().isoformat(), "stage": "factor_extraction",
               "trade_id": trade_id, "error": str(e), "tb": traceback.format_exc()}
        _learning_errors.append(err)
        if len(_learning_errors) > 20:
            _learning_errors.pop(0)
        print(f"[SELF-IMPROVE] Factor extraction failed: {e}")

    try:
        await sie.record_trade_result(
            trade_id       = trade_id,
            market_type    = trade.get("market_type", "UNKNOWN"),
            direction      = direction,
            entry_price    = trade.get("entry_price", 0),
            exit_price     = exit_price,
            pnl            = pnl,
            won            = won,
            signal_factors = factors,
        )
    except Exception as e:
        err = {"ts": datetime.utcnow().isoformat(), "stage": "record_trade_result",
               "trade_id": trade_id, "error": str(e), "tb": traceback.format_exc()}
        _learning_errors.append(err)
        if len(_learning_errors) > 20:
            _learning_errors.pop(0)
        print(f"[SELF-IMPROVE] Record failed: {e}")

    # ââ TRADE EXPLANATION & LESSON EXTRACTION ââââââââââââââââââââââââââââ
    try:
        weights = await db.get_signal_weights()
        trade["exit_price"] = exit_price
        trade["pnl"]        = pnl

        factors = _extract_factors(trade_id, await db.get_trade_explanations(200))

        exit_expl = explain_exit(trade, reason, pnl)
        lesson    = generate_lesson(factors, pnl, weights, reason)
        await db.update_trade_explanation_exit(trade_id, exit_expl, lesson, outcome, pnl)

        # Adjust signal weights based on this trade's factors
        if factors:
            await _adjust_weights(factors, won, weights)
    except Exception as e:
        print(f"[EXPLAINER] Exit failed: {e}")

    # ââ LOG ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
    pnl_sign = "+" if pnl >= 0 else ""
    emoji    = "+" if won else "X"
    mode     = trade.get("market_type", "?")
    print(f"[TRADE] {emoji} CLOSE [{mode}] {direction} "
          f"'{trade['market_question'][:38]}' "
          f"@ {exit_price:.3f} | PNL={pnl_sign}{pnl:.2f} | {outcome}")


def _extract_factors(trade_id: int, explanations: list) -> dict:
    """Extract signal factors for a trade from explanations."""
    for ex in explanations:
        if ex.get("trade_id") == trade_id:
            raw = ex.get("factors_json", ex.get("factors", "{}"))
            if isinstance(raw, dict):
                return raw
            try:
                return json.loads(raw) if isinstance(raw, str) else {}
            except Exception:
                return {}
    return {}


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


# ââ Leverage (disabled) ââââââââââââââââââââââââââââââââââââââââââââââââââââââ

async def maybe_enter_leverage_trade(signal: dict) -> Optional[dict]:
    return None

async def check_leverage_exits(markets_by_id: dict):
    open_lev = await db.get_open_leverage_trades()
    if not open_lev:
        return
    now = datetime.utcnow()
    for trade in open_lev:
        try:
            created = datetime.fromisoformat(trade["created_at"])
            age_hours = (now - created).total_seconds() / 3600
            if age_hours > 4:
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


# ââ Exit Scan âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

async def check_exits(markets_by_id: dict):
    """Check all open trades for TP / SL / timeout / resolution exits."""
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

        # Determine exit params by market type
        if market_type == "NEAR_CERTAINTY":
            take_profit_delta = None
            stop_loss_delta   = None
            max_hold_hours    = NEAR_CERTAINTY_HOLD_HOURS
        elif market_type == "SHORT_DURATION":
            take_profit_delta = None  # Hold to resolution
            stop_loss_delta   = None
            max_hold_hours    = SHORT_DURATION_HOLD_HOURS
        elif market_type == "VOLUME_SPIKE":
            take_profit_delta = VOLUME_SPIKE_TP
            stop_loss_delta   = VOLUME_SPIKE_SL
            max_hold_hours    = VOLUME_SPIKE_HOLD_HOURS
        elif market_type == "BINANCE_ARB":
            take_profit_delta = None
            stop_loss_delta   = None
            max_hold_hours    = BINANCE_ARB_HOLD_HOURS
        elif market_type == "LLM_ANALYSIS":
            take_profit_delta = LLM_ANALYSIS_TP
            stop_loss_delta   = LLM_ANALYSIS_SL
            max_hold_hours    = LLM_ANALYSIS_HOLD_HOURS
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

        # If market is closed/resolved, close the trade
        if market and market.get("closed"):
            reason = "TAKE_PROFIT" if cur_price > entry_px else "STOP_LOSS"
            await _close_at_price(trade, cur_price, reason)
            continue

        # Market not in current fetch = might be resolved
        if yes_price is None:
            if age_hours > 4:
                await _close_at_price(trade, entry_px, "TIMEOUT")
            continue

        # Early resolution guard
        if direction == "YES" and yes_price < 0.04:
            await _close_at_price(trade, cur_price, "STOP_LOSS")
            continue
        if direction == "NO" and yes_price > 0.96:
            await _close_at_price(trade, cur_price, "STOP_LOSS")
            continue

        # For NEAR_CERTAINTY, SHORT_DURATION, and BINANCE_ARB: close on resolution
        if market_type in ("NEAR_CERTAINTY", "SHORT_DURATION", "BINANCE_ARB"):
            if direction == "YES" and yes_price >= 0.97:
                await _close_at_price(trade, cur_price, "TAKE_PROFIT")
            elif direction == "NO" and yes_price <= 0.03:
                await _close_at_price(trade, cur_price, "TAKE_PROFIT")
            # Also close if clearly losing
            elif direction == "YES" and yes_price <= 0.05:
                await _close_at_price(trade, cur_price, "STOP_LOSS")
            elif direction == "NO" and yes_price >= 0.95 and entry_px < 0.90:
                # Only stop-loss NO trades if they're clearly losing
                pass
            continue

        # Standard TP/SL for other types
        if take_profit_delta is None or stop_loss_delta is None:
            continue

        move = cur_price - entry_px
        if move >= take_profit_delta:
            await _close_at_price(trade, cur_price, "TAKE_PROFIT")
        elif move <= -stop_loss_delta:
            await _close_at_price(trade, cur_price, "STOP_LOSS")
