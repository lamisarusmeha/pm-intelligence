"""
Paper Trader — TURBO LEARNING MODE (100k balance, max trade volume)

LOCK-IN MODE  (market_type == "LOCK_IN")  — betting WITH the crowd
  Entry:   price_zone ≥ 65 OR 1+ signal ≥ 50 (easier entry)
  Take-profit: +3¢ normal | +7¢ if resolving ≤2d | +12¢ if resolving today/tomorrow
     → Near-resolution TP widening: instead of exiting at 83¢ on an 80% market,
       we hold through the full drift to certainty (100¢ = 20¢ gain).
  Stop-loss:   -9¢  (wide — an 80% market needs to drop NINE cents to stop us out)
  Max hold:    4h (normal) | 24h if resolving ≤2d | 36h if resolving today
  Win logic:   80% market → ~80% win rate by design

COPY_TRADE MODE  (market_type == "COPY_TRADE")  — mirror smart money wallets
  Entry:   smart_wallet score ≥ 85
  Take-profit: +4¢  Stop-loss: -3¢  Hold: 2h (↓ from 3h for faster cycling)

BUY_NO_EARLY MODE  (market_type == "BUY_NO_EARLY")  — exploit YES overpricing bias
  Entry:   buy_no_early score ≥ 60 on sensational market with YES > 55%
  Take-profit: +6¢  Stop-loss: -5¢  Hold: 6h (↓ from 10h for faster cycling)

MOMENTUM MODE  (market_type == "MOMENTUM") — information edge plays
  Entry:   2+ signals ≥ 50, OR single news/wallet ≥ 78
  Take-profit: +6¢  Stop-loss: -4¢  Hold: 2h (↓ from 4h for faster cycling)

75 SIMULTANEOUS POSITIONS | Kelly Criterion sizing | 25% self-learning weight adjustment
$100k balance | max 3.5% per trade | lower entry threshold for more data
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
    """
    Return (take_profit_delta, stop_loss_delta, max_hold_hours) for a LOCK_IN trade,
    adjusted for how soon the market resolves.

    Near-resolution widening rationale:
      - A market at 80% YES will drift to ~100% at resolution (+20¢ gain).
      - If we exit at 83¢ (3¢ TP) we capture 15% of the available move.
      - By widening TP for near-resolution, we capture the full certainty drift.

    days_left > 2  : normal   — TP=5¢,  SL=9¢,  hold=12h
    days_left ≤ 2  : widened  — TP=7¢,  SL=9¢,  hold=24h
    days_left ≤ 0.5: resolving today/tomorrow — TP=12¢, SL=9¢, hold=36h
    """
    days = _market_days_left(market)
    if days <= 0.5:   # resolving within 12 hours — hold through for max gain
        return 0.12, LOCK_IN_SL, 36.0   # ↓ 48h→36h
    if days <= 2:     # resolving within 2 days — widen TP significantly
        return 0.07, LOCK_IN_SL, 24.0   # ↓ 36h→24h
    return LOCK_IN_TP, LOCK_IN_SL, LOCK_IN_HOLD_HOURS

# ── COPY_TRADE MODE constants ─────────────────────────────────────────────────
COPY_TRADE_TP         = 0.04   # 4¢ TP — follow smart money, exit quick
COPY_TRADE_SL         = 0.03   # 3¢ SL — tight: if wallet is wrong, exit fast
COPY_TRADE_HOLD_HOURS = 2      # ↓ 3h→2h: faster cycling = more closed trades/day

# ── BUY_NO_EARLY MODE constants ────────────────────────────────────────────────
BUY_NO_EARLY_TP         = 0.06   # 6¢ TP — wait for YES to deflate back to fair value
BUY_NO_EARLY_SL         = 0.05   # 5¢ SL — stop if crowd keeps buying YES
BUY_NO_EARLY_HOLD_HOURS = 6      # ↓ 10h→6h: faster cycling, still enough time

# ── LOCK-IN MODE constants ────────────────────────────────────────────────────
LOCK_IN_TP         = 0.05   # ↑ 3¢→5¢: wider TP — ≤14 day markets need room to drift
LOCK_IN_SL         = 0.09   # 9¢ stop-loss   (very wide — 80% markets rarely drop 9¢)
LOCK_IN_HOLD_HOURS = 12     # ↑ 4h→12h: more time for near-resolution drift to play out

# ── MOMENTUM MODE constants ───────────────────────────────────────────────────
MOMENTUM_TP        = 0.06   # 6¢ take-profit
MOMENTUM_SL        = 0.04   # 4¢ stop-loss
MOMENTUM_HOLD_HOURS = 2     # ↓ 4h→2h: fastest cycling — most data per day

# ── Shared constants — HIGH ACCURACY MODE ────────────────────────────────────
ENTRY_THRESHOLD        = 40   # ↑ 20→40: only quality signals enter (matches GENERATE_MIN_SCORE)
MAX_OPEN_TRADES        = 40   # ↓ 75→40: fewer, higher-conviction positions
BASE_RISK_PCT          = 0.005  # 0.5% floor — conservative for real-money safety
LEARN_RATE             = 0.20   # balanced: adapt fast but don't overfit to noise

# Win probability estimates per mode (for Kelly Criterion)
# Conservative estimates — better to undersize than oversize on real money
KELLY_WIN_PROBS = {
    "COPY_TRADE":    0.75,   # Smart money following — documented edge
    "BUY_NO_EARLY":  0.70,   # Historical NO base rate, conservative estimate
    "LOCK_IN":       0.78,   # 80% market wins ~78-80% — slight discount for uncertainty
    "MOMENTUM":      0.55,   # DISABLED but kept for Kelly math if ever re-enabled
}


# ── Kelly Criterion position sizing ───────────────────────────────────────────

def _kelly_position_size(portfolio: dict, signal: dict) -> float:
    """
    Kelly Criterion: f* = (b*p - q) / b
      b = net payout odds  = (1 - entry_price) / entry_price
      p = estimated win probability (by market type)
      q = 1 - p

    Uses 25% fractional Kelly to account for model uncertainty.
    Bounds: min 0.3% of capital, max 2.0% of capital.
    Higher confidence = larger bet. Copy trades bet most. Momentum bets least.
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
    b = (1 - entry_price) / entry_price  # net payout odds

    if b <= 0:
        return round(cash * BASE_RISK_PCT, 2)

    kelly     = (b * p - q) / b
    if kelly <= 0:
        return round(cash * 0.003, 2)   # no edge — minimum bet

    # 25% fractional Kelly — standard conservative sizing for real-money trading
    kelly_frac = kelly * 0.25

    # Score bonus: higher confidence = up to 20% more
    score_bonus = min(0.2, (signal.get("score", 50) - 50) / 250)
    kelly_frac  = kelly_frac * (1 + score_bonus)

    bet = cash * kelly_frac
    # Clamp: minimum 0.5%, maximum 0.5% per trade (hard $500 cap on $100k)
    # Hard cap prevents one bad stop-loss from wiping a large chunk of capital.
    # The $617 BUY_NO_EARLY loss showed Kelly can oversize dangerously — cap it.
    return round(max(cash * 0.001, min(cash * 0.002, bet)), 2)


# ── Entry ─────────────────────────────────────────────────────────────────────

async def maybe_enter_trade(signal: dict) -> Optional[dict]:
    """
    Enter a trade if it passes the dual-mode qualification gate.
    signal.can_enter is pre-computed by signal_engine._qualifies_for_entry().
    """
    score = signal.get("score", 0)
    if score < ENTRY_THRESHOLD:
        return None

    # Use pre-computed qualification from signal engine
    if not signal.get("can_enter", False):
        reason = signal.get("entry_reason", "no_reason")
        print(f"[GATE] SKIP '{signal['market_question'][:40]}' — {reason}")
        return None

    # Check open trade count
    open_trades = await db.get_open_paper_trades()
    if len(open_trades) >= MAX_OPEN_TRADES:
        return None

    # No double positions in same market
    if signal["market_id"] in {t["market_id"] for t in open_trades}:
        return None

    # Portfolio check — Kelly Criterion sizing
    portfolio = await db.get_portfolio()
    cost = _kelly_position_size(portfolio, signal)
    if cost < 1.0:
        return None
    if cost > portfolio.get("cash_balance", 0):
        return None

    direction   = signal.get("direction", "YES")
    market_type = signal.get("market_type", "MOMENTUM")

    # Block raw NO on MOMENTUM/LOCK_IN/COPY_TRADE — only 50% WR historically.
    # BUY_NO_EARLY is EXEMPT: its NO bet has documented 78% NO base rate edge.
    if direction == "NO" and market_type not in ("BUY_NO_EARLY", "LOCK_IN", "LLM_ANALYSIS"):
        return None

    # Hard block EXTREME mode — no TP room (price already at 92%+)
    if market_type == "EXTREME":
        return None

    yes_price   = signal.get("yes_price", 0.5)
    entry_price = yes_price if direction == "YES" else (1 - yes_price)
    if entry_price <= 0:
        return None

    shares      = round(cost / entry_price, 4)
    signal_id   = await db.save_signal(signal)
    now         = datetime.utcnow().isoformat()
    # market_type already set above — no need to re-fetch

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

    mode_emoji = "🔒" if market_type == "LOCK_IN" else "📈"
    print(f"[TRADE] {mode_emoji} {market_type} {direction} "
          f"'{signal['market_question'][:48]}' "
          f"@ {entry_price:.3f} | ${cost:.2f} | score={score:.0f} | "
          f"{signal.get('entry_reason','')}")
    return trade


# ── Exit helpers ──────────────────────────────────────────────────────────────

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

    # ── SELF-IMPROVEMENT: Record result and trigger learning cycle ─────────────
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
        print(f"[SELF-IMPROVE] Failed to record trade result: {e}")

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
    emoji    = "✅" if won else "❌"
    mode     = trade.get("market_type", "?")
    print(f"[TRADE] {emoji} CLOSE [{mode}] {direction} "
          f"'{trade['market_question'][:38]}' "
          f"@ {exit_price:.3f} | PNL={pnl_sign}{pnl:.2f} | {outcome}")


async def _adjust_weights(factors: dict, won: bool, current_weights: dict):
    """15% weight adjustment per trade — aggressive self-learning."""
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


# ── LEVERAGE MODE ─────────────────────────────────────────────────────────────
# Separate $10k virtual portfolio. Same Polymarket signals, amplified P&L.
# Tighter stops to protect against amplified downside.

LEV_ENTRY_THRESHOLD = 70    # Only high-confidence signals (vs 33 for normal)
LEV_TP              = 0.07  # 7¢ take-profit
LEV_SL              = 0.04  # 4¢ stop-loss (tighter — leverage amplifies losses)
LEV_HOLD_HOURS      = 4
LEV_RISK_PCT        = 0.015 # 1.5% actual capital per trade
MAX_OPEN_LEV        = 5     # Max simultaneous leveraged positions


async def maybe_enter_leverage_trade(signal: dict) -> Optional[dict]:
    """
    Enter a leveraged trade if signal score ≥ 70 and can_enter=True.
    P&L is amplified by leverage_multiplier (2x/3x/5x).
    Actual capital risked: 1.5% per trade. Effective exposure = cost × multiplier.
    """
    if signal.get("score", 0) < LEV_ENTRY_THRESHOLD:
        return None
    if not signal.get("can_enter", False):
        return None
    if signal.get("market_type") == "EXTREME":
        return None

    open_lev = await db.get_open_leverage_trades()
    if len(open_lev) >= MAX_OPEN_LEV:
        return None
    if signal["market_id"] in {t["market_id"] for t in open_lev}:
        return None

    lev_portfolio = await db.get_leverage_portfolio()
    multiplier    = int(lev_portfolio.get("leverage_multiplier", 2))
    cash          = lev_portfolio.get("cash_balance", 10000)
    cost          = round(cash * LEV_RISK_PCT, 2)
    if cost < 1.0 or cost > cash:
        return None

    direction   = signal.get("direction", "YES")
    yes_price   = signal.get("yes_price", 0.5)
    entry_price = yes_price if direction == "YES" else (1 - yes_price)
    if entry_price <= 0:
        return None

    # Effective exposure = cost × multiplier; shares scaled accordingly
    effective_cost = cost * multiplier
    shares         = round(effective_cost / entry_price, 4)
    now            = datetime.utcnow().isoformat()

    trade = {
        "signal_id":           signal.get("signal_id", 0),
        "market_id":           signal["market_id"],
        "market_question":     signal["market_question"],
        "direction":           direction,
        "entry_price":         entry_price,
        "shares":              shares,
        "cost":                cost,
        "leverage_multiplier": multiplier,
        "status":              "OPEN",
        "created_at":          now,
    }
    trade_id    = await db.save_leverage_trade(trade)
    trade["id"] = trade_id
    await db.update_leverage_portfolio(cash_delta=-cost, invested_delta=cost)

    print(f"[LEV] ⚡ {multiplier}x {direction} '{signal['market_question'][:45]}' "
          f"@ {entry_price:.3f} | actual=${cost:.2f} exposure=${effective_cost:.2f} | score={signal['score']:.0f}")
    return trade


async def check_leverage_exits(markets_by_id: dict):
    """Check all open leveraged trades for TP / SL / timeout exits."""
    open_lev = await db.get_open_leverage_trades()
    if not open_lev:
        return

    now = datetime.utcnow()
    for trade in open_lev:
        market_id   = trade["market_id"]
        entry_px    = trade.get("entry_price", 0)
        direction   = trade.get("direction", "YES")
        multiplier  = trade.get("leverage_multiplier", 2)
        cost        = trade.get("cost", 0)

        try:
            created   = datetime.fromisoformat(trade["created_at"])
        except Exception:
            continue

        market    = markets_by_id.get(market_id)
        yes_price = market.get("yes_price") if market else None
        cur_price = (yes_price if direction == "YES" else (1 - yes_price)) if yes_price is not None else entry_px

        age_hours = (now - created).total_seconds() / 3600
        if age_hours > LEV_HOLD_HOURS:
            # Leverage P&L = (price_move) × shares (shares already reflect multiplier)
            pnl = round((cur_price - entry_px) * trade["shares"], 2)
            won = pnl > 0
            outcome = "WIN" if won else "LOSS"
            await db.close_leverage_trade(trade["id"], cur_price, pnl, outcome)
            payout = cost + pnl
            await db.update_leverage_portfolio(cash_delta=payout, invested_delta=-cost)
            print(f"[LEV] ⏱ TIMEOUT {direction} '{trade['market_question'][:35]}' PNL=${pnl:+.2f}")
            continue

        if yes_price is None:
            continue

        move = cur_price - entry_px
        if move >= LEV_TP:
            pnl  = round(move * trade["shares"], 2)
            await db.close_leverage_trade(trade["id"], cur_price, pnl, "WIN")
            payout = cost + pnl
            await db.update_leverage_portfolio(cash_delta=payout, invested_delta=-cost)
            print(f"[LEV] ✅ TP {direction} '{trade['market_question'][:35]}' PNL=${pnl:+.2f} ({multiplier}x)")
        elif move <= -LEV_SL:
            pnl  = round(move * trade["shares"], 2)
            await db.close_leverage_trade(trade["id"], cur_price, pnl, "STOP_LOSS")
            payout = max(0, cost + pnl)
            await db.update_leverage_portfolio(cash_delta=payout, invested_delta=-cost)
            print(f"[LEV] ❌ SL {direction} '{trade['market_question'][:35]}' PNL=${pnl:+.2f} ({multiplier}x)")


# ── Exit scan ─────────────────────────────────────────────────────────────────

async def check_exits(markets_by_id: dict):
    """
    Check all open trades for TP / SL / timeout exits.
    Uses ADAPTIVE thresholds based on market_type stored on the trade.

    CRITICAL FIX: timeout now uses CURRENT market price, not entry price.
    Previously all timeouts exited at entry price → PNL always 0 → useless data.
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

        # Get current market data (needed for near-resolution TP adjustment)
        market = markets_by_id.get(market_id)

        # Determine mode-specific thresholds
        if market_type == "COPY_TRADE":
            take_profit_delta = COPY_TRADE_TP
            stop_loss_delta   = COPY_TRADE_SL
            max_hold_hours    = COPY_TRADE_HOLD_HOURS
        elif market_type == "BUY_NO_EARLY":
            take_profit_delta = BUY_NO_EARLY_TP
            stop_loss_delta   = BUY_NO_EARLY_SL
            max_hold_hours    = BUY_NO_EARLY_HOLD_HOURS
        elif market_type == "LOCK_IN":
            # Near-resolution widening: capture the full drift to certainty
            take_profit_delta, stop_loss_delta, max_hold_hours = _lock_in_exit_params(market)
        else:
            take_profit_delta = MOMENTUM_TP
            stop_loss_delta   = MOMENTUM_SL
            max_hold_hours    = MOMENTUM_HOLD_HOURS

        # Get current price (market already fetched above for near-res TP calc)
        yes_price = None
        if market:
            yes_price = market.get("yes_price")
        cur_price = (yes_price if direction == "YES" else (1 - yes_price)) if yes_price is not None else entry_px

        # Timeout check — use CURRENT price (was using entry_px before → always 0 PNL)
        age_hours = (now - created).total_seconds() / 3600
        if age_hours > max_hold_hours:
            await _close_at_price(trade, cur_price, "TIMEOUT")
            continue

        if yes_price is None:
            continue

        # FIX 1 — Early resolution guard.
        # Binary markets can resolve from 0.40 → 0.01 between loop ticks,
        # blowing past the 4¢ SL and losing 97% instead of 8%.
        # If YES price has collapsed to near-zero (market resolving NO) on a YES
        # trade, close immediately at current price — same as stop loss but caught
        # before the position goes to pennies.
        if direction == "YES" and yes_price < 0.04:
            await _close_at_price(trade, cur_price, "STOP_LOSS")
            continue

        move = cur_price - entry_px

        if move >= take_profit_delta:
            await _close_at_price(trade, cur_price, "TAKE_PROFIT")
        elif move <= -stop_loss_delta:
            await _close_at_price(trade, cur_price, "STOP_LOSS")
