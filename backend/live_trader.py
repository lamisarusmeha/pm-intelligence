"""
Live Trader — Real money execution on Polymarket via CLOB API.

SAFETY GATES (all must pass before any real order is placed):
  1. LIVE_MODE env var must be explicitly set to "true"
  2. Min 100 closed paper trades required (proven track record)
  3. Min 75% paper win rate required over last 50 trades
  4. Max 2% of live capital per trade (hard cap)
  5. Market must have ≥ $10,000 liquidity (can absorb our order)
  6. Only LOCK_IN, BUY_NO_EARLY, COPY_TRADE modes (no MOMENTUM)

HOW IT WORKS:
  - Runs alongside paper trading (paper trades continue for comparison)
  - Uses py-clob-client to place real limit orders on Polymarket CLOB
  - Tracks live positions in separate live_trades / live_portfolio DB tables
  - Same TP/SL/timeout logic as paper trader, but closes with real sell orders

CREDENTIALS (set in backend/.env):
  POLYMARKET_API_KEY        — from polymarket.com Settings > API Keys
  POLYMARKET_API_SECRET     — from polymarket.com Settings > API Keys
  POLYMARKET_API_PASSPHRASE — from polymarket.com Settings > API Keys
  POLYMARKET_PRIVATE_KEY    — your Polygon wallet private key (0x...)
  LIVE_MODE                 — set to "true" to enable real trading
  LIVE_MAX_POSITIONS        — max simultaneous live positions (default: 10)
  LIVE_MAX_PCT_PER_TRADE    — max % of balance per trade (default: 0.02 = 2%)
"""

import os
import json
import asyncio
from datetime import datetime, timedelta
from typing import Optional

import database as db

# ── Live mode guard ───────────────────────────────────────────────────────────
LIVE_MODE = os.getenv("LIVE_MODE", "false").lower() == "true"

# Safety limits
MIN_PAPER_TRADES    = 100    # must have at least this many closed paper trades
MIN_PAPER_WIN_RATE  = 0.75   # must be at least 75% WR over last 50 paper trades
MAX_LIVE_POSITIONS  = int(os.getenv("LIVE_MAX_POSITIONS", "10"))
MAX_PCT_PER_TRADE   = float(os.getenv("LIVE_MAX_PCT_PER_TRADE", "0.02"))  # 2%
MIN_MARKET_LIQUIDITY = 10_000  # $10k minimum liquidity to absorb our order

# CLOB API
CLOB_HOST = "https://clob.polymarket.com"
POLYGON_CHAIN_ID = 137

# Allowed modes for live trading
LIVE_ALLOWED_MODES = {"LOCK_IN", "BUY_NO_EARLY", "COPY_TRADE"}

# TP/SL constants (mirrors paper_trader.py)
LIVE_TP = {
    "COPY_TRADE":   0.04,
    "BUY_NO_EARLY": 0.06,
    "LOCK_IN":      0.03,
}
LIVE_SL = {
    "COPY_TRADE":   0.03,
    "BUY_NO_EARLY": 0.05,
    "LOCK_IN":      0.09,
}
LIVE_HOLD = {
    "COPY_TRADE":   2,
    "BUY_NO_EARLY": 6,
    "LOCK_IN":      4,
}

_clob_client = None


def _get_clob_client():
    """Lazy-load CLOB client. Returns None if credentials not set."""
    global _clob_client
    if _clob_client is not None:
        return _clob_client

    api_key        = os.getenv("POLYMARKET_API_KEY", "")
    api_secret     = os.getenv("POLYMARKET_API_SECRET", "")
    api_passphrase = os.getenv("POLYMARKET_API_PASSPHRASE", "")
    private_key    = os.getenv("POLYMARKET_PRIVATE_KEY", "")

    if not all([api_key, api_secret, api_passphrase, private_key]):
        print("[LIVE] ⚠️  Credentials not set — live trading disabled")
        return None

    try:
        from py_clob_client.client import ClobClient
        from py_clob_client.clob_types import ApiCreds

        creds = ApiCreds(
            api_key=api_key,
            api_secret=api_secret,
            api_passphrase=api_passphrase,
        )
        _clob_client = ClobClient(
            host=CLOB_HOST,
            chain_id=POLYGON_CHAIN_ID,
            private_key=private_key,
            creds=creds,
        )
        print("[LIVE] ✅ CLOB client initialized")
        return _clob_client
    except ImportError:
        print("[LIVE] ❌ py-clob-client not installed. Run: pip install py-clob-client")
        return None
    except Exception as e:
        print(f"[LIVE] ❌ CLOB client init failed: {e}")
        return None


async def _check_safety_gates() -> tuple[bool, str]:
    """
    All gates must pass before any live trade is placed.
    Returns (ok, reason).
    """
    if not LIVE_MODE:
        return False, "LIVE_MODE not enabled"

    if _get_clob_client() is None:
        return False, "CLOB credentials not set"

    # Gate 1: minimum paper trade history
    portfolio = await db.get_portfolio()
    total_closed = (portfolio.get("win_count", 0) or 0) + (portfolio.get("loss_count", 0) or 0)
    if total_closed < MIN_PAPER_TRADES:
        remaining = MIN_PAPER_TRADES - total_closed
        return False, f"Need {remaining} more paper trades (have {total_closed}/{MIN_PAPER_TRADES})"

    # Gate 2: minimum win rate over last 50 paper trades
    recent_trades = await db.get_all_paper_trades(50)
    closed_recent = [t for t in recent_trades if t.get("status") in ("WIN", "LOSS", "STOP_LOSS")]
    if len(closed_recent) >= 20:
        wins = sum(1 for t in closed_recent if t.get("status") == "WIN")
        wr   = wins / len(closed_recent)
        if wr < MIN_PAPER_WIN_RATE:
            return False, f"Paper WR {wr:.1%} below required {MIN_PAPER_WIN_RATE:.0%}"

    return True, "all_gates_passed"


async def _get_live_portfolio() -> dict:
    """Get live portfolio from DB."""
    return await db.get_live_portfolio()


def _position_size(portfolio: dict, signal: dict) -> float:
    """
    Conservative live position sizing.
    Max 2% of live capital. No Kelly — flat % for predictability.
    """
    cash       = portfolio.get("cash_balance", 0)
    max_bet    = cash * MAX_PCT_PER_TRADE
    min_bet    = max(50.0, cash * 0.005)  # $50 minimum or 0.5%
    return round(min(max_bet, max(min_bet, max_bet)), 2)


async def maybe_enter_live_trade(signal: dict) -> Optional[dict]:
    """
    Attempt to place a real Polymarket order if all safety gates pass.
    Returns trade dict if order placed, None otherwise.
    """
    # Mode check — only high-accuracy modes
    market_type = signal.get("market_type", "")
    if market_type not in LIVE_ALLOWED_MODES:
        return None

    # Safety gates
    gates_ok, gate_reason = await _check_safety_gates()
    if not gates_ok:
        return None

    # Liquidity check — must have enough book depth to absorb our order
    if signal.get("liquidity", 0) < MIN_MARKET_LIQUIDITY:
        return None

    # Max positions check
    open_live = await db.get_open_live_trades()
    if len(open_live) >= MAX_LIVE_POSITIONS:
        return None

    # No duplicate positions in same market
    if signal["market_id"] in {t["market_id"] for t in open_live}:
        return None

    # Direction check — NO only allowed for BUY_NO_EARLY
    direction = signal.get("direction", "YES")
    if direction == "NO" and market_type != "BUY_NO_EARLY":
        return None

    live_portfolio = await _get_live_portfolio()
    cost           = _position_size(live_portfolio, signal)
    if cost < 50 or cost > live_portfolio.get("cash_balance", 0):
        return None

    # Get CLOB token IDs from signal (passed from market fetch)
    clob_token_ids = signal.get("clob_token_ids", [])
    if not clob_token_ids or len(clob_token_ids) < 2:
        print(f"[LIVE] ⚠️  No CLOB token IDs for '{signal['market_question'][:40]}' — skipping")
        return None

    yes_price    = signal.get("yes_price", 0.5)
    entry_price  = yes_price if direction == "YES" else (1 - yes_price)
    token_id     = clob_token_ids[0] if direction == "YES" else clob_token_ids[1]
    shares       = round(cost / entry_price, 4)

    # Place the real order
    client = _get_clob_client()
    if not client:
        return None

    try:
        from py_clob_client.clob_types import OrderArgs, OrderType
        from py_clob_client.constants import BUY

        order_args = OrderArgs(
            token_id=token_id,
            price=round(entry_price, 4),
            size=round(shares, 4),
            side=BUY,
        )
        signed_order = client.create_order(order_args)
        resp         = client.post_order(signed_order, OrderType.GTC)

        if not resp or resp.get("errorMsg"):
            err = resp.get("errorMsg", "unknown") if resp else "no response"
            print(f"[LIVE] ❌ Order rejected: {err}")
            return None

        order_id = resp.get("orderID", resp.get("id", ""))
        now      = datetime.utcnow().isoformat()

        trade = {
            "market_id":       signal["market_id"],
            "market_question": signal["market_question"],
            "direction":       direction,
            "market_type":     market_type,
            "entry_price":     entry_price,
            "shares":          shares,
            "cost":            cost,
            "clob_order_id":   order_id,
            "token_id":        token_id,
            "status":          "OPEN",
            "created_at":      now,
        }
        trade_id    = await db.save_live_trade(trade)
        trade["id"] = trade_id

        await db.update_live_portfolio(cash_delta=-cost, invested_delta=cost)

        print(f"[LIVE] 🟢 REAL ORDER [{market_type}] {direction} "
              f"'{signal['market_question'][:45]}' "
              f"@ {entry_price:.3f} | ${cost:.2f} | order={order_id[:12]}...")
        return trade

    except Exception as e:
        print(f"[LIVE] ❌ Order placement error: {e}")
        return None


async def close_live_trade(trade: dict, exit_price: float, reason: str):
    """Place a real sell order to close a live position."""
    client = _get_clob_client()
    if not client:
        return

    direction  = trade.get("direction", "YES")
    shares     = trade.get("shares", 0)
    token_id   = trade.get("token_id", "")
    cost       = trade.get("cost", 0)

    try:
        from py_clob_client.clob_types import OrderArgs, OrderType
        from py_clob_client.constants import SELL

        order_args = OrderArgs(
            token_id=token_id,
            price=round(exit_price, 4),
            size=round(shares, 4),
            side=SELL,
        )
        signed_order = client.create_order(order_args)
        resp         = client.post_order(signed_order, OrderType.GTC)

        payout  = shares * exit_price
        pnl     = round(payout - cost, 2)
        won     = pnl > 0
        outcome = "WIN" if won else ("STOP_LOSS" if reason == "STOP_LOSS" else "LOSS")
        if reason == "TIMEOUT":
            outcome = "TIMEOUT"

        await db.close_live_trade(trade["id"], exit_price, pnl, outcome)
        await db.update_live_portfolio(
            cash_delta=payout, pnl_delta=pnl, invested_delta=-cost, win=won
        )

        sign = "+" if pnl >= 0 else ""
        emoji = "✅" if won else "❌"
        print(f"[LIVE] {emoji} CLOSE [{trade.get('market_type','?')}] {direction} "
              f"'{trade['market_question'][:38]}' "
              f"@ {exit_price:.3f} | PNL={sign}{pnl:.2f} | {outcome}")

    except Exception as e:
        print(f"[LIVE] ❌ Close order error: {e}")


async def check_live_exits(markets_by_id: dict):
    """
    Scan all open live trades for TP / SL / timeout.
    Mirrors paper_trader.check_exits() but executes real sell orders.
    """
    open_trades = await db.get_open_live_trades()
    if not open_trades:
        return

    now = datetime.utcnow()
    for trade in open_trades:
        market_id   = trade["market_id"]
        entry_px    = trade.get("entry_price", 0)
        direction   = trade.get("direction", "YES")
        market_type = trade.get("market_type", "LOCK_IN")

        try:
            created = datetime.fromisoformat(trade["created_at"])
        except Exception:
            continue

        market    = markets_by_id.get(market_id)
        yes_price = market.get("yes_price") if market else None
        cur_price = (yes_price if direction == "YES" else (1 - yes_price)) if yes_price is not None else entry_px

        age_hours = (now - created).total_seconds() / 3600
        max_hold  = LIVE_HOLD.get(market_type, 4)

        # Timeout
        if age_hours > max_hold:
            await close_live_trade(trade, cur_price, "TIMEOUT")
            continue

        if yes_price is None:
            continue

        # Early resolution guard (same as paper trader)
        if direction == "YES" and yes_price < 0.04:
            await close_live_trade(trade, cur_price, "STOP_LOSS")
            continue

        move = cur_price - entry_px
        tp   = LIVE_TP.get(market_type, 0.05)
        sl   = LIVE_SL.get(market_type, 0.04)

        if move >= tp:
            await close_live_trade(trade, cur_price, "TAKE_PROFIT")
        elif move <= -sl:
            await close_live_trade(trade, cur_price, "STOP_LOSS")


async def get_live_status() -> dict:
    """Return live trading status for dashboard."""
    gates_ok, gate_reason = await _check_safety_gates()
    portfolio = await db.get_live_portfolio()
    open_trades = await db.get_open_live_trades()

    wins   = portfolio.get("win_count", 0) or 0
    losses = portfolio.get("loss_count", 0) or 0
    total  = wins + losses
    wr     = round(wins / total * 100, 1) if total > 0 else 0

    return {
        "live_mode_enabled": LIVE_MODE,
        "gates_passed":      gates_ok,
        "gate_status":       gate_reason,
        "portfolio":         portfolio,
        "open_positions":    len(open_trades),
        "max_positions":     MAX_LIVE_POSITIONS,
        "win_rate":          wr,
        "total_trades":      total,
    }
