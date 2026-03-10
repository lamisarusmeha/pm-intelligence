"""
Crypto Leverage Agent — Self-Learning Engine

Every LEARN_EVERY_N_TRADES closed trades, the learner:
  1. Loads the last HISTORY_WINDOW closed trades + their entry-time feature snapshots
  2. Compares factor-score breakdowns between winning and losing trades
  3. Adjusts 8 factor weights via gradient-style update (winners get more weight)
  4. Tracks per-strategy and per-regime win rates
  5. Dynamically raises/lowers MIN_SETUP_SCORE based on overall win rate
  6. Persists all weights to SQLite so they survive server restarts

The learned weights flow back into score_setup() in crypto_trader.py,
making the agent progressively more selective in the conditions that historically win.
"""

import json
import math
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from database import (
    get_closed_trades_with_meta,
    get_crypto_factor_weights,
    save_crypto_factor_weights,
    count_closed_crypto_trades,
)

# ── Hyperparameters ────────────────────────────────────────────────────────────
LEARN_EVERY_N_TRADES = 5         # trigger learning after every 5 new closed trades
HISTORY_WINDOW       = 40        # how many recent trades to analyze
LEARNING_RATE        = 0.12      # step size for weight gradient update
MOMENTUM             = 0.3       # fraction of last delta to carry forward (momentum SGD)
MAX_WEIGHT           = 3.0       # clamp upper bound
MIN_WEIGHT           = 0.15      # clamp lower bound

# Maximum possible points per factor in score_setup() — used for normalization
FACTOR_MAX_SCORES = {
    "htf_alignment":  15,
    "regime_clarity": 15,
    "structure":      15,
    "volume":         10,
    "book_flow":      10,
    "r2r":            15,
    "liquidity":      10,
    "event_risk":     10,
}

DEFAULT_WEIGHTS: Dict[str, float] = {f: 1.0 for f in FACTOR_MAX_SCORES}

# ── In-memory learning state ───────────────────────────────────────────────────
_factor_weights:    Dict[str, float]         = dict(DEFAULT_WEIGHTS)
_weight_deltas:     Dict[str, float]         = {f: 0.0 for f in DEFAULT_WEIGHTS}  # for momentum
_strategy_stats:    Dict[str, List[int]]     = {}  # strategy → [wins, total]
_regime_stats:      Dict[str, List[int]]     = {}  # regime   → [wins, total]
_asset_stats:       Dict[str, List[int]]     = {}  # symbol   → [wins, total]
_learning_log:      List[Dict]               = []  # last 20 learning cycles
_last_learn_count:  int                      = 0   # closed count at last learn
_dynamic_threshold: int                      = 65  # adaptive MIN_SETUP_SCORE


def get_factor_weights() -> Dict[str, float]:
    """Return current factor weights (called by crypto_trader.score_setup)."""
    return dict(_factor_weights)


def get_dynamic_threshold() -> int:
    """Return the current adaptive minimum setup score."""
    return _dynamic_threshold


def get_learning_stats() -> Dict:
    """Return full learning state for WebSocket broadcast and UI."""
    strategy_perf = {
        s: {
            "wins":     v[0],
            "total":    v[1],
            "win_rate": round(v[0] / max(v[1], 1) * 100, 1),
        }
        for s, v in _strategy_stats.items() if v[1] > 0
    }
    regime_perf = {
        r: {
            "wins":     v[0],
            "total":    v[1],
            "win_rate": round(v[0] / max(v[1], 1) * 100, 1),
        }
        for r, v in _regime_stats.items() if v[1] > 0
    }
    asset_perf = {
        a: {
            "wins":     v[0],
            "total":    v[1],
            "win_rate": round(v[0] / max(v[1], 1) * 100, 1),
        }
        for a, v in _asset_stats.items() if v[1] > 0
    }
    return {
        "factor_weights":      {k: round(v, 3) for k, v in _factor_weights.items()},
        "strategy_performance": strategy_perf,
        "regime_performance":   regime_perf,
        "asset_performance":    asset_perf,
        "dynamic_threshold":    _dynamic_threshold,
        "recent_cycles":        _learning_log[:5],
        "total_learn_cycles":   len(_learning_log),
        "last_learn_at":        _learning_log[0]["time"] if _learning_log else None,
    }


async def load_weights_from_db():
    """Load persisted weights from DB on startup (called once in main.py startup)."""
    global _factor_weights
    saved = await get_crypto_factor_weights()
    if saved:
        for factor, weight in saved.items():
            if factor in _factor_weights:
                _factor_weights[factor] = weight
        print(f"[LEARN] Loaded {len(saved)} factor weights from DB")


async def maybe_learn():
    """
    Check if enough new trades have closed to trigger a learning cycle.
    Call this from the main market loop after updating open trades.
    """
    global _last_learn_count
    closed_count = await count_closed_crypto_trades()
    if closed_count - _last_learn_count >= LEARN_EVERY_N_TRADES and closed_count >= 5:
        _last_learn_count = closed_count
        await run_learning_cycle()


async def run_learning_cycle():
    """
    Core learning algorithm:
      - Pull last HISTORY_WINDOW closed trades with entry snapshots
      - Separate into wins / losses
      - Compute per-factor gradient and apply weighted update
      - Update strategy/regime/asset win-rate trackers
      - Adjust dynamic score threshold
      - Persist weights to DB
    """
    global _factor_weights, _weight_deltas, _strategy_stats, \
           _regime_stats, _asset_stats, _learning_log, _dynamic_threshold

    trades = await get_closed_trades_with_meta(HISTORY_WINDOW)
    if len(trades) < 5:
        return

    # ── Parse and split ───────────────────────────────────────────────────────
    wins, losses = [], []
    for t in trades:
        try:
            snap = json.loads(t["snapshot_json"]) if t.get("snapshot_json") else {}
        except Exception:
            snap = {}
        t["_snap"] = snap
        if (t.get("pnl") or 0) > 0:
            wins.append(t)
        else:
            losses.append(t)

    total     = len(trades)
    win_rate  = round(len(wins) / total * 100, 1)

    # ── Factor weight gradient update ─────────────────────────────────────────
    win_snaps  = [t["_snap"] for t in wins  if t["_snap"].get("breakdown")]
    loss_snaps = [t["_snap"] for t in losses if t["_snap"].get("breakdown")]

    adjustments: Dict[str, float] = {}

    if win_snaps and loss_snaps:
        for factor, max_pts in FACTOR_MAX_SCORES.items():
            # Average normalized factor score for wins vs losses
            win_scores  = [s["breakdown"].get(factor, 0) / max_pts
                           for s in win_snaps  if "breakdown" in s]
            loss_scores = [s["breakdown"].get(factor, 0) / max_pts
                           for s in loss_snaps if "breakdown" in s]

            if not win_scores or not loss_scores:
                adjustments[factor] = 0.0
                continue

            avg_win  = sum(win_scores)  / len(win_scores)
            avg_loss = sum(loss_scores) / len(loss_scores)
            norm_diff = avg_win - avg_loss   # in [-1, +1]

            # Momentum SGD: delta = lr * grad + momentum * prev_delta
            grad  = norm_diff * LEARNING_RATE
            delta = grad + MOMENTUM * _weight_deltas.get(factor, 0.0)
            _weight_deltas[factor] = delta

            old_w = _factor_weights[factor]
            new_w = max(MIN_WEIGHT, min(MAX_WEIGHT, old_w + delta))
            _factor_weights[factor] = round(new_w, 5)
            adjustments[factor] = round(new_w - old_w, 5)

    # ── Strategy / regime / asset trackers ────────────────────────────────────
    # Reset with full history window (not cumulative) for freshness
    _strategy_stats.clear()
    _regime_stats.clear()
    _asset_stats.clear()

    for t in trades:
        snap     = t["_snap"]
        strategy = snap.get("strategy", "unknown")
        regime   = snap.get("regime", "unknown")
        symbol   = t.get("symbol", "unknown")
        won      = (t.get("pnl") or 0) > 0

        for store, key in [(_strategy_stats, strategy),
                           (_regime_stats,   regime),
                           (_asset_stats,    symbol)]:
            if key not in store:
                store[key] = [0, 0]
            store[key][0] += 1 if won else 0
            store[key][1] += 1

    # ── Adaptive score threshold ───────────────────────────────────────────────
    # If win rate < 45% → raise bar (be more selective)
    # If win rate > 65% → lower bar slightly (capture more good setups)
    # Clamp between 55 and 80
    if win_rate < 40:
        _dynamic_threshold = min(80, _dynamic_threshold + 3)
    elif win_rate < 50:
        _dynamic_threshold = min(78, _dynamic_threshold + 1)
    elif win_rate > 65:
        _dynamic_threshold = max(55, _dynamic_threshold - 1)
    elif win_rate > 75:
        _dynamic_threshold = max(55, _dynamic_threshold - 2)

    # ── Persist weights ────────────────────────────────────────────────────────
    await save_crypto_factor_weights(_factor_weights)

    # ── Log this cycle ─────────────────────────────────────────────────────────
    top_increased = sorted(
        [(f, d) for f, d in adjustments.items() if d > 0.001],
        key=lambda x: x[1], reverse=True
    )[:3]
    top_decreased = sorted(
        [(f, d) for f, d in adjustments.items() if d < -0.001],
        key=lambda x: x[1]
    )[:2]

    cycle_entry = {
        "time":             datetime.utcnow().isoformat(),
        "trades_analyzed":  total,
        "wins":             len(wins),
        "losses":           len(losses),
        "win_rate":         win_rate,
        "threshold":        _dynamic_threshold,
        "top_boosted":      [f"{f}(+{d:.3f})" for f, d in top_increased],
        "top_reduced":      [f"{f}({d:.3f})" for f, d in top_decreased],
        "weights_snapshot": {k: round(v, 3) for k, v in _factor_weights.items()},
        "adjustments":      {k: round(v, 4) for k, v in adjustments.items()},
    }
    _learning_log.insert(0, cycle_entry)
    _learning_log = _learning_log[:20]

    print(
        f"[LEARN] Cycle #{len(_learning_log)}: {win_rate}% WR ({len(wins)}W/{len(losses)}L) | "
        f"threshold → {_dynamic_threshold} | "
        f"boosted: {[f for f, _ in top_increased]}"
    )


def apply_weights_to_breakdown(breakdown: Dict[str, int]) -> float:
    """
    Apply current learned weights to a raw factor breakdown dict.
    Returns a weighted score (comparable to original 0-100 range via renormalization).
    """
    raw_total     = 0.0
    weighted_total = 0.0
    weight_sum     = 0.0

    for factor, max_pts in FACTOR_MAX_SCORES.items():
        raw_score = breakdown.get(factor, 0)
        weight    = _factor_weights.get(factor, 1.0)
        weighted_total += raw_score * weight
        raw_total      += raw_score
        weight_sum     += max_pts * weight

    # Renormalize to keep score in familiar 0-100 range
    if weight_sum > 0:
        return round(weighted_total / weight_sum * 100, 1)
    return round(raw_total, 1)
