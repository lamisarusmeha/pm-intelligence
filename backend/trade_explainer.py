"""
Trade Explainer — natural-language reasoning for every trade the agent makes.

Generates three pieces of text per trade:
  1. entry_explanation  — why the agent entered (which signals fired and why)
  2. exit_explanation   — what happened (outcome, what went right/wrong)
  3. lesson             — what the agent will do differently next time

All text is built entirely from the signal data already collected;
no external AI API is needed.
"""

from typing import Optional

# ── Factor descriptors ────────────────────────────────────────────────────────

FACTOR_NAMES = {
    "volume_spike": "volume spike",
    "price_zone":   "price positioning",
    "liquidity":    "liquidity pressure",
    "momentum":     "price momentum",
    "category":     "category insight",
}

def _level(score: float) -> str:
    if score >= 85: return "exceptional"
    if score >= 70: return "strong"
    if score >= 50: return "moderate"
    if score >= 30: return "weak"
    return "negligible"

def _factor_phrase(factor: str, score: float) -> str:
    level = _level(score)
    name  = FACTOR_NAMES.get(factor, factor)
    return f"{name} was {level} ({score:.0f}/100)"

def _top_factors(factors: dict, n: int = 2) -> list:
    """Return the n highest-scoring factor names."""
    return sorted(factors, key=lambda k: factors.get(k, 0), reverse=True)[:n]

def _bottom_factors(factors: dict, n: int = 2) -> list:
    """Return the n lowest-scoring factor names."""
    return sorted(factors, key=lambda k: factors.get(k, 0))[:n]


# ── Entry explanation ─────────────────────────────────────────────────────────

def explain_entry(signal: dict, trade: dict) -> str:
    """
    Generate a plain-English explanation of why the agent entered this trade.
    """
    question  = signal.get("market_question", trade.get("market_question", "Unknown market"))
    direction = trade.get("direction", "YES")
    score     = signal.get("score", 0)
    factors   = signal.get("factors", {})
    yes_price = signal.get("yes_price", 0.5)
    category  = signal.get("category", "general")
    cost      = trade.get("cost", 0)

    # Strength label
    strength = "strong" if score >= 70 else "moderate" if score >= 50 else "cautious"

    # Direction rationale
    if direction == "YES":
        price_pct = round(yes_price * 100)
        if yes_price < 0.42:
            dir_reason = (f"The market is pricing YES at {price_pct}¢ — "
                          f"the agent believes it's underpriced and expects an upward move.")
        else:
            dir_reason = (f"Recent momentum is driving YES higher (currently {price_pct}¢); "
                          f"the agent is riding the trend.")
    else:
        no_price_pct = round((1 - yes_price) * 100)
        dir_reason = (f"The market's YES price looks overextended at "
                      f"{round(yes_price*100)}¢; the agent is betting NO at {no_price_pct}¢.")

    # Top contributing factors
    tops = _top_factors(factors, 3)
    factor_lines = [f"  • {_factor_phrase(f, factors[f])}" for f in tops if f in factors]
    factor_block = "\n".join(factor_lines) if factor_lines else "  • (insufficient factor data)"

    # Position sizing note
    size_note = f"Position size: ${cost:.2f} ({round(cost/100, 1)}% of portfolio)"

    lines = [
        f"📥 ENTRY — {direction} on \"{question[:70]}\"",
        f"",
        f"The agent made a {strength} {direction} trade (signal score {score:.0f}/100).",
        f"",
        f"Reasoning:",
        f"{dir_reason}",
        f"",
        f"Key signals that fired:",
        factor_block,
        f"",
        f"Category: {category.title()} | {size_note}",
    ]
    return "\n".join(lines)


# ── Exit explanation ──────────────────────────────────────────────────────────

def explain_exit(trade: dict, reason: str, pnl: float,
                 entry_signal: Optional[dict] = None) -> str:
    """
    Generate a plain-English explanation of how and why the trade closed.
    """
    question  = trade.get("market_question", "Unknown market")
    direction = trade.get("direction", "YES")
    entry_px  = trade.get("entry_price", 0)
    exit_px   = trade.get("exit_price", 0)
    cost      = trade.get("cost", 0)
    won       = pnl > 0
    outcome   = "WIN" if won else "LOSS"

    pnl_pct = round((pnl / cost) * 100, 1) if cost else 0
    move    = round((exit_px - entry_px) * 100, 1)  # in cents

    # Reason narrative
    reason_map = {
        "WIN":         "The market moved in the agent's favour and the position was closed profitably.",
        "LOSS":        "The market moved against the agent's position.",
        "STOP_LOSS":   "The price moved more than 20¢ against the position — the agent cut losses early.",
        "TAKE_PROFIT": "The position gained more than 30¢ per share — the agent locked in profits.",
        "TIMEOUT":     f"The trade was held for the maximum allowed period and auto-closed.",
        "RESOLVED":    "The underlying market resolved and the position was settled.",
    }
    reason_text = reason_map.get(reason, f"Trade closed: {reason}.")

    icon = "✅" if won else "❌"

    lines = [
        f"{icon} EXIT ({reason}) — {direction} | {outcome}",
        f"",
        reason_text,
        f"",
        f"Entry: {round(entry_px*100, 1)}¢  →  Exit: {round(exit_px*100, 1)}¢  "
        f"({'▲' if move >= 0 else '▼'} {abs(move):.1f}¢ move)",
        f"P&L: {'+'if pnl>=0 else ''}{pnl:.2f} ({'+' if pnl_pct>=0 else ''}{pnl_pct}% return)",
    ]
    return "\n".join(lines)


# ── Lesson generator ──────────────────────────────────────────────────────────

def generate_lesson(factors: dict, pnl: float,
                    current_weights: dict,
                    reason: str = "") -> str:
    """
    Produce a plain-English lesson the agent 'learned' from this trade.
    Compares which factors were high/low against the outcome so the user
    can understand how weights are being adjusted.
    """
    won = pnl > 0

    tops    = _top_factors(factors, 2)
    bottoms = _bottom_factors(factors, 2)

    if won:
        top_names   = [FACTOR_NAMES.get(f, f) for f in tops]
        strong_note = " and ".join(top_names)
        lesson_core = (
            f"The {strong_note} signal{'s' if len(top_names)>1 else ''} "
            f"correctly predicted the outcome. "
            f"The agent will give {'these' if len(top_names)>1 else 'this'} "
            f"factor{'s' if len(top_names)>1 else ''} slightly more weight going forward."
        )
        if reason == "TAKE_PROFIT":
            lesson_core += (
                " The take-profit trigger worked well — the agent is calibrated "
                "to capture gains before reversals."
            )
    else:
        bot_names   = [FACTOR_NAMES.get(f, f) for f in bottoms]
        weak_note   = " and ".join(bot_names)
        top_names   = [FACTOR_NAMES.get(f, f) for f in tops]
        strong_note = " and ".join(top_names)
        lesson_core = (
            f"Despite {strong_note} scoring well, the trade lost. "
            f"The {weak_note} signal{'s were' if len(bot_names)>1 else ' was'} low, "
            f"which may have been a warning sign. "
            f"The agent will reduce the weight of factors that didn't predict this move."
        )
        if reason == "STOP_LOSS":
            lesson_core += (
                " The stop-loss protected the portfolio from a larger loss — "
                "risk management is working as intended."
            )
        elif reason == "TIMEOUT":
            lesson_core += (
                " The market did not move within the holding window. "
                "The agent may need higher confidence thresholds for slow-moving markets."
            )

    # Weight delta hints
    weight_hints = []
    for f in tops[:2]:
        w = current_weights.get(f, 1.0)
        new_w = round(w * 1.05 if won else w * 0.95, 2)
        direction_str = "↑" if won else "↓"
        weight_hints.append(
            f"{FACTOR_NAMES.get(f, f)}: {w:.2f} {direction_str} {new_w}"
        )

    hint_block = "  " + "  |  ".join(weight_hints) if weight_hints else ""

    lines = [
        f"🧠 LESSON",
        f"",
        lesson_core,
    ]
    if hint_block:
        lines += [f"", f"Weight adjustments:", hint_block]

    return "\n".join(lines)
