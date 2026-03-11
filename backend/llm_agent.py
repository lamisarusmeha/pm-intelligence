"""
LLM Agent â The brain of PM Intelligence.

COST-EFFICIENT DUAL MODEL STRATEGY:
- Haiku ($0.25/M input, $1.25/M output) for initial screening of ALL markets
- Sonnet ($3/M input, $15/M output) ONLY for high-edge opportunities (>15% edge)
- This saves ~90% on API costs vs using Sonnet for everything

Uses Claude API to:
1. Analyze markets with real reasoning (not just threshold checks)
2. Incorporate news, volume spikes, and wallet activity
3. Produce structured trade decisions with confidence scores
4. Learn from past mistakes via memory system
"""

import os
import json
import asyncio
from datetime import datetime
from typing import Optional

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
SCREENING_MODEL = os.getenv("LLM_SCREEN_MODEL", "claude-haiku-4-5-20251001")
DEEP_MODEL = os.getenv("LLM_DEEP_MODEL", "claude-sonnet-4-20250514")

# Edge threshold for upgrading to Sonnet analysis
DEEP_ANALYSIS_EDGE = 0.12  # 12%+ edge triggers Sonnet re-analysis

# Cost tracking (per model)
_cost_tracking = {
    "haiku_calls": 0, "haiku_input": 0, "haiku_output": 0,
    "sonnet_calls": 0, "sonnet_input": 0, "sonnet_output": 0,
}


def get_cost_summary() -> dict:
    """Return detailed cost tracking per model."""
    haiku_cost = (
        (_cost_tracking["haiku_input"] / 1_000_000) * 0.25 +
        (_cost_tracking["haiku_output"] / 1_000_000) * 1.25
    )
    sonnet_cost = (
        (_cost_tracking["sonnet_input"] / 1_000_000) * 3.0 +
        (_cost_tracking["sonnet_output"] / 1_000_000) * 15.0
    )
    return {
        "haiku_calls": _cost_tracking["haiku_calls"],
        "sonnet_calls": _cost_tracking["sonnet_calls"],
        "total_calls": _cost_tracking["haiku_calls"] + _cost_tracking["sonnet_calls"],
        "haiku_cost_usd": round(haiku_cost, 4),
        "sonnet_cost_usd": round(sonnet_cost, 4),
        "total_cost_usd": round(haiku_cost + sonnet_cost, 4),
        "savings_vs_all_sonnet": round(
            max(0, ((_cost_tracking["haiku_calls"] * 0.08) - haiku_cost)), 4
        ),  # approx savings
    }


async def analyze_market(
    market: dict,
    news_context: str,
    volume_profile: dict,
    memory_lessons: list,
    portfolio_state: dict,
) -> Optional[dict]:
    """
    Two-stage analysis:
    Stage 1: Haiku screens the market (cheap, fast)
    Stage 2: If Haiku finds >12% edge, Sonnet does deep analysis (expensive, accurate)
    """
    if not HAS_ANTHROPIC or not ANTHROPIC_API_KEY:
        return _fallback_analysis(market, volume_profile)

    # Stage 1: Haiku screening
    haiku_result = await _call_llm(
        market, news_context, volume_profile, memory_lessons,
        portfolio_state, model=SCREENING_MODEL, max_tokens=500
    )

    if not haiku_result or haiku_result["action"] == "SKIP":
        return haiku_result

    # Stage 2: If Haiku found a big edge, confirm with Sonnet
    if abs(haiku_result.get("edge", 0)) >= DEEP_ANALYSIS_EDGE:
        print(f"[LLM] ð¬ Edge={haiku_result['edge']:.1%} â upgrading to Sonnet analysis")
        sonnet_result = await _call_llm(
            market, news_context, volume_profile, memory_lessons,
            portfolio_state, model=DEEP_MODEL, max_tokens=800
        )
        if sonnet_result:
            # Sonnet overrides Haiku â it's the more accurate model
            return sonnet_result

    return haiku_result


async def _call_llm(
    market: dict,
    news_context: str,
    volume_profile: dict,
    memory_lessons: list,
    portfolio_state: dict,
    model: str,
    max_tokens: int = 500,
) -> Optional[dict]:
    """Core LLM call â works with any Claude model."""
    prompt = _build_analysis_prompt(
        market, news_context, volume_profile, memory_lessons, portfolio_state
    )

    is_haiku = "haiku" in model.lower()
    cost_key = "haiku" if is_haiku else "sonnet"

    try:
        client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
        response = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}],
            system=(
                "You are a prediction market analyst. You analyze Polymarket markets "
                "and estimate true probabilities based on available evidence. "
                "You must respond ONLY with valid JSON. No markdown, no explanation outside JSON. "
                "Be calibrated: if you're unsure, say confidence is low. "
                "Never recommend a trade unless you see a clear edge (>10% mispricing). "
                "CRITICAL SANITY CHECKS: "
                "1) NEVER bet NO on a condition that is ALREADY TRUE (e.g. if BTC is $69k, do NOT buy NO on 'BTC above $64k'). "
                "2) NEVER recommend trades where entry price would be >$0.95 or <$0.05 — there is almost no room for profit. "
                "3) For price/threshold markets, always check if the current value already satisfies the condition."
            ),
        )

        _cost_tracking[f"{cost_key}_calls"] += 1
        _cost_tracking[f"{cost_key}_input"] += response.usage.input_tokens
        _cost_tracking[f"{cost_key}_output"] += response.usage.output_tokens

        raw = response.content[0].text.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()

        decision = json.loads(raw)

        action = decision.get("action", "SKIP")
        if action not in ("BUY_YES", "BUY_NO", "SKIP"):
            action = "SKIP"

        confidence = min(1.0, max(0.0, float(decision.get("confidence", 0))))
        est_prob = min(1.0, max(0.0, float(decision.get("estimated_probability", 0.5))))

        market_price = market.get("yes_price", 0.5)
        if action == "BUY_YES":
            edge = est_prob - market_price
        elif action == "BUY_NO":
            edge = (1 - est_prob) - (1 - market_price)
        else:
            edge = 0.0

        return {
            "action": action,
            "confidence": confidence,
            "reasoning": decision.get("reasoning", "No reasoning provided"),
            "estimated_probability": est_prob,
            "edge": round(edge, 4),
            "risk_factors": decision.get("risk_factors", []),
            "key_evidence": decision.get("key_evidence", []),
            "model": model,
            "tokens_used": response.usage.input_tokens + response.usage.output_tokens,
        }

    except json.JSONDecodeError as e:
        print(f"[LLM] JSON parse error ({model}): {e}")
        return None
    except Exception as e:
        print(f"[LLM] API error ({model}): {e}")
        return None


def _build_analysis_prompt(
    market: dict,
    news_context: str,
    volume_profile: dict,
    memory_lessons: list,
    portfolio_state: dict,
) -> str:
    """Build the full analysis prompt for Claude."""

    vol_alerts = volume_profile.get("recent_alerts", [])
    vol_text = "No unusual volume activity."
    if vol_alerts:
        vol_lines = []
        for a in vol_alerts[:3]:
            vol_lines.append(f"- {a.get('alert_type', 'UNKNOWN')}: {a.get('description', '')}")
        vol_text = "\n".join(vol_lines)

    has_spike = volume_profile.get("has_recent_spike", False)
    spike_note = ""
    if has_spike:
        spike_note = (
            "\n**IMPORTANT: This market has a recent volume spike. "
            "This could indicate insider knowledge or informed trading. "
            "Weight this heavily in your analysis.**"
        )

    lessons_text = "No previous lessons for similar markets."
    if memory_lessons:
        lesson_lines = [f"- {l}" for l in memory_lessons[:5]]
        lessons_text = "\n".join(lesson_lines)

    cash = portfolio_state.get("cash_balance", 10000)
    open_positions = portfolio_state.get("invested", 0)
    win_rate = portfolio_state.get("win_rate", 0)

    prompt = f"""Analyze this Polymarket prediction market and decide whether to trade.

## Market
- **Question:** {market.get('question', 'Unknown')}
- **Category:** {market.get('category', 'Unknown')}
- **Current YES price:** ${market.get('yes_price', 0.5):.4f} (market thinks {market.get('yes_price', 0.5)*100:.1f}% likely)
- **Current NO price:** ${market.get('no_price', 0.5):.4f}
- **24h Volume:** ${market.get('volume24hr', 0):,.0f}
- **Total Volume:** ${market.get('volume', 0):,.0f}
- **Liquidity:** ${market.get('liquidity', 0):,.0f}
- **End Date:** {market.get('end_date', 'Unknown')}

## Recent News Context
{news_context if news_context else "No relevant news found."}

## Volume Activity
{vol_text}
{spike_note}

## Lessons from Past Trades
{lessons_text}

## Portfolio
- Cash: ${cash:,.0f}
- Open positions value: ${open_positions:,.0f}
- Current win rate: {win_rate:.0f}%

## Your Task
1. Estimate the TRUE probability of YES based on all evidence
2. Compare to the market price to identify mispricing
3. Decide: BUY_YES, BUY_NO, or SKIP
4. Only recommend BUY if you see >5% edge AND confidence >= 0.4 (we are in learning phase — explore more markets)

Respond with ONLY this JSON (no other text):
{{
    "action": "BUY_YES" or "BUY_NO" or "SKIP",
    "confidence": 0.0 to 1.0,
    "estimated_probability": 0.0 to 1.0,
    "reasoning": "2-3 sentence explanation of your analysis",
    "risk_factors": ["list", "of", "risks"],
    "key_evidence": ["list", "of", "evidence", "that", "informed", "your", "decision"]
}}"""

    return prompt


def _fallback_analysis(market: dict, volume_profile: dict) -> Optional[dict]:
    """Fallback when LLM is unavailable."""
    has_spike = volume_profile.get("has_recent_spike", False)
    price = market.get("yes_price", 0.5)

    if not has_spike:
        return {
            "action": "SKIP",
            "confidence": 0.0,
            "reasoning": "No LLM available and no volume spike detected. Skipping.",
            "estimated_probability": price,
            "edge": 0.0,
            "risk_factors": ["No LLM analysis available"],
            "key_evidence": [],
            "model": "fallback_heuristic",
            "tokens_used": 0,
        }

    alerts = volume_profile.get("recent_alerts", [])
    spike_alert = alerts[0] if alerts else {}
    price_at_spike = spike_alert.get("price_at_alert", price)

    if price > price_at_spike:
        action = "BUY_YES"
        est_prob = min(0.95, price + 0.10)
    else:
        action = "BUY_NO"
        est_prob = max(0.05, price - 0.10)

    return {
        "action": action,
        "confidence": 0.5,
        "reasoning": "Volume spike detected (fallback mode, no LLM). Following smart money direction.",
        "estimated_probability": est_prob,
        "edge": 0.10,
        "risk_factors": ["Fallback analysis â no LLM reasoning"],
        "key_evidence": [spike_alert.get("description", "Volume spike")],
        "model": "fallback_heuristic",
        "tokens_used": 0,
    }


async def evaluate_trade_outcome(
    trade: dict,
    original_reasoning: str,
    outcome: str,
    pnl: float,
) -> Optional[str]:
    """
    After a trade closes, use Haiku (cheapest) to extract a lesson.
    Only called on losses or significant wins to save money.
    """
    if not HAS_ANTHROPIC or not ANTHROPIC_API_KEY:
        if outcome == "LOSS":
            return f"Lost ${abs(pnl):.2f} on {trade.get('market_question', 'unknown')}. No LLM analysis available."
        return None

    # LEARNING MODE: Extract lessons from ALL trades for maximum data
    # (Previously skipped small wins — but every data point matters during learning phase)

    prompt = f"""A trade just closed. Analyze what happened and extract a lesson.

## Trade Details
- Market: {trade.get('market_question', 'Unknown')}
- Direction: {trade.get('direction', 'Unknown')}
- Entry price: {trade.get('entry_price', 0)}
- Exit price: {trade.get('exit_price', 0)}
- P&L: ${pnl:.2f}
- Outcome: {outcome}

## Original Reasoning
{original_reasoning}

## Your Task
Write ONE concise lesson (1-2 sentences) that the agent should remember for future trades.
Focus on what the reasoning got wrong (if loss) or right (if win).
Be specific â mention the market type, category, or signal type.
Return ONLY the lesson text, nothing else."""

    try:
        client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
        response = await client.messages.create(
            model=SCREENING_MODEL,  # Use Haiku for lessons (cheapest)
            max_tokens=150,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
        )
        _cost_tracking["haiku_calls"] += 1
        _cost_tracking["haiku_input"] += response.usage.input_tokens
        _cost_tracking["haiku_output"] += response.usage.output_tokens

        return response.content[0].text.strip()

    except Exception as e:
        print(f"[LLM] Lesson extraction error: {e}")
        return f"{'Won' if outcome == 'WIN' else 'Lost'} ${abs(pnl):.2f} on {trade.get('market_question', 'unknown')}."
"""
LLM Agent â The brain of PM Intelligence.

COST-EFFICIENT DUAL MODEL STRATEGY:
- Haiku ($0.25/M input, $1.25/M output) for initial screening of ALL markets
- Sonnet ($3/M input, $15/M output) ONLY for high-edge opportunities (>15% edge)
- This saves ~90% on API costs vs using Sonnet for everything

Uses Claude API to:
1. Analyze markets with real reasoning (not just threshold checks)
2. Incorporate news, volume spikes, and wallet activity
3. Produce structured trade decisions with confidence scores
4. Learn from past mistakes via memory system
"""

import os
import json
import asyncio
from datetime import datetime
from typing import Optional

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
SCREENING_MODEL = os.getenv("LLM_SCREEN_MODEL", "claude-haiku-4-5-20251001")
DEEP_MODEL = os.getenv("LLM_DEEP_MODEL", "claude-sonnet-4-20250514")

# Edge threshold for upgrading to Sonnet analysis
DEEP_ANALYSIS_EDGE = 0.12  # 12%+ edge triggers Sonnet re-analysis

# Cost tracking (per model)
_cost_tracking = {
    "haiku_calls": 0, "haiku_input": 0, "haiku_output": 0,
    "sonnet_calls": 0, "sonnet_input": 0, "sonnet_output": 0,
}


def get_cost_summary() -> dict:
    """Return detailed cost tracking per model."""
    haiku_cost = (
        (_cost_tracking["haiku_input"] / 1_000_000) * 0.25 +
        (_cost_tracking["haiku_output"] / 1_000_000) * 1.25
    )
    sonnet_cost = (
        (_cost_tracking["sonnet_input"] / 1_000_000) * 3.0 +
        (_cost_tracking["sonnet_output"] / 1_000_000) * 15.0
    )
    return {
        "haiku_calls": _cost_tracking["haiku_calls"],
        "sonnet_calls": _cost_tracking["sonnet_calls"],
        "total_calls": _cost_tracking["haiku_calls"] + _cost_tracking["sonnet_calls"],
        "haiku_cost_usd": round(haiku_cost, 4),
        "sonnet_cost_usd": round(sonnet_cost, 4),
        "total_cost_usd": round(haiku_cost + sonnet_cost, 4),
        "savings_vs_all_sonnet": round(
            max(0, ((_cost_tracking["haiku_calls"] * 0.08) - haiku_cost)), 4
        ),  # approx savings
    }


async def analyze_market(
    market: dict,
    news_context: str,
    volume_profile: dict,
    memory_lessons: list,
    portfolio_state: dict,
) -> Optional[dict]:
    """
    Two-stage analysis:
    Stage 1: Haiku screens the market (cheap, fast)
    Stage 2: If Haiku finds >12% edge, Sonnet does deep analysis (expensive, accurate)
    """
    if not HAS_ANTHROPIC or not ANTHROPIC_API_KEY:
        return _fallback_analysis(market, volume_profile)

    # Stage 1: Haiku screening
    haiku_result = await _call_llm(
        market, news_context, volume_profile, memory_lessons,
        portfolio_state, model=SCREENING_MODEL, max_tokens=500
    )

    if not haiku_result or haiku_result["action"] == "SKIP":
        return haiku_result

    # Stage 2: If Haiku found a big edge, confirm with Sonnet
    if abs(haiku_result.get("edge", 0)) >= DEEP_ANALYSIS_EDGE:
        print(f"[LLM] ð¬ Edge={haiku_result['edge']:.1%} â upgrading to Sonnet analysis")
        sonnet_result = await _call_llm(
            market, news_context, volume_profile, memory_lessons,
            portfolio_state, model=DEEP_MODEL, max_tokens=800
        )
        if sonnet_result:
            # Sonnet overrides Haiku â it's the more accurate model
            return sonnet_result

    return haiku_result


async def _call_llm(
    market: dict,
    news_context: str,
    volume_profile: dict,
    memory_lessons: list,
    portfolio_state: dict,
    model: str,
    max_tokens: int = 500,
) -> Optional[dict]:
    """Core LLM call â works with any Claude model."""
    prompt = _build_analysis_prompt(
        market, news_context, volume_profile, memory_lessons, portfolio_state
    )

    is_haiku = "haiku" in model.lower()
    cost_key = "haiku" if is_haiku else "sonnet"

    try:
        client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
        response = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}],
            system=(
                "You are a prediction market analyst. You analyze Polymarket markets "
                "and estimate true probabilities based on available evidence. "
                "You must respond ONLY with valid JSON. No markdown, no explanation outside JSON. "
                "Be calibrated: if you're unsure, say confidence is low. "
                "Never recommend a trade unless you see a clear edge (>10% mispricing)."
            ),
        )

        _cost_tracking[f"{cost_key}_calls"] += 1
        _cost_tracking[f"{cost_key}_input"] += response.usage.input_tokens
        _cost_tracking[f"{cost_key}_output"] += response.usage.output_tokens

        raw = response.content[0].text.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()

        decision = json.loads(raw)

        action = decision.get("action", "SKIP")
        if action not in ("BUY_YES", "BUY_NO", "SKIP"):
            action = "SKIP"

        confidence = min(1.0, max(0.0, float(decision.get("confidence", 0))))
        est_prob = min(1.0, max(0.0, float(decision.get("estimated_probability", 0.5))))

        market_price = market.get("yes_price", 0.5)
        if action == "BUY_YES":
            edge = est_prob - market_price
        elif action == "BUY_NO":
            edge = (1 - est_prob) - (1 - market_price)
        else:
            edge = 0.0

        return {
            "action": action,
            "confidence": confidence,
            "reasoning": decision.get("reasoning", "No reasoning provided"),
            "estimated_probability": est_prob,
            "edge": round(edge, 4),
            "risk_factors": decision.get("risk_factors", []),
            "key_evidence": decision.get("key_evidence", []),
            "model": model,
            "tokens_used": response.usage.input_tokens + response.usage.output_tokens,
        }

    except json.JSONDecodeError as e:
        print(f"[LLM] JSON parse error ({model}): {e}")
        return None
    except Exception as e:
        print(f"[LLM] API error ({model}): {e}")
        return None


def _build_analysis_prompt(
    market: dict,
    news_context: str,
    volume_profile: dict,
    memory_lessons: list,
    portfolio_state: dict,
) -> str:
    """Build the full analysis prompt for Claude."""

    vol_alerts = volume_profile.get("recent_alerts", [])
    vol_text = "No unusual volume activity."
    if vol_alerts:
        vol_lines = []
        for a in vol_alerts[:3]:
            vol_lines.append(f"- {a.get('alert_type', 'UNKNOWN')}: {a.get('description', '')}")
        vol_text = "\n".join(vol_lines)

    has_spike = volume_profile.get("has_recent_spike", False)
    spike_note = ""
    if has_spike:
        spike_note = (
            "\n**IMPORTANT: This market has a recent volume spike. "
            "This could indicate insider knowledge or informed trading. "
            "Weight this heavily in your analysis.**"
        )

    lessons_text = "No previous lessons for similar markets."
    if memory_lessons:
        lesson_lines = [f"- {l}" for l in memory_lessons[:5]]
        lessons_text = "\n".join(lesson_lines)

    cash = portfolio_state.get("cash_balance", 10000)
    open_positions = portfolio_state.get("invested", 0)
    win_rate = portfolio_state.get("win_rate", 0)

    prompt = f"""Analyze this Polymarket prediction market and decide whether to trade.

## Market
- **Question:** {market.get('question', 'Unknown')}
- **Category:** {market.get('category', 'Unknown')}
- **Current YES price:** ${market.get('yes_price', 0.5):.4f} (market thinks {market.get('yes_price', 0.5)*100:.1f}% likely)
- **Current NO price:** ${market.get('no_price', 0.5):.4f}
- **24h Volume:** ${market.get('volume24hr', 0):,.0f}
- **Total Volume:** ${market.get('volume', 0):,.0f}
- **Liquidity:** ${market.get('liquidity', 0):,.0f}
- **End Date:** {market.get('end_date', 'Unknown')}

## Recent News Context
{news_context if news_context else "No relevant news found."}

## Volume Activity
{vol_text}
{spike_note}

## Lessons from Past Trades
{lessons_text}

## Portfolio
- Cash: ${cash:,.0f}
- Open positions value: ${open_positions:,.0f}
- Current win rate: {win_rate:.0f}%

## Your Task
1. Estimate the TRUE probability of YES based on all evidence
2. Compare to the market price to identify mispricing
3. Decide: BUY_YES, BUY_NO, or SKIP
4. Only recommend BUY if you see >10% edge AND confidence >= 0.6
5. NEVER bet NO on something ALREADY TRUE (e.g. if BTC=$69k, "BTC above $64k" is already true — do NOT buy NO)
6. NEVER recommend trades at extreme prices (YES price >$0.95 or <$0.05) — no profit room
7. For crypto/price markets: verify whether the current price already satisfies the condition BEFORE recommending

Respond with ONLY this JSON (no other text):
{{
    "action": "BUY_YES" or "BUY_NO" or "SKIP",
    "confidence": 0.0 to 1.0,
    "estimated_probability": 0.0 to 1.0,
    "reasoning": "2-3 sentence explanation of your analysis",
    "risk_factors": ["list", "of", "risks"],
    "key_evidence": ["list", "of", "evidence", "that", "informed", "your", "decision"]
}}"""

    return prompt


def _fallback_analysis(market: dict, volume_profile: dict) -> Optional[dict]:
    """Fallback when LLM is unavailable."""
    has_spike = volume_profile.get("has_recent_spike", False)
    price = market.get("yes_price", 0.5)

    if not has_spike:
        return {
            "action": "SKIP",
            "confidence": 0.0,
            "reasoning": "No LLM available and no volume spike detected. Skipping.",
            "estimated_probability": price,
            "edge": 0.0,
            "risk_factors": ["No LLM analysis available"],
            "key_evidence": [],
            "model": "fallback_heuristic",
            "tokens_used": 0,
        }

    alerts = volume_profile.get("recent_alerts", [])
    spike_alert = alerts[0] if alerts else {}
    price_at_spike = spike_alert.get("price_at_alert", price)

    if price > price_at_spike:
        action = "BUY_YES"
        est_prob = min(0.95, price + 0.10)
    else:
        action = "BUY_NO"
        est_prob = max(0.05, price - 0.10)

    return {
        "action": action,
        "confidence": 0.5,
        "reasoning": "Volume spike detected (fallback mode, no LLM). Following smart money direction.",
        "estimated_probability": est_prob,
        "edge": 0.10,
        "risk_factors": ["Fallback analysis â no LLM reasoning"],
        "key_evidence": [spike_alert.get("description", "Volume spike")],
        "model": "fallback_heuristic",
        "tokens_used": 0,
    }


async def evaluate_trade_outcome(
    trade: dict,
    original_reasoning: str,
    outcome: str,
    pnl: float,
) -> Optional[str]:
    """
    After a trade closes, use Haiku (cheapest) to extract a lesson.
    Only called on losses or significant wins to save money.
    """
    if not HAS_ANTHROPIC or not ANTHROPIC_API_KEY:
        if outcome == "LOSS":
            return f"Lost ${abs(pnl):.2f} on {trade.get('market_question', 'unknown')}. No LLM analysis available."
        return None

    # Only spend money on lesson extraction for losses or big wins
    if outcome == "WIN" and abs(pnl) < 50:
        return f"Won ${pnl:.2f} on {trade.get('market_question', 'unknown')}. Strategy worked as expected."

    prompt = f"""A trade just closed. Analyze what happened and extract a lesson.

## Trade Details
- Market: {trade.get('market_question', 'Unknown')}
- Direction: {trade.get('direction', 'Unknown')}
- Entry price: {trade.get('entry_price', 0)}
- Exit price: {trade.get('exit_price', 0)}
- P&L: ${pnl:.2f}
- Outcome: {outcome}

## Original Reasoning
{original_reasoning}

## Your Task
Write ONE concise lesson (1-2 sentences) that the agent should remember for future trades.
Focus on what the reasoning got wrong (if loss) or right (if win).
Be specific â mention the market type, category, or signal type.
Return ONLY the lesson text, nothing else."""

    try:
        client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
        response = await client.messages.create(
            model=SCREENING_MODEL,  # Use Haiku for lessons (cheapest)
            max_tokens=150,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
        )
        _cost_tracking["haiku_calls"] += 1
        _cost_tracking["haiku_input"] += response.usage.input_tokens
        _cost_tracking["haiku_output"] += response.usage.output_tokens

        return response.content[0].text.strip()

    except Exception as e:
        print(f"[LLM] Lesson extraction error: {e}")
        return f"{'Won' if outcome == 'WIN' else 'Lost'} ${abs(pnl):.2f} on {trade.get('market_question', 'unknown')}."
