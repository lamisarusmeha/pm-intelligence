"""
Research Agent â Gathers context for the LLM to reason about.

Sources:
1. Polymarket market metadata (description, resolution criteria)
2. Basic web search for recent news (using free APIs)
3. Market history and price trends
4. Cross-market correlation (related markets)

Keeps costs LOW by using free data sources, not LLM calls.
"""

import httpx
import asyncio
import os
import json
from datetime import datetime, timedelta
from typing import Optional

# Free news API (no key needed for basic usage)
NEWS_SOURCES = [
    "https://newsdata.io/api/1/news",  # Free tier: 200 req/day
]

# Cache to avoid redundant API calls
_news_cache = {}
_cache_ttl = timedelta(minutes=30)


async def gather_market_context(market: dict) -> str:
    """
    Gather all available context for a market.
    Returns a formatted string for the LLM prompt.
    """
    question = market.get("question", "")
    category = market.get("category", "")
    description = market.get("description", "")

    context_parts = []

    # 1. Market metadata
    if description:
        # Truncate long descriptions
        desc = description[:500] + "..." if len(description) > 500 else description
        context_parts.append(f"Market description: {desc}")

    # 2. Resolution criteria (from market data)
    end_date = market.get("end_date", "")
    if end_date:
        context_parts.append(f"Resolution date: {end_date}")

    # 3. Price trend analysis
    price = market.get("yes_price", 0.5)
    vol_24h = market.get("volume24hr", 0)
    total_vol = market.get("volume", 0)
    liquidity = market.get("liquidity", 0)

    if vol_24h > 0 and total_vol > 0:
        vol_ratio = vol_24h / total_vol if total_vol > 0 else 0
        if vol_ratio > 0.3:
            context_parts.append(
                f"HIGH ACTIVITY: 24h volume is {vol_ratio*100:.0f}% of total volume. "
                f"This market is seeing unusual recent interest."
            )

    if liquidity > 0:
        if liquidity < 5000:
            context_parts.append(
                "LOW LIQUIDITY WARNING: Less than $5,000 liquidity. "
                "Prices may not reflect true probability. Be cautious."
            )
        elif liquidity > 100000:
            context_parts.append(
                "High liquidity market (>$100k). Prices likely reflect "
                "well-informed consensus."
            )

    # 4. Try to get news context (free, no API key needed)
    news = await _search_news_free(question, category)
    if news:
        context_parts.append(f"Recent news context:\n{news}")

    # 5. Related market cross-reference
    related = await _find_related_markets(market)
    if related:
        context_parts.append(f"Related markets:\n{related}")

    return "\n\n".join(context_parts) if context_parts else ""


async def _search_news_free(question: str, category: str) -> str:
    """
    Search for relevant news using free sources.
    Uses DuckDuckGo instant answer API (no key needed).
    """
    cache_key = f"{question[:50]}_{category}"

    # Check cache
    if cache_key in _news_cache:
        cached_time, cached_result = _news_cache[cache_key]
        if datetime.utcnow() - cached_time < _cache_ttl:
            return cached_result

    # Extract key terms from market question
    search_terms = _extract_search_terms(question)
    if not search_terms:
        return ""

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # DuckDuckGo instant answer API (free, no key)
            resp = await client.get(
                "https://api.duckduckgo.com/",
                params={
                    "q": search_terms,
                    "format": "json",
                    "no_html": 1,
                    "skip_disambig": 1,
                }
            )
            if resp.status_code == 200:
                data = resp.json()
                abstract = data.get("Abstract", "")
                related_topics = data.get("RelatedTopics", [])

                parts = []
                if abstract:
                    parts.append(abstract[:300])

                for topic in related_topics[:3]:
                    if isinstance(topic, dict) and "Text" in topic:
                        parts.append(f"- {topic['Text'][:150]}")

                result = "\n".join(parts) if parts else ""
                _news_cache[cache_key] = (datetime.utcnow(), result)
                return result

    except Exception:
        pass

    return ""


def _extract_search_terms(question: str) -> str:
    """Extract key search terms from a market question."""
    # Remove common prediction market phrasing
    removals = [
        "will", "by", "before", "after", "in", "the",
        "be", "to", "of", "a", "an", "on", "at",
        "?", "!", ".", ",", "'", '"',
    ]
    words = question.lower().split()
    filtered = [w for w in words if w not in removals and len(w) > 2]
    return " ".join(filtered[:8])  # Keep top 8 keywords


async def _find_related_markets(market: dict) -> str:
    """
    Find related markets on Polymarket for cross-reference.
    If multiple markets point the same direction, that's a signal.
    """
    category = market.get("category", "")
    question = market.get("question", "")

    if not category:
        return ""

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(
                "https://gamma-api.polymarket.com/markets",
                params={
                    "closed": "false",
                    "limit": 5,
                    "order": "volume24hr",
                    "ascending": "false",
                    "tag_slug": category.lower().replace(" ", "-"),
                }
            )
            if resp.status_code == 200:
                markets = resp.json()
                related = []
                for m in markets:
                    q = m.get("question", "")
                    if q and q != question:
                        tokens = m.get("tokens", [])
                        yes_price = 0.5
                        for t in tokens:
                            if t.get("outcome", "").upper() == "YES":
                                yes_price = float(t.get("price", 0.5))
                        related.append(
                            f"- \"{q[:80]}\" â YES at ${yes_price:.2f}"
                        )

                return "\n".join(related[:3]) if related else ""

    except Exception:
        pass

    return ""


async def get_market_sentiment(markets: list) -> dict:
    """
    Analyze overall market sentiment from a batch of markets.
    Useful context for the LLM to understand market conditions.
    """
    if not markets:
        return {"sentiment": "neutral", "confidence": 0}

    # Simple sentiment: are markets pricing things higher or lower than 50%?
    bullish = sum(1 for m in markets if m.get("yes_price", 0.5) > 0.6)
    bearish = sum(1 for m in markets if m.get("yes_price", 0.5) < 0.4)
    neutral = len(markets) - bullish - bearish

    total = len(markets)

    if bullish > bearish * 2:
        sentiment = "bullish"
    elif bearish > bullish * 2:
        sentiment = "bearish"
    else:
        sentiment = "mixed"

    return {
        "sentiment": sentiment,
        "bullish_markets": bullish,
        "bearish_markets": bearish,
        "neutral_markets": neutral,
        "total_analyzed": total,
    }
