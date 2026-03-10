"""
News Engine — maps breaking news to Polymarket markets + infers trade direction.

Sources (free, no auth):
  10 general RSS feeds: BBC, Reuters, AP, Guardian, CNN, NYT, etc.
  4 crypto RSS feeds: CoinTelegraph, CoinDesk, Decrypt, Bitcoin Magazine
  CryptoPanic API: crowdsourced crypto news votes (no API key needed for public feed)
  Refreshed every 15s in TURBO mode.

For each market:
  1. Match headlines via keyword overlap (crypto headlines given 1.3x boost for crypto markets)
  2. Score impact: HIGH (85) / MEDIUM (55) / LOW (25)
  3. Infer trade direction: does the headline imply YES or NO resolution?

Direction inference is the critical layer — it's what separates informed
information-asymmetry trading from random signal noise.

Examples:
  "Trump wins Iowa" + "Will Trump win Iowa?" → YES
  "Ceasefire deal reached" + "Will there be a ceasefire in 2025?" → YES
  "Fed raises rates again" + "Will Fed cut rates in 2025?" → NO
  "Biden drops out of race" + "Will Biden win 2024 election?" → NO
"""

import asyncio
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import httpx

import database as db

# ── RSS feeds (free, no auth) ─────────────────────────────────────────────────

RSS_FEEDS = [
    "http://feeds.bbci.co.uk/news/world/rss.xml",
    "http://feeds.bbci.co.uk/news/politics/rss.xml",
    "https://feeds.reuters.com/reuters/topNews",
    "https://feeds.reuters.com/Reuters/worldNews",
    "https://feeds.reuters.com/reuters/businessNews",
    "https://rss.cnn.com/rss/edition.rss",
    "https://www.theguardian.com/world/rss",
    "https://feeds.ap.org/ap/TopNews",
    "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
    "https://feeds.feedburner.com/typepad/alleyinsider/silicon_alley_insider",
]

# ── Crypto-specific RSS feeds ─────────────────────────────────────────────────
# These are tagged "crypto_source" = True for 1.3x impact boost on crypto markets

CRYPTO_RSS_FEEDS = [
    "https://cointelegraph.com/rss",
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://decrypt.co/feed",
    "https://bitcoinmagazine.com/.rss/full/",
]

# CryptoPanic public API — crowdsourced crypto news voting, no auth needed
CRYPTOPANIC_URL = "https://cryptopanic.com/api/v1/posts/?auth_token=&public=true&filter=hot&kind=news"

NEWS_CACHE_TTL  = 120   # seconds before stale (matches main.py NEWS_EVERY * 5s)
_news_cache: List[dict] = []
_last_news_fetch: Optional[datetime] = None

# ── Impact keywords ───────────────────────────────────────────────────────────

IMPACT_KEYWORDS = {
    "HIGH": [
        "election", "elected", "wins", "won", "defeated", "result",
        "ceasefire", "peace deal", "war", "invasion", "attack", "strike",
        "arrested", "indicted", "charged", "convicted", "acquitted",
        "dead", "died", "killed", "assassination",
        "federal reserve", "interest rate", "fed", "inflation",
        "crash", "collapse", "bankrupt", "default",
        "impeach", "resign", "fired", "appointed",
        "nuclear", "missile", "sanctions",
        "supreme court", "ruling", "verdict",
        "bitcoin", "crypto", "sec", "etf approved",
        "championship", "winner", "champion", "playoffs",
        "drops out", "withdraws", "concedes",
    ],
    "MEDIUM": [
        "poll", "survey", "ahead", "leads", "behind", "race",
        "negotiation", "talks", "meeting", "summit",
        "protest", "riot", "unrest",
        "economy", "gdp", "unemployment", "jobs",
        "trade", "tariff", "deal", "agreement",
        "investigation", "probe", "lawsuit",
        "vote", "ballot", "primary",
        "market", "stocks", "rally", "selloff",
    ],
}

# ── Direction inference keywords ──────────────────────────────────────────────
# Maps headline language to YES/NO resolution direction.
# YES = the event happened / the person won / the thing was approved
# NO  = the event failed / the person lost / the thing was rejected

YES_INDICATORS = [
    "wins", "win", "won", "elected", "re-elected", "confirmed",
    "approved", "passed", "signed", "reached deal", "ceasefire",
    "acquitted", "not guilty", "champion", "championship",
    "raises rates", "rate hike", "hikes rates",
    "launches", "announces", "deal reached", "agreement signed",
    "advances", "advances to", "qualifies",
    "record high", "surges", "soars", "spikes",
    "found guilty",  # guilty verdict → typically YES for "will be convicted" markets
]

NO_INDICATORS = [
    "loses", "lose", "lost", "defeated", "drops out", "withdraws",
    "withdrawing", "cancels", "cancelled", "suspended", "blocked",
    "vetoed", "rejected", "fails", "failed", "collapse",
    "crashes", "plunges", "falls", "drops",
    "not guilty",  # acquittal → NO for "will be convicted" markets
    "rate cut", "cuts rates", "lowers rates", "pauses hikes",
    "resigns", "impeached", "fired",
    "no deal", "talks collapse", "deal falls apart",
    "concedes", "conceded",
]


def _extract_keywords(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    words = text.split()
    stops = {'the','a','an','in','on','at','is','are','was','were','to',
              'of','and','or','but','for','with','by','from','as','it',
              'be','has','had','have','will','would','could','should'}
    return [w for w in words if w not in stops and len(w) > 2]


def _news_impact_score(headline: str) -> Tuple[float, str]:
    hl = headline.lower()
    for word in IMPACT_KEYWORDS["HIGH"]:
        if word in hl:
            return 85.0, "HIGH"
    for word in IMPACT_KEYWORDS["MEDIUM"]:
        if word in hl:
            return 55.0, "MEDIUM"
    return 25.0, "LOW"


def _infer_direction_from_headline(headline: str) -> Optional[str]:
    """
    Return 'YES', 'NO', or None based on headline language.

    Examples:
      "Trump wins Iowa caucus"      → YES
      "Biden drops out of race"     → NO
      "Ceasefire deal reached"      → YES
      "Peace talks collapse"        → NO
      "Fed raises rates again"      → YES (for "will Fed raise rates" markets)
      "Fed cuts rates"              → NO  (for "will Fed raise rates" markets)
    """
    hl = headline.lower()
    yes_score = sum(1 for phrase in YES_INDICATORS if phrase in hl)
    no_score  = sum(1 for phrase in NO_INDICATORS  if phrase in hl)

    if yes_score > no_score:
        return "YES"
    if no_score > yes_score:
        return "NO"
    return None   # ambiguous — don't override


def _match_market(headline: str, market_question: str) -> float:
    """Return 0-1 relevance score between headline and market question."""
    h_words = set(_extract_keywords(headline))
    m_words = set(_extract_keywords(market_question))

    if not h_words or not m_words:
        return 0.0

    overlap = h_words & m_words
    if not overlap:
        return 0.0

    score = len(overlap) / min(len(h_words), len(m_words))
    # Bonus for proper nouns / numbers (names, years, specific entities)
    named = {w for w in overlap if len(w) > 4}
    score += len(named) * 0.08

    return min(1.0, score)


async def _fetch_cryptopanic(client: httpx.AsyncClient) -> List[dict]:
    """
    CryptoPanic public API — crowdsourced crypto news with vote counts.
    No API key needed for the public feed. Returns posts sorted by hot.
    Adds 'crypto_source': True flag so match scoring can apply boost.
    """
    try:
        resp = await client.get(CRYPTOPANIC_URL, timeout=8, follow_redirects=True)
        if resp.status_code != 200:
            return []
        data = resp.json()
        items = []
        for post in (data.get("results") or []):
            title = post.get("title", "")
            if not title:
                continue
            votes = post.get("votes", {})
            # Bull/bear votes are a proxy for direction signal
            bull = (votes.get("positive") or 0) + (votes.get("liked") or 0)
            bear = (votes.get("negative") or 0) + (votes.get("disliked") or 0)
            cp_direction: Optional[str] = None
            if bull > bear * 1.5:
                cp_direction = "YES"
            elif bear > bull * 1.5:
                cp_direction = "NO"
            items.append({
                "headline":     title,
                "published":    post.get("published_at", datetime.utcnow().isoformat()),
                "link":         post.get("url", ""),
                "source":       "cryptopanic",
                "crypto_source": True,
                "cp_direction": cp_direction,
                "vote_total":   bull + bear,
            })
        return items
    except Exception:
        return []


async def _fetch_rss(url: str, client: httpx.AsyncClient, crypto: bool = False) -> List[dict]:
    try:
        resp = await client.get(url, timeout=8, follow_redirects=True)
        if resp.status_code != 200:
            return []
        root  = ET.fromstring(resp.text)
        items = []
        for item in root.iter("item"):
            title = item.findtext("title", "")
            pub   = item.findtext("pubDate", datetime.utcnow().isoformat())
            link  = item.findtext("link", "")
            if title:
                items.append({
                    "headline":     title,
                    "published":    pub,
                    "link":         link,
                    "source":       url.split("/")[2],
                    "crypto_source": crypto,
                })
        return items
    except Exception:
        return []


async def refresh_news():
    """
    Fetch all sources: general RSS + crypto RSS + CryptoPanic API.
    Crypto sources are tagged so _match_market can apply a 1.3x boost for crypto markets.
    """
    global _news_cache, _last_news_fetch

    now = datetime.utcnow()
    if _last_news_fetch and (now - _last_news_fetch).total_seconds() < NEWS_CACHE_TTL:
        return

    _last_news_fetch = now
    all_items = []

    async with httpx.AsyncClient() as client:
        # General news RSS
        gen_tasks    = [_fetch_rss(url, client, crypto=False) for url in RSS_FEEDS]
        # Crypto RSS feeds
        crypto_tasks = [_fetch_rss(url, client, crypto=True) for url in CRYPTO_RSS_FEEDS]
        # CryptoPanic API
        cp_task      = _fetch_cryptopanic(client)

        results = await asyncio.gather(*gen_tasks, *crypto_tasks, cp_task, return_exceptions=True)
        for r in results:
            if isinstance(r, list):
                all_items.extend(r)

    cutoff = now - timedelta(hours=3)   # slightly wider window for crypto (24/7 market)
    fresh  = []
    from email.utils import parsedate_to_datetime
    for item in all_items:
        try:
            pub_str = item.get("published", "")
            try:
                pub = parsedate_to_datetime(pub_str).replace(tzinfo=None)
            except Exception:
                try:
                    pub = datetime.fromisoformat(pub_str.replace("Z", ""))
                except Exception:
                    pub = now   # assume fresh if unparseable
            if pub < cutoff:
                continue
            score, level = _news_impact_score(item["headline"])
            # CryptoPanic vote-based direction overrides keyword heuristic
            if item.get("cp_direction"):
                direction = item["cp_direction"]
            else:
                direction = _infer_direction_from_headline(item["headline"])
            item["impact_score"] = score
            item["impact_level"] = level
            item["direction"]    = direction
            fresh.append(item)
        except Exception:
            item["impact_score"] = 25.0
            item["impact_level"] = "LOW"
            item["direction"]    = None
            fresh.append(item)

    _news_cache = fresh
    if fresh:
        with_dir   = sum(1 for i in fresh if i.get("direction"))
        crypto_cnt = sum(1 for i in fresh if i.get("crypto_source"))
        print(f"[NEWS] {len(fresh)} headlines | {with_dir} with direction | {crypto_cnt} crypto-specific")
        await db.save_news_events(fresh)


def get_news_score(market_question: str, market_id: str) -> Tuple[float, List[str]]:
    """
    Return (0-100 news impact score, list of matching headlines) for a market.
    HIGH: 70-100, MEDIUM: 40-70, LOW: 10-40, NONE: 0

    Crypto markets get a 1.3x boost from crypto-source headlines (CryptoPanic, CoinTelegraph, etc.)
    since these sources are directly relevant to crypto market outcomes.
    """
    if not _news_cache:
        return 0.0, []

    q_lower = market_question.lower()
    is_crypto_market = any(kw in q_lower for kw in [
        "bitcoin", "btc", "ethereum", "eth", "crypto", "defi", "nft", "solana",
        "sol", "token", "coin", "blockchain", "altcoin", "usdt", "usdc", "binance"
    ])

    best_score        = 0.0
    matched_headlines = []

    for item in _news_cache:
        relevance = _match_market(item["headline"], market_question)
        if relevance < 0.12:
            continue
        base = item["impact_score"] * relevance
        # Apply 1.3x boost: crypto source + crypto market = higher conviction
        if is_crypto_market and item.get("crypto_source"):
            base *= 1.3
        if base > best_score:
            best_score = base
        if relevance > 0.18:
            matched_headlines.append(item["headline"][:80])

    return round(min(100.0, best_score), 1), matched_headlines[:3]


def get_news_direction(market_question: str,
                       matched_headlines: List[str]) -> Optional[str]:
    """
    Infer the trade direction from matched headlines and the market question.

    Strategy:
    1. Look at all matched headline directions
    2. Vote — if majority point YES, return YES; if majority NO, return NO
    3. Only return a direction if consensus is clear (no ties)

    This is only called when signal_engine already found a good match,
    so we trust the headlines are relevant.
    """
    if not matched_headlines or not _news_cache:
        return None

    yes_votes = 0
    no_votes  = 0

    for headline in matched_headlines:
        # Find this headline in cache to get its pre-computed direction
        for item in _news_cache:
            if item["headline"][:80] == headline:
                d = item.get("direction")
                if d == "YES":
                    yes_votes += 1
                elif d == "NO":
                    no_votes += 1
                break

    if yes_votes > no_votes:
        return "YES"
    if no_votes > yes_votes:
        return "NO"
    return None  # tie or no votes → let other signals decide


def get_cached_news_count() -> int:
    return len(_news_cache)
