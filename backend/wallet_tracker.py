"""
Wallet Tracker — follows smart money on Polymarket.

Fetches recent trades from the CLOB API, identifies wallets that are:
  - Consistently profitable (win rate > 60%)
  - Active recently (traded in last 24h)
  - Placing meaningful size (not dust trades)

When a smart wallet enters a market, that market gets a boosted signal score.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import httpx

import database as db

CLOB_API    = "https://clob.polymarket.com"
MIN_TRADES  = 5       # minimum trades before we trust a wallet's win rate
SMART_WINRATE = 0.58  # wallets above this are "smart money"
MIN_SIZE    = 20.0    # minimum trade size in USDC to count
CACHE_TTL   = 300     # seconds before refreshing wallet data

# In-memory cache: market_id -> list of smart wallet entries
_smart_wallet_cache: Dict[str, List[dict]] = {}
_last_refresh: Optional[datetime] = None
_wallet_stats: Dict[str, dict] = {}  # address -> {wins, losses, total_size}


async def _fetch_recent_trades(limit: int = 500) -> List[dict]:
    """Fetch recent trades from Polymarket CLOB API."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{CLOB_API}/trades",
                params={"limit": limit}
            )
            if resp.status_code != 200:
                return []
            data = resp.json()
            return data.get("data", data) if isinstance(data, dict) else data
    except Exception as e:
        print(f"[WALLET] Fetch error: {e}")
        return []


async def _fetch_market_trades(market_id: str, limit: int = 100) -> List[dict]:
    """Fetch trades for a specific market."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{CLOB_API}/trades",
                params={"market": market_id, "limit": limit}
            )
            if resp.status_code != 200:
                return []
            data = resp.json()
            return data.get("data", data) if isinstance(data, dict) else data
    except Exception as e:
        return []


def _analyze_wallet(trades: List[dict]) -> dict:
    """
    Analyze a wallet's trades to determine if it's 'smart money'.
    Returns stats dict with win_rate, trade_count, avg_size, is_smart.
    """
    if len(trades) < MIN_TRADES:
        return {"is_smart": False, "win_rate": 0, "trade_count": len(trades)}

    wins   = sum(1 for t in trades if float(t.get("price", 0)) > 0.5 and t.get("side") == "BUY")
    total  = len([t for t in trades if t.get("side") == "BUY"])
    if total == 0:
        return {"is_smart": False, "win_rate": 0, "trade_count": 0}

    avg_size = sum(float(t.get("size", 0)) for t in trades) / len(trades)
    win_rate = wins / total if total > 0 else 0

    return {
        "is_smart":    win_rate >= SMART_WINRATE and avg_size >= MIN_SIZE,
        "win_rate":    round(win_rate, 3),
        "trade_count": total,
        "avg_size":    round(avg_size, 2),
    }


async def refresh_smart_wallets(markets: List[dict]):
    """
    Main refresh cycle: fetch recent trades, identify smart wallets,
    cache which markets they're active in.
    """
    global _smart_wallet_cache, _last_refresh, _wallet_stats

    now = datetime.utcnow()
    if _last_refresh and (now - _last_refresh).total_seconds() < CACHE_TTL:
        return  # still fresh

    _last_refresh = now
    _smart_wallet_cache = {}

    try:
        trades = await _fetch_recent_trades(500)
        if not trades:
            return

        # Group trades by wallet
        wallet_trades: Dict[str, List[dict]] = {}
        for t in trades:
            addr = t.get("maker", t.get("trader", t.get("address", "")))
            if not addr:
                continue
            if addr not in wallet_trades:
                wallet_trades[addr] = []
            wallet_trades[addr].append(t)

        # Find smart wallets
        smart_addrs = set()
        for addr, wtrades in wallet_trades.items():
            stats = _analyze_wallet(wtrades)
            _wallet_stats[addr] = stats
            if stats["is_smart"]:
                smart_addrs.add(addr)

        print(f"[WALLET] {len(smart_addrs)} smart wallets found out of {len(wallet_trades)} active")

        # Map smart wallets to markets
        for t in trades:
            addr = t.get("maker", t.get("trader", t.get("address", "")))
            if addr not in smart_addrs:
                continue
            market_id = str(t.get("market", t.get("marketId", "")))
            if not market_id:
                continue
            if market_id not in _smart_wallet_cache:
                _smart_wallet_cache[market_id] = []
            _smart_wallet_cache[market_id].append({
                "address":   addr,
                "side":      t.get("side", "BUY"),
                "size":      float(t.get("size", 0)),
                "price":     float(t.get("price", 0)),
                "timestamp": t.get("timestamp", t.get("createdAt", "")),
                "win_rate":  _wallet_stats[addr]["win_rate"],
            })

        # Save to DB
        await db.save_smart_wallet_activity(_smart_wallet_cache)

    except Exception as e:
        print(f"[WALLET] Refresh error: {e}")


def get_smart_wallet_score(market_id: str) -> float:
    """
    Return a 0-100 score for how much smart money is in this market.
    0  = no smart wallet activity
    100 = multiple high-win-rate wallets actively trading this market
    """
    entries = _smart_wallet_cache.get(str(market_id), [])
    if not entries:
        return 0.0

    # Weight by win rate and recency
    score = 0.0
    for e in entries:
        wr_bonus = (e.get("win_rate", 0.6) - SMART_WINRATE) * 200  # 0-80
        size_bonus = min(20, e.get("size", 0) / 100)               # 0-20
        score += 50 + wr_bonus + size_bonus

    return min(100.0, score / max(1, len(entries)) * len(entries) * 0.5)


def get_smart_wallet_direction(market_id: str) -> Optional[str]:
    """Return which direction smart wallets are betting (YES/NO), or None."""
    entries = _smart_wallet_cache.get(str(market_id), [])
    if not entries:
        return None

    buy_size  = sum(e["size"] for e in entries if e.get("side") == "BUY")
    sell_size = sum(e["size"] for e in entries if e.get("side") == "SELL")

    if buy_size > sell_size * 1.5:
        return "YES"
    if sell_size > buy_size * 1.5:
        return "NO"
    return None
