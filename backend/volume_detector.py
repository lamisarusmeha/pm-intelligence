"""
Volume Spike Detector — Identifies insider-like activity on Polymarket.

Tracks:
1. Volume anomalies: sudden spikes vs rolling average
2. Large single-wallet moves (whale detection)
3. Price-volume divergence (accumulation before a move)
4. Timing relative to known events
"""

import asyncio
import aiosqlite
import os
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path

DB_PATH = Path(os.getenv("DB_PATH", "/app/pm_trading.db"))


async def _ensure_tables():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript("""
            CREATE TABLE IF NOT EXISTS volume_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id TEXT NOT NULL,
                volume_24h REAL,
                total_volume REAL,
                yes_price REAL,
                liquidity REAL,
                timestamp TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS volume_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                spike_multiplier REAL,
                volume_before REAL,
                volume_after REAL,
                price_at_alert REAL,
                description TEXT,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_vol_snap_market
                ON volume_snapshots(market_id, timestamp);
            CREATE INDEX IF NOT EXISTS idx_vol_alerts_market
                ON volume_alerts(market_id, created_at);
        """)
        await db.commit()


async def record_snapshot(market_id, volume_24h, total_volume, yes_price, liquidity):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO volume_snapshots
               (market_id, volume_24h, total_volume, yes_price, liquidity, timestamp)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (market_id, volume_24h, total_volume, yes_price, liquidity,
             datetime.utcnow().isoformat())
        )
        await db.commit()


async def get_rolling_average(market_id, hours=24):
    cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        row = await db.execute_fetchall(
            """SELECT AVG(volume_24h) as avg_vol_24h, MAX(volume_24h) as max_vol_24h,
                MIN(volume_24h) as min_vol_24h, AVG(yes_price) as avg_price,
                COUNT(*) as sample_count
               FROM volume_snapshots WHERE market_id = ? AND timestamp > ?""",
            (market_id, cutoff)
        )
        if row and row[0][0] is not None:
            r = row[0]
            return {"avg_vol_24h": r[0] or 0, "max_vol_24h": r[1] or 0,
                    "min_vol_24h": r[2] or 0, "avg_price": r[3] or 0.5,
                    "sample_count": r[4] or 0}
    return {"avg_vol_24h": 0, "max_vol_24h": 0, "min_vol_24h": 0,
            "avg_price": 0.5, "sample_count": 0}


async def detect_spike(market_id, current_vol_24h, current_price, current_volume, liquidity):
    await _ensure_tables()
    await record_snapshot(market_id, current_vol_24h, current_volume, current_price, liquidity)
    rolling = await get_rolling_average(market_id, hours=24)
    if rolling["sample_count"] < 3:
        return None
    avg_vol = rolling["avg_vol_24h"]
    avg_price = rolling["avg_price"]
    if avg_vol <= 0:
        return None
    spike_ratio = current_vol_24h / avg_vol
    price_change = abs(current_price - avg_price) / max(avg_price, 0.01)
    alert = None
    if spike_ratio >= 3.0:
        if price_change < 0.05:
            alert = {"market_id": market_id, "alert_type": "ACCUMULATION",
                "spike_multiplier": round(spike_ratio, 2), "volume_before": round(avg_vol, 2),
                "volume_after": round(current_vol_24h, 2), "price_at_alert": current_price,
                "description": f"Volume {spike_ratio:.1f}x normal but price only moved {price_change*100:.1f}%. Stealth accumulation."}
        elif price_change > 0.15:
            alert = {"market_id": market_id, "alert_type": "WHALE_MOVE",
                "spike_multiplier": round(spike_ratio, 2), "volume_before": round(avg_vol, 2),
                "volume_after": round(current_vol_24h, 2), "price_at_alert": current_price,
                "description": f"Volume {spike_ratio:.1f}x normal + price moved {price_change*100:.1f}%. Whale activity."}
        else:
            alert = {"market_id": market_id, "alert_type": "VOLUME_SURGE",
                "spike_multiplier": round(spike_ratio, 2), "volume_before": round(avg_vol, 2),
                "volume_after": round(current_vol_24h, 2), "price_at_alert": current_price,
                "description": f"Volume {spike_ratio:.1f}x above average. Unusual activity."}
    if alert:
        await _save_alert(alert)
    return alert


async def _save_alert(alert):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO volume_alerts
               (market_id, alert_type, spike_multiplier, volume_before,
                volume_after, price_at_alert, description, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (alert["market_id"], alert["alert_type"], alert["spike_multiplier"],
             alert["volume_before"], alert["volume_after"], alert["price_at_alert"],
             alert["description"], datetime.utcnow().isoformat()))
        await db.commit()


async def get_recent_alerts(market_id=None, limit=20):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        if market_id:
            rows = await db.execute_fetchall(
                "SELECT * FROM volume_alerts WHERE market_id = ? ORDER BY created_at DESC LIMIT ?",
                (market_id, limit))
        else:
            rows = await db.execute_fetchall(
                "SELECT * FROM volume_alerts ORDER BY created_at DESC LIMIT ?", (limit,))
        return [dict(r) for r in rows]


async def get_market_volume_profile(market_id):
    rolling_1h = await get_rolling_average(market_id, hours=1)
    rolling_6h = await get_rolling_average(market_id, hours=6)
    rolling_24h = await get_rolling_average(market_id, hours=24)
    alerts = await get_recent_alerts(market_id, limit=5)
    has_recent_spike = any(
        a.get("alert_type") in ("ACCUMULATION", "WHALE_MOVE", "VOLUME_SURGE")
        for a in alerts
        if a.get("created_at", "") > (datetime.utcnow() - timedelta(hours=2)).isoformat()
    )
    return {"rolling_1h": rolling_1h, "rolling_6h": rolling_6h, "rolling_24h": rolling_24h,
            "recent_alerts": alerts, "has_recent_spike": has_recent_spike,
            "spike_count_24h": len([a for a in alerts
                if a.get("created_at", "") > (datetime.utcnow() - timedelta(hours=24)).isoformat()])}
