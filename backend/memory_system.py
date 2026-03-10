"""
Memory System â The agent's long-term learning brain.

Stores:
1. Trade reasoning chains (why the agent entered each trade)
2. Outcome analysis (what actually happened)
3. Extracted lessons (what the agent learned)
4. Category-specific insights (e.g., "I'm bad at sports markets")
5. Strategy effectiveness tracking

The LLM queries this before every trade decision to avoid repeating mistakes.
"""

import aiosqlite
import json
import os
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path

DB_PATH = Path(os.getenv("DB_PATH", "/app/pm_trading.db"))


async def init_memory():
    """Create memory tables."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript("""
            CREATE TABLE IF NOT EXISTS trade_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id INTEGER,
                market_id TEXT,
                market_question TEXT,
                category TEXT,
                direction TEXT,
                action TEXT,
                entry_price REAL,
                exit_price REAL,
                pnl REAL,
                outcome TEXT,
                confidence REAL,
                estimated_probability REAL,
                edge REAL,
                reasoning TEXT,
                key_evidence TEXT,
                risk_factors TEXT,
                lesson TEXT,
                volume_spike INTEGER DEFAULT 0,
                model_used TEXT,
                tokens_used INTEGER DEFAULT 0,
                created_at TEXT,
                resolved_at TEXT
            );

            CREATE TABLE IF NOT EXISTS agent_lessons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lesson TEXT NOT NULL,
                category TEXT,
                market_type TEXT,
                outcome TEXT,
                importance REAL DEFAULT 1.0,
                times_referenced INTEGER DEFAULT 0,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS category_stats (
                category TEXT PRIMARY KEY,
                total_trades INTEGER DEFAULT 0,
                wins INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                total_pnl REAL DEFAULT 0,
                avg_confidence REAL DEFAULT 0,
                avg_edge REAL DEFAULT 0,
                last_updated TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_memory_category
                ON trade_memory(category, outcome);
            CREATE INDEX IF NOT EXISTS idx_memory_outcome
                ON trade_memory(outcome, created_at);
            CREATE INDEX IF NOT EXISTS idx_lessons_category
                ON agent_lessons(category, importance DESC);
        """)
        await db.commit()


async def store_trade_reasoning(
    trade_id: int,
    market_id: str,
    market_question: str,
    category: str,
    direction: str,
    action: str,
    entry_price: float,
    confidence: float,
    estimated_probability: float,
    edge: float,
    reasoning: str,
    key_evidence: list,
    risk_factors: list,
    had_volume_spike: bool,
    model_used: str,
    tokens_used: int,
):
    """Store the full reasoning chain when entering a trade."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO trade_memory
               (trade_id, market_id, market_question, category, direction,
                action, entry_price, confidence, estimated_probability, edge,
                reasoning, key_evidence, risk_factors, volume_spike,
                model_used, tokens_used, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (trade_id, market_id, market_question, category, direction,
             action, entry_price, confidence, estimated_probability, edge,
             reasoning, json.dumps(key_evidence), json.dumps(risk_factors),
             1 if had_volume_spike else 0, model_used, tokens_used,
             datetime.utcnow().isoformat())
        )
        await db.commit()


async def record_trade_outcome(
    trade_id: int,
    exit_price: float,
    pnl: float,
    outcome: str,
    lesson: str = None,
):
    """Record the outcome of a trade and store the lesson."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """UPDATE trade_memory
               SET exit_price = ?, pnl = ?, outcome = ?,
                   lesson = ?, resolved_at = ?
               WHERE trade_id = ?""",
            (exit_price, pnl, outcome, lesson,
             datetime.utcnow().isoformat(), trade_id)
        )

        # Get trade details for category stats
        row = await db.execute_fetchall(
            "SELECT category, confidence, edge FROM trade_memory WHERE trade_id = ?",
            (trade_id,)
        )
        if row:
            cat = row[0][0] or "General"
            conf = row[0][1] or 0
            edge = row[0][2] or 0
            await _update_category_stats(db, cat, outcome, pnl, conf, edge)

        # Store lesson if provided
        if lesson:
            cat = row[0][0] if row else "General"
            importance = 1.5 if outcome == "LOSS" else 1.0  # Losses are more important to remember
            await db.execute(
                """INSERT INTO agent_lessons
                   (lesson, category, outcome, importance, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (lesson, cat, outcome, importance, datetime.utcnow().isoformat())
            )

        await db.commit()


async def _update_category_stats(db, category: str, outcome: str,
                                  pnl: float, confidence: float, edge: float):
    """Update running stats per market category."""
    existing = await db.execute_fetchall(
        "SELECT * FROM category_stats WHERE category = ?", (category,)
    )
    if existing:
        win_delta = 1 if outcome == "WIN" else 0
        loss_delta = 1 if outcome == "LOSS" else 0
        await db.execute(
            """UPDATE category_stats
               SET total_trades = total_trades + 1,
                   wins = wins + ?,
                   losses = losses + ?,
                   total_pnl = total_pnl + ?,
                   avg_confidence = (avg_confidence * total_trades + ?) / (total_trades + 1),
                   avg_edge = (avg_edge * total_trades + ?) / (total_trades + 1),
                   last_updated = ?
               WHERE category = ?""",
            (win_delta, loss_delta, pnl, confidence, edge,
             datetime.utcnow().isoformat(), category)
        )
    else:
        await db.execute(
            """INSERT INTO category_stats
               (category, total_trades, wins, losses, total_pnl,
                avg_confidence, avg_edge, last_updated)
               VALUES (?, 1, ?, ?, ?, ?, ?, ?)""",
            (category,
             1 if outcome == "WIN" else 0,
             1 if outcome == "LOSS" else 0,
             pnl, confidence, edge, datetime.utcnow().isoformat())
        )


async def get_relevant_lessons(category: str = None, limit: int = 10) -> list:
    """
    Get the most relevant lessons for the LLM to consider before trading.
    Prioritizes: high importance, recent, matching category.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        if category:
            rows = await db.execute_fetchall(
                """SELECT lesson, category, outcome, importance
                   FROM agent_lessons
                   WHERE category = ? OR category = 'General'
                   ORDER BY importance DESC, created_at DESC
                   LIMIT ?""",
                (category, limit)
            )
        else:
            rows = await db.execute_fetchall(
                """SELECT lesson, category, outcome, importance
                   FROM agent_lessons
                   ORDER BY importance DESC, created_at DESC
                   LIMIT ?""",
                (limit,)
            )

        lessons = []
        for r in rows:
            lessons.append(r["lesson"])
            # Mark as referenced
            await db.execute(
                """UPDATE agent_lessons
                   SET times_referenced = times_referenced + 1
                   WHERE lesson = ?""",
                (r["lesson"],)
            )
        await db.commit()
        return lessons


async def get_category_performance() -> dict:
    """Get win/loss stats per category. Helps agent avoid weak categories."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        rows = await db.execute_fetchall(
            "SELECT * FROM category_stats ORDER BY total_trades DESC"
        )
        result = {}
        for r in rows:
            total = r["total_trades"] or 1
            result[r["category"]] = {
                "total_trades": r["total_trades"],
                "wins": r["wins"],
                "losses": r["losses"],
                "win_rate": round((r["wins"] / total) * 100, 1),
                "total_pnl": round(r["total_pnl"], 2),
                "avg_confidence": round(r["avg_confidence"], 3),
                "avg_edge": round(r["avg_edge"], 4),
            }
        return result


async def get_memory_summary() -> dict:
    """Full summary of agent's memory and learning state."""
    async with aiosqlite.connect(DB_PATH) as db:
        # Total trades analyzed
        row = await db.execute_fetchall(
            "SELECT COUNT(*), SUM(CASE WHEN outcome='WIN' THEN 1 ELSE 0 END), "
            "SUM(CASE WHEN outcome='LOSS' THEN 1 ELSE 0 END), "
            "SUM(pnl), AVG(confidence), AVG(edge) "
            "FROM trade_memory WHERE outcome IS NOT NULL"
        )
        total = row[0][0] or 0
        wins = row[0][1] or 0
        losses = row[0][2] or 0
        total_pnl = row[0][3] or 0

        # Total lessons learned
        lesson_row = await db.execute_fetchall(
            "SELECT COUNT(*) FROM agent_lessons"
        )
        total_lessons = lesson_row[0][0] or 0

        # Pending trades (entered but not resolved)
        pending_row = await db.execute_fetchall(
            "SELECT COUNT(*) FROM trade_memory WHERE outcome IS NULL"
        )
        pending = pending_row[0][0] or 0

        # Volume spike trades performance
        spike_row = await db.execute_fetchall(
            "SELECT COUNT(*), SUM(CASE WHEN outcome='WIN' THEN 1 ELSE 0 END) "
            "FROM trade_memory WHERE volume_spike = 1 AND outcome IS NOT NULL"
        )
        spike_total = spike_row[0][0] or 0
        spike_wins = spike_row[0][1] or 0

        return {
            "total_analyzed_trades": total,
            "wins": wins,
            "losses": losses,
            "win_rate": round((wins / total) * 100, 1) if total else 0,
            "total_pnl": round(total_pnl, 2),
            "pending_trades": pending,
            "total_lessons_learned": total_lessons,
            "volume_spike_trades": spike_total,
            "volume_spike_win_rate": round((spike_wins / spike_total) * 100, 1) if spike_total else 0,
            "category_performance": await get_category_performance(),
        }
