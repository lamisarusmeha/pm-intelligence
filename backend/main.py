"""
PM Intelligence v3.1 â 4-Strategy Self-Learning Trading Agent

Strategies:
1. Near-Certainty Grinder â scans for 80-97% likely outcomes
2. Volume Spike Trading â detects 3x+ volume anomalies
3. Binance Price Lag Arbitrage â exploits Polymarket lag behind Binance
4. Short-Duration 5m/15m Markets â rolling crypto up/down markets (NEW)

Plus: LLM analysis cycle for general market screening
"""

import asyncio
import base64
import json
import os
import secrets
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Set

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response

import database as db
from paper_trader import (
    maybe_enter_trade, check_exits,
    check_leverage_exits,
    _learning_errors,
)
from near_certainty_grinder import generate_near_certainty_signals
from volume_spike_trader import generate_spike_signals
from binance_arb import generate_arb_signals
from short_duration_trader import generate_short_duration_signals
from binance_feed import (
    binance_websocket_loop,
    binance_prices,
    get_status as get_binance_status,
)
import telegram_alerts
import volume_detector
import memory_system
import self_improvement_engine as sie

try:
    from llm_agent import analyze_market, get_cost_summary, evaluate_trade_outcome
    HAS_LLM = True
except ImportError:
    HAS_LLM = False
    def get_cost_summary(): return {"haiku_calls": 0, "sonnet_calls": 0, "total_cost_usd": 0}

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ââ Configuration âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

DASHBOARD_PASSWORD = os.getenv("DASHBOARD_PASSWORD", "")

# Frontend path resolution (3-level fallback for Railway)
BASE_DIR     = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
INDEX_HTML   = FRONTEND_DIR / "index.html"
if not INDEX_HTML.exists():
    BASE_DIR     = Path(os.getcwd())
    FRONTEND_DIR = BASE_DIR / "frontend"
    INDEX_HTML   = FRONTEND_DIR / "index.html"
if not INDEX_HTML.exists():
    BASE_DIR     = Path("/app")
    FRONTEND_DIR = BASE_DIR / "frontend"
    INDEX_HTML   = FRONTEND_DIR / "index.html"

# Polymarket API
GAMMA_API      = "https://gamma-api.polymarket.com"
FETCH_LIMIT    = 1000
NEAR_RES_LIMIT = 300
MIN_DAYS       = 0
MAX_DAYS       = 30
NEAR_RES_DAYS  = 7

# Loop timing
LOOP_SLEEP     = 3
GRINDER_EVERY  = 20    # Run grinder every 20th loop (~60s)
LLM_EVERY      = 30    # Run LLM analysis every 30th loop (~90s)
SHORT_DUR_EVERY = 3    # Run short-duration check every 3rd loop (~9s)

# Global state
active_connections: Set[WebSocket] = set()
_loop_count = 0
_strategy_debug = {
    "last_loop": None,
    "arb_signals": 0,
    "spike_signals": 0,
    "grinder_signals": 0,
    "short_duration_signals": 0,
    "llm_signals": 0,
    "total_entered": 0,
    "trades_closed_this_session": 0,
    "loops_run": 0,
}


# ââ App Lifecycle ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

@asynccontextmanager
async def lifespan(app):
    print("[v3.1] PM Intelligence v3.1 â Starting up")
    await db.init_db()
    await memory_system.init_memory()

    # ââ Memory system startup diagnostic ââââââââââââââââââââââââââââââââââ
    try:
        import aiosqlite
        async with aiosqlite.connect(db.DB_PATH) as conn:
            tables = await conn.execute_fetchall(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            table_names = [t[0] for t in tables]
            print(f"[STARTUP] SQLite tables: {table_names}")

            if "signal_performance" in table_names:
                count = await conn.execute_fetchall("SELECT COUNT(*) FROM signal_performance")
                print(f"[STARTUP] signal_performance rows: {count[0][0]}")
            else:
                print("[STARTUP] WARNING: signal_performance table missing!")

        # Test self_improvement_engine.record_trade_result() with dummy data
        await sie.record_trade_result(
            trade_id=-1, market_type="TEST", direction="YES",
            entry_price=0.5, exit_price=0.6, pnl=0.1, won=True,
            signal_factors={},
        )
        # Clean up test record
        async with aiosqlite.connect(db.DB_PATH) as conn:
            await conn.execute("DELETE FROM signal_performance WHERE trade_id = -1")
            await conn.commit()
        print("[STARTUP] Memory system diagnostic: PASS")
    except Exception as e:
        err_msg = f"[STARTUP] Memory system diagnostic FAILED: {e}"
        print(err_msg)
        import traceback
        _learning_errors.append({
            "error": err_msg,
            "traceback": traceback.format_exc(),
            "time": datetime.utcnow().isoformat(),
        })

    try:
        await volume_detector._ensure_tables()
    except Exception as e:
        print(f"[STARTUP] Volume tables: {e}")

    # Close trades stuck from previous deployments
    await close_stuck_trades()

    # Seed default signal weights
    await seed_weights()

    # Ensure strategy_params and signal_performance tables exist
    await _ensure_self_learning_tables()

    # Populate arb entered markets from open BINANCE_ARB trades (prevents double-entry after redeploy)
    try:
        from binance_arb import _arb_entered_markets
        open_trades = await db.get_open_paper_trades()
        for trade in open_trades:
            if trade.get("market_type") == "BINANCE_ARB":
                _arb_entered_markets.add(trade["market_id"])
        if _arb_entered_markets:
            print(f"[STARTUP] Populated {len(_arb_entered_markets)} BINANCE_ARB entered markets from DB")
    except Exception as e:
        print(f"[STARTUP] Arb market population: {e}")

    # Launch background tasks
    asyncio.create_task(binance_websocket_loop())
    asyncio.create_task(trading_loop())

    telegram_alerts.alert_startup()
    print(f"[v3.1] Dashboard: {'auth-protected' if DASHBOARD_PASSWORD else 'open'}")
    print(f"[v3.1] Frontend: {INDEX_HTML} (exists={INDEX_HTML.exists()})")

    yield

    print("[v3.1] Shutting down")


app = FastAPI(lifespan=lifespan)


# ââ Helpers âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

def _check_auth(request: Request) -> bool:
    if not DASHBOARD_PASSWORD:
        return True
    auth = request.headers.get("authorization", "")
    if auth.startswith("Basic "):
        try:
            decoded = base64.b64decode(auth[6:]).decode()
            _, password = decoded.split(":", 1)
            return secrets.compare_digest(password, DASHBOARD_PASSWORD)
        except Exception:
            pass
    return False


def _auth_required(request: Request):
    if not _check_auth(request):
        raise HTTPException(
            status_code=401,
            headers={"WWW-Authenticate": 'Basic realm="PM Intelligence"'},
        )


def _days_left(end_date_str: str) -> float:
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


def _is_good_date(end_date_str: str) -> bool:
    dl = _days_left(end_date_str)
    return MIN_DAYS <= dl <= MAX_DAYS


def _parse_market(raw: dict) -> Optional[dict]:
    try:
        mid = raw.get("id") or raw.get("conditionId") or raw.get("condition_id")
        if not mid:
            return None
        question = raw.get("question", "")
        if not question:
            return None

        # Parse outcome prices
        yes_price = 0.5
        outcomes_raw = raw.get("outcomePrices") or raw.get("outcome_prices")
        if outcomes_raw:
            if isinstance(outcomes_raw, str):
                try:
                    prices = json.loads(outcomes_raw)
                except Exception:
                    prices = []
            else:
                prices = outcomes_raw
            if prices and len(prices) >= 1:
                yes_price = float(prices[0])

        end_date = raw.get("endDate") or raw.get("end_date") or ""
        category = (raw.get("groupItemTitle") or raw.get("category") or "").lower()
        slug = raw.get("slug", "")

        # Parse clob token IDs
        clob_raw = raw.get("clobTokenIds") or raw.get("clob_token_ids")
        clob_ids = []
        if clob_raw:
            if isinstance(clob_raw, str):
                try:
                    clob_ids = json.loads(clob_raw)
                except Exception:
                    clob_ids = []
            else:
                clob_ids = list(clob_raw)

        return {
            "id": str(mid),
            "question": question,
            "slug": slug,
            "category": category,
            "yes_price": yes_price,
            "no_price": round(1 - yes_price, 4),
            "volume": float(raw.get("volume", 0) or 0),
            "volume24hr": float(raw.get("volume24hr", 0) or 0),
            "liquidity": float(raw.get("liquidity", 0) or 0),
            "active": raw.get("active", True),
            "closed": raw.get("closed", False),
            "end_date": end_date,
            "condition_id": raw.get("conditionId", mid),
            "clob_token_ids": clob_ids,
            "last_updated": datetime.utcnow().isoformat(),
        }
    except Exception:
        return None


async def close_stuck_trades():
    """Close trades older than 720h or with extreme entry prices from previous deploys."""
    open_trades = await db.get_open_paper_trades()
    now = datetime.utcnow()
    for trade in open_trades:
        try:
            created = datetime.fromisoformat(trade["created_at"])
            age_hours = (now - created).total_seconds() / 3600

            # Close ancient trades
            if age_hours > 720:
                entry = trade.get("entry_price", 0)
                await db.close_paper_trade(trade["id"], entry, 0.0, "TIMEOUT")
                await db.update_portfolio(
                    cash_delta=trade["cost"], pnl_delta=0,
                    invested_delta=-trade["cost"], win=False,
                )
                print(f"[STARTUP] Closed stuck trade #{trade['id']} (age={age_hours:.0f}h)")

            # Close trades with extreme entry prices that slipped through
            elif trade.get("entry_price", 0) < 0.02:
                await db.close_paper_trade(trade["id"], 0.01, -trade["cost"], "STOP_LOSS")
                await db.update_portfolio(
                    cash_delta=0, pnl_delta=-trade["cost"],
                    invested_delta=-trade["cost"], win=False,
                )
                print(f"[STARTUP] Closed bad-price trade #{trade['id']}")
        except Exception as e:
            print(f"[STARTUP] Error closing trade: {e}")


async def seed_weights():
    """Seed default signal weights if not already set."""
    weights = await db.get_signal_weights()
    defaults = {
        "near_certainty": 1.5,
        "volume_spike": 1.5,
        "binance_arb": 1.2,
        "short_duration": 1.3,
        "price_zone": 1.0,
        "liquidity": 1.0,
        "momentum": 1.0,
        "category": 1.0,
        "news_impact": 1.5,
        "smart_wallet": 1.5,
        "end_date": 1.2,
        "buy_no_early": 2.0,
    }
    for factor, default_w in defaults.items():
        if factor not in weights:
            await db.update_signal_weight(factor, default_w)


async def _ensure_self_learning_tables():
    """Create tables needed by self-improvement engine if missing."""
    import aiosqlite
    async with aiosqlite.connect(db.DB_PATH) as conn:
        await conn.executescript("""
            CREATE TABLE IF NOT EXISTS signal_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id INTEGER,
                market_type TEXT,
                direction TEXT,
                entry_price REAL,
                exit_price REAL,
                pnl REAL,
                won INTEGER,
                signal_factors_json TEXT DEFAULT '{}',
                created_at TEXT
            );
            CREATE TABLE IF NOT EXISTS strategy_params (
                param_name TEXT PRIMARY KEY,
                param_value TEXT,
                updated_at TEXT
            );
            CREATE TABLE IF NOT EXISTS improvement_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                overall_win_rate REAL,
                gap_to_target REAL,
                stats_json TEXT,
                changes_json TEXT,
                created_at TEXT
            );
        """)
        await conn.commit()


# ââ Market Fetching ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

async def fetch_markets() -> list:
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            tier1_task = client.get(f"{GAMMA_API}/markets", params={
                "active": "true", "closed": "false",
                "limit": FETCH_LIMIT, "order": "volume24hr", "ascending": "false",
            })
            tier2_task = client.get(f"{GAMMA_API}/markets", params={
                "active": "true", "closed": "false",
                "limit": NEAR_RES_LIMIT, "order": "endDate", "ascending": "true",
            })
            r1, r2 = await asyncio.gather(tier1_task, tier2_task, return_exceptions=True)

        seen_ids: set = set()
        near_res: list = []
        regular:  list = []

        if not isinstance(r2, Exception) and r2.status_code == 200:
            for m in r2.json():
                parsed = _parse_market(m)
                if parsed and parsed["id"] not in seen_ids:
                    dl = _days_left(parsed["end_date"])
                    if dl <= NEAR_RES_DAYS:
                        near_res.append(parsed)
                    elif dl <= MAX_DAYS:
                        regular.append(parsed)
                    seen_ids.add(parsed["id"])

        if not isinstance(r1, Exception) and r1.status_code == 200:
            for m in r1.json():
                parsed = _parse_market(m)
                if parsed and parsed["id"] not in seen_ids:
                    regular.append(parsed)
                    seen_ids.add(parsed["id"])

        near_res.sort(key=lambda m: _days_left(m["end_date"]))
        regular.sort(key=lambda m: m.get("volume24hr", 0), reverse=True)
        return near_res + regular

    except Exception as e:
        print(f"[FETCH] Error: {e}")
        return []


async def fetch_market_by_id(market_id: str) -> Optional[dict]:
    """Fetch a single market by ID."""
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(f"{GAMMA_API}/markets/{market_id}")
            if r.status_code == 200:
                parsed = _parse_market(r.json())
                return parsed
    except Exception:
        pass
    return None


async def backfill_open_trade_markets(markets_by_id: dict):
    """Fetch current prices for open trades whose markets weren't in main fetch."""
    open_trades = await db.get_open_paper_trades()
    missing_ids = []
    for trade in open_trades:
        if trade["market_id"] not in markets_by_id:
            missing_ids.append(trade["market_id"])

    for mid in missing_ids[:10]:
        market = await fetch_market_by_id(mid)
        if market:
            markets_by_id[mid] = market


# ââ LLM Analysis Cycle ââââââââââââââââââââââââââââââââââââââââââââââââââââââ

async def llm_analysis_cycle(markets: list) -> list:
    """
    General-purpose LLM screening of markets.
    Haiku screens all candidates; Sonnet deep-dives on high-edge opportunities.
    """
    if not HAS_LLM:
        return []

    signals = []

    # Select candidates: prefer short-duration, then near-resolution
    candidates = sorted(
        markets,
        key=lambda m: (_days_left(m.get("end_date", "")), -m.get("volume24hr", 0)),
    )[:15]  # Screen top 15 candidates

    portfolio = await db.get_portfolio()
    lessons = []
    try:
        lessons = await memory_system.get_relevant_lessons(limit=5)
    except Exception:
        pass

    for market in candidates:
        try:
            # Skip if we already have a position
            open_trades = await db.get_open_paper_trades()
            if market["id"] in {t["market_id"] for t in open_trades}:
                continue

            # Get volume profile for this market
            vol_profile = {}
            try:
                vol_profile = await volume_detector.get_market_volume_profile(market["id"])
            except Exception:
                pass

            # Get news context (empty string if none)
            news_context = ""

            # Run LLM analysis
            result = await analyze_market(
                market=market,
                news_context=news_context,
                volume_profile=vol_profile,
                memory_lessons=lessons,
                portfolio_state={
                    "cash_balance": portfolio.get("cash_balance", 100000),
                    "invested": portfolio.get("total_invested", 0),
                    "win_rate": (
                        (portfolio.get("win_count", 0) /
                         max(1, portfolio.get("win_count", 0) + portfolio.get("loss_count", 0)))
                        * 100
                    ),
                },
            )

            if not result or result.get("action") == "SKIP":
                continue

            action = result["action"]
            edge = abs(result.get("edge", 0))
            confidence = result.get("confidence", 0)

            # Thresholds for learning phase
            if edge < 0.05 or confidence < 0.4:
                continue

            direction = "YES" if action == "BUY_YES" else "NO"
            yes_price = market.get("yes_price", 0.5)
            entry_price = yes_price if direction == "YES" else (1 - yes_price)

            # Sanity: reject extreme prices
            if entry_price > 0.95 or entry_price < 0.05:
                print(f"[LLM] EXTREME PRICE {entry_price:.4f} â skip")
                continue

            score = int(60 + edge * 200 + confidence * 20)
            score = min(99, max(60, score))

            signal = {
                "market_id": market["id"],
                "market_question": market["question"],
                "score": score,
                "confidence": confidence,
                "direction": direction,
                "yes_price": yes_price,
                "market_type": "LLM_ANALYSIS",
                "can_enter": True,
                "entry_reason": (
                    f"LLM: {direction}@{entry_price:.2f}, "
                    f"edge={edge:.1%}, conf={confidence:.1%}, "
                    f"model={result.get('model', 'unknown')}"
                ),
                "factors_json": json.dumps({
                    "edge": round(edge, 4),
                    "confidence": round(confidence, 3),
                    "estimated_probability": result.get("estimated_probability", 0.5),
                    "reasoning": result.get("reasoning", ""),
                    "model": result.get("model", "unknown"),
                }),
                "created_at": datetime.utcnow().isoformat(),
                "clob_token_ids": market.get("clob_token_ids", []),
                "condition_id": market.get("condition_id", ""),
                "liquidity": market.get("liquidity", 0),
            }
            signals.append(signal)

            print(
                f"[LLM] Signal: {direction}@{entry_price:.2f} "
                f"edge={edge:.1%} conf={confidence:.1%} "
                f"'{market['question'][:45]}'"
            )

        except Exception as e:
            print(f"[LLM] Error analyzing market: {e}")
            continue

    if signals:
        print(f"[LLM] Generated {len(signals)} LLM signals")
    return signals


# ââ Trading Loop ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

async def trading_loop():
    global _loop_count
    print("[v3.1] PM Intelligence v3.1 â 4-Strategy Trading Agent")
    print("[v3.1] Strategy 1: Near-Certainty Grinder")
    print("[v3.1] Strategy 2: Volume Spike Trading")
    print("[v3.1] Strategy 3: Binance Price Lag Arbitrage")
    print("[v3.1] Strategy 4: Short-Duration 5m/15m Markets")
    print("[v3.1] + LLM Analysis Cycle (general screening)")

    while True:
        try:
            _loop_count += 1
            _strategy_debug["loops_run"] = _loop_count
            _strategy_debug["last_loop"] = datetime.utcnow().isoformat()

            markets = await fetch_markets()
            if not markets:
                await asyncio.sleep(LOOP_SLEEP)
                continue

            markets_by_id = {}
            for m in markets:
                await db.upsert_market(m)
                await db.save_market_snapshot(
                    m["id"], m["yes_price"], m["volume"],
                    m["volume24hr"], m["liquidity"]
                )
                markets_by_id[m["id"]] = m

            # Backfill resolved markets for open trades
            await backfill_open_trade_markets(markets_by_id)

            # Check exits FIRST (before entering new trades)
            await check_exits(markets_by_id)
            await check_leverage_exits(markets_by_id)

            # ââ Strategy 3: Binance Arb (EVERY loop â speed critical) ââââ
            arb_signals = generate_arb_signals(markets)
            _strategy_debug["arb_signals"] = len(arb_signals)

            # ââ Strategy 4: Short-Duration (every 3rd loop) ââââââââââââââ
            short_signals = []
            if _loop_count % SHORT_DUR_EVERY == 0:
                try:
                    short_signals = generate_short_duration_signals(markets)
                except Exception as e:
                    print(f"[SHORT] Error: {e}")
            _strategy_debug["short_duration_signals"] = len(short_signals)

            # ââ Strategy 2: Volume Spike (EVERY loop) ââââââââââââââââââââ
            try:
                spike_signals = await generate_spike_signals(markets)
            except Exception as e:
                spike_signals = []
                if _loop_count <= 3:
                    print(f"[SPIKE] Init phase: {e}")
            _strategy_debug["spike_signals"] = len(spike_signals)

            # ââ Strategy 1: Near-Certainty Grinder (every 20th loop) âââââ
            grinder_signals = []
            if _loop_count % GRINDER_EVERY == 0:
                try:
                    grinder_signals = await generate_near_certainty_signals(
                        markets, binance_prices
                    )
                except Exception as e:
                    print(f"[GRIND] Error: {e}")
            _strategy_debug["grinder_signals"] = len(grinder_signals)

            # ââ LLM Analysis Cycle (every 30th loop) âââââââââââââââââââââ
            llm_signals = []
            if _loop_count % LLM_EVERY == 0:
                try:
                    llm_signals = await llm_analysis_cycle(markets)
                except Exception as e:
                    print(f"[LLM] Cycle error: {e}")
            _strategy_debug["llm_signals"] = len(llm_signals)

            # ââ Enter trades from ALL strategies âââââââââââââââââââââââââ
            all_signals = (
                arb_signals + short_signals + spike_signals +
                grinder_signals + llm_signals
            )
            entered = 0
            for sig in all_signals:
                trade = await maybe_enter_trade(sig)
                if trade:
                    entered += 1
                    telegram_alerts.alert_trade_entry(trade)
                    # Track BINANCE_ARB entries to prevent double-entry
                    if sig.get("market_type") == "BINANCE_ARB":
                        try:
                            from binance_arb import _arb_entered_markets
                            _arb_entered_markets.add(sig["market_id"])
                        except ImportError:
                            pass

            _strategy_debug["total_entered"] += entered

            # ââ Logging (every 10 loops or when something happens) ââââââââ
            if _loop_count % 10 == 0 or entered > 0 or len(all_signals) > 0:
                btc_status = get_binance_status()
                btc_price = btc_status.get("BTC", {}).get("price", 0)
                portfolio = await db.get_portfolio()
                wins   = portfolio.get("win_count", 0) or 0
                losses = portfolio.get("loss_count", 0) or 0
                total  = wins + losses
                wr_pct = round(wins / total * 100, 1) if total else 0

                print(
                    f"[v3.1] #{_loop_count}: {len(markets)} mkts | "
                    f"ARB={len(arb_signals)} SHORT={len(short_signals)} "
                    f"SPIKE={len(spike_signals)} GRIND={len(grinder_signals)} "
                    f"LLM={len(llm_signals)} | "
                    f"entered={entered} | BTC=${btc_price:,.0f} | "
                    f"${portfolio.get('cash_balance',0):,.0f} "
                    f"{wins}W/{losses}L WR={wr_pct}%"
                )

            # ââ Track trade closures via Telegram âââââââââââââââââââââââââ
            open_now = await db.get_open_paper_trades()
            open_ids_now = {t["id"] for t in open_now}
            if hasattr(trading_loop, '_prev_open_ids'):
                closed_ids = trading_loop._prev_open_ids - open_ids_now
                if closed_ids:
                    _strategy_debug["trades_closed_this_session"] += len(closed_ids)
                    all_trades = await db.get_all_paper_trades(200)
                    for t in all_trades:
                        if t["id"] in closed_ids:
                            telegram_alerts.alert_trade_exit(t)
            trading_loop._prev_open_ids = open_ids_now

            # ââ Health summary every ~30 min ââââââââââââââââââââââââââââââ
            if _loop_count % 600 == 0:
                try:
                    h_portfolio = await db.get_portfolio()
                    h_trades = await db.get_all_paper_trades(200)
                    telegram_alerts.alert_health_summary(
                        h_portfolio, h_trades, get_binance_status(), _loop_count
                    )
                except Exception:
                    pass

            # ââ Broadcast to dashboard (every 5 loops) ââââââââââââââââââââ
            if _loop_count % 5 == 0:
                portfolio      = await db.get_portfolio()
                trades         = await db.get_all_paper_trades(50)
                recent_sigs    = await db.get_recent_signals(30)
                weights        = await db.get_signal_weights()
                explanations   = await db.get_trade_explanations(30)
                lev_portfolio  = await db.get_leverage_portfolio()
                lev_trades     = await db.get_all_leverage_trades(30)
                costs          = get_cost_summary()

                memory_stats = {}
                try:
                    memory_stats = await memory_system.get_memory_summary()
                except Exception:
                    pass

                # Self-improvement performance summary
                self_improve_stats = {}
                try:
                    import self_improvement_engine as sie
                    self_improve_stats = await sie.get_performance_summary()
                except Exception:
                    pass

                await broadcast({
                    "type":               "update",
                    "portfolio":          portfolio,
                    "trades":             trades,
                    "signals":            recent_sigs,
                    "weights":            weights,
                    "markets":            markets[:200],
                    "total_markets":      len(markets),
                    "trade_explanations": explanations,
                    "leverage_portfolio": lev_portfolio,
                    "leverage_trades":    lev_trades,
                    "llm_costs":          costs,
                    "memory_stats":       memory_stats,
                    "self_improve":       self_improve_stats,
                    "binance_status":     get_binance_status(),
                    "strategy_debug":     _strategy_debug,
                    "timestamp":          datetime.utcnow().isoformat(),
                })

        except Exception as e:
            print(f"[v3.1] Loop error: {e}")
            telegram_alerts.alert_error("trading_loop", str(e))

        await asyncio.sleep(LOOP_SLEEP)


# ââ WebSocket âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

async def broadcast(payload: dict):
    dead = set()
    msg = json.dumps(payload)
    for ws in active_connections:
        try:
            await ws.send_text(msg)
        except Exception:
            dead.add(ws)
    active_connections.difference_update(dead)


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        active_connections.discard(websocket)


# ââ API Endpoints âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

@app.get("/api/portfolio")
async def api_portfolio():
    return await db.get_portfolio()


@app.get("/api/trades")
async def api_trades():
    return await db.get_all_paper_trades(200)


@app.get("/api/signals")
async def api_signals():
    return await db.get_recent_signals(50)


@app.get("/api/insights")
async def api_insights():
    explanations = await db.get_trade_explanations(50)
    return {"explanations": explanations}


@app.get("/api/weights")
async def api_weights():
    return await db.get_signal_weights()


@app.get("/api/stats")
async def api_stats():
    portfolio = await db.get_portfolio()
    trades = await db.get_all_paper_trades(200)
    costs = get_cost_summary()
    memory = {}
    try:
        memory = await memory_system.get_memory_summary()
    except Exception:
        pass
    self_improve = {}
    try:
        import self_improvement_engine as sie
        self_improve = await sie.get_performance_summary()
    except Exception:
        pass

    return {
        "portfolio": portfolio,
        "total_trades": len(trades),
        "open_trades": len([t for t in trades if t.get("status") == "OPEN"]),
        "closed_trades": len([t for t in trades if t.get("status") != "OPEN"]),
        "llm_costs": costs,
        "memory": memory,
        "self_improvement": self_improve,
        "strategy_debug": _strategy_debug,
    }


@app.get("/api/strategy-performance")
async def api_strategy_performance():
    """Performance breakdown by strategy type."""
    try:
        import self_improvement_engine as sie
        return await sie.get_performance_summary()
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/llm/costs")
async def api_llm_costs():
    return get_cost_summary()


@app.get("/api/llm/test")
async def api_llm_test():
    try:
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
        model = os.getenv("LLM_SCREEN_MODEL", "claude-haiku-4-5-20251001")
        response = await client.messages.create(
            model=model,
            max_tokens=20,
            messages=[{"role": "user", "content": "Say hello in 5 words"}],
        )
        return {
            "success": True,
            "response": response.content[0].text,
            "model": model,
            "sdk_version": anthropic.__version__,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/llm/debug")
async def api_llm_debug():
    costs = get_cost_summary()
    has_key = bool(os.getenv("ANTHROPIC_API_KEY", ""))
    key_preview = ""
    if has_key:
        key = os.getenv("ANTHROPIC_API_KEY", "")
        key_preview = f"{key[:8]}...{key[-4:]}" if len(key) > 12 else "***"
    return {
        "loop_count": _loop_count,
        "has_anthropic": HAS_LLM,
        "has_api_key": has_key,
        "api_key_preview": key_preview,
        "costs": costs,
        "strategy_debug": _strategy_debug,
        "binance": get_binance_status(),
        "last_error": None,
        "learning_errors": _learning_errors,
    }


@app.get("/api/llm/memory")
async def api_llm_memory():
    try:
        return await memory_system.get_memory_summary()
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/llm/categories")
async def api_llm_categories():
    try:
        return await memory_system.get_category_performance()
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/trades/explained")
async def api_trades_explained():
    """Return trades enriched with entry reasons and signal factors."""
    trades = await db.get_all_paper_trades(200)
    signals = await db.get_recent_signals(200)
    signal_map = {s["id"]: s for s in signals}

    explained = []
    for trade in trades:
        signal = signal_map.get(trade.get("signal_id"))
        explained.append({
            **trade,
            "entry_reason": signal.get("entry_reason", "Unknown") if signal else "Unknown",
            "factors_json": signal.get("factors_json") if signal else None,
            "score": signal.get("score") if signal else None,
        })
    return explained


@app.get("/api/leverage/portfolio")
async def api_leverage_portfolio():
    return await db.get_leverage_portfolio()


@app.get("/api/leverage/trades")
async def api_leverage_trades():
    return await db.get_all_leverage_trades(50)


@app.post("/api/leverage/multiplier/{multiplier}")
async def api_set_multiplier(multiplier: int):
    return {"status": "disabled", "msg": "Leverage trading is disabled in v3.1"}


@app.get("/api/live/status")
async def api_live_status():
    return {"live_mode": False, "msg": "Paper trading only"}


@app.get("/api/live/portfolio")
async def api_live_portfolio():
    return {"cash_balance": 0, "total_invested": 0, "total_pnl": 0}


@app.get("/api/live/trades")
async def api_live_trades():
    return []


@app.post("/api/live/set-balance/{balance}")
async def api_set_live_balance(balance: float):
    return {"status": "disabled"}


@app.get("/api/binance")
async def api_binance():
    return get_binance_status()


@app.get("/api/paths")
async def api_paths():
    """Diagnostic: show resolved paths."""
    return {
        "base_dir": str(BASE_DIR),
        "frontend_dir": str(FRONTEND_DIR),
        "index_html": str(INDEX_HTML),
        "index_exists": INDEX_HTML.exists(),
        "cwd": os.getcwd(),
        "file": __file__,
    }


# ââ Frontend Serving âââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

@app.get("/")
async def serve_index(request: Request):
    _auth_required(request)
    if INDEX_HTML.exists():
        return FileResponse(INDEX_HTML, media_type="text/html")
    return JSONResponse({"error": "Frontend not found", "path": str(INDEX_HTML)})


@app.get("/{path:path}")
async def serve_static(path: str, request: Request):
    _auth_required(request)
    file_path = FRONTEND_DIR / path
    if file_path.exists() and file_path.is_file():
        return FileResponse(file_path)
    if INDEX_HTML.exists():
        return FileResponse(INDEX_HTML, media_type="text/html")
    raise HTTPException(status_code=404)
