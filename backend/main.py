"""
PM Intelligence v3 â 3-Strategy Trading Agent

Architecture:
  3-second loop:
    - Strategy 3: Binance Arb (EVERY loop â speed critical)
    - Strategy 2: Volume Spike (EVERY loop)
    - Strategy 1: Near-Certainty Grinder (every 20th loop â less time-sensitive)

  Background:
    - Binance WebSocket feed (real-time BTC, ETH, SOL prices)
    - Volume detector (accumulates snapshots)

Cost control:
  - Haiku only for verification (~$0.001/call)
  - No Sonnet needed
  - ~$2-5/day target spend
"""

import asyncio
import base64
import json
import os
import secrets
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, Set

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response

import database as db
from paper_trader import maybe_enter_trade, check_exits, check_leverage_exits

# 3 Strategy modules
from near_certainty_grinder import generate_near_certainty_signals
from volume_spike_trader import generate_spike_signals
from binance_arb import generate_arb_signals
from binance_feed import binance_websocket_loop, binance_prices, get_status as get_binance_status

# Telegram alerts
import telegram_alerts

# Volume detector (used by Strategy 2)
from volume_detector import _ensure_tables as init_volume_tables

# Memory system (kept for learning)
from memory_system import init_memory, get_memory_summary

# LLM cost tracking (kept for dashboard)
try:
    from llm_agent import get_cost_summary, HAS_ANTHROPIC, ANTHROPIC_API_KEY
except ImportError:
    HAS_ANTHROPIC = False
    ANTHROPIC_API_KEY = ""
    def get_cost_summary():
        return {"haiku_calls": 0, "sonnet_calls": 0, "total_cost_usd": 0}

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

# ââ Dashboard password protection ââââââââââââââââââââââââââââââââââââââââââââ
DASHBOARD_PASSWORD = os.getenv("DASHBOARD_PASSWORD", "")

def _check_auth(request: Request) -> bool:
    if not DASHBOARD_PASSWORD:
        return True
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Basic "):
        return False
    try:
        decoded = base64.b64decode(auth[6:]).decode("utf-8")
        _, password = decoded.split(":", 1)
        return secrets.compare_digest(password, DASHBOARD_PASSWORD)
    except Exception:
        return False

def _auth_required():
    return Response(
        content="Authentication required",
        status_code=401,
        headers={"WWW-Authenticate": 'Basic realm="PM Intelligence"'},
    )

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

# ââ CONFIG âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
GAMMA_API    = "https://gamma-api.polymarket.com"
FETCH_LIMIT  = 1000
LOOP_SLEEP   = 3
NEAR_RES_LIMIT = 300
MIN_DAYS = 0
MAX_DAYS = 30
NEAR_RES_DAYS = 7

# Strategy 1 runs every N loops (less time-sensitive than arb)
GRINDER_EVERY = 20  # Every ~60 seconds

active_connections: Set[WebSocket] = set()
_loop_count = 0
_strategy_debug = {
    "last_loop": None,
    "arb_signals": 0,
    "spike_signals": 0,
    "grinder_signals": 0,
    "total_entered": 0,
    "loops_run": 0,
}


async def broadcast(payload: dict):
    dead = set()
    for ws in list(active_connections):
        try:
            await ws.send_json(payload)
        except Exception:
            dead.add(ws)
    active_connections.difference_update(dead)


async def close_stuck_trades():
    open_trades = await db.get_open_paper_trades()
    if not open_trades:
        return
    now = datetime.utcnow()
    closed = 0
    for t in open_trades:
        try:
            created   = datetime.fromisoformat(t["created_at"])
            age_hours = (now - created).total_seconds() / 3600
            if age_hours > 720 or (t.get("entry_price") or 1) < 0.02:
                exit_price = max(t.get("entry_price") or 0.01, 0.01)
                payout     = t["shares"] * exit_price
                pnl        = round(payout - t["cost"], 2)
                outcome    = "TIMEOUT" if pnl >= 0 else "STOP_LOSS"
                await db.close_paper_trade(t["id"], exit_price, pnl, outcome)
                won = pnl > 0
                await db.update_portfolio(cash_delta=payout, pnl_delta=pnl,
                                          invested_delta=-t["cost"], win=won)
                closed += 1
        except Exception as e:
            print(f"[STARTUP] Stuck trade error {t.get('id')}: {e}")
    if closed:
        print(f"[STARTUP] Closed {closed} stuck trades.")


async def seed_weights():
    try:
        weights = await db.get_signal_weights()
        defaults = {"near_certainty": 1.5, "volume_spike": 1.5, "binance_arb": 1.2}
        for factor, default in defaults.items():
            if factor not in weights:
                await db.update_signal_weight(factor, default)
    except Exception as e:
        print(f"[STARTUP] Weight seed error: {e}")


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
    if not end_date_str:
        return True
    try:
        end_date_str = end_date_str.replace("Z", "+00:00")
        if "T" in end_date_str:
            end_dt = datetime.fromisoformat(end_date_str).replace(tzinfo=None)
        else:
            end_dt = datetime.strptime(end_date_str[:10], "%Y-%m-%d")
        days_left = (end_dt - datetime.utcnow()).days
        return MIN_DAYS <= days_left <= MAX_DAYS
    except Exception:
        return True


def _parse_market(m: dict) -> Optional[dict]:
    try:
        outcome_prices = m.get("outcomePrices", [0.5])
        if isinstance(outcome_prices, str):
            outcome_prices = json.loads(outcome_prices)
        yes_price = float(outcome_prices[0])

        end_date = m.get("endDate", "")
        if not _is_good_date(end_date):
            return None

        category = "other"
        events = m.get("events") or []
        if events and isinstance(events, list) and events[0].get("category"):
            category = events[0]["category"]
        elif m.get("category"):
            category = m["category"]

        clob_token_ids = m.get("clobTokenIds", [])
        if isinstance(clob_token_ids, str):
            try:
                clob_token_ids = json.loads(clob_token_ids)
            except Exception:
                clob_token_ids = []

        market = {
            "id":             str(m.get("id", "")),
            "question":       m.get("question", ""),
            "slug":           m.get("slug", ""),
            "category":       category,
            "yes_price":      round(yes_price, 4),
            "no_price":       round(1 - yes_price, 4),
            "volume":         float(m.get("volume", 0) or 0),
            "volume24hr":     float(m.get("volume24hr", 0) or 0),
            "liquidity":      float(m.get("liquidity", 0) or 0),
            "active":         1 if m.get("active", True) else 0,
            "closed":         1 if m.get("closed", False) else 0,
            "end_date":       end_date,
            "last_updated":   datetime.utcnow().isoformat(),
            "condition_id":   str(m.get("conditionId", "")),
            "clob_token_ids": clob_token_ids,
        }
        if market["id"] and market["question"]:
            return market
        return None
    except Exception:
        return None


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

        combined = near_res + regular
        return combined

    except Exception as e:
        print(f"[FETCH] Error: {e}")
        return []


# ââ Main Trading Loop â 3 Strategies âââââââââââââââââââââââââââââââââââââââââ

async def fetch_market_by_id(market_id: str) -> Optional[dict]:
    """Fetch a single market by ID — used to check resolved markets for open trades."""
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(f"{GAMMA_API}/markets/{market_id}")
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list) and data:
                    data = data[0]
                if isinstance(data, dict):
                    parsed = _parse_market(data)
                    if parsed:
                        return parsed
                    # Market might be closed — parse manually
                    outcome_prices = data.get("outcomePrices", [0.5])
                    if isinstance(outcome_prices, str):
                        import json as _j
                        outcome_prices = _j.loads(outcome_prices)
                    yes_price = float(outcome_prices[0]) if outcome_prices else 0.5
                    return {
                        "id": str(data.get("id", market_id)),
                        "question": data.get("question", ""),
                        "yes_price": round(yes_price, 4),
                        "no_price": round(1 - yes_price, 4),
                        "closed": 1 if data.get("closed", False) else 0,
                        "active": 1 if data.get("active", True) else 0,
                        "end_date": data.get("endDate", ""),
                        "volume": float(data.get("volume", 0) or 0),
                        "liquidity": float(data.get("liquidity", 0) or 0),
                    }
    except Exception as e:
        print(f"[FETCH] Market {market_id} lookup failed: {e}")
    return None


async def backfill_open_trade_markets(markets_by_id: dict):
    """Fetch current prices for open trades whose markets aren't in the main fetch."""
    open_trades = await db.get_open_paper_trades()
    missing_ids = [t["market_id"] for t in open_trades if t["market_id"] not in markets_by_id]
    if not missing_ids:
        return
    print(f"[v3] Fetching {len(missing_ids)} missing markets for open trades")
    for mid in missing_ids[:10]:  # Cap at 10 to avoid rate limits
        m = await fetch_market_by_id(mid)
        if m:
            markets_by_id[m["id"]] = m
            print(f"[v3] Backfilled market {mid}: YES={m['yes_price']:.3f} closed={m.get('closed', 0)}")


async def trading_loop():
    global _loop_count
    print("[v3] PM Intelligence v3 â 3-Strategy Trading Agent")
    print("[v3] Strategy 1: Near-Certainty Grinder")
    print("[v3] Strategy 2: Volume Spike Trading")
    print("[v3] Strategy 3: Binance Price Lag Arbitrage")

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

            # ââ Strategy 3: Binance Arb (EVERY loop â speed critical) âââââ
            arb_signals = generate_arb_signals(markets)  # Synchronous for speed
            _strategy_debug["arb_signals"] = len(arb_signals)

            # ââ Strategy 2: Volume Spike (EVERY loop) âââââââââââââââââââââ
            try:
                spike_signals = await generate_spike_signals(markets)
            except Exception as e:
                spike_signals = []
                if _loop_count <= 3:
                    print(f"[SPIKE] Init phase: {e}")
            _strategy_debug["spike_signals"] = len(spike_signals)

            # ââ Strategy 1: Near-Certainty Grinder (every 20th loop) ââââââ
            grinder_signals = []
            if _loop_count % GRINDER_EVERY == 0:
                try:
                    grinder_signals = await generate_near_certainty_signals(
                        markets, binance_prices
                    )
                except Exception as e:
                    print(f"[GRIND] Error: {e}")
            _strategy_debug["grinder_signals"] = len(grinder_signals)

            # ââ Enter trades from all strategies ââââââââââââââââââââââââââ
            all_signals = arb_signals + spike_signals + grinder_signals
            entered = 0
            for sig in all_signals:
                trade = await maybe_enter_trade(sig)
                if trade:
                    entered += 1
                    telegram_alerts.alert_trade_entry(trade)

            _strategy_debug["total_entered"] += entered

            # Log summary (every 10th loop to reduce spam)
            if _loop_count % 10 == 0 or entered > 0 or len(all_signals) > 0:
                binance_status = get_binance_status()
                btc_price = binance_status.get("BTC", {}).get("price", 0)
                portfolio = await db.get_portfolio()
                wins   = portfolio.get("win_count", 0) or 0
                losses = portfolio.get("loss_count", 0) or 0
                total  = wins + losses
                wr_pct = round(wins / total * 100, 1) if total else 0

                print(f"[v3] #{_loop_count}: {len(markets)} mkts | "
                      f"ARB={len(arb_signals)} SPIKE={len(spike_signals)} "
                      f"GRIND={len(grinder_signals)} | "
                      f"entered={entered} | BTC=${btc_price:,.0f} | "
                      f"${portfolio.get('cash_balance',0):,.0f} {wins}W/{losses}L WR={wr_pct}%")

            # -- Telegram: track exits + health summary --------
            # Check for closed trades (exit alerts)
            open_now = await db.get_open_paper_trades()
            open_ids_now = {t["id"] for t in open_now}
            if hasattr(trading_loop, '_prev_open_ids'):
                closed_ids = trading_loop._prev_open_ids - open_ids_now
                if closed_ids:
                    all_trades = await db.get_all_paper_trades(200)
                    for t in all_trades:
                        if t["id"] in closed_ids:
                            telegram_alerts.alert_trade_exit(t)
            trading_loop._prev_open_ids = open_ids_now

            # Health summary every ~30 min (600 loops * 3s)
            if _loop_count % 600 == 0:
                try:
                    h_portfolio = await db.get_portfolio()
                    h_trades = await db.get_all_paper_trades(200)
                    telegram_alerts.alert_health_summary(h_portfolio, h_trades, get_binance_status(), _loop_count)
                except Exception:
                    pass

            # ââ Broadcast to dashboard ââââââââââââââââââââââââââââââââââââ
            if _loop_count % 5 == 0:  # Broadcast every 15s (not every 3s)
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
                    memory_stats = await get_memory_summary()
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
                    "binance_status":     get_binance_status(),
                    "strategy_debug":     _strategy_debug,
                    "timestamp":          datetime.utcnow().isoformat(),
                })

        except Exception as e:
            print(f"[v3] Loop error: {e}")
            telegram_alerts.alert_error("trading_loop", str(e))

        await asyncio.sleep(LOOP_SLEEP)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.init_db()

    # Initialize modules
    try:
        await init_memory()
        await init_volume_tables()
        print("[v3] Memory + Volume detector initialized")
    except Exception as e:
        print(f"[v3] Init warning: {e}")

    await close_stuck_trades()
    await seed_weights()

    # Launch Binance WebSocket feed (background task)
    binance_task = asyncio.create_task(binance_websocket_loop())
    print("[v3] Binance WebSocket feed launched")

    # Check API key
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if api_key:
        print("[v3] Anthropic API key configured â Haiku verification active")
    else:
        print("[v3] No ANTHROPIC_API_KEY â running without Haiku verification")

    loop_task = asyncio.create_task(trading_loop())
    print("[v3] Trading loop started â 3 strategies active")

    # Telegram alerts
    if telegram_alerts.is_configured():
        telegram_alerts.alert_startup()
        print("[v3] Telegram alerts active")
    else:
        print("[v3] Telegram not configured (set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID)")
    yield
    loop_task.cancel()
    binance_task.cancel()
    try:
        await loop_task
    except asyncio.CancelledError:
        pass
    try:
        await binance_task
    except asyncio.CancelledError:
        pass


app = FastAPI(title="PM Intelligence v3 â 3-Strategy Agent", lifespan=lifespan)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    active_connections.add(ws)
    try:
        memory_stats = {}
        try:
            memory_stats = await get_memory_summary()
        except Exception:
            pass
        await ws.send_json({
            "type":               "init",
            "portfolio":          await db.get_portfolio(),
            "trades":             await db.get_all_paper_trades(50),
            "signals":            await db.get_recent_signals(30),
            "weights":            await db.get_signal_weights(),
            "markets":            await db.get_all_markets(200),
            "trade_explanations": await db.get_trade_explanations(30),
            "leverage_portfolio": await db.get_leverage_portfolio(),
            "leverage_trades":    await db.get_all_leverage_trades(30),
            "llm_costs":          get_cost_summary(),
            "memory_stats":       memory_stats,
            "binance_status":     get_binance_status(),
            "strategy_debug":     _strategy_debug,
            "timestamp":          datetime.utcnow().isoformat(),
        })
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        active_connections.discard(ws)


@app.get("/api/portfolio")
async def api_portfolio():
    return await db.get_portfolio()

@app.get("/api/trades")
async def api_trades(limit: int = 50):
    return await db.get_all_paper_trades(limit)

@app.get("/api/signals")
async def api_signals(limit: int = 30):
    return await db.get_recent_signals(limit)

@app.get("/api/insights")
async def api_insights(limit: int = 30):
    return await db.get_trade_explanations(limit)

@app.get("/api/weights")
async def api_weights():
    return await db.get_signal_weights()

@app.get("/api/news")
async def api_news():
    try:
        return await db.get_recent_news(30)
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/stats")
async def api_stats():
    try:
        p = await db.get_portfolio()
        wins   = p.get("win_count", 0) or 0
        losses = p.get("loss_count", 0) or 0
        total  = wins + losses
        return {
            "win_rate":      round(wins / total * 100, 1) if total else 0,
            "total_trades":  total,
            "wins":          wins,
            "losses":        losses,
            "total_pnl":     p.get("total_pnl", 0),
            "cash":          p.get("cash_balance", 0),
            "target_wr":     80,
            "gap":           max(0, round(80 - (wins / total * 100 if total else 0), 1)),
        }
    except Exception as e:
        return {"error": str(e)}

# v2/v3 endpoints
@app.get("/api/llm/costs")
async def api_llm_costs():
    return get_cost_summary()

@app.get("/api/llm/test")
async def api_llm_test():
    """Direct LLM test â calls Haiku with a simple prompt to verify API works."""
    try:
        import anthropic as _anth
        key = os.getenv("ANTHROPIC_API_KEY", "")
        client = _anth.AsyncAnthropic(api_key=key)
        response = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=50,
            messages=[{"role": "user", "content": "Say hello in exactly 5 words."}],
        )
        return {
            "success": True,
            "response": response.content[0].text,
            "model": response.model,
            "sdk_version": _anth.__version__,
        }
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()[-500:],
        }

@app.get("/api/llm/debug")
async def api_llm_debug():
    return {
        "loop_count": _loop_count,
        "strategy_debug": _strategy_debug,
        "binance_status": get_binance_status(),
        "has_anthropic": HAS_ANTHROPIC,
        "has_api_key": bool(os.getenv("ANTHROPIC_API_KEY", "")),
        "costs": get_cost_summary(),
    }

@app.get("/api/llm/memory")
async def api_llm_memory():
    try:
        return await get_memory_summary()
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/llm/categories")
async def api_llm_categories():
    try:
        from memory_system import get_category_performance
        return await get_category_performance()
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/leverage/portfolio")
async def api_leverage_portfolio():
    return await db.get_leverage_portfolio()

@app.get("/api/leverage/trades")
async def api_leverage_trades(limit: int = 50):
    return await db.get_all_leverage_trades(limit)

@app.post("/api/leverage/multiplier/{multiplier}")
async def api_set_leverage(multiplier: int):
    if multiplier not in (2, 3, 5):
        return JSONResponse({"error": "multiplier must be 2, 3, or 5"}, status_code=400)
    await db.set_leverage_multiplier(multiplier)
    return {"ok": True, "leverage_multiplier": multiplier}

@app.get("/api/live/status")
async def api_live_status():
    try:
        from live_trader import get_live_status
        return await get_live_status()
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/live/portfolio")
async def api_live_portfolio():
    return await db.get_live_portfolio()

@app.get("/api/live/trades")
async def api_live_trades(limit: int = 50):
    return await db.get_all_live_trades(limit)

@app.post("/api/live/set-balance/{balance}")
async def api_set_live_balance(balance: float):
    if balance < 0:
        return JSONResponse({"error": "balance must be positive"}, status_code=400)
    await db.set_live_balance(balance)
    return {"ok": True, "live_balance": balance}

@app.get("/api/binance")
async def api_binance():
    """Binance feed status â prices, age, change percentages."""
    return get_binance_status()

@app.get("/")
async def serve_index(request: Request):
    if not _check_auth(request):
        return _auth_required()
    if INDEX_HTML.exists():
        return FileResponse(INDEX_HTML, headers={
            "Cache-Control": "no-store, no-cache, must-revalidate"
        })
    return JSONResponse({"error": "Frontend not found", "base_dir": str(BASE_DIR), "frontend_dir": str(FRONTEND_DIR), "index_html": str(INDEX_HTML), "index_exists": INDEX_HTML.exists(), "frontend_exists": FRONTEND_DIR.exists(), "cwd": os.getcwd(), "file": str(Path(__file__).resolve())}, status_code=404)

@app.get("/{path:path}")
async def serve_static(path: str, request: Request):
    file_path = FRONTEND_DIR / path
    if file_path.exists() and file_path.is_file():
        return FileResponse(file_path)
    return await serve_index(request)
