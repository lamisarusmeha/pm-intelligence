"""
PM Intelligence — FastAPI backend. TURBO LEARNING MODE.

Loop: every 3s | 1000 markets (incl. near-resolution priority batch) | news every 9s | wallets every 6s
Four modes: COPY_TRADE > BUY_NO_EARLY > LOCK-IN > MOMENTUM
Near-resolution: second market fetch targets markets ending within 7 days → pushed to front of queue
$100k balance | 75 max positions | entry threshold 20 | 25% learn rate
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
from signal_engine import generate_signals
from paper_trader import maybe_enter_trade, check_exits, maybe_enter_leverage_trade, check_leverage_exits
from live_trader import maybe_enter_live_trade, check_live_exits, get_live_status
import news_engine
import wallet_tracker

# Load .env credentials if present
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

# ── Dashboard password protection ─────────────────────────────────────────────
# Set DASHBOARD_PASSWORD env var on Railway to password-protect the dashboard.
# If not set, no auth required (local development).
DASHBOARD_PASSWORD = os.getenv("DASHBOARD_PASSWORD", "")

def _check_auth(request: Request) -> bool:
    """Returns True if request is authenticated (or no password is set)."""
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
    """Return a 401 response that triggers the browser's basic auth prompt."""
    return Response(
        content="Authentication required",
        status_code=401,
        headers={"WWW-Authenticate": 'Basic realm="PM Intelligence"'},
    )

BASE_DIR     = Path(__file__).parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
INDEX_HTML   = FRONTEND_DIR / "index.html"

# ── SPEED CONFIG ──────────────────────────────────────────────────────────────
GAMMA_API    = "https://gamma-api.polymarket.com"
FETCH_LIMIT  = 1000   # massive market pool for maximum signal coverage
LOOP_SLEEP   = 3      # ↓ 5s→3s: fastest safe polling without rate-limit risk
NEWS_EVERY   = 3      # refresh news every ~9s (3 loops × 3s)
WALLET_EVERY = 2      # refresh wallets every ~6s (2 loops × 3s)

# Date filter — focus on the 0–30 day sweet spot (highest win rate per end_date scoring)
MIN_DAYS = 0    # include today (lock-in plays can resolve today)
MAX_DAYS = 30   # ↓ was 90: ignore slow-moving far-out markets that don't drift

# Near-resolution batch: pull additional markets ending ≤ 7 days
NEAR_RES_DAYS   = 7
NEAR_RES_LIMIT  = 300   # extra markets fetched from near-resolution batch

active_connections: Set[WebSocket] = set()
_loop_count = 0


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
            if age_hours > 8 or (t.get("entry_price") or 1) < 0.02:
                # Use entry_price as best proxy at startup (no market data yet).
                # This is conservative — avoids recording false wins from stale data.
                # Real exits use live prices via check_exits() during the main loop.
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
        defaults = {"news_impact": 1.5, "smart_wallet": 1.5, "end_date": 1.2}
        for factor, default in defaults.items():
            if factor not in weights:
                await db.update_signal_weight(factor, default)
    except Exception as e:
        print(f"[STARTUP] Weight seed error: {e}")


def _days_left(end_date_str: str) -> float:
    """Return number of days until market end. Returns 9999 if unknown."""
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
    """Parse a raw Gamma API market dict into our internal format. Returns None on error."""
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

        # Parse CLOB token IDs for live trading (YES token [0], NO token [1])
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
            "clob_token_ids": clob_token_ids,  # [YES_token_id, NO_token_id]
        }
        if market["id"] and market["question"]:
            return market
        return None
    except Exception:
        return None


async def fetch_markets() -> list:
    """
    Dual-fetch strategy for maximum win rate:
      Tier 1 — Top FETCH_LIMIT markets by 24h volume (mainstream activity)
      Tier 2 — Top NEAR_RES_LIMIT markets ending within NEAR_RES_DAYS (near-resolution priority)

    Near-resolution markets are sorted to the FRONT of the combined list so they get
    processed first in each scoring cycle (highest expected win rate zone).
    """
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            # Tier 1: high-volume markets
            tier1_task = client.get(f"{GAMMA_API}/markets", params={
                "active":    "true",
                "closed":    "false",
                "limit":     FETCH_LIMIT,
                "order":     "volume24hr",
                "ascending": "false",
            })
            # Tier 2: markets ending soonest (near-resolution priority)
            tier2_task = client.get(f"{GAMMA_API}/markets", params={
                "active":    "true",
                "closed":    "false",
                "limit":     NEAR_RES_LIMIT,
                "order":     "endDate",
                "ascending": "true",    # soonest first
            })
            r1, r2 = await asyncio.gather(tier1_task, tier2_task, return_exceptions=True)

        seen_ids: set = set()
        near_res: list = []
        regular:  list = []

        # Process Tier 2 first — near-resolution markets go to front
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

        # Process Tier 1 — high-volume markets, skip already seen
        if not isinstance(r1, Exception) and r1.status_code == 200:
            for m in r1.json():
                parsed = _parse_market(m)
                if parsed and parsed["id"] not in seen_ids:
                    regular.append(parsed)
                    seen_ids.add(parsed["id"])

        # Near-res first (sorted by days_left ASC), then regular by volume DESC
        near_res.sort(key=lambda m: _days_left(m["end_date"]))
        regular.sort(key=lambda m: m.get("volume24hr", 0), reverse=True)

        combined = near_res + regular
        print(f"[FETCH] {len(near_res)} near-resolution (<={NEAR_RES_DAYS}d) + {len(regular)} regular = {len(combined)} total")
        return combined

    except Exception as e:
        print(f"[FETCH] Error: {e}")
        return []


async def trading_loop():
    """
    10s cycle. LOCK-IN plays sorted first (highest expected win rate).
    Tries top 25 signals per cycle — maximises trades → faster learning.
    """
    global _loop_count
    print("[TURBO] 🚀 Started — LOCK-IN + MOMENTUM dual mode | 1000 markets | 20 open max")

    while True:
        try:
            _loop_count += 1

            if _loop_count % NEWS_EVERY == 1:
                try:
                    await news_engine.refresh_news()
                except Exception as e:
                    print(f"[NEWS] {e}")

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

            if _loop_count % WALLET_EVERY == 1:
                try:
                    await wallet_tracker.refresh_smart_wallets(markets)
                except Exception as e:
                    print(f"[WALLET] {e}")

            await check_exits(markets_by_id)
            await check_leverage_exits(markets_by_id)
            await check_live_exits(markets_by_id)   # live money exit scan

            signals = await generate_signals(markets)

            # Pass CLOB token IDs into each signal (needed for live order placement)
            for sig in signals:
                m = markets_by_id.get(sig.get("market_id", ""), {})
                sig["clob_token_ids"] = m.get("clob_token_ids", [])
                sig["condition_id"]   = m.get("condition_id", "")
                sig["liquidity"]      = m.get("liquidity", 0)

            lock_in_sigs  = [s for s in signals if s.get("market_type") == "LOCK_IN"]
            bne_sigs      = [s for s in signals if s.get("market_type") == "BUY_NO_EARLY"]
            momentum_sigs = [s for s in signals if s.get("market_type") == "MOMENTUM"]
            ct_sigs       = [s for s in signals if s.get("market_type") == "COPY_TRADE"]
            enterable     = [s for s in signals if s.get("can_enter", False)]

            print(f"[TURBO] #{_loop_count}: {len(markets)} mkts | "
                  f"{len(lock_in_sigs)} lock-in | {len(bne_sigs)} bne | "
                  f"{len(ct_sigs)} copy | {len(enterable)} enterable")

            # Save top 30 signals to feed
            for sig in signals[:30]:
                await db.save_signal(sig)

            # Paper + live trading loop — LOCK_IN first, then BUY_NO_EARLY, COPY_TRADE
            entered = 0
            live_entered = 0
            lev_entered = 0
            for sig in signals[:50]:
                trade = await maybe_enter_trade(sig)
                if trade:
                    entered += 1
                # Live trade (runs only if LIVE_MODE=true + all gates pass)
                live_trade = await maybe_enter_live_trade(sig)
                if live_trade:
                    live_entered += 1
                # Leverage: only high-confidence signals (score ≥ 70)
                lev_trade = await maybe_enter_leverage_trade(sig)
                if lev_trade:
                    lev_entered += 1
            if entered:
                print(f"[PAPER] ✅ {entered} trade(s) entered")
            if live_entered:
                print(f"[LIVE] 🟢 {live_entered} REAL trade(s) entered")
            if lev_entered:
                print(f"[LEV] ⚡ {lev_entered} leverage trade(s) entered")

            portfolio      = await db.get_portfolio()
            trades         = await db.get_all_paper_trades(50)
            recent_sigs    = await db.get_recent_signals(30)
            weights        = await db.get_signal_weights()
            explanations   = await db.get_trade_explanations(30)
            lev_portfolio  = await db.get_leverage_portfolio()
            lev_trades     = await db.get_all_leverage_trades(30)

            wins   = portfolio.get("win_count", 0) or 0
            losses = portfolio.get("loss_count", 0) or 0
            total  = wins + losses
            wr_pct = round(wins / total * 100, 1) if total else 0
            print(f"[TURBO] 💰 ${portfolio.get('cash_balance',0):,.0f} | "
                  f"{wins}W/{losses}L | WR={wr_pct}% (target: 80%)")

            await broadcast({
                "type":               "update",
                "portfolio":          portfolio,
                "trades":             trades,
                "signals":            recent_sigs,
                "weights":            weights,
                "markets":            markets[:40],
                "trade_explanations": explanations,
                "lock_in_count":      len(lock_in_sigs),
                "momentum_count":     len(momentum_sigs),
                "leverage_portfolio": lev_portfolio,
                "leverage_trades":    lev_trades,
                "timestamp":          datetime.utcnow().isoformat(),
            })

        except Exception as e:
            print(f"[TURBO] Loop error: {e}")

        await asyncio.sleep(LOOP_SLEEP)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.init_db()
    await close_stuck_trades()
    await seed_weights()
    print("[STARTUP] Pre-loading news...")
    try:
        await news_engine.refresh_news()
    except Exception as e:
        print(f"[STARTUP] News: {e}")
    loop_task = asyncio.create_task(trading_loop())
    yield
    loop_task.cancel()
    try:
        await loop_task
    except asyncio.CancelledError:
        pass


app = FastAPI(title="PM Intelligence — 80% Target", lifespan=lifespan)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    active_connections.add(ws)
    try:
        await ws.send_json({
            "type":               "init",
            "portfolio":          await db.get_portfolio(),
            "trades":             await db.get_all_paper_trades(50),
            "signals":            await db.get_recent_signals(30),
            "weights":            await db.get_signal_weights(),
            "markets":            await db.get_all_markets(40),
            "trade_explanations": await db.get_trade_explanations(30),
            "leverage_portfolio": await db.get_leverage_portfolio(),
            "leverage_trades":    await db.get_all_leverage_trades(30),
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
    return await get_live_status()

@app.get("/api/live/portfolio")
async def api_live_portfolio():
    return await db.get_live_portfolio()

@app.get("/api/live/trades")
async def api_live_trades(limit: int = 50):
    return await db.get_all_live_trades(limit)

@app.post("/api/live/set-balance/{balance}")
async def api_set_live_balance(balance: float):
    """Call this once after funding your Polymarket account to set starting balance."""
    if balance < 0:
        return JSONResponse({"error": "balance must be positive"}, status_code=400)
    await db.set_live_balance(balance)
    return {"ok": True, "live_balance": balance}

@app.get("/")
async def serve_index(request: Request):
    if not _check_auth(request):
        return _auth_required()
    if INDEX_HTML.exists():
        return FileResponse(INDEX_HTML, headers={
            "Cache-Control": "no-store, no-cache, must-revalidate"
        })
    return JSONResponse({"error": "Frontend not found"}, status_code=404)

@app.get("/{path:path}")
async def serve_static(path: str, request: Request):
    file_path = FRONTEND_DIR / path
    if file_path.exists() and file_path.is_file():
        return FileResponse(file_path)
    return await serve_index(request)
