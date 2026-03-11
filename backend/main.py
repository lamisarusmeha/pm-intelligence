"""
PM Intelligence v2 â Self-Learning LLM Agent.

Architecture:
  Existing loop (3s) â fetches markets, generates heuristic signals, paper trades
  NEW: Every 60s, LLM agent analyzes top 10 markets with volume spikes
  NEW: Memory system tracks reasoning chains and learns from outcomes
  NEW: Volume spike detector identifies insider-like activity
  NEW: Research agent gathers news context (free APIs, no LLM cost)

Cost control:
  - Haiku ($0.25/M) screens all markets
  - Sonnet ($3/M) only for >12% edge opportunities
  - ~$1-2/day target spend
  - Lesson extraction only on losses or big wins
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

# v2 modules â LLM brain
from llm_agent import analyze_market, evaluate_trade_outcome, get_cost_summary
from volume_detector import detect_spike, get_market_volume_profile, _ensure_tables as init_volume_tables
from memory_system import (
    init_memory, store_trade_reasoning, record_trade_outcome,
    get_relevant_lessons, get_category_performance, get_memory_summary
)
from research_agent import gather_market_context

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

# ââ Dashboard password protection âââââââââââââââââââââââââââââââââââââââââââââ
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

BASE_DIR     = Path(__file__).parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
INDEX_HTML   = FRONTEND_DIR / "index.html"

# ââ SPEED CONFIG ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
GAMMA_API    = "https://gamma-api.polymarket.com"
FETCH_LIMIT  = 1000
LOOP_SLEEP   = 3
NEWS_EVERY   = 3
WALLET_EVERY = 2

# LLM analysis frequency â every N loops (60s = 20 loops Ã 3s)
LLM_EVERY        = 20   # Run LLM analysis every ~60 seconds
LLM_TOP_MARKETS  = 8    # Analyze top 8 markets per cycle (cost control)

MIN_DAYS = 0
MAX_DAYS = 30
NEAR_RES_DAYS   = 7
NEAR_RES_LIMIT  = 300

active_connections: Set[WebSocket] = set()
_loop_count = 0
_llm_debug = {"last_cycle": None, "last_error": None, "candidates_found": 0, "spikes_found": 0, "cycles_run": 0}


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
        print(f"[FETCH] {len(near_res)} near-res + {len(regular)} regular = {len(combined)} total")
        return combined

    except Exception as e:
        print(f"[FETCH] Error: {e}")
        return []


# ââ NEW: LLM Analysis Cycle ââââââââââââââââââââââââââââââââââââââââââââââââââ

async def llm_analysis_cycle(markets: list, markets_by_id: dict):
    """
    Run LLM analysis on top markets. Called every ~60 seconds.

    Strategy:
    1. Detect volume spikes across all markets
    2. Pick top N markets (by volume spike + 24h volume)
    3. Gather free research context for each
    4. Feed to LLM for analysis
    5. If LLM says BUY, create a paper trade with LLM reasoning stored in memory
    """
    global _llm_debug
    _llm_debug["cycles_run"] += 1
    _llm_debug["last_cycle"] = datetime.utcnow().isoformat()
    print(f"[LLM] === Starting LLM cycle #{_llm_debug['cycles_run']} with {len(markets)} markets ===")
    import sys; sys.stdout.flush()

    portfolio = await db.get_portfolio()
    cash = portfolio.get("cash_balance", 0)
    wins = portfolio.get("win_count", 0) or 0
    losses = portfolio.get("loss_count", 0) or 0
    total = wins + losses
    win_rate = round(wins / total * 100, 1) if total else 0

    portfolio_state = {
        "cash_balance": cash,
        "invested": portfolio.get("total_invested", 0),
        "win_rate": win_rate,
    }

    # Step 1: Detect volume spikes across markets
    spike_markets = []
    spike_errors = 0
    for m in markets[:100]:  # Check top 100 by volume
        try:
            spike = await detect_spike(
                m["id"], m["volume24hr"], m["yes_price"],
                m["volume"], m["liquidity"]
            )
            if spike:
                spike_markets.append(m)
                print(f"[SPIKE] ð {spike['alert_type']}: '{m['question'][:50]}' â {spike['description']}")
        except Exception as e:
            spike_errors += 1
            if spike_errors <= 2:
                print(f"[SPIKE] Error for {m.get('id','?')}: {e}")

    _llm_debug["spikes_found"] = len(spike_markets)
    print(f"[LLM] Spike scan: {len(spike_markets)} spikes, {spike_errors} errors out of {min(len(markets), 100)} markets")
    sys.stdout.flush()

    # Step 2: Select markets for LLM analysis
    # Priority: volume spike markets first, then highest 24h volume
    candidates = spike_markets[:4]  # Up to 4 spike markets
    remaining_slots = LLM_TOP_MARKETS - len(candidates)

    # Add high-volume non-spike markets
    seen = {m["id"] for m in candidates}
    for m in markets[:50]:
        if len(candidates) >= LLM_TOP_MARKETS:
            break
        if m["id"] not in seen and m.get("volume24hr", 0) > 10000:
            candidates.append(m)
            seen.add(m["id"])

    _llm_debug["candidates_found"] = len(candidates)

    if not candidates:
        print(f"[LLM] No candidates â top 5 vol24h: {[m.get('volume24hr',0) for m in markets[:5]]}")
        sys.stdout.flush()
        return

    print(f"[LLM] ð§  Analyzing {len(candidates)} markets ({len(spike_markets)} with volume spikes)")
    sys.stdout.flush()

    # Step 3: Analyze each candidate
    llm_trades = 0
    for market in candidates:
        try:
            # Gather context (free â no LLM cost)
            news_context = await gather_market_context(market)
            vol_profile = await get_market_volume_profile(market["id"])
            lessons = await get_relevant_lessons(market.get("category", ""), limit=5)

            # LLM analysis (Haiku screening, Sonnet if big edge)
            decision = await analyze_market(
                market, news_context, vol_profile, lessons, portfolio_state
            )

            if not decision:
                continue

            action = decision["action"]
            confidence = decision["confidence"]
            edge = decision["edge"]
            reasoning = decision["reasoning"]

            if action == "SKIP":
                print(f"[LLM] â­ SKIP '{market['question'][:45]}' â {reasoning[:80]}")
                continue

            if confidence < 0.6 or abs(edge) < 0.10:
                print(f"[LLM] â  LOW-CONF '{market['question'][:40]}' conf={confidence:.2f} edge={edge:.1%}")
                continue

            # Build signal compatible with existing paper_trader
            direction = "YES" if action == "BUY_YES" else "NO"
            yes_price = market.get("yes_price", 0.5)
            entry_price = yes_price if direction == "YES" else (1 - yes_price)

            signal = {
                "market_id":       market["id"],
                "market_question": market["question"],
                "score":           int(confidence * 100),  # Convert to 0-100 scale
                "confidence":      confidence,
                "direction":       direction,
                "yes_price":       yes_price,
                "market_type":     "LLM_ANALYSIS",  # New type for LLM trades
                "can_enter":       True,
                "entry_reason":    f"LLM: edge={edge:.1%}, conf={confidence:.2f}",
                "factors_json":    json.dumps({
                    "llm_confidence": confidence,
                    "llm_edge": edge,
                    "volume_spike": 1.0 if vol_profile.get("has_recent_spike") else 0.0,
                    "news_context": 1.0 if news_context else 0.0,
                }),
                "created_at":      datetime.utcnow().isoformat(),
                "clob_token_ids":  market.get("clob_token_ids", []),
                "condition_id":    market.get("condition_id", ""),
                "liquidity":       market.get("liquidity", 0),
            }

            # Enter via existing paper_trader (uses Kelly sizing)
            trade = await maybe_enter_trade(signal)
            if trade:
                llm_trades += 1

                # Store reasoning in memory system
                try:
                    await store_trade_reasoning(
                        trade_id=trade["id"],
                        market_id=market["id"],
                        market_question=market["question"],
                        category=market.get("category", ""),
                        direction=direction,
                        action=action,
                        entry_price=entry_price,
                        confidence=confidence,
                        estimated_probability=decision["estimated_probability"],
                        edge=edge,
                        reasoning=reasoning,
                        key_evidence=decision.get("key_evidence", []),
                        risk_factors=decision.get("risk_factors", []),
                        had_volume_spike=vol_profile.get("has_recent_spike", False),
                        model_used=decision.get("model", "unknown"),
                        tokens_used=decision.get("tokens_used", 0),
                    )
                except Exception as e:
                    print(f"[MEMORY] Store error: {e}")

                print(f"[LLM] â TRADE {action} '{market['question'][:40]}' "
                      f"edge={edge:.1%} conf={confidence:.2f} model={decision.get('model','?')}")

        except Exception as e:
            import traceback
            _llm_debug["last_error"] = f"{e} | {traceback.format_exc()[-200:]}"
            print(f"[LLM] Analysis error for '{market.get('question','?')[:40]}': {e}")
            print(f"[LLM] Traceback: {traceback.format_exc()[-300:]}")
            sys.stdout.flush()

    # Print cost summary
    costs = get_cost_summary()
    print(f"[LLM] ð° Cycle done: {llm_trades} trades | "
          f"API cost: ${costs['total_cost_usd']:.4f} "
          f"(Haiku: {costs['haiku_calls']} calls, Sonnet: {costs['sonnet_calls']} calls)")


# ââ NEW: Post-Trade Learning âââââââââââââââââââââââââââââââââââââââââââââââââ

async def learn_from_closed_trade(trade: dict, pnl: float, outcome: str):
    """
    After a trade closes, extract a lesson using the LLM and store it.
    Only runs for LLM-originated trades (has reasoning in memory).
    """
    try:
        # Get the original reasoning from memory
        from memory_system import get_memory_summary
        # For now, use trade explanation as reasoning proxy
        explanations = await db.get_trade_explanations(50)
        reasoning = ""
        for ex in explanations:
            if ex.get("trade_id") == trade.get("id"):
                reasoning = ex.get("entry_explanation", "")
                break

        if not reasoning:
            reasoning = f"Trade entered on {trade.get('market_type', 'unknown')} signal"

        # Extract lesson via LLM (uses Haiku â cheapest)
        lesson = await evaluate_trade_outcome(trade, reasoning, outcome, pnl)

        if lesson:
            # Store in memory system
            await record_trade_outcome(
                trade_id=trade["id"],
                exit_price=trade.get("exit_price", 0),
                pnl=pnl,
                outcome=outcome,
                lesson=lesson,
            )
            print(f"[LEARN] ð Lesson: {lesson[:100]}")

    except Exception as e:
        print(f"[LEARN] Error: {e}")


# ââ Main Trading Loop âââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

async def trading_loop():
    global _loop_count
    print("[v2] ð§  PM Intelligence v2 â Self-Learning LLM Agent")
    print("[v2] ð¡ Haiku screening + Sonnet deep analysis | Memory system active")

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
            await check_live_exits(markets_by_id)

            signals = await generate_signals(markets)

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

            print(f"[LOOP] #{_loop_count}: {len(markets)} mkts | "
                  f"{len(lock_in_sigs)} lock-in | {len(bne_sigs)} bne | "
                  f"{len(ct_sigs)} copy | {len(enterable)} enterable")

            for sig in signals[:30]:
                await db.save_signal(sig)

            # Regular heuristic trading
            entered = 0
            live_entered = 0
            lev_entered = 0
            for sig in signals[:50]:
                trade = await maybe_enter_trade(sig)
                if trade:
                    entered += 1
                live_trade = await maybe_enter_live_trade(sig)
                if live_trade:
                    live_entered += 1
                lev_trade = await maybe_enter_leverage_trade(sig)
                if lev_trade:
                    lev_entered += 1
            if entered:
                print(f"[PAPER] â {entered} heuristic trade(s) entered")
            if live_entered:
                print(f"[LIVE] ð¢ {live_entered} REAL trade(s) entered")
            if lev_entered:
                print(f"[LEV] â¡ {lev_entered} leverage trade(s) entered")

            # ââ NEW: LLM Analysis Cycle (every ~60 seconds) ââ
            if _loop_count % LLM_EVERY == 0:
                try:
                    print(f"[LLM] Triggering cycle at loop #{_loop_count}")
                    import sys; sys.stdout.flush()
                    await llm_analysis_cycle(markets, markets_by_id)
                    print(f"[LLM] Cycle complete. Costs: {get_cost_summary()}")
                    sys.stdout.flush()
                except Exception as e:
                    import traceback
                    _llm_debug["last_error"] = f"Cycle: {e} | {traceback.format_exc()[-200:]}"
                    print(f"[LLM] Cycle error: {e}")
                    print(f"[LLM] Traceback: {traceback.format_exc()[-300:]}")
                    sys.stdout.flush()

            # Build broadcast payload
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
            costs  = get_cost_summary()

            print(f"[v2] ð° ${portfolio.get('cash_balance',0):,.0f} | "
                  f"{wins}W/{losses}L | WR={wr_pct}% | "
                  f"API=${costs['total_cost_usd']:.4f}")

            # Include LLM stats in broadcast
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
                "markets":            markets[:40],
                "trade_explanations": explanations,
                "lock_in_count":      len(lock_in_sigs),
                "momentum_count":     len(momentum_sigs),
                "leverage_portfolio": lev_portfolio,
                "leverage_trades":    lev_trades,
                "llm_costs":          costs,
                "memory_stats":       memory_stats,
                "timestamp":          datetime.utcnow().isoformat(),
            })

        except Exception as e:
            print(f"[v2] Loop error: {e}")

        await asyncio.sleep(LOOP_SLEEP)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.init_db()
    # Initialize v2 modules
    try:
        await init_memory()
        await init_volume_tables()
        print("[v2] â Memory system initialized")
        print("[v2] â Volume detector initialized")
    except Exception as e:
        print(f"[v2] Init warning: {e}")

    await close_stuck_trades()
    await seed_weights()
    print("[STARTUP] Pre-loading news...")
    try:
        await news_engine.refresh_news()
    except Exception as e:
        print(f"[STARTUP] News: {e}")

    # Check if API key is configured
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if api_key:
        print("[v2] ð Anthropic API key configured â LLM brain ACTIVE")
    else:
        print("[v2] â  No ANTHROPIC_API_KEY â running in fallback (heuristic-only) mode")

    loop_task = asyncio.create_task(trading_loop())
    yield
    loop_task.cancel()
    try:
        await loop_task
    except asyncio.CancelledError:
        pass


app = FastAPI(title="PM Intelligence v2 â Self-Learning Agent", lifespan=lifespan)


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
            "markets":            await db.get_all_markets(40),
            "trade_explanations": await db.get_trade_explanations(30),
            "leverage_portfolio": await db.get_leverage_portfolio(),
            "leverage_trades":    await db.get_all_leverage_trades(30),
            "llm_costs":          get_cost_summary(),
            "memory_stats":       memory_stats,
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

# v2 endpoints
@app.get("/api/llm/costs")
async def api_llm_costs():
    return get_cost_summary()

@app.get("/api/llm/debug")
async def api_llm_debug():
    return {
        "loop_count": _loop_count,
        "llm_every": LLM_EVERY,
        "next_llm_at": LLM_EVERY - (_loop_count % LLM_EVERY),
        "debug": _llm_debug,
        "has_anthropic": "check_llm_agent",
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
    return await get_live_status()

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
