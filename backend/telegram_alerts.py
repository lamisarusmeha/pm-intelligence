"""
Telegram Alerts for PM Intelligence Trading Bot.

Sends push notifications for:
- Trade entries (new position opened)
- Trade exits (position closed with P&L)
- Strategy errors or going silent
- Binance feed status changes
- Periodic health summaries

Uses Telegram Bot API via urllib (no extra dependencies).
"""

import json
import os
import time
import urllib.request
import urllib.error
from datetime import datetime

# Config â set via environment variables on Railway
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Rate limiting â don't spam
_last_sent = {}
MIN_INTERVAL = 10  # seconds between same-type messages
_error_count = 0
_max_silent_errors = 5  # stop trying after 5 consecutive failures


def _send_message(text: str, parse_mode: str = "HTML") -> bool:
    """Send a message via Telegram Bot API."""
    global _error_count

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False

    if _error_count >= _max_silent_errors:
        return False  # Stop trying after too many failures

    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = json.dumps({
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json", "User-Agent": "PM-Intelligence/1.0"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            result = json.loads(resp.read().decode())
            if result.get("ok"):
                _error_count = 0
                return True
            else:
                _error_count += 1
                return False

    except Exception as e:
        _error_count += 1
        if _error_count <= 2:
            print(f"[TELEGRAM] Send failed: {e}")
        return False


def _rate_limit(key: str, interval: int = None) -> bool:
    """Returns True if we should skip (too recent)."""
    now = time.time()
    min_gap = interval or MIN_INTERVAL
    if now - _last_sent.get(key, 0) < min_gap:
        return True
    _last_sent[key] = now
    return False


# ============================================================
# PUBLIC API â call these from main.py / paper_trader.py
# ============================================================

def alert_trade_entry(trade: dict):
    """Alert when a new trade is entered."""
    if _rate_limit("entry", 5):
        return

    market_type = trade.get("market_type", "UNKNOWN")
    direction = trade.get("direction", "?")
    entry_price = trade.get("entry_price", 0)
    cost = trade.get("cost", 0)
    question = trade.get("market_question", "")[:60]

    emoji = {"NEAR_CERTAINTY": "\U0001f3af", "VOLUME_SPIKE": "\U0001f4c8", "BINANCE_ARB": "\u26a1"}.get(market_type, "\U0001f4b0")

    text = (
        f"{emoji} <b>NEW TRADE</b>\n"
        f"<b>{market_type}</b> | {direction} @ ${entry_price:.3f}\n"
        f"Cost: ${cost:.0f}\n"
        f"{question}"
    )
    _send_message(text)


def alert_trade_exit(trade: dict):
    """Alert when a trade is closed."""
    if _rate_limit("exit", 5):
        return

    pnl = trade.get("pnl", 0)
    market_type = trade.get("market_type", "UNKNOWN")
    direction = trade.get("direction", "?")
    entry_price = trade.get("entry_price", 0)
    exit_price = trade.get("exit_price", 0)
    question = trade.get("market_question", "")[:60]
    reason = trade.get("exit_reason", "")

    if pnl >= 0:
        emoji = "\u2705"
        pnl_str = f"+${pnl:.2f}"
    else:
        emoji = "\u274c"
        pnl_str = f"-${abs(pnl):.2f}"

    text = (
        f"{emoji} <b>TRADE CLOSED</b> ({pnl_str})\n"
        f"<b>{market_type}</b> | {direction} | {entry_price:.3f} \u2192 {exit_price:.3f}\n"
        f"{reason}\n"
        f"{question}"
    )
    _send_message(text)


def alert_error(source: str, error: str):
    """Alert on errors (rate-limited to avoid spam)."""
    if _rate_limit(f"error_{source}", 300):  # Max 1 error per source per 5 min
        return

    text = (
        f"\u26a0\ufe0f <b>ERROR</b> in {source}\n"
        f"<code>{error[:200]}</code>"
    )
    _send_message(text)


def alert_feed_status(status: str, details: str = ""):
    """Alert on Binance feed status changes."""
    if _rate_limit("feed", 600):  # Max 1 per 10 min
        return

    text = (
        f"\U0001f4e1 <b>FEED {status.upper()}</b>\n"
        f"{details}"
    )
    _send_message(text)


def alert_health_summary(portfolio: dict, trades: list, binance_status: dict, loop_count: int):
    """Periodic health summary (call every ~30 min)."""
    if _rate_limit("health", 1800):  # Max 1 per 30 min
        return

    balance = portfolio.get("balance", 100000)
    total_pnl = portfolio.get("total_pnl", 0)
    open_trades = len([t for t in trades if t.get("status") == "OPEN"])
    closed_trades = len([t for t in trades if t.get("status") == "CLOSED"])

    wins = len([t for t in trades if t.get("status") == "CLOSED" and (t.get("pnl", 0) or 0) > 0])
    losses = len([t for t in trades if t.get("status") == "CLOSED" and (t.get("pnl", 0) or 0) < 0])
    win_rate = (wins / max(1, wins + losses)) * 100

    btc = binance_status.get("BTC", {}).get("price", 0)
    feed = binance_status.get("feed_source", "unknown")

    pnl_emoji = "\U0001f4c8" if total_pnl >= 0 else "\U0001f4c9"

    text = (
        f"\U0001f4ca <b>HEALTH CHECK</b>\n"
        f"Balance: ${balance:,.0f} ({pnl_emoji} ${total_pnl:+,.2f})\n"
        f"Open: {open_trades} | Closed: {closed_trades}\n"
        f"Win rate: {win_rate:.0f}% ({wins}W/{losses}L)\n"
        f"BTC: ${btc:,.0f} | Feed: {feed}\n"
        f"Loops: {loop_count}"
    )
    _send_message(text)


def alert_startup():
    """Alert when bot starts up."""
    text = (
        f"\U0001f680 <b>PM Intelligence v3 STARTED</b>\n"
        f"Strategies: Near-Certainty + Volume Spike + Binance Arb\n"
        f"Feed: Binance.us REST + WebSocket backup\n"
        f"Mode: Paper Trading"
    )
    _send_message(text)


def is_configured() -> bool:
    """Check if Telegram alerts are configured."""
    return bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)
