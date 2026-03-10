#!/bin/bash
# ─────────────────────────────────────────────────────────────
#  Prediction Market Intelligence Platform — Launcher
# ─────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"

echo ""
echo "📡  Prediction Market Intelligence Platform"
echo "──────────────────────────────────────────"

# Check Python 3.10+
if ! command -v python3 &>/dev/null; then
  echo "❌  Python 3 not found. Please install Python 3.10 or higher."
  exit 1
fi

PYTHON_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "✅  Python $PYTHON_VER detected"

# Kill any existing process on port 8000
EXISTING=$(lsof -ti :8000 2>/dev/null)
if [ -n "$EXISTING" ]; then
  echo "⚠️   Port 8000 in use — killing old process (PID $EXISTING)..."
  kill -9 $EXISTING 2>/dev/null
  sleep 1
  echo "✅  Old process cleared"
fi

# Install dependencies
echo "📦  Installing dependencies…"
cd "$BACKEND_DIR"
pip3 install -r requirements.txt --quiet --break-system-packages 2>/dev/null \
  || pip3 install -r requirements.txt --quiet

echo "✅  Dependencies installed"
echo ""
echo "🚀  Starting server on http://localhost:8000"
echo "    Dashboard: http://localhost:8000"
echo "    API docs:  http://localhost:8000/docs"
echo ""
echo "    Press Ctrl+C to stop."
echo "──────────────────────────────────────────"
echo ""

# Load .env if present (sets DB_PATH etc.)
if [ -f "$BACKEND_DIR/.env" ]; then
  set -a; source "$BACKEND_DIR/.env"; set +a
fi

# Launch FastAPI (must run from backend/ dir so imports resolve)
cd "$BACKEND_DIR"
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
