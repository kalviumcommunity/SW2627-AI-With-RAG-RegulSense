#!/usr/bin/env bash
# ==============================================================================
# Launch RegulSense FastAPI Backend (Bash)
# ==============================================================================
set -e

echo "[RegulSense] Starting Backend REST API Service..."

# Activate virtual environment if present
if [ -d ".venv" ]; then
    echo "[RegulSense] Activating virtual environment .venv..."
    source .venv/bin/activate
fi

# Run FastAPI Backend with Uvicorn
exec uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
