#!/usr/bin/env bash
# ==============================================================================
# Launch RegulSense Streamlit Chat Interface (Bash)
# ==============================================================================
set -e

echo "[RegulSense] Starting Streamlit Chat Interface..."

# Activate virtual environment if present
if [ -d ".venv" ]; then
    echo "[RegulSense] Activating virtual environment .venv..."
    source .venv/bin/activate
fi

# Run Streamlit Application
exec streamlit run app.py --server.port 8501
