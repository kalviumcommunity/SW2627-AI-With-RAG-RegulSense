# ==============================================================================
# Launch RegulSense Streamlit Chat Interface (PowerShell)
# ==============================================================================
Write-Host "[RegulSense] Starting Streamlit Chat Interface..." -ForegroundColor Cyan

# Activate virtual environment if present
if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    Write-Host "[RegulSense] Activating virtual environment .venv..." -ForegroundColor Gray
    & ".\.venv\Scripts\Activate.ps1"
}

# Run Streamlit Application
streamlit run app.py --server.port 8501
