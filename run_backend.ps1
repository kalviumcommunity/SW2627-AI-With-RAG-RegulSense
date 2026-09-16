# ==============================================================================
# Launch RegulSense FastAPI Backend (PowerShell)
# ==============================================================================
Write-Host "[RegulSense] Starting Backend REST API Service..." -ForegroundColor Cyan

# Activate virtual environment if present
if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    Write-Host "[RegulSense] Activating virtual environment .venv..." -ForegroundColor Gray
    & ".\.venv\Scripts\Activate.ps1"
}

# Run FastAPI Backend with Uvicorn
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
