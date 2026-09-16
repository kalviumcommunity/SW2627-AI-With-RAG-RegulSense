# ==============================================================================
# RegulSense AI Regulatory Compliance Assistant - Production Dockerfile
# ==============================================================================
FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install minimal system utilities required for network healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /app/requirements.txt

# Copy application directories
COPY src/ /app/src/
COPY prompts/ /app/prompts/
COPY data/ /app/data/
COPY app.py /app/app.py
COPY .env.example /app/.env.example

# Create directories for runtime uploads and outputs
RUN mkdir -p /app/data/uploads /app/data/chroma_db /app/outputs

# Expose FastAPI backend (8000) and Streamlit frontend (8501)
EXPOSE 8000 8501

# Default command: Launch FastAPI backend
# Overridable via docker-compose or docker run
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
