# Multi-stage Dockerfile for production deployment
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application code
COPY app ./app
COPY env.example .env

# Create necessary directories
RUN mkdir -p /tmp/badminton_analyzer/uploads /tmp/badminton_analyzer/results logs && \
    chown -R appuser:appuser /app /tmp/badminton_analyzer

# Switch to app user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/api/v1/health', timeout=5)"

# Run application
CMD ["python", "-m", "app.main"]


# ==================== Development Stage ====================
FROM base as development

USER root

# Install development dependencies
RUN pip install pytest pytest-asyncio pytest-cov black flake8 mypy isort ipython

USER appuser

# Override CMD for development
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]


# ==================== Production Stage ====================
FROM base as production

# Production-optimized settings
ENV DEBUG=False \
    LOG_LEVEL=INFO \
    WORKERS=4

# Run with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

