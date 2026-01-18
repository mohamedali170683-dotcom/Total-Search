# Keyword Research Tool - Multi-Platform Data Aggregator
# Python 3.11 base image

FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser

# ---------------------
# Builder stage
# ---------------------
FROM base as builder

# Copy requirements first for better caching
COPY pyproject.toml ./

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install .

# ---------------------
# Development stage
# ---------------------
FROM builder as development

# Install development dependencies
RUN pip install ".[dev]"

# Copy source code
COPY --chown=appuser:appuser . .

USER appuser

# Default command for development
CMD ["python", "-m", "pytest", "-v"]

# ---------------------
# Production stage
# ---------------------
FROM base as production

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser scripts/ ./scripts/
COPY --chown=appuser:appuser migrations/ ./migrations/
COPY --chown=appuser:appuser alembic.ini ./
COPY --chown=appuser:appuser pyproject.toml ./

# Create data directories
RUN mkdir -p data/input data/output data/checkpoints && \
    chown -R appuser:appuser data/

USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from src.config import get_settings; get_settings()" || exit 1

# Default command
CMD ["python", "scripts/run_pipeline.py", "--help"]
