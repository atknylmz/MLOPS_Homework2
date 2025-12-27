# =============================================================================
# Dockerfile for High-Cardinality Prediction Service
# =============================================================================
# 
# MLOps Principle: "Only build your binaries once"
# 
# This Dockerfile creates a deployable container that:
# 1. Packages the complete prediction service
# 2. Includes all dependencies
# 3. Can be deployed to any container runtime
# 
# The same image built in CI is used for:
# - Integration testing
# - Smoke testing
# - Production deployment
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder Stage
# -----------------------------------------------------------------------------
# Use multi-stage build for smaller final image
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 2: Production Stage
# -----------------------------------------------------------------------------
FROM python:3.11-slim as production

# Labels for container metadata
LABEL maintainer="MLOps Student"
LABEL version="1.0.0"
LABEL description="High-Cardinality Prediction Service"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY src/ ./src/
COPY setup.py ./
COPY README.md ./

# Create data directory (model files will be mounted or copied separately)
RUN mkdir -p ./data

# Change ownership to non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose the application port
EXPOSE 8000

# Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health').raise_for_status()" || exit 1

# Default command to run the application
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
