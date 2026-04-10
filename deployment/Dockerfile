# Multi-stage build for the FastAPI inference service.
#
# WHY MULTI-STAGE?
#   Stage 1 (builder): installs ALL dependencies including build tools (~2GB)
#   Stage 2 (runtime): copies only the installed packages (~400MB)
#   Result: production image is ~5x smaller — faster deploys, less attack surface.
#
# HOW TO BUILD:
#   docker build -f deployment/Dockerfile -t fraud-scoring:latest .
#
# HOW TO RUN:
#   docker run -p 8000:8000 \
#     -v $(pwd)/src/models/artifacts:/app/src/models/artifacts:ro \
#     -e MODEL_VERSION=v1 \
#     -e REDIS_URL=redis://redis:6379/0 \
#     fraud-scoring:latest

# ---- Stage 1: Builder ----
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker layer caching: this layer only rebuilds when requirements change)
COPY deployment/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ---- Stage 2: Runtime ----
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder (no build tools in final image)
COPY --from=builder /install /usr/local

# Copy application code
COPY src/ ./src/
COPY api/ ./api/
COPY configs/ ./configs/

# Model artifacts are mounted at runtime (not baked into the image)
# This lets you swap model versions without rebuilding the image
RUN mkdir -p src/models/artifacts

# Non-root user for security
RUN useradd --create-home --shell /bin/bash app && chown -R app:app /app
USER app

# Environment variables with sensible defaults
ENV MODEL_VERSION=v1 \
    REVIEW_THRESHOLD=0.3 \
    BLOCK_THRESHOLD=0.7 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

# Health check: Kubernetes liveness probe calls this
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Gunicorn with Uvicorn workers: 4 workers handle concurrent requests
# Each worker is a full copy of the app (model loaded in each)
CMD ["gunicorn", "api.main:app", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "30", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
