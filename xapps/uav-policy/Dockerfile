# Multi-stage build for uav-policy xApp
# Stage 1: Builder
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy source code
COPY . .

# Install package in build stage
RUN pip install --no-cache-dir --user \
    flask>=2.3.0 \
    werkzeug>=2.3.0

# Stage 2: Runtime
FROM python:3.11-slim

LABEL maintainer="O-RAN Alliance"
LABEL description="UAV Policy xApp - E2 Indication Handler for O-RAN"
LABEL version="0.1.0"

# Create non-root user for security
RUN useradd -m -u 1000 xapp

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /home/xapp/.local

# Copy application code
COPY --chown=xapp:xapp src/uav_policy /app/uav_policy

# Set environment variables
ENV PATH=/home/xapp/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    LOG_LEVEL=INFO \
    SERVER_HOST=0.0.0.0 \
    SERVER_PORT=5000

# Switch to non-root user
USER xapp

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')" || exit 1

# Expose port
EXPOSE 5000

# Run application
CMD ["python3", "-m", "uav_policy.main"]
