# QServer Dockerfile
# Multi-stage build for Python web server with Qlib support

FROM python:3.10-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create and set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --user -r requirements.txt

# Final stage
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/root/.local/bin:$PATH" \
    FLASK_APP=run.py \
    FLASK_ENV=production \
    QLIB_DATA_PATH=/app/data/stock/cn_data \
    MODEL_OUTPUT_PATH=/app/data/models \
    TRAIN_OUTPUT_PATH=/app/data/train

# Install runtime dependencies and cron
RUN apt-get update && apt-get install -y --no-install-recommends \
    cron \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data/stock/cn_data \
    /app/data/models \
    /app/data/train \
    /app/data/logs

# Setup cron job
COPY cron/crontab /etc/cron.d/qserver-cron
RUN chmod 0644 /etc/cron.d/qserver-cron \
    && crontab /etc/cron.d/qserver-cron \
    && touch /var/log/cron.log

# Expose port
EXPOSE 5000

# Create entrypoint script
RUN echo '#!/bin/bash\n\
# Start cron in background\n\
service cron start\n\
\n\
# Start Flask application\n\
exec gunicorn --bind 0.0.0.0:5000 --workers 4 --threads 2 run:app\n\
' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
