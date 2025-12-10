FROM python:3.11-slim

LABEL maintainer="Skolyn LLC <engineering@skolyn.com>"
LABEL description="Rhenium OS - AI Operating System for Diagnostic Imaging"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for layer caching
COPY pyproject.toml .

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Copy application code
COPY rhenium/ rhenium/

# Create non-root user
RUN useradd -m rhenium && chown -R rhenium:rhenium /app
USER rhenium

# Set environment variables
ENV RHENIUM_DATA_DIR=/data
ENV RHENIUM_MODELS_DIR=/models
ENV RHENIUM_LOGS_DIR=/logs
ENV RHENIUM_CACHE_DIR=/cache

# Health check
HEALTHCHECK CMD python -c "import rhenium; print(rhenium.__version__)"

# Default command
CMD ["rhenium", "--help"]
