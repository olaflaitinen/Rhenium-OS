FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ src/

RUN pip install --no-cache-dir -e ".[server]"

RUN useradd -m rhenium
USER rhenium

ENV RHENIUM_DATA_DIR=/data
ENV RHENIUM_MODELS_DIR=/models
ENV RHENIUM_LOGS_DIR=/logs

EXPOSE 8000

CMD ["uvicorn", "rhenium.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
