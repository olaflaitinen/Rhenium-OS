# Configuration Guide

## Overview

Rhenium OS uses Pydantic-based configuration with support for:
- Environment variables (RHENIUM_ prefix)
- `.env` files
- Profile-based configuration files

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RHENIUM_DATA_DIR` | `./data` | Data storage directory |
| `RHENIUM_MODELS_DIR` | `./models` | Model weights directory |
| `RHENIUM_LOGS_DIR` | `./logs` | Log files directory |
| `RHENIUM_CACHE_DIR` | `./cache` | Cache directory |
| `RHENIUM_DEVICE` | `auto` | Device type (auto/cpu/cuda) |
| `RHENIUM_BATCH_SIZE` | `1` | Batch size for inference |
| `RHENIUM_LOG_LEVEL` | `INFO` | Logging level |
| `RHENIUM_MEDGEMMA_BACKEND` | `stub` | MedGemma backend |
| `RHENIUM_MEDGEMMA_ENDPOINT` | `http://localhost:8080` | Remote endpoint |

## Example .env File

```bash
RHENIUM_DATA_DIR=/mnt/data/rhenium
RHENIUM_MODELS_DIR=/mnt/models/rhenium
RHENIUM_DEVICE=cuda
RHENIUM_CUDA_DEVICE_IDS=0,1
RHENIUM_MEDGEMMA_BACKEND=local
RHENIUM_MEDGEMMA_MODEL_PATH=/mnt/models/medgemma
```

## Python API

```python
from rhenium.core.config import get_settings

settings = get_settings()
print(settings.data_dir)
print(settings.get_device_string())
```
