# Installation

## Requirements

| Component | Minimum Version |
|-----------|-----------------|
| Python | 3.10+ |
| PyTorch | 2.1+ |
| CUDA | 11.8+ (optional) |

## Installation Methods

=== "pip (Recommended)"
    ```bash
    pip install rhenium-os
    ```

=== "From Source"
    ```bash
    git clone https://github.com/rhenium-os/rhenium-os.git
    cd rhenium-os
    pip install -e ".[dev]"
    ```

=== "Docker"
    ```bash
    docker pull rhenium-os/rhenium-os:latest
    docker run -p 8000:8000 rhenium-os/rhenium-os
    ```

## Optional Dependencies

```bash
# Training support
pip install rhenium-os[training]

# Documentation
pip install rhenium-os[docs]

# Server/API
pip install rhenium-os[server]

# All extras
pip install rhenium-os[all]
```

## Verify Installation

```python
import rhenium
print(f"Rhenium OS v{rhenium.__version__}")

from rhenium.core.config import get_settings
settings = get_settings()
print(f"Device: {settings.get_effective_device()}")
```

## GPU Support

For GPU acceleration, install PyTorch with CUDA:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```
