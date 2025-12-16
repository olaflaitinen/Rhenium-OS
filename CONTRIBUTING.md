# Contributing to Rhenium OS

Thank you for your interest in contributing to Rhenium OS! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Documentation](#documentation)

## Development Setup

```bash
# Clone the repository
git clone https://github.com/rhenium-os/rhenium-os.git
cd rhenium-os

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or .venv\Scripts\activate  # Windows

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Code Style

We follow these coding standards:

| Tool | Purpose | Configuration |
|------|---------|---------------|
| **Black** | Code formatting | Line length: 100 |
| **Ruff** | Linting | See `pyproject.toml` |
| **mypy** | Type checking | Strict mode |
| **isort** | Import sorting | Via Ruff |

### Type Hints

All functions must have type hints:

```python
def process_volume(
    volume: ImageVolume,
    threshold: float = 0.5,
) -> tuple[np.ndarray, dict[str, Any]]:
    ...
```

### Docstrings

Use Google-style docstrings:

```python
def compute_dice(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Compute Dice similarity coefficient.

    Args:
        pred: Predicted binary mask
        target: Ground truth binary mask

    Returns:
        Dice score in range [0, 1]

    Raises:
        ValueError: If arrays have different shapes
    """
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=rhenium --cov-report=term-missing

# Run specific test file
pytest tests/test_core.py

# Run tests matching pattern
pytest -k "test_dice"
```

### Test Categories

- **Unit tests**: `tests/unit/` - Fast, isolated tests
- **Integration tests**: `tests/integration/` - Component interaction
- **E2E tests**: `tests/e2e/` - Full pipeline tests

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with appropriate tests
3. Ensure all tests pass: `pytest`
4. Ensure linting passes: `ruff check src/`
5. Update documentation if needed
6. Submit PR with clear description

### Commit Messages

Use conventional commits:

```
feat: add new segmentation model
fix: resolve memory leak in data loader
docs: update API documentation
test: add tests for XAI module
```

## Documentation

- Update docstrings for any public API changes
- Add examples for new features
- Update relevant `.md` files in `docs/`

## License

By contributing, you agree that your contributions will be licensed under the EUPL-1.1 license.
