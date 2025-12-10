# Contributing to Rhenium OS

**
---

## Introduction

Thank you for your interest in contributing to Skolyn Rhenium OS. This document provides guidelines for contributing to the project.

---

## Code of Conduct

Contributors are expected to maintain a professional, respectful environment. Discrimination, harassment, and unprofessional behavior are not tolerated.

---

## How to Contribute

### Reporting Issues

1. Search existing issues to avoid duplicates
2. Use issue templates when available
3. Provide detailed reproduction steps
4. Include relevant system information

### Submitting Changes

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/your-feature`
3. **Make changes** following the style guidelines
4. **Add tests** for new functionality
5. **Update documentation** as needed
6. **Run the test suite**: `pytest`
7. **Submit a pull request**

---

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/rhenium-os.git
cd rhenium-os

# Set up upstream remote
git remote add upstream https://github.com/skolyn/rhenium-os.git

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

---

## Code Style

### Python Style

- Follow PEP 8
- Use type hints for all public functions
- Maximum line length: 100 characters
- Use docstrings (Google style)

### Formatting Tools

```bash
# Format code
black rhenium tests

# Sort imports
isort rhenium tests

# Lint
ruff check rhenium

# Type check
mypy rhenium
```

---

## Testing Requirements

- All new code must have tests
- Maintain or improve test coverage
- Tests must pass before merge

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=rhenium --cov-report=term-missing
```

---

## Documentation

- Update relevant documentation for changes
- Use clear, professional language
- No emojis in documentation
- Include examples where appropriate

---

## Pull Request Process

1. Ensure all tests pass
2. Update CHANGELOG.md if applicable
3. Request review from maintainers
4. Address feedback promptly
5. Squash commits before merge

---

## License

By contributing, you agree that your contributions will be licensed under the European Union Public License 1.1 (EUPL-1.1).

---

**Copyright (c) 2025 Skolyn LLC. All rights reserved.**

**SPDX-License-Identifier: EUPL-1.1**
