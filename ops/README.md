# Rhenium OS Operations (Ops)

This directory contains operational scripts, CI/CD pipeline configurations, and maintenance tools for Rhenium OS.

## Directory Structure

```
ops/
├── ci-cd/               # CI/CD pipeline definitions
│   └── ci-pipeline-outline.md  # Overview of the CI pipeline
├── scripts/             # Maintenance and utility scripts
└── README.md            # This file
```

## CI/CD Pipeline

The Continuous Integration/Continuous Deployment guidelines are detailed in [**CI Pipeline Outline**](ci-cd/ci-pipeline-outline.md).

### Key Workflows

1.  **Build & Test**: Runs on every push to `main` and pull requests.
    -   Linting (`flake8`, `black`)
    -   Type Checking (`mypy`)
    -   Unit Tests (`pytest`)
2.  **Verification**: Validates notebook generation and documentation links.
3.  **Release**: Automated tagging and packaging upon version bump.

## Deployment

Rhenium OS is typically deployed as a Python package or Docker container.

### Local Development
Refer to the [**Developer Guide**](../docs/usage/developer-guide.md) for local environment setup.

### Production Environment
Production deployments utilize Docker containers orchestrated by Kubernetes (or similar).
*(Specific production deployment scripts to be added)*

---

**Copyright (c) 2025 Skolyn LLC. All rights reserved.**
