# Core API Reference

## Configuration

::: rhenium.core.config.RheniumSettings
    options:
      show_root_heading: true
      show_source: false

### Environment Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `RHENIUM_DATA_DIR` | Path | `./data` | Data directory |
| `RHENIUM_MODELS_DIR` | Path | `./models` | Model weights |
| `RHENIUM_DEVICE` | str | `auto` | Compute device |
| `RHENIUM_SEED` | int | `42` | Random seed |
| `RHENIUM_LOG_LEVEL` | str | `INFO` | Logging level |

---

## Component Registry

```mermaid
classDiagram
    class Registry {
        +register(type, name, version)
        +get(type, name)
        +list_components(type)
    }
    class ComponentType {
        MODEL
        PIPELINE
        PREPROCESSOR
        RECONSTRUCTOR
    }
    Registry --> ComponentType
```

### Usage

```python
from rhenium.core.registry import registry, ComponentType

# Register a model
@registry.register(ComponentType.MODEL, "my_model", version="1.0.0")
class MyModel:
    pass

# Retrieve
model_cls = registry.get(ComponentType.MODEL, "my_model")
```

---

## Error Taxonomy

```mermaid
graph TD
    RheniumError --> ConfigurationError
    RheniumError --> DataError
    RheniumError --> ModelError
    RheniumError --> PipelineError
    RheniumError --> ValidationError
    DataError --> DICOMError
    DataError --> NIfTIError
    ModelError --> ReconstructionError
    ModelError --> XAIError
```

### Exception Classes

| Exception | Use Case |
|-----------|----------|
| `ConfigurationError` | Invalid settings |
| `DataError` | Data loading/format issues |
| `ModelError` | Model inference errors |
| `PipelineError` | Pipeline execution failures |
| `ValidationError` | Input/output validation |
