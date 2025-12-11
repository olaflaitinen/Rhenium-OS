# Dashboard Engine API Reference

This document provides API documentation for the Rhenium OS Dashboard Engine.

---

## Overview

The Dashboard Engine provides a unified API for dashboard applications to access all Rhenium OS capabilities.

```mermaid
graph TD
    subgraph Dashboard["Dashboard Application"]
        UI[Web UI]
        Mobile[Mobile App]
        PACS[PACS Integration]
    end
    
    subgraph Engine["Dashboard Engine"]
        API[API Layer]
        Cache[Results Cache]
        Queue[Request Queue]
    end
    
    subgraph Core["Rhenium OS Core"]
        Pipelines[Pipeline Manager]
        Disease[Disease Reasoner]
        XAI[XAI Generator]
        MedGemma[MedGemma Client]
    end
    
    subgraph Models["Model Registry"]
        Perception[Perception Models]
        Reconstruction[Reconstruction Models]
        Generative[Generative Models]
    end
    
    Dashboard --> Engine
    Engine --> Core
    Core --> Models
```

---

## Engine Initialization

```mermaid
sequenceDiagram
    participant App as Dashboard App
    participant Engine as DashboardEngine
    participant Config as EngineConfig
    participant Pipelines as PipelineLoader
    
    App->>Config: Create EngineConfig
    App->>Engine: DashboardEngine(config)
    App->>Engine: initialize()
    Engine->>Pipelines: _load_pipelines()
    Pipelines-->>Engine: PipelineInfo[]
    Engine-->>App: status = READY
```

---

## Analysis Workflow

```mermaid
sequenceDiagram
    participant App as Dashboard
    participant Engine as DashboardEngine
    participant Pipeline as Pipeline
    participant Disease as DiseaseReasoner
    participant XAI as XAIEngine
    participant MG as MedGemma
    
    App->>Engine: analyze_study(request)
    Engine->>Pipeline: run(study_id)
    Pipeline->>Pipeline: load_input()
    Pipeline->>Pipeline: run_reconstruction()
    Pipeline->>Pipeline: run_analysis()
    Pipeline->>Disease: run_disease_reasoning()
    Disease-->>Pipeline: DiseaseReasoningOutput
    Pipeline->>XAI: run_xai()
    XAI-->>Pipeline: EvidenceDossier
    Pipeline->>MG: run_medgemma_explanation()
    MG-->>Pipeline: ReportDraft
    Pipeline-->>Engine: PipelineResult
    Engine-->>App: AnalysisResponse
```

---

## API Reference

### DashboardEngine

Main engine class for dashboard integration.

```mermaid
classDiagram
    class DashboardEngine {
        +EngineConfig config
        +EngineStatus status
        +bool is_ready
        +initialize() void
        +shutdown() void
        +analyze_study(request) AnalysisResponse
        +get_disease_assessment(study_id) DiseaseReasoningOutput
        +get_findings(study_id) List~Finding~
        +get_evidence_dossier(study_id) Dict
        +get_report_draft(study_id) Dict
        +list_pipelines() List~PipelineInfo~
        +get_pipeline(pipeline_id) PipelineInfo
        +get_status() HealthStatus
    }
    
    class EngineConfig {
        +Path data_dir
        +Path models_dir
        +Path cache_dir
        +str log_level
        +int max_concurrent_requests
        +bool enable_gpu
        +str medgemma_backend
    }
    
    class AnalysisRequest {
        +str study_id
        +str pipeline_id
        +str priority
        +Dict options
        +Dict clinical_context
    }
    
    class AnalysisResponse {
        +str request_id
        +str study_id
        +str status
        +int findings_count
        +bool has_critical_findings
        +float processing_time_ms
        +Dict disease_output
    }
    
    class HealthStatus {
        +EngineStatus status
        +str version
        +float uptime_seconds
        +int pipelines_loaded
        +int models_loaded
        +bool gpu_available
    }
    
    DashboardEngine --> EngineConfig
    DashboardEngine --> AnalysisRequest
    DashboardEngine --> AnalysisResponse
    DashboardEngine --> HealthStatus
```

---

## Usage Examples

### Basic Analysis

```python
from rhenium.engine import DashboardEngine, EngineConfig, AnalysisRequest

# Initialize engine
config = EngineConfig(enable_gpu=True)
engine = DashboardEngine(config)
engine.initialize()

# Check status
status = engine.get_status()
print(f"Engine status: {status.status.value}")
print(f"Pipelines loaded: {status.pipelines_loaded}")

# Analyze a study
request = AnalysisRequest(
    study_id="study_001",
    pipeline_id="ct_head_ich_detection",
    priority="urgent",
)
response = engine.analyze_study(request)

# Get disease assessment
disease = engine.get_disease_assessment("study_001")
if disease and disease.has_disease:
    print(f"Primary diagnosis: {disease.primary_diagnosis.disease_name}")
    if disease.has_critical_flags:
        print("CRITICAL: Requires immediate attention")

# Cleanup
engine.shutdown()
```

---

## Error Handling

```mermaid
flowchart TD
    A[analyze_study] --> B{Valid Request?}
    B -->|No| C[Return Error Response]
    B -->|Yes| D{Pipeline Available?}
    D -->|No| E[Return Pipeline Not Found]
    D -->|Yes| F[Execute Pipeline]
    F --> G{Success?}
    G -->|No| H[Log Error]
    H --> I[Return Error Response]
    G -->|Yes| J[Cache Result]
    J --> K[Return Success Response]
```

---

## Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `data_dir` | Path | `./data` | Data storage directory |
| `models_dir` | Path | `./models` | Model weights directory |
| `cache_dir` | Path | `./cache` | Results cache directory |
| `log_level` | str | `INFO` | Logging level |
| `max_concurrent_requests` | int | `4` | Maximum parallel requests |
| `enable_gpu` | bool | `True` | Enable GPU acceleration |
| `medgemma_backend` | str | `stub` | MedGemma backend type |
| `enable_audit_logging` | bool | `True` | Enable audit logging |

---

**Copyright (c) 2025 Skolyn LLC. All rights reserved.**

