# Governance

## Overview

Rhenium OS includes governance tools for responsible AI development in medical imaging.

## Model Cards

Document your models for transparency:

```python
from rhenium.governance import ModelCard

card = ModelCard(
    name="brain_lesion_segmentation",
    version="1.0.0",
    task="Semantic segmentation of brain lesions",
    architecture="3D U-Net with attention gates",
    training_data="Synthetic + BraTS 2021",
    metrics={"dice": 0.87, "hd95": 4.2},
    limitations=[
        "Trained primarily on adult brain MRI",
        "May underperform on pediatric cases",
    ],
    regulatory_status="Research use only - not FDA approved",
)

card.save(Path("model_card.yaml"))
```

---

## Dataset Cards

```python
from rhenium.governance import DatasetCard

card = DatasetCard(
    name="synthetic_brain_mri",
    version="1.0",
    modality="MRI",
    num_samples=1000,
    demographics={"age_range": "25-75", "sex_ratio": "50/50"},
    known_biases=["Limited representation of non-European populations"],
)
```

---

## Risk Register

Track and mitigate AI risks:

| Risk ID | Category | Likelihood | Impact | Status |
|---------|----------|------------|--------|--------|
| R001 | Bias | Medium | High | Monitoring |
| R002 | Privacy | Low | Critical | Mitigated |
| R003 | Safety | Medium | High | Open |

```python
from rhenium.governance import RiskRegister, Risk, RiskCategory

register = RiskRegister()
register.add(Risk(
    id="R001",
    title="Demographic bias in training data",
    category=RiskCategory.BIAS,
    likelihood="medium",
    impact="high",
    mitigation=["Collect diverse validation sets", "Monitor subgroup performance"],
))
```

---

## Audit Logging

Immutable audit trail for compliance:

```python
from rhenium.governance import AuditLogger, AuditEvent, AuditEventType

logger = AuditLogger(log_dir=Path("./audit_logs"))

logger.log(AuditEvent(
    event_id="ev001",
    event_type=AuditEventType.PREDICTION,
    study_uid="1.2.3.4.5",
    details={"model": "brain_lesion_v1", "confidence": 0.92},
))
```

---

## Fairness Analysis

```python
from rhenium.governance.fairness import FairnessAnalyzer

analyzer = FairnessAnalyzer(protected_attribute="sex")
metrics = analyzer.compute_metrics(predictions, labels, demographics)

print(f"Demographic Parity: {metrics.demographic_parity:.3f}")
print(f"Equalized Odds: {metrics.equalized_odds:.3f}")
```

### Fairness Thresholds

| Metric | Threshold | Interpretation |
|--------|-----------|----------------|
| Demographic Parity | < 0.1 | Prediction rates similar across groups |
| Equalized Odds | < 0.1 | TPR/FPR consistent across groups |
