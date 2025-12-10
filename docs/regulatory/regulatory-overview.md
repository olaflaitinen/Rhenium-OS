# Regulatory Overview: Skolyn Rhenium OS

**Last Updated: December 2025**

---

## Introduction

Skolyn Rhenium OS is designed with regulatory compliance as a foundational principle. This document outlines the regulatory landscape applicable to AI-based medical imaging software in the European Union and describes how Rhenium OS addresses key requirements.

---

## Applicable Regulations

### EU Medical Device Regulation (MDR) 2017/745

The EU MDR governs medical devices, including Software as a Medical Device (SaMD). Key aspects:

- **Classification**: AI diagnostic software typically falls under Class IIa or IIb depending on clinical risk
- **Conformity Assessment**: Requires notified body involvement for Class IIa and above
- **Technical Documentation**: Comprehensive documentation of design, development, and validation
- **Post-Market Surveillance**: Ongoing monitoring of device performance in clinical use
- **Unique Device Identification (UDI)**: Traceability requirements

### EU Artificial Intelligence Act

The AI Act establishes requirements for high-risk AI systems. Medical diagnostic AI is explicitly classified as high-risk under Annex III.

Key requirements for high-risk AI systems:

1. **Risk Management System**: Identification and mitigation of risks
2. **Data Governance**: Quality requirements for training and validation data
3. **Technical Documentation**: Detailed description of system design and performance
4. **Record Keeping**: Logging capabilities for traceability
5. **Transparency**: Clear information for users about AI capabilities and limitations
6. **Human Oversight**: Design enabling effective human control
7. **Accuracy, Robustness, Cybersecurity**: Performance and security requirements

### GDPR (General Data Protection Regulation)

Processing of health data requires:

- Legal basis for processing (typically explicit consent or healthcare provision)
- Data minimization and purpose limitation
- Data subject rights (access, rectification, erasure)
- Data protection by design and default
- Data Protection Impact Assessment for high-risk processing

---

## Rhenium OS Regulatory Alignment

### Risk Management

| MDR/AI Act Requirement | Rhenium OS Implementation |
|------------------------|---------------------------|
| Risk identification | `governance/risk_tracking.py` - structured risk registry |
| Risk mitigation | Documented mitigation strategies per identified risk |
| Residual risk acceptance | Clear criteria and approval workflow |
| Continuous monitoring | Performance monitoring with drift detection |

### Data Governance

| Requirement | Implementation |
|-------------|----------------|
| Data quality | Preprocessing validation, anomaly detection |
| Data traceability | `data/metadata.py` - lineage tracking |
| PHI protection | PHI sanitization in logging, patient ID hashing |
| Access control | Role-based access hooks in governance layer |

### Technical Documentation

| Document Type | Location |
|---------------|----------|
| System description | `docs/architecture/` |
| Design rationale | `docs/vision/` |
| Performance validation | Evaluation framework, benchmark results |
| Intended use | README.md, model cards |

### Record Keeping and Audit

| Capability | Implementation |
|------------|----------------|
| Comprehensive logging | `governance/audit_log.py` |
| Tamper evidence | Checksum-based log integrity |
| Model versioning | Registry with semantic versioning |
| Configuration tracking | Settings logging per pipeline run |

### Transparency

| Requirement | Implementation |
|-------------|----------------|
| User information | Evidence Dossiers with explanations |
| Capability disclosure | Model cards with limitations |
| Uncertainty communication | Confidence scores, uncertainty metrics |
| Limitation statements | Integrated into MedGemma outputs |

### Human Oversight

| Design Principle | Implementation |
|------------------|----------------|
| AI as decision support | All outputs require radiologist verification |
| Clear AI labeling | Findings marked as AI-generated |
| Override capability | Radiologist can reject/modify all AI findings |
| Escalation paths | High-risk findings flagged for review |

### Accuracy and Robustness

| Quality Dimension | Implementation |
|-------------------|----------------|
| Performance metrics | Comprehensive evaluation framework |
| Calibration | Calibration curves and Brier scores |
| Fairness | Stratified metrics across subgroups |
| Robustness testing | Evaluation on diverse datasets |

---

## Conformity Assessment Pathway

### Pre-Market

1. **Classification Determination**: Self-classification with rationale documentation
2. **Quality Management System**: ISO 13485 alignment
3. **Technical Documentation**: Per MDR Annex II/III
4. **Clinical Evaluation**: Per MEDDEV 2.7/1 rev 4
5. **Notified Body Review**: For Class IIa/IIb devices

### Post-Market

1. **Post-Market Surveillance Plan**: Active monitoring strategy
2. **PMCF Study**: Post-market clinical follow-up
3. **Periodic Safety Update Reports**: Regular safety reviews
4. **Vigilance**: Serious incident reporting

---

## Conclusion

Rhenium OS is architected to support regulatory compliance through:

- Built-in audit and traceability capabilities
- Comprehensive documentation infrastructure
- Fairness and bias monitoring
- Transparent, explainable AI outputs
- Human oversight by design

Actual regulatory approval requires completing the conformity assessment process with appropriate notified bodies.

---

**Copyright (c) 2025 Skolyn LLC. All rights reserved.**

**SPDX-License-Identifier: EUPL-1.1**
