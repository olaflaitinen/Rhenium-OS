# Feature Taxonomy: Skolyn Rhenium OS

---

## Overview

This document provides a hierarchical taxonomy of features implemented or planned for Skolyn Rhenium OS. The taxonomy is organized by functional layer and demonstrates the platform's capacity for 1000+ fine-grained features across modalities, organs, and clinical tasks.

---

## 1. Data Ingestion and Management (100+ features)

### 1.1 Format Support
- DICOM ingestion (all SOP classes)
- NIfTI ingestion (NIfTI-1, NIfTI-2)
- Raw k-space ingestion (vendor-agnostic)
- Raw sinogram ingestion (CT)
- RF/IQ data ingestion (Ultrasound)
- DICOM-SR parsing
- DICOM-SEG support

### 1.2 Metadata Extraction
- Patient demographics extraction (pseudonymized)
- Study/series/instance hierarchy parsing
- Acquisition parameter extraction
- Vendor-specific tag interpretation
- Protocol name normalization
- Body part detection from metadata
- Laterality detection

### 1.3 Preprocessing
- Intensity normalization (z-score, min-max, percentile)
- Spatial resampling (isotropic, target resolution)
- Orientation standardization (RAS+)
- Cropping and padding (centered, ROI-based)
- Bias field correction (MRI)
- Skull stripping (brain MRI)
- Motion artifact detection

### 1.4 Data Lineage
- Transformation tracking
- Provenance chain documentation
- Checksum verification
- Version control integration

---

## 2. Reconstruction Features (150+ features)

### 2.1 MRI Reconstruction
- Inverse FFT baseline
- GRAPPA parallel imaging
- SENSE parallel imaging
- Compressed sensing (L1-wavelet)
- Deep learning reconstruction (U-Net, VarNet)
- Physics-informed reconstruction (PINN-MRI)
- Coil sensitivity estimation
- Motion correction
- Gibbs ringing removal
- Quantitative mapping (T1, T2, T2*)
- Diffusion tensor estimation
- Perfusion quantification

### 2.2 CT Reconstruction
- Filtered back-projection (FBP)
- Iterative reconstruction
- Deep learning denoising
- Metal artifact reduction
- Beam hardening correction
- Scatter correction
- Low-dose optimization
- Dual-energy decomposition
- Cardiac gating integration

### 2.3 X-ray Enhancement
- Noise reduction
- Contrast enhancement (CLAHE-like)
- Grid line artifact removal
- Bone suppression
- Rib unfolding (virtual)
- Dynamic range optimization

### 2.4 Ultrasound Processing
- Envelope detection
- Log compression
- Speckle reduction
- Compound imaging
- Doppler processing
- Elastography strain estimation

### 2.5 Physics-Informed Models (PINNs)
- MR signal equation constraints
- Bloch equation modeling
- CT projection physics
- Attenuation modeling
- Regularization via physics priors

---

## 3. Perception Features (300+ features)

### 3.1 Segmentation
#### Anatomical Structures
- Organ segmentation (liver, kidney, spleen, pancreas)
- Bone segmentation (vertebrae, femur, tibia)
- Muscle segmentation
- Fat compartment segmentation
- Brain structure segmentation (cortex, white matter, ventricles)
- Cardiac chamber segmentation

#### Pathological Structures
- Lesion segmentation
- Tumor segmentation
- Hemorrhage segmentation
- Infarct segmentation
- Edema segmentation

### 3.2 Detection
#### Lesion Detection
- Lung nodule detection
- Liver lesion detection
- Breast lesion detection
- Brain lesion detection
- Bone lesion detection
- Lymph node detection

#### Anatomical Landmark Detection
- Vertebral body localization
- Joint center detection
- Organ boundary detection

### 3.3 Classification
#### Pathology Classification
- Benign vs malignant
- Tumor grade prediction
- Histological subtype prediction
- Treatment response assessment

#### Scoring Systems
- PI-RADS (prostate)
- BI-RADS (breast)
- LI-RADS (liver)
- Lung-RADS (lung)
- ASPECTS (stroke)
- Kellgren-Lawrence (osteoarthritis)
- Outerbridge (cartilage)

### 3.4 Quantification
- Volume measurement
- Diameter measurement (RECIST)
- Shape descriptors
- Texture features (radiomics)
- Intensity statistics
- Temporal change quantification

---

## 4. Organ-Specific Modules (200+ features)

### 4.1 Knee MRI Module
- Meniscus segmentation (medial, lateral)
- Meniscal tear detection
- Tear type classification (horizontal, vertical, radial, complex)
- Cartilage segmentation (femoral, tibial, patellar)
- Cartilage thickness mapping
- Outerbridge grading
- ACL integrity assessment
- PCL integrity assessment
- MCL/LCL assessment
- Bone marrow edema detection
- Effusion quantification
- Osteophyte detection

### 4.2 Brain MRI Module
- White matter lesion segmentation
- Lesion load quantification
- Hemorrhage detection (epidural, subdural, SAH, IPH, IVH)
- Tumor segmentation (enhancing, non-enhancing, edema)
- Tumor volume estimation
- Midline shift measurement
- Ventricle volume measurement
- Atrophy assessment
- Microbleed detection
- Infarct detection and aging

### 4.3 Prostate MRI Module
- Prostate gland segmentation
- Zonal anatomy segmentation (PZ, TZ, CZ)
- PI-RADS lesion detection
- PI-RADS score prediction
- Lesion volume estimation
- ADC value extraction
- Extracapsular extension assessment
- Seminal vesicle invasion assessment

### 4.4 Breast Imaging Module
- Breast segmentation
- Fibroglandular tissue segmentation
- Density classification (a, b, c, d)
- Lesion detection (mass, non-mass enhancement)
- BI-RADS assessment
- Lesion characterization (morphology, kinetics)
- Calcification detection (mammography)
- Architectural distortion detection

---

## 5. Generative Model Features (50+ features)

### 5.1 Super-Resolution
- MRI super-resolution (2x, 4x)
- CT super-resolution
- Slice interpolation

### 5.2 Denoising
- GAN-based denoising
- Diffusion-based denoising
- Noise level estimation
- Adaptive denoising

### 5.3 Anomaly Detection
- Reconstruction-based anomaly detection
- Out-of-distribution detection
- Uncertainty-based anomaly scoring

### 5.4 Data Augmentation
- Realistic artifact synthesis
- Pathology simulation (training only)
- Domain adaptation

---

## 6. Explainability Features (100+ features)

### 6.1 Visual Evidence
- Saliency maps (Grad-CAM, Integrated Gradients)
- Attention maps
- Segmentation overlays
- Bounding box visualization
- Uncertainty heatmaps
- Comparison views (before/after, with/without finding)

### 6.2 Quantitative Evidence
- Measurement tables
- Radiomics feature reports
- Confidence scores
- Uncertainty quantification
- Calibration statistics
- Comparison to reference values

### 6.3 Narrative Evidence
- Finding descriptions
- Clinical significance statements
- Differential diagnosis suggestions
- Guideline references
- Limitation statements
- Recommendation suggestions

### 6.4 Evidence Dossier
- Dossier compilation
- Dossier serialization (JSON, PDF)
- Dossier versioning
- Audit trail integration

---

## 7. MedGemma Reasoning Features (75+ features)

### 7.1 Report Generation
- Structured report drafting
- Section-wise generation (technique, findings, impression)
- Template-based generation
- Free-form generation
- Multi-finding synthesis

### 7.2 Explanation
- Finding-specific explanation
- Reasoning chain exposition
- Uncertainty communication
- Limitation acknowledgment

### 7.3 Validation
- Cross-finding consistency checking
- Measurement plausibility checking
- Laterality verification
- Guideline alignment checking

### 7.4 Interaction
- Question answering over cases
- Multi-turn dialog
- Clarification requests
- Alternative hypothesis exploration

---

## 8. Fairness and Governance Features (75+ features)

### 8.1 Fairness Metrics
- Stratified AUC
- Stratified sensitivity/specificity
- Calibration by subgroup
- Performance disparity metrics
- Equalized odds assessment

### 8.2 Bias Mitigation
- Reweighting hooks
- Augmentation strategies
- Algorithmic fairness constraints
- Post-hoc calibration

### 8.3 Audit and Compliance
- Comprehensive logging
- Tamper-evident logs
- Access control logging
- Model version tracking
- Configuration logging

### 8.4 Documentation
- Model cards
- Dataset documentation
- Risk management records
- Clinical evaluation documentation

---

## 9. Evaluation Features (50+ features)

### 9.1 Classification Metrics
- AUC-ROC
- AUC-PR
- Sensitivity/Specificity at thresholds
- F1 score
- Calibration (Brier, reliability diagrams)

### 9.2 Segmentation Metrics
- Dice coefficient
- IoU / Jaccard
- Hausdorff distance (95th percentile)
- Surface distance metrics

### 9.3 Detection Metrics
- FROC curves
- Sensitivity at fixed FP rates
- Per-lesion and per-patient metrics

### 9.4 Image Quality Metrics
- PSNR
- SSIM
- LPIPS (perceptual)
- FID (for generative models)

---

## Summary

| Category | Feature Count (Approximate) |
|----------|----------------------------|
| Data Ingestion | 100+ |
| Reconstruction | 150+ |
| Perception | 300+ |
| Organ Modules | 200+ |
| Generative | 50+ |
| Explainability | 100+ |
| MedGemma | 75+ |
| Fairness/Governance | 75+ |
| Evaluation | 50+ |
| **Total** | **1100+** |

---

**Copyright (c) 2025 Skolyn LLC. All rights reserved.**

---

## Documentation Update Notes

- Last updated: December 2025.
- Aligned with Rhenium OS disease reasoning layer.
- Revised markdown structure for consistency.
