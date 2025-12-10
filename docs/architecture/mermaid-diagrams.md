# Mermaid Diagrams Reference

**
---

## Overview

This document provides a centralized reference for all major architectural diagrams in Rhenium OS using Mermaid syntax.

---

## 1. Global System Architecture

```mermaid
graph TD
    subgraph Interfaces["User Interfaces"]
        CLI[Rhenium CLI]
        PyAPI[Python API]
    end
    
    subgraph RheniumOS["Rhenium OS Core"]
        direction TB
        DataLayer[Data & IO Layer]
        ReconLayer[Reconstruction Layer]
        PerceptionLayer[Perception Layer]
        XAILayer[XAI Layer]
        ReasoningLayer[MedGemma Reasoning]
        GovLayer[Governance Layer]
    end
    
    subgraph Models["Model Ecosystem"]
        PINN[PINNs]
        GAN[GANs]
        UNet[U-Net Family]
        ViT[Vision Transformers]
        MG27B[MedGemma 27B]
    end
    
    subgraph Modalities["Supported Modalities"]
        MRI[MRI - All Sequences]
        CT[CT - All Protocols]
        US[Ultrasound - All Modes]
        XR[X-ray - All Types]
    end
    
    Interfaces --> RheniumOS
    DataLayer --> Modalities
    ReconLayer --> PINN & GAN
    PerceptionLayer --> UNet & ViT
    ReasoningLayer --> MG27B
```

---

## 2. Data Flow Pipeline

```mermaid
flowchart LR
    subgraph Ingest["1. Data Ingestion"]
        DICOM[DICOM Files]
        RAW[Raw Data]
        META[Metadata]
    end
    
    subgraph Preprocess["2. Preprocessing"]
        Norm[Normalization]
        Reg[Registration]
        QC[Quality Check]
    end
    
    subgraph Recon["3. Reconstruction"]
        Phys[Physics-Informed]
        DL[Deep Learning]
        Enh[Enhancement]
    end
    
    subgraph Analyze["4. Analysis"]
        Seg[Segmentation]
        Det[Detection]
        Cls[Classification]
        Quant[Quantification]
    end
    
    subgraph Explain["5. Explanation"]
        Vis[Visual XAI]
        Num[Quantitative XAI]
        Narr[Narrative XAI]
    end
    
    subgraph Report["6. Reporting"]
        Dossier[Evidence Dossier]
        MG[MedGemma Report]
        Audit[Audit Log]
    end
    
    Ingest --> Preprocess --> Recon --> Analyze --> Explain --> Report
```

---

## 3. MRI Processing Flow

```mermaid
flowchart TD
    subgraph Input["MRI Input"]
        KSpace[K-space Data]
        DICOM[DICOM Images]
        Params[Sequence Parameters]
    end
    
    subgraph Recon["Reconstruction"]
        FFT[FFT Baseline]
        GRAPPA[GRAPPA/SENSE]
        PINN[PINN Recon]
        DL[DL Recon]
    end
    
    subgraph Enhance["Enhancement"]
        Bias[Bias Correction]
        Denoise[Denoising]
        SR[Super-Resolution]
    end
    
    subgraph MultiSeq["Multi-Sequence Processing"]
        T1[T1-weighted]
        T2[T2-weighted]
        FLAIR[FLAIR]
        DWI[DWI/ADC]
        Reg[Registration]
    end
    
    KSpace --> FFT & GRAPPA
    KSpace --> PINN & DL
    FFT & GRAPPA & PINN & DL --> Bias
    Bias --> Denoise --> SR
    SR --> T1 & T2 & FLAIR & DWI --> Reg
```

---

## 4. CT Processing Flow

```mermaid
flowchart TD
    subgraph Input["CT Input"]
        Sino[Sinogram Data]
        DICOM[DICOM Images]
        Phases[Contrast Phases]
    end
    
    subgraph Recon["Reconstruction"]
        FBP[FBP]
        IR[Iterative Recon]
        PINN[PINN Recon]
        DL[DL Recon]
    end
    
    subgraph Enhance["Enhancement"]
        Denoise[Denoising]
        MAR[Metal Artifact Reduction]
        SR[Super-Resolution]
    end
    
    subgraph Protocol["Protocol-Specific"]
        NCCT[Non-Contrast]
        CTA[CT Angiography]
        LDCT[Low-Dose]
        DECT[Dual-Energy]
    end
    
    Sino --> FBP & IR & PINN & DL
    DICOM --> Enhance
    FBP & IR & PINN & DL --> Denoise
    Denoise --> MAR --> SR
    SR --> NCCT & CTA & LDCT & DECT
```

---

## 5. Ultrasound Processing Flow

```mermaid
flowchart TD
    subgraph Input["US Input"]
        RF[RF Data]
        IQ[IQ Data]
        DICOM[DICOM Images]
    end
    
    subgraph Beam["Beamforming"]
        DAS[Delay-and-Sum]
        Adapt[Adaptive BF]
        PINN[PINN BF]
    end
    
    subgraph Enhance["Enhancement"]
        Speckle[Speckle Reduction]
        Log[Log Compression]
        Norm[Normalization]
    end
    
    subgraph Modes["Imaging Modes"]
        Bmode[B-mode]
        Color[Color Doppler]
        Spectral[Spectral Doppler]
        Elasto[Elastography]
    end
    
    RF & IQ --> DAS & Adapt & PINN
    DAS & Adapt & PINN --> Speckle
    DICOM --> Speckle
    Speckle --> Log --> Norm
    Norm --> Bmode & Color & Spectral & Elasto
```

---

## 6. X-ray Processing Flow

```mermaid
flowchart TD
    subgraph Input["X-ray Input"]
        Proj[Projection Data]
        DICOM[DICOM Images]
    end
    
    subgraph Preprocess["Preprocessing"]
        Flat[Flat-field Correction]
        Norm[Normalization]
        Enhance[Contrast Enhancement]
    end
    
    subgraph Types["X-ray Types"]
        CXR[Chest X-ray]
        MSK[Musculoskeletal]
        Mammo[Mammography]
        Fluoro[Fluoroscopy]
    end
    
    Proj --> Flat --> Norm
    DICOM --> Norm
    Norm --> Enhance
    Enhance --> CXR & MSK & Mammo & Fluoro
```

---

## 7. Perception Model Architecture

```mermaid
flowchart TD
    subgraph Input["Preprocessed Images"]
        Vol2D[2D Slices]
        Vol3D[3D Volumes]
        Multi[Multi-sequence]
    end
    
    subgraph Models["Model Zoo"]
        UNet[U-Net/nnU-Net]
        Trans[Transformers]
        CNN[3D CNNs]
        Hybrid[Hybrid Models]
    end
    
    subgraph Tasks["Perception Tasks"]
        Seg[Segmentation]
        Det[Detection]
        Cls[Classification]
        Reg[Regression]
    end
    
    subgraph Output["Outputs"]
        Mask[Segmentation Masks]
        Box[Bounding Boxes]
        Prob[Probabilities]
        Val[Quantitative Values]
    end
    
    Vol2D & Vol3D & Multi --> UNet & Trans & CNN & Hybrid
    UNet & Trans & CNN & Hybrid --> Seg & Det & Cls & Reg
    Seg --> Mask
    Det --> Box
    Cls --> Prob
    Reg --> Val
```

---

## 8. XAI and Evidence Dossier Flow

```mermaid
flowchart TD
    subgraph Findings["Model Findings"]
        Pred[Predictions]
        Conf[Confidence Scores]
        Mask[Segmentation Masks]
    end
    
    subgraph VisualXAI["Visual Evidence"]
        Sal[Saliency Maps]
        Att[Attention Maps]
        Overlay[Overlays]
    end
    
    subgraph QuantXAI["Quantitative Evidence"]
        Meas[Measurements]
        Radio[Radiomics]
        Unc[Uncertainty]
    end
    
    subgraph NarrXAI["Narrative Evidence"]
        MG[MedGemma Explanation]
        Limit[Limitations]
        Refs[References]
    end
    
    subgraph Dossier["Evidence Dossier"]
        JSON[JSON Schema]
        Export[Export Formats]
    end
    
    Pred & Conf & Mask --> VisualXAI & QuantXAI
    VisualXAI & QuantXAI --> NarrXAI
    NarrXAI --> MG
    VisualXAI & QuantXAI & NarrXAI --> Dossier
```

---

## 9. MedGemma Integration

```mermaid
sequenceDiagram
    participant Pipeline as Pipeline Runner
    participant XAI as XAI Engine
    participant Adapter as MedGemma Adapter
    participant Model as MedGemma 27B
    participant Validator as Validators
    participant Log as Audit Log
    
    Pipeline->>XAI: Findings + Evidence
    XAI->>Adapter: request_explanation()
    Adapter->>Adapter: Build prompt from template
    Adapter->>Model: Generate
    Model-->>Adapter: Raw response
    Adapter->>Validator: Validate response
    Validator-->>Adapter: Validation result
    
    alt Validation Failed
        Adapter->>Adapter: Flag for review
    end
    
    Adapter->>Log: Log call details
    Adapter-->>XAI: Narrative evidence
    XAI-->>Pipeline: Complete dossier
```

---

## 10. Governance and Audit Flow

```mermaid
flowchart TD
    subgraph Actions["System Actions"]
        Ingest[Data Ingestion]
        Infer[Model Inference]
        Explain[Explanation Gen]
        Report[Report Draft]
    end
    
    subgraph Audit["Audit System"]
        Log[Audit Logger]
        Store[Secure Storage]
        Query[Query Interface]
    end
    
    subgraph Governance["Governance"]
        Card[Model Cards]
        Risk[Risk Tracking]
        Fair[Fairness Reports]
        Access[Access Control]
    end
    
    Actions --> Log
    Log --> Store
    Store --> Query
    Query --> Card & Risk & Fair
    Access --> Actions
```

---

## 11. Fairness Evaluation Flow

```mermaid
flowchart TD
    subgraph Data["Evaluation Data"]
        Test[Test Dataset]
        Demo[Demographics]
        Meta[Metadata]
    end
    
    subgraph Stratify["Stratification"]
        Age[Age Groups]
        Sex[Sex/Gender]
        Site[Scanner/Site]
        Vendor[Vendor]
    end
    
    subgraph Metrics["Per-Stratum Metrics"]
        AUC[AUC]
        Dice[Dice]
        Cal[Calibration]
    end
    
    subgraph Analysis["Disparity Analysis"]
        Diff[Absolute Difference]
        Ratio[Disparity Ratio]
        Sig[Significance Test]
    end
    
    subgraph Report["Fairness Report"]
        Table[Summary Table]
        Flag[Disparity Flags]
        Rec[Recommendations]
    end
    
    Test & Demo & Meta --> Age & Sex & Site & Vendor
    Stratify --> Metrics
    Metrics --> Analysis
    Analysis --> Report
```

---

**Copyright (c) 2025 Skolyn LLC. All rights reserved.**

**SPDX-License-Identifier: EUPL-1.1**
