# MRI Benchmarks

**
****

---

## Overview

This document defines benchmark targets for MRI perception tasks across organ systems within Rhenium OS.

---

## Reconstruction Quality Benchmarks

| Task | Acceleration | Metric | Target | Baseline DL |
|------|--------------|--------|--------|-------------|
| Brain MRI (T1/T2) | 4x | PSNR | > 36 dB | 34 dB |
| Brain MRI (T1/T2) | 4x | SSIM | > 0.96 | 0.94 |
| Knee MRI (PD-FSE) | 4x | PSNR | > 35 dB | 33 dB |
| Cardiac Cine | 8x | PSNR | > 32 dB | 30 dB |
| Prostate (T2) | 4x | SSIM | > 0.95 | 0.92 |

---

## Brain MRI Benchmarks

| Task | Sequences | Metric | Target | Baseline |
|------|-----------|--------|--------|----------|
| Tumor segmentation | T1c, T2, FLAIR | Dice | > 0.88 | 0.84 |
| White matter lesion | FLAIR, T2 | Dice | > 0.75 | 0.70 |
| Acute stroke (DWI) | DWI, ADC | Sensitivity | > 0.95 | 0.90 |
| Brain volumetry | T1 MP-RAGE | Volume MAE | < 3% | 5% |
| Hemorrhage detection | SWI, T2* | AUC | > 0.95 | 0.90 |

---

## Knee MRI Benchmarks

| Task | Sequences | Metric | Target | Baseline |
|------|-----------|--------|--------|----------|
| Meniscus tear detection | PD-FSE, T2 | AUC | > 0.95 | 0.92 |
| Meniscus tear detection | PD-FSE, T2 | Sensitivity | > 0.90 | 0.85 |
| ACL tear detection | PD-FSE, T2 | AUC | > 0.95 | 0.93 |
| Cartilage lesion seg | PD-FSE | Dice | > 0.85 | 0.80 |
| Bone marrow edema | STIR | Dice | > 0.80 | 0.75 |

---

## Prostate MRI Benchmarks

| Task | Sequences | Metric | Target | Baseline |
|------|-----------|--------|--------|----------|
| Lesion detection | T2, DWI, DCE | AUC | > 0.92 | 0.88 |
| PI-RADS >= 3 detection | mpMRI | Sensitivity | > 0.90 | 0.85 |
| Zonal segmentation | T2 | Dice | > 0.90 | 0.85 |
| Lesion segmentation | T2, DWI | Dice | > 0.75 | 0.70 |

---

## Cardiac MRI Benchmarks

| Task | Sequences | Metric | Target | Baseline |
|------|-----------|--------|--------|----------|
| LV segmentation | Cine SSFP | Dice | > 0.92 | 0.90 |
| RV segmentation | Cine SSFP | Dice | > 0.88 | 0.85 |
| EF estimation | Cine | MAE | < 5% | 7% |
| LGE scar seg | LGE | Dice | > 0.85 | 0.80 |
| Myocardial T1 | T1 Map | CV | < 5% | 8% |

---

## Spine MRI Benchmarks

| Task | Sequences | Metric | Target | Baseline |
|------|-----------|--------|--------|----------|
| Disc herniation detection | T2 Sag/Ax | AUC | > 0.90 | 0.85 |
| Stenosis grading | T2 | Kappa | > 0.80 | 0.70 |
| Vertebral fracture | T1, STIR | Sensitivity | > 0.95 | 0.90 |
| Disc segmentation | T2 | Dice | > 0.85 | 0.80 |

---

## Breast MRI Benchmarks

| Task | Sequences | Metric | Target | Baseline |
|------|-----------|--------|--------|----------|
| Lesion detection | DCE | Sensitivity | > 0.90 | 0.85 |
| BI-RADS classification | DCE | AUC | > 0.88 | 0.82 |
| Enhancement kinetics | DCE | Accuracy | > 0.85 | 0.80 |

---

## Liver MRI Benchmarks

| Task | Sequences | Metric | Target | Baseline |
|------|-----------|--------|--------|----------|
| Focal lesion detection | T2, DWI, DCE | Sensitivity | > 0.90 | 0.85 |
| HCC detection | DCE multiphasic | AUC | > 0.92 | 0.88 |
| Liver volumetry | T1 | Volume MAE | < 5% | 8% |

---

## Signal Quality Metrics

| Metric | Definition | Target |
|--------|------------|--------|
| SNR | mean(signal) / std(noise) | > 20 |
| CNR | |mean1 - mean2| / std(noise) | > 5 |
| Ghosting | Ghost signal / main signal | < 5% |

---

## Fairness Stratification

All benchmarks are evaluated across:

| Dimension | Categories |
|-----------|------------|
| Age | < 40, 40-65, > 65 |
| Sex | Male, Female |
| Field strength | 1.5T, 3T |
| Vendor | Siemens, GE, Philips |
| Institution | Multi-site validation |

---

**Copyright (c) 2025 Skolyn LLC. All rights reserved.**

**SPDX-License-Identifier: EUPL-1.1**
