# CT Benchmarks

**
****

---

## Overview

This document defines benchmark targets for CT perception tasks across organ systems within Rhenium OS.

---

## Brain CT Benchmarks

| Task | Protocol | Metric | Target | Baseline |
|------|----------|--------|--------|----------|
| ICH detection | Non-contrast head | AUC | > 0.97 | 0.94 |
| ICH subtype classification | Non-contrast head | Accuracy | > 0.90 | 0.85 |
| Midline shift estimation | Non-contrast head | MAE (mm) | < 2.0 | 3.0 |
| Acute stroke detection | Non-contrast head | Sensitivity | > 0.90 | 0.80 |
| Skull fracture detection | Non-contrast head | Sensitivity | > 0.95 | 0.90 |

---

## Lung CT Benchmarks

| Task | Protocol | Metric | Target | Baseline |
|------|----------|--------|--------|----------|
| Nodule detection | Low-dose screening | Sens@1FP/case | > 0.92 | 0.88 |
| Nodule malignancy | Low-dose screening | AUC | > 0.90 | 0.85 |
| Lung segmentation | Any chest | Dice | > 0.98 | 0.96 |
| Lobe segmentation | Any chest | Dice | > 0.95 | 0.92 |
| PE detection | CTA pulmonary | AUC | > 0.95 | 0.90 |
| Emphysema quantification | Non-contrast chest | Correlation | > 0.95 | 0.90 |

---

## Liver CT Benchmarks

| Task | Protocol | Metric | Target | Baseline |
|------|----------|--------|--------|----------|
| Liver segmentation | Any abdomen | Dice | > 0.96 | 0.94 |
| Lesion detection | Multi-phase | Sensitivity | > 0.90 | 0.85 |
| Lesion segmentation | Multi-phase | Dice | > 0.86 | 0.80 |
| HCC detection | Multi-phase | AUC | > 0.92 | 0.88 |

---

## Cardiac CT Benchmarks

| Task | Protocol | Metric | Target | Baseline |
|------|----------|--------|--------|----------|
| Calcium score | Non-contrast gated | Correlation | > 0.95 | 0.92 |
| CACS risk category | Non-contrast gated | Kappa | > 0.90 | 0.85 |
| Coronary stenosis | CCTA | AUC | > 0.90 | 0.85 |
| Plaque detection | CCTA | Sensitivity | > 0.85 | 0.80 |

---

## Colon CT Benchmarks

| Task | Protocol | Metric | Target | Baseline |
|------|----------|--------|--------|----------|
| Polyp detection (>6mm) | CT colonography | Sensitivity | > 0.95 | 0.90 |
| Polyp detection (>10mm) | CT colonography | Sensitivity | > 0.98 | 0.95 |
| Colon segmentation | CT colonography | Dice | > 0.95 | 0.92 |

---

## Bone CT Benchmarks

| Task | Protocol | Metric | Target | Baseline |
|------|----------|--------|--------|----------|
| Fracture detection | Any bone | Sensitivity | > 0.95 | 0.90 |
| Spine fracture | Spine CT | AUC | > 0.93 | 0.88 |
| Bone segmentation | Any CT | Dice | > 0.95 | 0.92 |

---

## Reconstruction Quality Benchmarks

| Task | Protocol | Metric | Target |
|------|----------|--------|--------|
| Low-dose denoising | Low-dose abdomen | PSNR | > 34 dB |
| Low-dose denoising | Low-dose abdomen | SSIM | > 0.94 |
| Sparse-view recon | 90 views | PSNR | > 30 dB |
| Super-resolution | 5mm to 1mm | PSNR | > 32 dB |

---

## Fairness Stratification

All benchmarks evaluated across:

| Dimension | Categories |
|-----------|------------|
| Body habitus | Normal, Obese |
| Age | < 50, 50-70, > 70 |
| Dose level | Standard, Low-dose |
| Scanner vendor | Multi-vendor |
| Institution | Multi-site |

---

**Copyright (c) 2025 Skolyn LLC. All rights reserved.**

---

## Documentation Update Notes

- Last updated: December 2025.
- Aligned with Rhenium OS disease reasoning layer.
- Revised markdown structure for consistency.
