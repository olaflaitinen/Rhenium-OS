# Ultrasound Benchmarks

**Last Updated: December 2025**

---

## Cardiac Ultrasound (Echocardiography)

| Task | Metric | Target | Baseline |
|------|--------|--------|----------|
| LV Segmentation | Dice | > 0.92 | 0.88 |
| EF Estimation | MAE (%) | < 5.0 | 7.5 |
| View Classification | Accuracy | > 0.98 | 0.95 |
| GLS Estimation | Correlation | > 0.90 | 0.85 |

---

## Obstetric Ultrasound

| Task | Metric | Target | Baseline |
|------|--------|--------|----------|
| Head Circumference | MAE (mm) | < 2.5 | 4.0 |
| Femur Length | MAE (mm) | < 1.5 | 2.5 |
| Plane Detection | Accuracy | > 0.95 | 0.90 |
| EFW Error | % Error | < 8.0% | 12.0% |

---

## Abdominal & Vascular

| Task | Metric | Target | Baseline |
|------|--------|--------|----------|
| Liver Segmentation | Dice | > 0.94 | 0.90 |
| Gallstone Detection | Sensitivity | > 0.95 | 0.88 |
| CIMT Measurement | MAE (mm) | < 0.05 | 0.10 |
| DVT Detection | Sensitivity | > 0.96 | 0.92 |

---

## Small Parts (Thyroid/Breast)

| Task | Metric | Target | Baseline |
|------|--------|--------|----------|
| Thyroid Nodule Det | Sensitivity | > 0.93 | 0.88 |
| Breast Lesion Seg | Dice | > 0.88 | 0.82 |
| BI-RADS Class | Accuracy | > 0.85 | 0.80 |

---

## Image Quality Benchmarks

| Task | Metric | Target |
|------|--------|--------|
| Speckle Reduction | CNR Improvement | > 3.0 dB |
| Edge Preservation | SSIM | > 0.95 |
| Beamforming Speed | FPS (Software) | > 30 |

---

**Copyright (c) 2025 Skolyn LLC. All rights reserved.**
