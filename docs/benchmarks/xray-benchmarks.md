# X-ray Benchmarks

**Last Updated: December 2025**

---

## Key Formulas

**AUC (Area Under Curve)**:
$$\text{AUC} = \int_0^1 \text{TPR}(t) \, d\text{FPR}(t)$$

**Sensitivity (True Positive Rate)**:
$$\text{Sensitivity} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

**Dice Coefficient**:
$$\text{Dice}(A, B) = \frac{2|A \cap B|}{|A| + |B|}$$

**Mean Absolute Error**:
$$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$

---

## Chest X-ray (CXR)

| Task | Metric | Target | Baseline |
|------|--------|--------|----------|
| Triage (Normal/Abnormal) | AUC | > 0.95 | 0.85 |
| Pneumothorax Detection | AUC | > 0.96 | 0.92 |
| Nodule Detection | Sensitivity | > 0.90 | 0.82 |
| Lung Segmentation | Dice | > 0.97 | 0.95 |

---


## Musculoskeletal (MSK)

| Task | Metric | Target | Baseline |
|------|--------|--------|----------|
| Fracture Detection | Sensitivity | > 0.92 | 0.80 |
| Knee OA Grading (KL) | Accuracy | > 0.85 | 0.70 |
| Bone Age Est. | MAE (Months) | < 5.0 | 8.0 |

---

## Mammography

| Task | Metric | Target | Baseline |
|------|--------|--------|----------|
| Lesion Detection | AUC | > 0.94 | 0.89 |
| Malignancy Class | AUC | > 0.90 | 0.80 |
| Density Est. | Quadratic Kappa | > 0.85 | 0.75 |

---

## Dental & Abdomen

| Task | Metric | Target | Baseline |
|------|--------|--------|----------|
| Caries Detection | F1-Score | > 0.88 | 0.80 |
| Free Air Detection | Sensitivity | > 0.95 | 0.90 |
| Tooth Numbering | Accuracy | > 0.98 | 0.95 |

---

## Image Quality Benchmarks

| Task | Metric | Target |
|------|--------|--------|
| Bone Suppression | SSIM (vs Dual Energy) | > 0.92 |
| Denoising | PSNR Gain | > 5 dB |

---

**Copyright (c) 2025 Skolyn LLC. All rights reserved.**
