# Metrics Mathematical Appendix

**Last Updated: December 2025**

---

## Overview

This document provides mathematical definitions for all core metrics used in Rhenium OS evaluation, benchmarking, and quality assessment.

---

## 1. Segmentation Metrics

### 1.1 Dice Coefficient (Sorensen-Dice Index)

The Dice coefficient measures the overlap between predicted segmentation $A$ and ground truth $B$:

$$\text{Dice}(A, B) = \frac{2|A \cap B|}{|A| + |B|}$$

**Range**: $[0, 1]$ where $1$ indicates perfect overlap.

### 1.2 Intersection over Union (Jaccard Index)

IoU measures the ratio of intersection to union:

$$\text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

**Relationship to Dice**:

$$\text{IoU} = \frac{\text{Dice}}{2 - \text{Dice}}$$

### 1.3 Hausdorff Distance

The Hausdorff distance measures the maximum distance between boundaries:

$$\text{HD}(A, B) = \max\left(\sup_{a \in \partial A} \inf_{b \in \partial B} d(a, b), \sup_{b \in \partial B} \inf_{a \in \partial A} d(a, b)\right)$$

**HD95**: The 95th percentile Hausdorff distance, more robust to outliers.

### 1.4 Volumetric Error

Absolute and relative volume error:

$$\text{AVE} = |V_{\text{pred}} - V_{\text{true}}|$$

$$\text{RVE} = \frac{|V_{\text{pred}} - V_{\text{true}}|}{V_{\text{true}}} \times 100\%$$

---

## 2. Detection Metrics

### 2.1 Precision and Recall

$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$

$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

Where:
- **TP**: True Positives
- **FP**: False Positives
- **FN**: False Negatives

### 2.2 F1 Score

Harmonic mean of precision and recall:

$$\text{F1} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \cdot \text{TP}}{2 \cdot \text{TP} + \text{FP} + \text{FN}}$$

### 2.3 Area Under ROC Curve (AUC)

The AUC represents the probability that a classifier ranks a random positive instance higher than a random negative:

$$\text{AUC} = \int_0^1 \text{TPR}(t) \, d\text{FPR}(t)$$

Where:
- $\text{TPR}(t) = \frac{\text{TP}(t)}{\text{TP}(t) + \text{FN}(t)}$ (Sensitivity)
- $\text{FPR}(t) = \frac{\text{FP}(t)}{\text{FP}(t) + \text{TN}(t)}$ (1 - Specificity)

### 2.4 Free-Response ROC (FROC)

For lesion detection, FROC plots sensitivity vs. false positives per image:

$$\text{Sensitivity}(\lambda) = \frac{\text{TP}(\lambda)}{N_{\text{lesions}}}$$

$$\text{FP per image}(\lambda) = \frac{\text{FP}(\lambda)}{N_{\text{images}}}$$

### 2.5 Mean Average Precision (mAP)

For multi-class detection:

$$\text{mAP} = \frac{1}{K} \sum_{k=1}^K \text{AP}_k$$

Where $\text{AP}_k$ is the average precision for class $k$.

---

## 3. Classification Metrics

### 3.1 Accuracy

$$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$

### 3.2 Sensitivity and Specificity

$$\text{Sensitivity} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

$$\text{Specificity} = \frac{\text{TN}}{\text{TN} + \text{FP}}$$

### 3.3 Youden's J Statistic

$$J = \text{Sensitivity} + \text{Specificity} - 1$$

---

## 4. Calibration Metrics

### 4.1 Brier Score

Measures calibration for probabilistic predictions:

$$\text{Brier} = \frac{1}{N} \sum_{i=1}^N (p_i - y_i)^2$$

Where $p_i$ is predicted probability and $y_i \in \{0, 1\}$ is the true label.

**Range**: $[0, 1]$ where lower is better.

### 4.2 Expected Calibration Error (ECE)

Measures calibration across probability bins:

$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

Where:
- $B_m$ is the set of samples in bin $m$
- $\text{acc}(B_m)$ is the accuracy within bin $m$
- $\text{conf}(B_m)$ is the mean confidence within bin $m$

---

## 5. Reconstruction Metrics

### 5.1 Mean Squared Error (MSE)

$$\text{MSE}(x, \hat{x}) = \frac{1}{N} \sum_{i=1}^N (x_i - \hat{x}_i)^2$$

### 5.2 Peak Signal-to-Noise Ratio (PSNR)

$$\text{PSNR} = 10 \log_{10} \left( \frac{\text{MAX}^2}{\text{MSE}} \right) \text{ dB}$$

Where $\text{MAX}$ is the maximum possible pixel value (e.g., 255 for 8-bit).

### 5.3 Structural Similarity Index (SSIM)

$$\text{SSIM}(x, y) = \frac{(2\mu_x \mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$

Where:
- $\mu_x, \mu_y$: local means
- $\sigma_x^2, \sigma_y^2$: local variances
- $\sigma_{xy}$: local covariance
- $C_1, C_2$: stabilization constants

### 5.4 Normalized Mean Squared Error (NMSE)

$$\text{NMSE} = \frac{\|x - \hat{x}\|_2^2}{\|x\|_2^2}$$

---

## 6. Physics-Based Formulas

### 6.1 MRI Signal Equation (Spin Echo)

$$S = \rho \cdot e^{-\text{TE}/T_2} \cdot (1 - e^{-\text{TR}/T_1})$$

Where:
- $\rho$: proton density
- $T_1, T_2$: relaxation times
- $\text{TE}$: echo time
- $\text{TR}$: repetition time

### 6.2 CT Attenuation (Hounsfield Units)

$$\text{HU} = 1000 \times \frac{\mu - \mu_{\text{water}}}{\mu_{\text{water}}}$$

Where $\mu$ is the linear attenuation coefficient.

### 6.3 Beer-Lambert Law (X-ray)

$$I = I_0 \cdot e^{-\int \mu(s) \, ds}$$

### 6.4 Cardiothoracic Ratio (CTR)

$$\text{CTR} = \frac{W_{\text{cardiac}}}{W_{\text{thoracic}}}$$

Normal: CTR < 0.5

---

## 7. Fairness Metrics

### 7.1 Demographic Parity Difference

$$\Delta_{\text{DP}} = |P(\hat{Y}=1|G=0) - P(\hat{Y}=1|G=1)|$$

### 7.2 Equalized Odds Difference

$$\Delta_{\text{EO}} = |P(\hat{Y}=1|Y=1,G=0) - P(\hat{Y}=1|Y=1,G=1)|$$

### 7.3 Disparity Ratio

$$\text{Ratio} = \frac{\text{Metric}_{G=0}}{\text{Metric}_{G=1}}$$

Target: Ratio close to 1.0

---

## 8. Summary Table

| Category | Metric | Formula Reference | Interpretation |
|----------|--------|-------------------|----------------|
| Segmentation | Dice | Section 1.1 | Higher is better (1 = perfect) |
| Segmentation | IoU | Section 1.2 | Higher is better |
| Segmentation | HD95 | Section 1.3 | Lower is better (mm) |
| Detection | AUC | Section 2.3 | Higher is better (1 = perfect) |
| Detection | F1 | Section 2.2 | Higher is better |
| Classification | Accuracy | Section 3.1 | Higher is better |
| Calibration | Brier | Section 4.1 | Lower is better |
| Calibration | ECE | Section 4.2 | Lower is better |
| Reconstruction | PSNR | Section 5.2 | Higher is better (dB) |
| Reconstruction | SSIM | Section 5.3 | Higher is better (1 = identical) |

---

**Copyright (c) 2025 Skolyn LLC. All rights reserved.**

**SPDX-License-Identifier: EUPL-1.1**
