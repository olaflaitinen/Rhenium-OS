# Mathematical Foundations

## Sampling Theory

### Nyquist Criterion

For signal \(s(t)\) with maximum frequency \(f_{max}\), the sampling frequency must satisfy:

\[
f_s \geq 2 f_{max}
\]

In medical imaging, this relates directly to **spatial resolution** and **aliasing artifacts**.

---

## MRI Reconstruction

### Forward Model

The MRI signal model in k-space:

\[
\mathbf{y} = \mathbf{F}_u \mathbf{S} \mathbf{x} + \boldsymbol{\eta}
\]

Where:

| Symbol | Description |
|--------|-------------|
| \(\mathbf{y}\) | Measured k-space data |
| \(\mathbf{F}_u\) | Undersampled Fourier transform |
| \(\mathbf{S}\) | Coil sensitivity maps |
| \(\mathbf{x}\) | True image |
| \(\boldsymbol{\eta}\) | Noise |

### Compressed Sensing

Reconstruction as optimization:

\[
\hat{\mathbf{x}} = \arg\min_{\mathbf{x}} \frac{1}{2}\|\mathbf{A}\mathbf{x} - \mathbf{y}\|_2^2 + \lambda \|\Psi \mathbf{x}\|_1
\]

---

## CT Reconstruction

### Radon Transform

The Radon transform of image \(f(x, y)\):

\[
R_\theta(s) = \int_{-\infty}^{\infty} f(s\cos\theta - t\sin\theta, s\sin\theta + t\cos\theta) dt
\]

### Filtered Backprojection

Reconstruction via FBP:

\[
f(x, y) = \int_0^{\pi} \left[ R_\theta * h \right](x\cos\theta + y\sin\theta) d\theta
\]

Where \(h\) is the ramp filter in frequency domain:

\[
H(\omega) = |\omega|
\]

---

## Loss Functions

### Dice Loss

For segmentation with predictions \(p\) and ground truth \(g\):

\[
\mathcal{L}_{Dice} = 1 - \frac{2 \sum_i p_i g_i + \epsilon}{\sum_i p_i + \sum_i g_i + \epsilon}
\]

### Perceptual Loss

Using VGG features \(\phi^l\):

\[
\mathcal{L}_{perc} = \sum_l \frac{1}{C_l H_l W_l} \|\phi^l(\hat{x}) - \phi^l(x)\|_2^2
\]

---

## Evaluation Metrics

### Dice Coefficient

\[
DSC = \frac{2|A \cap B|}{|A| + |B|}
\]

### Hausdorff Distance (95th percentile)

\[
HD_{95} = \max\left( h_{95}(A, B), h_{95}(B, A) \right)
\]

Where:

\[
h_{95}(A, B) = \text{percentile}_{95}\left( \min_{b \in B} d(a, b) : a \in A \right)
\]

### PSNR

\[
PSNR = 10 \log_{10}\left( \frac{MAX^2}{MSE} \right)
\]

### SSIM

\[
SSIM(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}
\]

---

## Physics-Informed Neural Networks

PINN loss combines data fidelity with physics:

\[
\mathcal{L}_{total} = \lambda_1 \mathcal{L}_{data} + \lambda_2 \mathcal{L}_{physics} + \lambda_3 \mathcal{L}_{reg}
\]

For MRI:

\[
\mathcal{L}_{physics} = \|M(\mathbf{F}\hat{x} - y)\|_2^2
\]

Where \(M\) is the sampling mask and \(\mathbf{F}\) is the Fourier operator.
