# Architecture Decision Records (ADRs)

This document captures the key design choices, theoretical trade-offs, and design justifications made in `quarot-hadamard-gemm`.

---

## ADR-001: Fast Walsh-Hadamard Transform (FWHT) vs. Dense Haar/Givens Orthogonal Rotations

* **Status:** Accepted
* **Date:** 2026-10-25
* **Decider:** Aditya Bhadoria

### Context
To eliminate activation outliers prior to 4-bit quantization, activations $\mathbf{X}$ must be rotated into an orthogonal basis: $\tilde{\mathbf{X}} = \mathbf{H} \mathbf{X}$.
Any orthogonal matrix $\mathbf{H} \in \mathbb{R}^{d \times d}$ ($\mathbf{H}^T \mathbf{H} = \mathbf{I}$) preserves inner products and Euclidean norms. Candidates include:
1. Dense Haar-random orthogonal matrices (sampled via QR decomposition of Gaussian matrices).
2. Givens rotations (sequences of 2D coordinate plane rotations).
3. Randomized Walsh-Hadamard matrices ($\mathbf{H} = \frac{1}{\sqrt{d}} \mathbf{H}_d \mathbf{S}$).

### Decision
We choose **Randomized Walsh-Hadamard matrices evaluated via the Fast Walsh-Hadamard Transform (FWHT)**.

### Rationale & Trade-offs
* **Computational Complexity:** A dense Haar matrix requires $\mathcal{O}(d^2)$ floating-point operations ($d^2 = 16.7\text{M}$ for $d=4096$).
* **Butterfly Efficiency (arithmetic):** FWHT needs $d \log_2 d$ additions/subtractions and one scalar normalisation ($49{,}152$ + $4{,}096$ ops for $d=4096$) versus $16.8\text{M}$ multiply-adds for a dense rotation — $341\times$ fewer arithmetic operations.
* **Measured wall-clock (this implementation, eager PyTorch):** the FWHT allocates three tensors per stage and is memory-bound. On Apple M4, rotating a $128 \times 4096$ activation costs $2.43\text{ ms}$ against a $4.17\text{ ms}$ float32 GEMM ($58.3\%$ overhead on CPU; $89.8\%$ overhead on MPS) and is $1.81\times$ faster than a dense `X @ H` on CPU ($1.06\times$ on MPS). The $<2\%$ overhead reported by QuaRot requires a fused CUDA/Metal kernel; it is **not** achieved here and is recorded as an open item.
* **Equal Outlier Diffusion:** Because Hadamard matrices consist of elements with uniform absolute magnitude $|H_{ij}| = 1/\sqrt{d}$, multiplying by $\mathbf{H}$ uniformly distributes energy across all coordinates. Combining with a random diagonal sign matrix $\mathbf{S} = \text{diag}(\pm 1)$ breaks symmetry and prevents worst-case alignment with coordinate-aligned outliers.

---

## ADR-002: Symmetric Uniform INT4 Quantization vs. Asymmetric Zero-Point Quantization

* **Status:** Accepted
* **Date:** 2026-10-25
* **Decider:** Aditya Bhadoria

### Context
Quantizing rotated activations and weights to 4-bit can use either:
1. **Asymmetric Affine Quantization:** $q = \text{round}(x / \Delta) + z$, mapping $[\min(X), \max(X)]$ to $[0, 15]$.
2. **Symmetric Uniform Quantization:** $q = \text{clip}(\text{round}(x / \Delta), -8, 7)$, with zero-point $z = 0$.

### Decision
We standardize on **Symmetric Uniform INT4 Quantization** for both weights and activations.

### Rationale & Trade-offs
* **Hardware Integer GEMM Alignment:** Asymmetric quantization introduces a non-zero integer offset $z$. In matrix multiplication $\mathbf{W} \mathbf{X}$, non-zero offsets produce cross-terms: $\sum_k (w_k - z_w)(x_k - z_x) = \sum w_k x_k - z_x \sum w_k - z_w \sum x_k + K z_w z_x$. These terms require runtime reduction across rows and columns, adding memory stalls.
* **Symmetric Simplicity:** Symmetric quantization maps real-value zero to integer zero ($z=0$). Hardware tensor cores (NVIDIA INT4, Apple Metal MPSGraph, ARM NEON) natively execute dot products of signed integers without correction terms.
* **Post-Rotation Symmetry:** Because randomized Hadamard rotation centers the activation distribution and drives skewness close to zero, asymmetric zero-point offsets provide negligible benefit (not measured in this repository; QuaRot [1] and SpinQuant [2] likewise use symmetric quantisation for activations) while adding runtime complexity.

---

## ADR-003: Block-Diagonal Decomposition for Non-Power-of-Two Hidden Dimensions

* **Status:** Accepted
* **Date:** 2026-10-25
* **Decider:** Aditya Bhadoria

### Context
The standard recursive Fast Walsh-Hadamard Transform strictly requires the vector dimension to be an exact power of two: $d = 2^k$.
While primary attention dimensions are powers of two ($d_{\text{model}} \in \{2048, 4096, 8192\}$), MLP intermediate feed-forward projections are frequently non-powers-of-two (e.g., LLaMA-3 $d_{\text{ffn}} = 14,336 = 7 \times 2048$, Mistral-7B $d_{\text{ffn}} = 14,336$).

### Decision
We implement a **Block-Diagonal Hadamard Decomposition** for non-power-of-two dimensions:

$$
\mathbf{H}_{\text{block}} = \begin{pmatrix} \mathbf{H}_{d_1} & \mathbf{0} & \dots \\ \mathbf{0} & \mathbf{H}_{d_2} & \dots \\ \vdots & \vdots & \ddots \end{pmatrix}
$$

Where $d = \sum_j d_j$, with each $d_j = 2^{k_j}$ (e.g., $14,336 = 7 \times 2048$).

### Rationale & Trade-offs
* **Zero Padding Overhead:** Zero-padding $d=14,336$ up to the next power of two ($d=16,384$) wastes $2,048$ channels ($+14.3\%$ memory and compute bloat).
* **Block Orthogonality:** A block-diagonal matrix composed of orthogonal blocks is itself strictly orthogonal ($\mathbf{H}_{\text{block}}^T \mathbf{H}_{\text{block}} = \mathbf{I}$).
* **Sufficient Outlier Dispersion:** In transformer feed-forward layers, outlier channels are distributed throughout the vector. A block size of $d_j = 2048$ provides ample dimensional diffusion ($\approx \sqrt{2048} \approx 45.2\times$ theoretical peak reduction), completely eliminating outlier saturation without padding.
