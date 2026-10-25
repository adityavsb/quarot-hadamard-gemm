# Project Plan: Outlier-Free Matrix Rotations for 4-Bit LLM Quantization (`quarot-hadamard-gemm`)

## Benchmarking Randomized Fast Walsh-Hadamard Rotations for Lossless W4A4 GEMM and KV-Cache Compression

An empirical systems research harness and prototype implementation replicating and benchmarking **randomized orthogonal Hadamard rotations** (based on the *QuaRot* and *SpinQuant* methodologies) to eliminate activation outliers prior to 4-bit post-training quantization (PTQ).

---

## 1. Executive Summary & Problem Formulation

### 1.1 The Activation Outlier Bottleneck
Large language models (LLMs) running at FP16/BF16 precision place severe pressure on GPU memory capacity and memory bandwidth. While model weights can easily be quantized to INT4 using offline methods (e.g., AWQ, GPTQ), **quantizing activations (W4A4) and the KV-cache to 4-bit remains notoriously unstable**.

In modern transformer architectures (such as LLaMA and Mistral):
* Approximately $0.1\%$ of hidden-state feature channels develop massive **activation outliers**—dimensions with values $50\times$ to $100\times$ larger than typical channel magnitudes.
* When applying standard uniform affine or symmetric INT4 quantization:
  $$\Delta = \frac{\max(|X|)}{2^{b-1} - 1}$$
  A single extreme outlier forces the step size $\Delta$ to become excessively coarse, crushing the dynamic range and quantization resolution for the remaining $99.9\%$ of features in that tensor.
* This dynamic range collapse leads to severe perplexity degradation, output gibberish, and token collapse under standard W4A4 and W4KV4 quantization.

### 1.2 The Mathematical Solution: Orthogonal Rotations
Because matrix multiplication is associative, we can insert any orthogonal matrix $\mathbf{H} \in \mathbb{R}^{d \times d}$ ($\mathbf{H}^T \mathbf{H} = \mathbf{I}$) between weights $\mathbf{W}$ and activations $\mathbf{X}$ without altering the mathematical output:
$$\mathbf{Y} = \mathbf{W} \mathbf{X} = (\mathbf{W} \mathbf{H}^T) (\mathbf{H} \mathbf{X}) = \tilde{\mathbf{W}} \tilde{\mathbf{X}}$$

Where:
* $\tilde{\mathbf{W}} = \mathbf{W} \mathbf{H}^T$ is pre-computed offline and quantized to INT4.
* $\tilde{\mathbf{X}} = \mathbf{H} \mathbf{X}$ rotates activations into an orthogonal basis where extreme channel energy is evenly distributed across all $d$ dimensions.
* By using **randomized Walsh-Hadamard matrices** $\mathbf{H} = \frac{1}{\sqrt{d}} \mathbf{H}_d \mathbf{S}$ (where $\mathbf{S} = \text{diag}(\pm 1)$ is a random sign flip vector):
  1. The rotation preserves Euclidean distance and inner products (isometry).
  2. Maximum channel amplitude drops by $\approx \sqrt{d}$, reducing the peak-to-average ratio (PAR) and kurtosis.
  3. The Fast Walsh-Hadamard Transform (FWHT) computes $\mathbf{H} \mathbf{X}$ in $\mathcal{O}(d \log d)$ arithmetic operations using butterfly addition/subtraction circuits, avoiding dense $\mathcal{O}(d^2)$ matrix multiplications.

---

## 2. Technical Objectives & Success Criteria

| Objective | Baseline Target (Without Rotation) | Target with Randomized FWHT Rotation |
| :--- | :--- | :--- |
| **Peak-to-Average Ratio (PAR)** | $> 25.0$ (severe outlier concentration) | $< 4.0$ (diffuse Gaussian-like distribution) |
| **Excess Kurtosis** | $> 80.0$ (heavy-tailed) | $< 3.5$ (near-standard distribution) |
| **Reconstruction Error (Frobenius)** | Relative error $> 0.25$ under INT4 | Relative error $< 0.05$ under INT4 |
| **Signal-to-Quantization-Noise (SQNR)** | $< 12\text{ dB}$ | $> 24\text{ dB}$ ($+12\text{ dB}$ improvement) |
| **Rotation Kernel Overhead** | Dense GEMM: $\mathcal{O}(d^2)$ ($> 15\%$ compute overhead) | Butterfly FWHT: $\mathcal{O}(d \log d)$ ($< 2\%$ latency overhead) |
| **Memory Footprint** | $16\text{ GB}$ (FP16 baseline for 8B model) | $4.8\text{ GB}$ (W4A4 + 4-bit KV-cache, $70\%$ reduction) |

---

## 3. Architecture & Repository Layout

```
quarot-hadamard-gemm/
├── plan.md                            # Comprehensive engineering and research plan
├── README.md                          # Technical report, business case, and benchmarks
├── Makefile                           # Developer automation (install, test, bench)
├── pyproject.toml                     # Project packaging and dependencies
├── LICENSE                            # MIT License with explicit liability limitation
├── docs/
│   └── decisions.md                   # Architecture Decision Records (ADRs)
├── src/
│   ├── __init__.py
│   ├── rotations/
│   │   ├── __init__.py
│   │   ├── hadamard.py                # Sylvester matrix construction & random sign diagonal
│   │   └── fwht.py                    # Vectorized Fast Walsh-Hadamard Transform (O(d log d))
│   ├── quantization/
│   │   ├── __init__.py
│   │   ├── uniform_int4.py            # Symmetric per-tensor & per-channel INT4 quantizers
│   │   └── bit_packing.py             # 4-bit nibble packing utilities
│   ├── kernels/
│   │   ├── __init__.py
│   │   └── rotated_linear.py          # PyTorch module applying W_rot and online FWHT
│   └── analysis/
│       ├── __init__.py
│       ├── outlier_metrics.py         # Kurtosis, peak-to-average ratio (PAR), dynamic range
│       └── error_profiler.py          # Relative Frobenius error and SQNR calculations
├── experiments/
│   ├── run_synthetic_bench.py         # Outlier distribution sweep across dimensions
│   └── run_ptq_comparison.py          # Comparative evaluation: FP16 vs INT4 vs Rotated-INT4
├── tests/
│   ├── test_fwht.py                   # Validates FWHT against dense matrix multiplication
│   ├── test_orthogonality.py          # Validates isometry and H.T @ H = I
│   └── test_quantization.py           # Validates INT4 accuracy recovery on heavy-tailed tensors
└── results/
    └── .gitkeep
```

---

## 4. Implementation Phases

### Phase 1: Orthogonal Transforms & Vectorized FWHT
* Implement recursive Sylvester construction for Hadamard matrices:
  $$\mathbf{H}_{2k} = \begin{pmatrix} \mathbf{H}_k & \mathbf{H}_k \\ \mathbf{H}_k & -\mathbf{H}_k \end{pmatrix}$$
* Implement randomized sign diagonal $\mathbf{S} \in \{-1, +1\}^d$ to guarantee orthogonal mixing across all basis vectors.
* Implement in-place / non-allocating Fast Walsh-Hadamard Transform (FWHT) exploiting power-of-two recursive butterfly additions/subtractions in $\mathcal{O}(d \log d)$.
* Implement block-diagonal padding/chunking for dimensions where $d \neq 2^k$ (e.g., $d = 4096$ vs $d = 2880$).
* **Verification:** Unit tests verifying $\mathbf{H}^T \mathbf{H} = \mathbf{I}$, $\|\mathbf{H} \mathbf{x}\|_2 = \|\mathbf{x}\|_2$ (energy conservation), and $\text{FWHT}(\mathbf{x}) == \mathbf{H} \mathbf{x}$.

### Phase 2: Quantization Engine & Outlier Profiling
* Implement symmetric uniform INT4 quantizer:
  $$q = \text{clip}\left(\left\lfloor \frac{x}{\Delta} \right\rceil, -8, 7\right), \quad \hat{x} = q \cdot \Delta$$
* Support per-tensor, per-channel, and per-group ($G=128$) quantization granularities.
* Build statistical profiler measuring:
  * Maximum absolute amplitude: $\|X\|_\infty$
  * Peak-to-Average Ratio: $\text{PAR} = \frac{\|X\|_\infty}{\frac{1}{d} \sum |X_i|}$
  * Excess Kurtosis: $\kappa = \frac{\mathbb{E}[(X - \mu)^4]}{\sigma^4} - 3$
  * Signal-to-Quantization-Noise Ratio: $\text{SQNR} = 10 \log_{10} \frac{\|X\|_2^2}{\|X - \hat{X}\|_2^2}$

### Phase 3: Rotated Linear Layer & End-to-End GEMM
* Implement `RotatedLinear` PyTorch module:
  * Offline: Pre-multiplies weight $\tilde{\mathbf{W}} = \mathbf{W} \mathbf{H}^T$ and quantizes $\tilde{\mathbf{W}}$ to INT4.
  * Online forward pass:
    1. Apply randomized sign flip $\mathbf{X}' = \mathbf{S} \odot \mathbf{X}$.
    2. Apply in-place Fast Walsh-Hadamard Transform $\tilde{\mathbf{X}} = \text{FWHT}(\mathbf{X}')$.
    3. Quantize $\tilde{\mathbf{X}}$ to INT4.
    4. Compute integer GEMM $\tilde{\mathbf{Y}} = \tilde{\mathbf{W}}_{\text{INT4}} \times \tilde{\mathbf{X}}_{\text{INT4}}$.
* Test reconstruction fidelity against unrotated INT4 baseline on synthetic tensors with injected power-law outliers (Zipfian and log-normal distributions).

### Phase 4: Systems Profiling, Telemetry & Documentation
* Measure runtime latency breakdown: FWHT transform time vs GEMM execution time.
* Verify memory reduction and peak resident set size (RSS).
* Write Architecture Decision Records (`docs/decisions.md`) detailing algorithmic decisions and trade-offs.
* Record raw benchmark outputs into `results/` on Apple Silicon (M4) to guarantee physical provenance.

---

## 5. Engineering Standards & Human Provenance
To ensure authentic engineering craftsmanship:
1. **Document Real Hardware Friction:** Explicitly measure the boundary where FWHT butterfly computation overhead begins to exceed the memory bandwidth savings of INT4.
2. **Machine Provenance:** Record exact hardware environment (`Apple M4`, 16GB unified memory, Python 3.12, PyTorch MPS/CPU flags).
3. **No Fluff:** Clean technical prose focused on signal processing, linear algebra, and systems constraints.
