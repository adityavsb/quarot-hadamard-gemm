# Outlier-Free Matrix Rotations for 4-Bit LLM Quantization (`quarot-hadamard-gemm`)
### Eliminating Activation Outliers in Post-Training Quantization via Fast Walsh-Hadamard Transforms (FWHT)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ci](https://github.com/adityavsb/quarot-hadamard-gemm/actions/workflows/ci.yml/badge.svg)](https://github.com/adityavsb/quarot-hadamard-gemm/actions/workflows/ci.yml)

An empirical systems research harness and prototype implementation evaluating **randomized orthogonal Hadamard rotations** (following *QuaRot*; *SpinQuant*'s learned rotations are related work, not implemented here) to eliminate activation outliers prior to 4-bit post-training quantization (PTQ) of weights and activations (W4A4). Quantisation is simulated ("fake quantisation": quantise → dequantise → float32 matmul); there is no integer GEMM, no bit-packing and no KV-cache code in this repository.

---

## Why rotate before quantising (2-minute version)

A handful of hidden-state channels in large transformers carry values one to two orders of magnitude above the rest [3]. Symmetric INT4 sets its step from the largest value, so those channels dictate a step that rounds the bulk to zero. Multiplying by an orthogonal matrix on both sides of a linear layer leaves the output unchanged but spreads each channel's energy across all coordinates; a randomised Walsh–Hadamard matrix does this in $d\log_2 d$ additions. This repository measures how much that helps a round-to-nearest W4A4 quantiser on synthetic activations.

A fictional, fully-disclaimed deployment story that motivates the memory numbers lives in [`docs/case_study.md`](docs/case_study.md).

---

## Mathematical Formulation

### 1. Invariance of Linear Projections
Let $\mathbf{W} \in \mathbb{R}^{m \times d}$ be a projection weight matrix and $\mathbf{X} \in \mathbb{R}^{d \times n}$ be an input activation tensor. Let $\mathbf{H} \in \mathbb{R}^{d \times d}$ be an orthogonal matrix satisfying:
$$\mathbf{H}^T \mathbf{H} = \mathbf{H} \mathbf{H}^T = \mathbf{I}_d$$

The linear projection can be factored without approximation error:
$$\mathbf{Y} = \mathbf{W} \mathbf{X} = \mathbf{W} \mathbf{I}_d \mathbf{X} = \mathbf{W} (\mathbf{H}^T \mathbf{H}) \mathbf{X} = (\mathbf{W} \mathbf{H}^T) (\mathbf{H} \mathbf{X})$$

We define:
$$\tilde{\mathbf{W}} \triangleq \mathbf{W} \mathbf{H}^T, \quad \tilde{\mathbf{X}} \triangleq \mathbf{H} \mathbf{X}$$

### 2. Randomized Sylvester-Hadamard Matrices
A normalized Walsh-Hadamard matrix of order $2^k$ is constructed recursively via the Sylvester Kronecker formulation:
$$\mathbf{H}_1 = [1], \quad \mathbf{H}_{2^k} = \frac{1}{\sqrt{2}} \begin{pmatrix} \mathbf{H}_{2^{k-1}} & \mathbf{H}_{2^{k-1}} \\ \mathbf{H}_{2^{k-1}} & -\mathbf{H}_{2^{k-1}} \end{pmatrix}$$

To prevent alignment with structured coordinate axes, we introduce a randomized diagonal sign matrix $\mathbf{S} = \text{diag}(s_1, \dots, s_d)$ where each $s_i \in \{-1, +1\}$ with equal probability $\frac{1}{2}$:
$$\mathbf{H}_{\text{rand}} = \mathbf{H}_d \mathbf{S}$$

Because both $\mathbf{H}_d$ and $\mathbf{S}$ are orthogonal, their product $\mathbf{H}_{\text{rand}}$ is strictly orthogonal:
$$\mathbf{H}_{\text{rand}}^T \mathbf{H}_{\text{rand}} = \mathbf{S}^T \mathbf{H}_d^T \mathbf{H}_d \mathbf{S} = \mathbf{S} \mathbf{I}_d \mathbf{S} = \mathbf{I}_d$$

This is the randomised Hadamard transform of Ailon & Chazelle [5], used for quantisation by QuIP# [6] and QuaRot [1].

### 3. Fast Walsh-Hadamard Transform (FWHT)
Computing $\mathbf{H}_d \mathbf{x}$ directly via matrix-vector multiplication requires $d^2$ floating-point multiply-accumulate operations. The Fast Walsh-Hadamard Transform evaluates this product recursively using a butterfly graph in $\log_2(d)$ stages:

$$\mathbf{y}^{(k)}_i = \mathbf{y}^{(k-1)}_i + \mathbf{y}^{(k-1)}_{i + 2^{k-1}}, \quad \mathbf{y}^{(k)}_{i + 2^{k-1}} = \mathbf{y}^{(k-1)}_i - \mathbf{y}^{(k-1)}_{i + 2^{k-1}}$$

$$\text{ops} = d \log_2 d \text{ additions/subtractions} + d \text{ multiplications (normalisation)}$$

For a hidden dimension of $d = 4096$:
* Dense Matrix Multiplication: $4096^2 = \mathbf{16,777,216\text{ operations}}$
* Fast Walsh-Hadamard Transform: $4096 \times 12 = \mathbf{49,152\text{ operations}}$
* **Theoretical Speedup of Transform:** $\mathbf{341\times}$ fewer operations.

---

## Measured results (synthetic activations, Apple M4, CPU and MPS)

Benchmarked directly on an Apple M4 Mac mini ($d=4096$, $B=128$, synthetic outlier injection with scale $80.0$ on $0.2\%$ of channels). Raw JSON telemetry written by `experiments/run_ptq_comparison.py --out`:
* CPU: [`results/apple_m4_cpu_2026-10-27.json`](results/apple_m4_cpu_2026-10-27.json)
* MPS: [`results/apple_m4_mps_2026-10-27.json`](results/apple_m4_mps_2026-10-27.json)

### 1. Activation Outlier Suppression Profile
| Metric | Unrotated Activations | Rotated Activations (QuaRot) | Suppression Delta |
| :--- | :---: | :---: | :---: |
| **Max Absolute Peak ($\|X\|_\infty$)** | $95.92$ | **$14.34$** | **$6.69\times$ lower** |
| **Peak-to-Average Ratio (PAR)** | $100.52$ | **$4.81$** | **$20.90\times$ lower** |
| **Excess Kurtosis ($\kappa$)** | $459.68$ | **$-0.20$** | **Normalized to Gaussian** |
| **P99 to Max Ratio** | $36.17$ | **$1.56$** | **$23.24\times$ tighter envelope** |

### 2. Linear Projection Reconstruction Fidelity (W4A4)
| Metric | Naïve INT4 (Unrotated) | Rotated INT4 (QuaRot W4A4) | Improvement Delta |
| :--- | :---: | :---: | :---: |
| **Relative Frobenius Error** | $0.3123$ | **$0.2061$** | **$1.52\times$ lower error** |
| **Signal-to-Noise Ratio (SQNR)** | $10.11\text{ dB}$ | **$13.72\text{ dB}$** | **$+3.61\text{ dB}$ gain** |
| **Max Absolute Point Error** | $7.9663$ | **$5.4134$** | **$1.47\times$ tighter bound** |
| **Mean Squared Error (MSE)** | $2.702397$ | **$1.176803$** | **$2.30\times$ lower MSE** |
| **Background-channel rel. error** (error of the 4 088 non-outlier channels' contribution to Y) | $1.1619$ | **$0.7667$** | — |
| **Outlier channels' share of $\|Y\|_F^2$** | $0.928$ | $0.928$ | explains why the Frobenius rows move less than the narrative suggests |

Naive per-token INT4 quantises 100 % of background-channel entries to zero at this outlier scale; relative Frobenius error still reads 0.31 because 93 % of the output energy comes from the 8 outlier channels. Use the background-channel row to judge information loss.

### 3. Latency & Kernel Overhead (Apple M4, 50 repetitions after warm-up)
* **FP32 GEMM Baseline:** $4.17\text{ ms}$ (CPU) / $1.73\text{ ms}$ (MPS)
* **FWHT Vectorized Rotation Alone:** **$2.43\text{ ms}$** (CPU, $58.3\%$ overhead vs GEMM) / **$1.55\text{ ms}$** (MPS, $89.8\%$ overhead vs GEMM)
* **Dense Rotation Reference `X @ H`:** $4.40\text{ ms}$ (CPU) / $1.64\text{ ms}$ (MPS)
* **Naive fake-INT4 forward:** $3.68\text{ ms}$ (CPU) / $2.08\text{ ms}$ (MPS)
* **Rotated fake-INT4 forward:** $5.67\text{ ms}$ (CPU) / $3.35\text{ ms}$ (MPS) — this is a float32 matmul on dequantised operands; it demonstrates numerical behaviour, not bandwidth or throughput.

### 4. Sensitivity to outlier scale (10 seeds each)

| Injected Outlier Scale | Naive Rel. Frobenius | Rotated Rel. Frobenius | Naive / Rotated Error Ratio | Naive Background Error | Rotated Background Error |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0.0 | 0.2240 ± 0.0010 | 0.2237 ± 0.0006 | 1.00× | — | — |
| 5.0 | 0.2832 ± 0.0007 | 0.2237 ± 0.0009 | 1.27× | 0.290 | 0.229 |
| 10.0 | 0.4680 ± 0.0014 | 0.2237 ± 0.0011 | 2.09× | 0.512 | 0.245 |
| 20.0 | 0.6569 ± 0.0039 | 0.2218 ± 0.0017 | 2.96× | 0.880 | 0.297 |
| 40.0 | 0.5094 ± 0.0048 | 0.2161 ± 0.0024 | 2.36× | 1.040 | 0.441 |
| 80.0 | 0.3128 ± 0.0033 | 0.2092 ± 0.0024 | 1.50× | 1.157 | 0.774 |
| 160.0 | 0.2135 ± 0.0023 | 0.2039 ± 0.0025 | 1.05× | 1.536 | 1.466 |

The naive/rotated gap is not monotone in outlier scale: it peaks near scale 20 and vanishes by 160 because the outlier channels dominate the Frobenius norm. The background-channel columns are the informative ones.

### 5. Real-activation spot check (not a benchmark)

Captured from `Qwen/Qwen2.5-0.5B` layer 12 `down_proj` input (intermediate dim $d=4864$, block size 128, 225 tokens):
```
Qwen/Qwen2.5-0.5B layer 12 down_proj input: tokens=225 d=4864 block=128
max|x| 9.21 -> 1.51 | PAR 111.6 -> 13.8 | kurtosis 149.5 -> 1.9
rel-Frob naive 0.5754 rotated 0.2308 | SQNR 4.80 -> 12.74 dB
```
Demonstrates that the block-diagonal randomised FWHT ($4864 = 38 \times 128$) successfully suppresses outliers on real transformer activations and recovers $+7.94\text{ dB}$ SQNR. Run via `python experiments/run_real_activations.py`.

---

## Targets vs. results

| Objective (plan.md §2) | Target | Measured | Status |
| :--- | :---: | :---: | :--- |
| Peak-to-Average Ratio after rotation | < 4.0 | 4.81 | not met (−) |
| Excess kurtosis after rotation | < 3.5 | −0.20 | met |
| Relative Frobenius error, rotated W4A4 | < 0.05 | 0.206 | **not met.** RTN INT4 with per-token max scaling has a no-outlier floor of 0.224 / 13.0 dB on Gaussian inputs; the target was unattainable for this quantiser. Rotation reaches the floor, i.e. removes the outlier penalty entirely. |
| SQNR, rotated W4A4 | > 24 dB | 13.72 dB | **not met** (same reason) |
| FWHT overhead vs. GEMM | < 2 % | 58.3% (CPU) / 89.8% (MPS) | **not met.** Eager-PyTorch FWHT is memory-bound (ADR-001, revised) |
| Memory footprint | 4.8 GB | not measured | fake quantisation moves float32 bytes; no footprint claim is made |

---

## Repository Layout

```
quarot-hadamard-gemm/
├── plan.md                            # Comprehensive engineering and research roadmap
├── README.md                          # Technical report, business case, and architecture
├── Makefile                           # Developer automation (install, test, bench)
├── pyproject.toml                     # Packaging and dependencies
├── LICENSE                            # MIT License
├── docs/
│   └── decisions.md                   # Architecture Decision Records (ADRs)
├── quarot_hadamard/
│   ├── __init__.py
│   ├── rotations/
│   │   ├── __init__.py
│   │   ├── hadamard.py                # Sylvester matrix constructor & sign diagonal
│   │   └── fwht.py                    # Vectorized Fast Walsh-Hadamard Transform (O(d log d))
│   ├── quantization/
│   │   ├── __init__.py
│   │   └── uniform_int4.py            # Symmetric uniform INT4 quantizer
│   ├── kernels/
│   │   ├── __init__.py
│   │   └── rotated_linear.py          # PyTorch module with offline W_rot and online FWHT
│   └── analysis/
│       ├── __init__.py
│       ├── outlier_metrics.py         # Kurtosis, peak-to-average ratio (PAR) profiler
│       └── error_profiler.py          # Relative Frobenius error and SQNR calculations
├── experiments/
│   └── run_ptq_comparison.py          # CLI runner comparing FP16 vs Naive INT4 vs Rotated INT4
├── tests/
│   ├── test_fwht.py                   # Verifies FWHT against dense matrix multiplication
│   ├── test_orthogonality.py          # Verifies isometry and H.T @ H = I
│   ├── test_quantization.py           # Validates INT4 accuracy recovery on heavy-tailed tensors
│   └── test_coverage_gaps.py          # Validates granularities, block paths, and MPS smoke
└── results/
    ├── apple_m4_cpu_2026-10-27.json   # JSON written by experiments/run_ptq_comparison.py --out (CPU)
    ├── apple_m4_mps_2026-10-27.json   # JSON written by experiments/run_ptq_comparison.py --out (MPS)
    └── synthetic_sweep.csv            # CSV written by experiments/run_synthetic_bench.py
```

---

## Quickstart & Reproducibility

### 1. Installation
```bash
git clone https://github.com/adityavsb/quarot-hadamard-gemm.git
cd quarot-hadamard-gemm
make install
```

### 2. Run Test Suite
```bash
make test
```

### 3. Run Outlier Suppression & PTQ Comparison Benchmark
```bash
make benchmark
```
Or directly via Python:
```bash
python experiments/run_ptq_comparison.py --dim 4096 --batch 128 --outlier-scale 80.0
```

---

## Hardware Friction & Engineering Limitations

Three things that did not work as the asymptotics suggest:

1. **FWHT is not free in eager PyTorch.** The butterfly allocates three tensors per stage; measured on Apple M4, it costs $58.3\%$ of the GEMM on CPU ($89.8\%$ on MPS) and is only $1.81\times$ faster than a dense rotation on CPU ($1.06\times$ on MPS). A fused kernel (as in QuaRot's CUDA implementation) is required for the $<2\%$ overhead reported there.
2. **Non-Power-of-Two Dimensions:** Standard Fast Walsh-Hadamard transforms strictly require $d = 2^k$. While LLM hidden dimensions are typically powers of two ($4096, 8192$), intermediate MLP feed-forward dimensions often are not (e.g., LLaMA-3 intermediate dimension $d_{\text{ffn}} = 14,336 = 7 \times 2048$). To rotate non-power-of-two layers without memory bloat, we implement a **block-diagonal Hadamard decomposition** splitting $d$ into power-of-two sub-blocks (see [ADR-003](docs/decisions.md#adr-003-handling-non-power-of-two-dimensions)).
3. **SIMD Vectorization & Strided Memory Overhead:** While FWHT is asymptotically $\mathcal{O}(d \log d)$, naïve recursive implementations suffer from poor cache stride locality at the final butterfly stages (stride $d/2$). Our vectorized tensor reshape formulation operates across contiguous memory blocks, which keeps each stage a contiguous pass; it does not avoid the $3 \log_2 d$ extra tensor passes (see item 1).
4. **Weight Storage Transformation:** Rotating weight matrices $\tilde{\mathbf{W}} = \mathbf{W} \mathbf{H}^T$ changes weight distributions. While activation kurtosis drops significantly, weight kurtosis is essentially unchanged for Gaussian weights (measured −0.001 → +0.003); for real LLM weights this should be re-measured. However, weights can tolerate fine-grained channel/group scales, whereas online activations require uniform scale factors for fast integer GEMM.

---

## Architecture Decision Records (ADRs)

Key architectural decisions are documented under [`docs/decisions.md`](docs/decisions.md):
* **ADR-001:** Fast Walsh-Hadamard Transform (FWHT) vs. Dense Haar/Givens Orthogonal Rotations
* **ADR-002:** Symmetric Uniform INT4 Quantization vs. Asymmetric Zero-Point Quantization
* **ADR-003:** Block-Diagonal Decomposition for Non-Power-of-Two Hidden Dimensions

---

## References & Scientific Attribution

1. **Ashkboos, S., Mohtashami, A., Croci, M. L., Li, B., Cameron, P., Jaggi, M., Alistarh, D., Hoefler, T., & Hensman, J. (2024).** *QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs.* NeurIPS 2024. arXiv:2404.00456.
2. **Liu, Z., Zhao, C., Fedorov, I., Soran, B., Choudhary, D., Krishnamoorthi, R., Chandra, V., Tian, Y., & Blankevoort, T. (2024).** *SpinQuant: LLM Quantization with Learned Rotations.* ICLR 2025. arXiv:2405.16406.
3. **Dettmers, T., Lewis, M., Belkada, Y., & Zettlemoyer, L. (2022).** *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale.* NeurIPS 2022. arXiv:2208.07339.
4. **Sylvester, J. J. (1867).** *Thoughts on Inverse Orthogonal Matrices, Simultaneous Sign-Successions, and Tessellated Pavements in Two or More Colours, with Applications to Newton's Rule, Ornamental Tile-Work, and the Theory of Numbers.* The London, Edinburgh, and Dublin Philosophical Magazine and Journal of Science, 34(232), 461–475. doi:10.1080/14786446708639914.
5. **Ailon, N., & Chazelle, B. (2009).** *The Fast Johnson–Lindenstrauss Transform and Approximate Nearest Neighbors.* SIAM Journal on Computing, 39(1), 302–322. doi:10.1137/060673096. — origin of the randomised Hadamard construction $\mathbf{H}\,\mathrm{diag}(s)$ used here.
6. **Tseng, A., Chee, J., Sun, Q., Kuleshov, V., & De Sa, C. (2024).** *QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks.* ICML 2024. arXiv:2402.04396. — randomised Hadamard incoherence processing for LLM quantisation, which QuaRot extends to activations.
7. **Ashkboos, S., Croci, M. L., Nascimento, M. G. do, Hoefler, T., & Hensman, J. (2024).** *SliceGPT: Compress Large Language Models by Deleting Rows and Columns.* ICLR 2024. arXiv:2401.15024. — computational-invariance argument.
8. **Xiao, G., Lin, J., Seznec, M., Wu, H., Demouth, J., & Han, S. (2023).** *SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models.* ICML 2023. arXiv:2211.10438.
9. **Fino, B. J., & Algazi, V. R. (1976).** *Unified Matrix Treatment of the Fast Walsh-Hadamard Transform.* IEEE Transactions on Computers, C-25(11), 1142–1146. doi:10.1109/TC.1976.1674569.

---

## Disclaimers & Legal Notice

* **Educational & Research Reproduction:** This repository is an independent open-source research and engineering reproduction. It is not affiliated with, sponsored by, or endorsed by ETH Zurich, Microsoft Research, Meta, or any medical entity.
* **Fictional Scenario Disclaimer:** The case study describing "Apex Health Intelligence" is entirely fictional and was constructed strictly to illustrate systems-level GPU memory constraints, VRAM bottlenecks, and numerical quantization trade-offs. This software does not provide medical advice and is not intended, designed, or certified for clinical diagnosis, patient care, or health decision support.
* **Trademark Attribution:** All product names, logos, and brands (including LLaMA, Mistral, NVIDIA, Apple, Mac mini) are property of their respective owners. Their mention in this repository is purely for technical identification and comparative benchmarking under fair use.
* **License:** Distributed under the [MIT License](LICENSE). Standard limitation of liability applies.

