# Fictional case study — "Apex Health Intelligence"

> **Everything in this file is invented.** No hospital, deployment, accuracy figure, latency figure or SLA below is real. It exists only to motivate the memory arithmetic. The repository's measured results are in the README; this software is not a medical device and gives no medical advice.

---

### 1. The Operational Context & Regulatory Envelope
**Apex Health Intelligence** operates an on-premise clinical copilot deployed across 14 hospital networks. The system ingests electronic health records (EHR), multi-page pathology narratives, and lab panels to generate real-time clinical discharge summaries and differential diagnosis suggestions during physician rounds.

Due to strict HIPAA patient privacy mandates, hospital air-gap policies, and sovereign data governance laws, **zero patient data may leave hospital premises**. External multi-tenant cloud APIs (OpenAI, Anthropic) are strictly prohibited. The system must run on hospital-owned, on-premise compute nodes deployed inside local server closets.

### 2. The Production Bottleneck: Memory & Concurrency Limits
Hospital server closets are constrained by space, power, and procurement budgets to standard edge workstations equipped with **single 24GB VRAM GPUs** (e.g., NVIDIA RTX 4090 / A10G / L4).

When running an open-weights 8-billion parameter foundation model (such as LLaMA-3-8B or Mistral-7B) in standard 16-bit floating point (FP16 / BF16):
* **Model Static Footprint:** $8.03 \times 10^9 \times 2\text{ bytes} \approx \mathbf{16.06\text{ GB}}$ VRAM just to load weights into memory.
* **Remaining VRAM for Inference Buffers:** $24.0\text{ GB} - 16.06\text{ GB} = \mathbf{7.94\text{ GB}}$.
* **The Long-Context KV-Cache Crisis:** Clinical summaries frequently involve historical chart ingestion spanning 8,192 to 16,384 tokens. For an 8B model with grouped-query attention (32 layers, 8 KV heads, head dimension 128 — LLaMA-3-8B and Mistral-7B both use 8 KV heads), the KV-cache memory requirement scales linearly with context length $L$ and batch size $B$:

  $$
  \text{KV-Cache (bytes)} = 2 \times B \times L \times N_{\text{layers}} \times (N_{\text{kv heads}} \cdot d_{\text{head}}) \times \text{bytes per element}
  $$

  At batch size $B=8$ and context length $L=16{,}384$, the FP16 KV-cache requires:

  $$
  2 \times 8 \times 16384 \times 32 \times 1024 \times 2 \approx \mathbf{17.18\text{ GB}}
  $$
* **The Consequence:** Total memory required is $16.06\text{ GB} + 17.18\text{ GB} = \mathbf{33.24\text{ GB}} \gg 24\text{ GB}$. The system suffers Out-Of-Memory (OOM) crashes, forcing the hospital to throttle concurrency to $B \le 3$ ($16.06 + 3 \times 2.15 = 22.5\text{ GB} < 24$). Doctors experience queuing delays exceeding 90 seconds per patient record, violating the clinical 15-second SLA.

### 3. The Failed Attempt: Standard INT4 Post-Training Quantization
To compress both weights and the KV-cache into the 24GB envelope, engineering applied standard uniform 4-bit post-training quantization (PTQ):
* **Weight Quantization (W4):** Compressed weights to $\sim 4.2\text{ GB}$.
* **Activation & KV-Cache Quantization (A4 & KV4):** Compressed KV-cache to $4.3\text{ GB}$, theoretically allowing $B=8$ concurrency within $12.5\text{ GB}$ total VRAM.

**The Catastrophic Failure Mode:**
Modern transformer architectures naturally develop **systematic activation outliers** in deep residual layers. Approximately $0.1\%$ of hidden-state channels exhibit peak amplitudes one to two orders of magnitude above the bulk (Dettmers et al. [3] report values up to ≈60 at 6.7B+ parameters), while $99.9\%$ of channel features stay tightly clustered within $[-1.5, +1.5]$:

$$
\Delta = \frac{\max(|X|)}{2^{b-1} - 1} = \frac{60}{7} \approx 8.57
$$

Under this quantization step size ($\Delta = 8.57$), every normal feature value between $-4.3$ and $+4.3$ rounds to **zero**. Fine-grained diagnostic signals (such as medication dosages and micro-differential symptoms) are completely erased.

* **Clinical Outcome:** Clinical extraction accuracy collapsed from $94.2\%$ to $21.4\%$. The quantized model output repetitive medical gibberish and hallucinated drug interactions.

---

### 4. The Engineering Solution: Randomized Orthogonal Matrix Rotations
Instead of undertaking an unsustainable multi-million dollar hardware upgrade to multi-GPU clusters, this project implements **randomized Fast Walsh-Hadamard Transform (FWHT) rotations** (QuaRot methodology):

```
Standard PTQ (Outliers destroy precision):
[X: Outliers up to ±60] ──> [Uniform INT4 Quantizer] ──> [Values < |4.3| round to 0] (Collapse)

Outlier-Free Rotated PTQ (Energy dispersed):
[X: Outliers up to ±60] ──> [FWHT Rotation: H @ X] ──> [X_rot: Max peak < ±9.2] ──> [INT4 Quantizer] ──> [Clean Reconstruction]
```

1. **Orthogonal Invariance:** By inserting orthogonal Hadamard matrices $\mathbf{H} \in \mathbb{R}^{d \times d}$ ($\mathbf{H}^T \mathbf{H} = \mathbf{I}$) between weights $\mathbf{W}$ and activations $\mathbf{X}$:

   $$
   \mathbf{Y} = \mathbf{W} \mathbf{X} = (\mathbf{W} \mathbf{H}^T) (\mathbf{H} \mathbf{X}) = \tilde{\mathbf{W}} \tilde{\mathbf{X}}
   $$
   * $\tilde{\mathbf{W}} = \mathbf{W} \mathbf{H}^T$ is computed once offline and quantized to INT4.
   * $\tilde{\mathbf{X}} = \mathbf{H} \mathbf{X}$ is computed on-the-fly during inference.
2. **Energy dispersion:** The randomised Hadamard transform spreads each coordinate's energy evenly over all $d$ coordinates. The energy of outlier channels is distributed uniformly across all $d$ dimensions.
   * Peak activation amplitude drops from $85.0$ to $< 9.2$ ($9.2\times$ reduction).
   * Peak-to-Average Ratio (PAR) drops from $28.4$ to $3.1$.
   * Excess kurtosis drops from $112.5$ to $3.2$ (approaching Gaussian behavior).
3. **Zero Dense GEMM Overhead via Butterfly FWHT:** Using the Fast Walsh-Hadamard Transform, $\mathbf{H} \mathbf{X}$ is computed in $\mathcal{O}(d \log d)$ butterfly additions/subtractions instead of a dense $\mathcal{O}(d^2)$ matrix multiplication.
   * Transform runtime overhead is under **$1.8\%$** of token generation latency.
   * Total model + KV-cache footprint drops from $33.2\text{ GB}$ to **$9.6\text{ GB}$**, comfortably supporting $B=8$ clinical concurrency on a single 24GB workstation with zero clinical accuracy loss.
