# Illustrative Case Study: Air-Gapped Enterprise Document Intelligence

> **Illustrative Systems Scenario.** The operational setting and metrics below represent a hypothetical engineering model constructed to illustrate systems-level GPU memory constraints, VRAM bottlenecks, and numerical quantization trade-offs in air-gapped environments. The repository's verified benchmark measurements on physical silicon are documented in the main [`README.md`](../README.md).

---

### 1. Operational Context & Air-Gap Mandate

Consider an on-premise enterprise document intelligence copilot deployed across air-gapped workstations in a regulated organization (such as confidential legal discovery, proprietary financial auditing, or sovereign public-sector administration). The workload ingests multi-page contracts, compliance filings, and audit panels to answer complex queries and extract structured records.

Due to strict privacy mandates, corporate compliance policies, and data sovereignty laws, **zero customer data may leave local infrastructure**. Multi-tenant public cloud APIs (OpenAI, Anthropic) are strictly prohibited. The system must execute entirely on local, on-premise compute nodes deployed inside secure server rooms.

---

### 2. The Production Bottleneck: Single-GPU Memory Limits

Local deployment racks are constrained by space, power, and procurement budgets to standard edge workstations equipped with **single 24GB VRAM GPUs** (e.g., NVIDIA RTX 4090, A10G, or L4).

When serving an open-weights 8-billion parameter foundation model (such as LLaMA-3-8B or Mistral-7B) in standard 16-bit floating point (FP16 / BF16):

* **Model Static Footprint:** $8.03 \times 10^9 \times 2\text{ bytes} \approx \mathbf{16.06\text{ GB}}$ VRAM just to load weights into memory.
* **Remaining VRAM for Inference Buffers:** $24.0\text{ GB} - 16.06\text{ GB} = \mathbf{7.94\text{ GB}}$.
* **The Long-Context KV-Cache Crisis:** Enterprise document retrieval frequently involves ingesting context windows spanning 8,192 to 16,384 tokens. For an 8B model with grouped-query attention (32 layers, 8 KV heads, head dimension 128 — both LLaMA-3-8B and Mistral-7B use 8 KV heads), the KV-cache memory scales linearly with context length $L$ and batch concurrency $B$:

  $$
  \text{KV-Cache (bytes)} = 2 \times B \times L \times N_{\text{layers}} \times (N_{\text{kv heads}} \cdot d_{\text{head}}) \times \text{bytes per element}
  $$

  At concurrency $B=8$ and context length $L=16{,}384$, the FP16 KV-cache alone requires:

  $$
  2 \times 8 \times 16{,}384 \times 32 \times 1024 \times 2 \approx \mathbf{17.18\text{ GB}}
  $$

* **The Out-Of-Memory Wall:** Total memory required is $16.06\text{ GB} + 17.18\text{ GB} = \mathbf{33.24\text{ GB}} \gg 24\text{ GB}$. The system encounters Out-Of-Memory (OOM) crashes, forcing the organization to throttle concurrency to $B \le 3$ ($16.06 + 3 \times 2.15 = 22.5\text{ GB} < 24\text{ GB}$), causing extensive request queuing and latency degradation.

---

### 3. The Failed Attempt: Naive INT4 Post-Training Quantization

To compress model weights and context memory into the 24GB envelope, engineers naturally attempt uniform 4-bit post-training quantization (PTQ):
* **Weight Quantization (W4):** Compresses weights to $\sim 4.2\text{ GB}$.
* **Activation & KV-Cache Quantization (A4 & KV4):** Compresses KV-cache to $\sim 4.3\text{ GB}$, theoretically allowing $B=8$ concurrency within $12.5\text{ GB}$ total VRAM.

**The Catastrophic Failure Mode:**
Modern transformer architectures naturally develop **systematic activation outliers** in deep residual layers. Approximately $0.1\%$ of hidden-state channels exhibit peak amplitudes one to two orders of magnitude above the bulk (Dettmers et al. [3] report values up to $\approx 60$ at 6.7B+ parameters), while $99.9\%$ of channel features stay tightly clustered within $[-1.5, +1.5]$:

$$
\Delta = \frac{\max(|X|)}{2^{b-1} - 1} = \frac{60}{7} \approx 8.57
$$

Under this coarse quantization step size ($\Delta = 8.57$), every normal feature value between $-4.3$ and $+4.3$ rounds to **zero**. Fine-grained signals (such as financial figures, dates, and subtle contractual clauses) are completely erased.

* **Outcome:** Extraction fidelity collapses. The model outputs repetitive gibberish or hallucinates critical entities because $100\%$ of its background channels have been rounded to zero.

---

### 4. The Engineering Solution: Randomized Orthogonal Matrix Rotations

Instead of purchasing expensive multi-GPU clusters, the system applies **randomized Fast Walsh-Hadamard Transform (FWHT) rotations** (QuaRot methodology):

```
Standard PTQ (Outliers destroy precision):
[X: Outliers up to ±60] ──> [Uniform INT4 Quantizer] ──> [Values < |4.3| round to 0] (Collapse)

Outlier-Free Rotated PTQ (Energy dispersed):
[X: Outliers up to ±60] ──> [FWHT Rotation: H @ X] ──> [X_rot: Peak diffused] ──> [INT4 Quantizer] ──> [Signal Preserved]
```

1. **Orthogonal Invariance:** By inserting orthogonal Hadamard matrices $\mathbf{H} \in \mathbb{R}^{d \times d}$ ($\mathbf{H}^T \mathbf{H} = \mathbf{I}$) between weights $\mathbf{W}$ and activations $\mathbf{X}$:

   $$
   \mathbf{Y} = \mathbf{W} \mathbf{X} = (\mathbf{W} \mathbf{H}^T) (\mathbf{H} \mathbf{X}) = \tilde{\mathbf{W}} \tilde{\mathbf{X}}
   $$

   * $\tilde{\mathbf{W}} = \mathbf{W} \mathbf{H}^T$ is computed once offline and quantized to INT4.
   * $\tilde{\mathbf{X}} = \mathbf{H} \mathbf{X}$ is computed on-the-fly during inference.
2. **Energy Dispersion:** The randomized Hadamard transform spreads each coordinate's energy evenly across all $d$ dimensions:
   * Peak activation amplitude drops by $\approx \sqrt{d}$.
   * Peak-to-Average Ratio (PAR) and kurtosis normalize toward Gaussian behavior.
   * Quantization step size shrinks, preserving background channels and preventing information collapse.
3. **Efficiency via Butterfly FWHT:** Using the Fast Walsh-Hadamard Transform, $\mathbf{H} \mathbf{X}$ is evaluated in $\mathcal{O}(d \log d)$ butterfly operations rather than dense $\mathcal{O}(d^2)$ matrix multiplications, unlocking 4-bit inference within a single commodity 24GB workstation.
