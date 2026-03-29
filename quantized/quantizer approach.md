# Quantizer Roadmap

## Mathematical Foundations

* ~~**[Whitepaper] [A Survey of Model Compression Techniques (2025)](https://arxiv.org/abs/2310.16795)** *Note: old links*~~
    * ~~**Focus:** Uniform vs. non-uniform quantization; calibration trade-offs (KL-divergence vs. MSE).~~
    * About MoE (mixture of experts) models  -> useless
* **[The Bible] [Quantizing Deep Convolutional Networks for Efficient Inference](https://arxiv.org/abs/1806.08342)**
    * **Focus:** Industry-standard $S$ (scale) and $Z$ (zero-point) equations; baseline requirement for any `int8` implementation.
    * **Formula:** $q = \text{clamp}\left(\lfloor \frac{r}{S} \rceil + Z, q_{min}, q_{max}\right)$


## Handling Mixed-Precision
Standard quantization degrades on LLMs because outlier features compress usable dynamic range. References below capture practical fixes.

* **[LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339)**
    * **Concept:** Vector-wise quantization; outlier dimensions isolated in FP16 while most workload remains in INT8.
* **[AWQ: Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978)**
    * **Concept:** Weights should not be quantized in isolation; activation magnitudes identify salient weights to protect for 4-bit accuracy.
* **[GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)**
    * **Concept:** Layer-wise quantization using Hessian-based corrections for residual error from prior quantization decisions.


## Micro-scaling & NVFP4

* **[Microscaling (MX) Specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)**
    * **Concept:** Shared exponent (scale) across small blocks (for example 16 or 32 values) instead of whole-tensor scaling.
* **[FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)**
    * **Concept:** Background for $E4M3$ and $E5M2$ formats, which motivate newer 4-bit floating-point logic (NVFP4).

---

## 4. Steps

| Phase | Component | Key Feature | Reference |
| :--- | :--- | :--- | :--- |
| **1** | **Affine Quantizer** | Symmetric/Asymmetric $S$ & $Z$ logic. | *Google Whitepaper (2018)* |
| **2** | **Calibration Suite** | Implement Max/Min, Entropy (KL), and Percentile clipping. | *TensorRT Docs / Survey Paper* |
| **3** | **Granularity Control** | Support Per-Tensor, Per-Channel, and Per-Group scaling. | *LLM.int8()* |
| **4** | **Error Compensation** | Implement a basic version of GPTQ (Hessian-based updates). | *GPTQ Paper* |
| **5** | **Micro-Block (MX)** | Implement block-based scaling for FP4/FP8. | *OCP MX Spec* |

---