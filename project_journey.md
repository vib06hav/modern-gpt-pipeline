# Project Development Journey: Modern LLM from Scratch

## 1. Initial State (GPT-2 Baseline)
*   **Architecture:** Vanilla GPT-2 implementation in PyTorch.
*   **Components:** Absolute Positional Embeddings, standard LayerNorm, GeLU-based Multi-Layer Perceptron (MLP), standard Multi-Head Attention (MHA).
*   **Limitation:** Computationally expensive inference, high memory overhead, and outdated positional representations.

## 2. Modernization Phase
The architecture was stripped down and rebuilt using state-of-the-art open-source LLM standards (mirroring Llama-2/3 paradigms).
*   **Rotary Positional Embeddings (RoPE):** Removed absolute position embeddings. Implemented complex coordinate rotation dynamically within the attention forward pass to improve sequence extrapolation.
*   **RMSNorm:** Replaced standard LayerNorm with Root Mean Square Normalization, eliminating mean-centering to reduce computational overhead.
*   **SwiGLU:** Replaced the standard GeLU MLP with a Swish Gated Linear Unit, separating the gating mechanism from the linear projection.
*   **Grouped-Query Attention (GQA):** Decoupled Query heads from Key/Value heads to drastically reduce the memory footprint required for attention mechanisms.

## 3. Cloud Infrastructure & Training (AWS)
Transitioned from local CPU training to a production-grade cloud GPU pipeline.
*   **Instance:** Deployed an AWS EC2 `g4dn.xlarge` instance (4 vCPUs, 16GB RAM, 1x NVIDIA T4 Tensor Core GPU with 16GB VRAM).
*   **Environment:** Utilized the "Deep Learning OSS Nvidia Driver AMI GPU PyTorch" for pre-installed CUDA/cuDNN stacks.
*   **Data Pipeline:** Authored `prepare_fineweb.py` to stream 50,000 documents from the HuggingFace `FineWeb-Edu` dataset, bypassing the need to download the entire terabyte-scale corpus.
*   **Training Execution:** 
    *   Executed training via `tmux` to ensure session persistence across SSH disconnects.
    *   Replaced arbitrary epoch loops with deterministic step-based training (`max-steps`) in `test_train.py`.
    *   Overrode default 124M config to a target ~30M parameter model: `emb_dim=512`, `n_layers=8`, `n_heads=8`, `n_kv_heads=2`.
*   **Artifact Retrieval:** Secured the trained checkpoint (`final_model.pth`, 233 MB) via secure copy (`scp`) and terminated the AWS instance.

## 4. Inference Optimization (KV Cache)
Engineered a persistent state-tracking Key-Value Cache to eliminate redundant $O(N^2)$ matrix multiplications during autoregressive generation.
*   **State Management:** Registered `cache_k` and `cache_v` persistent buffers inside the `MultiHeadAttention` class, implementing a `reset_cache()` pipeline invoked from the top-level `GPTModel`.
*   **Memory Efficiency:** Relocated the `transpose(1,2)` operation to occur *before* caching. This ensured only the minimal GQA Key/Value matrices (2 heads) were stored in VRAM. The `repeat_interleave` expansion to 8 heads was applied dynamically *after* the cache pull, preserving the GQA memory advantage.
*   **Dynamic RoPE Slicing:** Modified `rope.py` to track `ptr_current_pos`, dynamically slicing the precomputed Cosine/Sine tables to rotate single-token generations by their absolute sequence position rather than position zero.
*   **Generation Pipeline:** Refactored `generate.py` to separate generation into a bulk *Prefill Phase* (processing the prompt) and a token-by-token *Decode Phase* (passing only `[batch, 1]` tensors).

## 5. Benchmarking & Validation
Authored custom profiling scripts (`benchmark_matrix.py` and `benchmark_gqa_vs_mha.py`) featuring rigorous methodology (warmup runs, median averaging, `time.perf_counter`, and `torch.cuda.synchronize`).

*   **Latency Scaling (O(N) vs O(N²)):**
    *   *Uncached:* Speed degraded heavily as context grew, collapsing to 44.9 tokens/sec at 1,000 tokens (22.30 ms latency per token).
    *   *Cached:* Maintained a perfectly flat latency curve. Clocked at ~110 tokens/sec at 1,000 tokens (9.32 ms latency per token).
    *   *Result:* Proven 2.4x speedup during long-context generation.
*   **Memory Scaling (MHA vs GQA):**
    *   Benchmarked an 8-KV-head dummy model against the trained 2-KV-head model.
    *   *MHA:* Peaked at 34.1 MB VRAM for a 1,000-token cache.
    *   *GQA:* Peaked at 12.5 MB VRAM for a 1,000-token cache (comprising a theoretical 7.96 MB cache payload + ~4.34 MB PyTorch overhead).
    *   *Result:* Empirically proven ~75% reduction in KV-Cache memory footprint.
