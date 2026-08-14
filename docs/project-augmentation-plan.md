# Project Augmentation Plan — Quality Validation + Extrapolation + Batched Throughput

## Context for the executing LLM

This is a from-scratch decoder-only transformer project (GPT-2 baseline modernized with RoPE, RMSNorm, SwiGLU, and Grouped-Query Attention), trained on AWS (`g4dn.xlarge`, T4 GPU) on a streamed 50K-document slice of FineWeb-Edu. Current model: ~30M params, `emb_dim=512`, `n_layers=8`, `n_heads=8`, `n_kv_heads=2`. A KV-cache with prefill/decode split and dynamic RoPE slicing already exists and has been benchmarked (2.4x latency speedup cached vs. uncached at 1,000 tokens; ~75% KV-cache memory reduction for GQA vs. MHA at 1,000 tokens).

The gap: every existing benchmark measures speed and memory. None of them measure whether the model's output quality is any good, whether the RoPE extrapolation claim (the reason RoPE was chosen over absolute position embeddings) actually holds, or how the GQA memory saving translates into concurrent-request capacity — which is the number that actually matters in a serving context. This plan closes those three gaps, in this order, because each later step depends on infrastructure/artifacts produced by the step before it.

Work through the steps in order. Do not skip Step 0 — Steps 1 and 2 assume a working eval harness and a validated checkpoint.

---

## Step 0: Quality Validation (prerequisite for everything else)

**Goal:** Prove the modernized model (RoPE + RMSNorm + SwiGLU + GQA) did not sacrifice quality relative to the vanilla GPT-2 baseline it replaced, using a held-out slice of the *same* training distribution, and produce honestly-framed qualitative samples.

### 0.1 — Baseline model
Check whether a vanilla GPT-2 baseline checkpoint (pre-modernization, i.e. absolute positional embeddings + standard LayerNorm + GeLU MLP + standard MHA) already exists from earlier in the project.
- If it exists: reuse it as the comparison point. No retraining needed.
- If it does not exist: train one now, matched as closely as possible to the modernized model's training run — same dataset slice, same total training steps, same effective batch size, same optimizer/LR schedule, same parameter count (`emb_dim=512`, `n_layers=8`, `n_heads=8`; MHA uses 8 heads, not the 2-KV-head GQA config). This is a real time cost — budget for a full training run on the same instance type used originally. Do not skip this by using an untrained/randomly-initialized baseline; the comparison is meaningless unless both models are trained to convergence under matched conditions.

### 0.2 — Held-out evaluation set
Do NOT use WikiText or any other external dataset for evaluation — this introduces a domain-mismatch confound and makes the perplexity number undefendable ("why does your eval set not match your training distribution?").
- Pull a held-out slice from the *same* FineWeb-Edu stream used for training (e.g. the next N documents after the training cutoff in the streaming pipeline, where N gives a reasonably stable perplexity estimate — a few hundred documents is enough).
- Confirm zero overlap with the training slice.
- Save this held-out set to disk once so both models are evaluated on the exact same data.

### 0.3 — `eval_quality.py`
Write a script that:
- Loads a given checkpoint (`final_model.pth` for the modernized model, and the baseline checkpoint).
- Runs a forward pass over the held-out set in eval mode (no gradient computation, dropout off if applicable).
- Computes token-level cross-entropy loss and perplexity (`exp(loss)`), aggregated properly over the full held-out set (not just batch-averaged in a way that's sensitive to batch size — use total loss / total token count).
- Outputs both numbers to a results file (e.g. `quality_results.md`), clearly labeled per model (baseline vs. modernized).
- Report both models' numbers side by side. The deliverable is a one- or two-line comparison: modernized model's validation loss/perplexity vs. baseline's, on identical held-out data.

### 0.4 — `generate_samples.py`
- Use the sampling path already present in `generate.py` (temperature scaling + top-k), not greedy argmax — argmax was already observed to loop/repeat during benchmarking.
- Generate output from 3–5 fixed prompts (short, varied — e.g. a factual continuation prompt, a narrative-style prompt, an instructional-style prompt) at a reasonable temperature/top-k setting (pick one setting, document it, don't cherry-pick across many settings).
- Save raw outputs verbatim to the README/results doc.
- Framing requirement: present these as "sample outputs from a ~30M-parameter model trained on 50K documents" — do not claim this proves grammatical competence or fluency. State plainly that outputs will be locally coherent but not consistently fluent given the model/data scale. Accuracy of framing matters more than making the samples look impressive.

### 0.5 — Plots
Write a small `matplotlib` script that reads the existing `benchmark_results.md` (or equivalent raw numbers already produced during the original KV-cache benchmarking) and produces two PNGs:
- Latency vs. sequence length (uncached line curving up / degrading, cached line flat).
- Memory vs. sequence length (uncached memory growing, GQA-cached memory flat/low).
No new benchmark runs needed here — this is purely visualizing numbers that already exist.

### Step 0 deliverables
- `eval_quality.py`, `quality_results.md` (baseline vs. modernized loss/perplexity, same held-out set)
- `generate_samples.py`, sample outputs saved to README with honest framing
- Two PNG plots (latency, memory) generated from existing benchmark data
- README updated with: validation loss/perplexity comparison, sample generations, both plots

---

## Step 1: RoPE Extrapolation Test

**Goal:** Test the actual claim that motivated choosing RoPE over absolute positional embeddings — that it generalizes to sequence lengths beyond the training context — rather than asserting it unverified.

### 1.1 — Determine training context length
Confirm the exact sequence length / context window the model was trained on (from the training config).

### 1.2 — Extrapolation eval
- Using the same held-out FineWeb-Edu slice and the same `eval_quality.py` harness from Step 0 (extend it rather than duplicating), evaluate perplexity at multiple sequence lengths beyond the training context: e.g. 1.0x (baseline, already have this from Step 0), 1.5x, 2.0x training length. Use documents (or concatenated documents) long enough to support these lengths.
- Plot perplexity vs. sequence length. Expected pattern to report honestly (not assumed): RoPE-based model should degrade more gracefully than a fixed absolute-position embedding model would (which typically breaks hard past its trained length because it has no representation for out-of-range positions at all). If you have the baseline model from Step 0 and it's feasible to run it at extrapolated lengths too, include it on the same plot as a direct comparison — this is the strongest version of this result. If the baseline architecture can't run past its trained length at all (absolute position embeddings often can't), state that limitation explicitly instead of working around it — the inability to extrapolate *is* the comparison point.
- Also spot-check qualitative generation output (reusing `generate_samples.py`) at an extrapolated length to see whether output degrades into incoherence, repetition, or stays reasonable.

### Step 1 deliverables
- Perplexity-vs-sequence-length plot covering training length and 1.5x/2.0x beyond it (with baseline comparison if feasible, or an explicit note on baseline's inability to run at those lengths)
- A few qualitative samples generated at extrapolated length
- Short written finding in the README: does perplexity hold up, degrade gracefully, or break down at extrapolated lengths — report what actually happened, not what was expected

---

## Step 2: Batched Throughput / Memory Scaling

**Goal:** Reframe the existing single-sequence GQA memory result in terms of what it actually enables in a serving context — concurrent request capacity — rather than leaving it as an isolated per-sequence number.

### 2.1 — Extend the benchmark harness
Take the existing KV-cache benchmarking code (`benchmark_matrix.py` / `benchmark_gqa_vs_mha.py`) and extend it to sweep batch size instead of (or in addition to) sequence length:
- Fix a representative sequence length (e.g. the 1,000-token point already used in prior benchmarks).
- Sweep batch size across a reasonable range (e.g. 1, 2, 4, 8, 16 — whatever the T4's 16GB VRAM can support before OOM for both MHA and GQA configs).
- For each batch size, measure: peak GPU memory (`torch.cuda.max_memory_allocated`, matching the methodology already used — warmup runs, `torch.cuda.synchronize`), and throughput (tokens/sec across the batch).
- Run this for both the GQA (2 KV-heads, actual trained config) and MHA (8 KV-heads, dummy comparison config already used in prior benchmarking) to keep the comparison consistent with existing results.

### 2.2 — Find the practical ceiling
- Identify the maximum batch size each config (GQA vs MHA) can sustain at the fixed sequence length before hitting the T4's memory limit.
- This is the headline number: "GQA supports Nx more concurrent sequences than MHA at the same VRAM budget," not just "GQA uses less memory for one sequence."

### 2.3 — Plots
- Memory vs. batch size (GQA vs. MHA), same style as existing plots.
- Throughput (tokens/sec) vs. batch size (GQA vs. MHA).

### Step 2 deliverables
- Extended benchmark script producing batch-size-swept memory and throughput numbers for both GQA and MHA
- Two plots (memory vs. batch size, throughput vs. batch size)
- A stated maximum concurrent batch size for each config at the fixed sequence length, framed as the practical serving-capacity implication of the GQA result

---

## Explicit exclusions

Do not implement, and do not suggest implementing, any of the following as part of this plan — they were considered and deliberately deprioritized:
- Mixture-of-Experts (different project, not an extension of this one)
- Quantization (lower priority than Steps 1–2; only revisit if time remains after Step 2 is fully done)
- `torch.compile` (excluded — a speedup that's hard to explain the internals of under questioning is a liability, not an asset, for this project)

## Final README structure (target end state)
1. Architecture overview (existing)
2. Training setup (existing)
3. **Quality validation**: baseline vs. modernized loss/perplexity on held-out data, sample generations (honestly framed)
4. **RoPE extrapolation**: perplexity vs. sequence length plot, written finding
5. Inference optimization: KV-cache latency/memory results (existing, now with plots instead of just tables)
6. **Batched throughput**: memory/throughput vs. batch size plots, concurrent-capacity finding
7. (Optional, only if time remains) Quantization results
