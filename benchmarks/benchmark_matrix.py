import torch
import time
import tiktoken
import gc
import statistics
import sys
import os

# Add parent directory to path so we can import model and generate
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import GPTModel

def profile_uncached(model, idx, max_new_tokens, context_size):
    # Prefill
    if torch.cuda.is_available(): torch.cuda.synchronize()
    prefill_start = time.perf_counter()
    
    idx_cond = idx[:, -context_size:]
    with torch.no_grad():
        logits = model(idx_cond, use_cache=False)
    logits = logits[:, -1, :]
    idx_next = torch.argmax(logits, dim=-1, keepdim=True)
    idx = torch.cat((idx, idx_next), dim=1)
    
    if torch.cuda.is_available(): torch.cuda.synchronize()
    prefill_time = time.perf_counter() - prefill_start
    
    # Decode
    decode_start = time.perf_counter()
    for _ in range(max_new_tokens - 1):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond, use_cache=False)
        logits = logits[:, -1, :]
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
        
    if torch.cuda.is_available(): torch.cuda.synchronize()
    decode_time_total = time.perf_counter() - decode_start
    decode_latency_per_token = decode_time_total / (max_new_tokens - 1)
    
    total_time = prefill_time + decode_time_total
    return idx, prefill_time, decode_latency_per_token, total_time

def profile_cached(model, idx, max_new_tokens, context_size):
    model.reset_kv_cache()
    
    # Prefill
    if torch.cuda.is_available(): torch.cuda.synchronize()
    prefill_start = time.perf_counter()
    
    idx_cond = idx[:, -context_size:]
    with torch.no_grad():
        logits = model(idx_cond, use_cache=True)
    logits = logits[:, -1, :]
    idx_next = torch.argmax(logits, dim=-1, keepdim=True)
    idx = torch.cat((idx, idx_next), dim=1)
    
    if torch.cuda.is_available(): torch.cuda.synchronize()
    prefill_time = time.perf_counter() - prefill_start
    
    # Decode
    decode_start = time.perf_counter()
    for _ in range(max_new_tokens - 1):
        with torch.no_grad():
            logits = model(idx_next, use_cache=True)
        logits = logits[:, -1, :]
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
        
    if torch.cuda.is_available(): torch.cuda.synchronize()
    decode_time_total = time.perf_counter() - decode_start
    decode_latency_per_token = decode_time_total / (max_new_tokens - 1)
    
    total_time = prefill_time + decode_time_total
    return idx, prefill_time, decode_latency_per_token, total_time

def clear_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()

def get_peak_memory_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0

def benchmark_run(func, model, idx, max_new, ctx_size, runs=5, warmup=2):
    # Warmup runs to spin up the GPU
    for _ in range(warmup):
        func(model, idx, max_new, ctx_size)
    
    prefills = []
    decodes = []
    totals = []
    
    # explicitly clear the model's KV cache from the warmup runs 
    # so it doesn't inflate our baseline memory measurement
    model.reset_kv_cache()
    
    # Track memory exactly once cleanly
    clear_memory()
    start_mem = get_peak_memory_mb()
    
    for i in range(runs):
        _, pref, dec, tot = func(model, idx, max_new, ctx_size)
        prefills.append(pref)
        decodes.append(dec)
        totals.append(tot)
        
    peak_mem = get_peak_memory_mb() - start_mem
    return statistics.median(prefills), statistics.median(decodes), statistics.median(totals), peak_mem

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint_path = "../checkpoints/final_model.pth"
    print(f"Loading {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    
    model = GPTModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    tokenizer = tiktoken.get_encoding("gpt2")
    prompt = "Hello, my name is"
    encoded = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
    prompt_len = encoded.shape[1]
    
    # 1024 crashes because 1024 + 5 (prompt) > context_size (1024)
    # We will safely subtract the prompt length!
    lengths_to_test = [128, 256, 512, config["context_length"] - prompt_len]
    
    md_content = "# KV Cache Robust Benchmark Results\n\n"
    md_content += "Using 2 warmup runs and 5 measurement runs. Timed with perf_counter and CUDA sync.\n\n"
    md_content += "| Tokens | Mode | Prefill (s) | Decode/Tok (ms) | Tokens/Sec | Peak Mem (MB) |\n"
    md_content += "|---|---|---|---|---|---|\n"
    
    print("\n" + "="*80)
    print(f"{'Tokens':<10} | {'Mode':<10} | {'Prefill (s)':<12} | {'Decode/Tok (ms)':<15} | {'Tokens/Sec':<12} | {'Peak Mem (MB)':<12}")
    print("-" * 80)

    for seq_len in lengths_to_test:
        # ---- UNCACHED ----
        pref_uncached, dec_uncached, tot_uncached, peak_mem_uncached = benchmark_run(
            profile_uncached, model, encoded, seq_len, config["context_length"]
        )
        uncached_tps = seq_len / tot_uncached
        dec_uncached_ms = dec_uncached * 1000
        
        print(f"{seq_len:<10} | {'Uncached':<10} | {pref_uncached:<12.3f} | {dec_uncached_ms:<15.2f} | {uncached_tps:<12.1f} | {peak_mem_uncached:<12.1f}")
        md_content += f"| {seq_len} | Uncached | {pref_uncached:.3f} | {dec_uncached_ms:.2f} | {uncached_tps:.1f} | {peak_mem_uncached:.1f} |\n"
        
        # ---- CACHED ----
        pref_cached, dec_cached, tot_cached, peak_mem_cached = benchmark_run(
            profile_cached, model, encoded, seq_len, config["context_length"]
        )
        cached_tps = seq_len / tot_cached
        dec_cached_ms = dec_cached * 1000
        
        print(f"{seq_len:<10} | {'Cached':<10} | {pref_cached:<12.3f} | {dec_cached_ms:<15.2f} | {cached_tps:<12.1f} | {peak_mem_cached:<12.1f}")
        print("-" * 80)
        md_content += f"| {seq_len} | Cached | {pref_cached:.3f} | {dec_cached_ms:.2f} | {cached_tps:.1f} | {peak_mem_cached:.1f} |\n"
        
    with open("benchmark_results.md", "w") as f:
        f.write(md_content)
        
    print("\nBenchmark Complete! Results saved to benchmark_results.md")

if __name__ == "__main__":
    main()
