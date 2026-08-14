import torch
import time
import statistics
import tiktoken
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import GPTModel
from benchmark_matrix import profile_cached, clear_memory, get_peak_memory_mb, benchmark_run

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint_path = "../checkpoints/final_model.pth"
    print(f"Loading base config from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    base_config = checkpoint["config"]

    # 1. Create Dummy MHA Model (8 KV Heads)
    mha_config = base_config.copy()
    mha_config["n_kv_heads"] = 8
    model_mha = GPTModel(mha_config).to(device)
    model_mha.eval()

    # 2. Create Dummy GQA Model (2 KV Heads)
    gqa_config = base_config.copy()
    gqa_config["n_kv_heads"] = 2
    model_gqa = GPTModel(gqa_config).to(device)
    model_gqa.eval()

    tokenizer = tiktoken.get_encoding("gpt2")
    prompt = "Hello, my name is"
    encoded_single = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
    
    seq_len = 1000 # Fixed seq length
    batch_sizes = [1, 2, 4, 8, 16]
    
    print("\n" + "="*85)
    print(f"BATCH BENCHMARK (Context={seq_len} tokens)")
    print(f"{'Batch Size':<12} | {'Architecture':<12} | {'Tokens/Sec':<15} | {'Peak VRAM (MB)':<15}")
    print("-" * 85)
    
    md_content = f"# Batched Throughput & Memory Scaling (Context: {seq_len})\n\n"
    md_content += "| Batch Size | Architecture | Tokens/Sec | Peak VRAM (MB) |\n"
    md_content += "|---|---|---|---|\n"

    for bs in batch_sizes:
        # Create batched input
        encoded = encoded_single.repeat(bs, 1)
        
        # MHA
        try:
            pref_mha, dec_mha, tot_mha, peak_mem_mha = benchmark_run(
                profile_cached, model_mha, encoded, seq_len, base_config["context_length"], runs=3, warmup=1
            )
            # Tokens/Sec is (Batch Size * Tokens) / Total Time
            tps_mha = (bs * seq_len) / tot_mha
            print(f"{bs:<12} | {'MHA':<12} | {tps_mha:<15.1f} | {peak_mem_mha:<15.1f}")
            md_content += f"| {bs} | MHA | {tps_mha:.1f} | {peak_mem_mha:.1f} |\n"
        except RuntimeError as e: # Catch OOM
            if "out of memory" in str(e).lower():
                print(f"{bs:<12} | {'MHA':<12} | {'OOM':<15} | {'OOM':<15}")
                md_content += f"| {bs} | MHA | OOM | OOM |\n"
            else:
                raise e
                
        # GQA
        try:
            pref_gqa, dec_gqa, tot_gqa, peak_mem_gqa = benchmark_run(
                profile_cached, model_gqa, encoded, seq_len, base_config["context_length"], runs=3, warmup=1
            )
            tps_gqa = (bs * seq_len) / tot_gqa
            print(f"{bs:<12} | {'GQA':<12} | {tps_gqa:<15.1f} | {peak_mem_gqa:<15.1f}")
            md_content += f"| {bs} | GQA | {tps_gqa:.1f} | {peak_mem_gqa:.1f} |\n"
        except RuntimeError as e: # Catch OOM
            if "out of memory" in str(e).lower():
                print(f"{bs:<12} | {'GQA':<12} | {'OOM':<15} | {'OOM':<15}")
                md_content += f"| {bs} | GQA | OOM | OOM |\n"
            else:
                raise e
        print("-" * 85)

    with open("batch_scaling_results.md", "w") as f:
        f.write(md_content)
        
    print("\nBenchmark Complete! Saved to batch_scaling_results.md")

if __name__ == "__main__":
    main()
