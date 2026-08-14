import torch
import time
import gc
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
    print("Initializing Dummy MHA Model...")
    mha_config = base_config.copy()
    mha_config["n_kv_heads"] = 8
    model_mha = GPTModel(mha_config).to(device)
    model_mha.eval()

    # 2. Create Dummy GQA Model (2 KV Heads)
    print("Initializing Dummy GQA Model...")
    gqa_config = base_config.copy()
    gqa_config["n_kv_heads"] = 2
    model_gqa = GPTModel(gqa_config).to(device)
    model_gqa.eval()

    tokenizer = tiktoken.get_encoding("gpt2")
    prompt = "Hello, my name is"
    encoded = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
    
    lengths_to_test = [128, 512, 1000]
    
    md_content = "# Architecture Benchmark: MHA vs GQA\n\n"
    md_content += "| Tokens | Architecture | Decode/Tok (ms) | Tokens/Sec | Peak VRAM (MB) | Theoretical KV Size (MB) |\n"
    md_content += "|---|---|---|---|---|---|\n"
    
    print("\n" + "="*95)
    print(f"{'Tokens':<10} | {'Arch':<10} | {'Decode/Tok (ms)':<15} | {'Tokens/Sec':<12} | {'Peak VRAM':<12} | {'Theoretical':<12}")
    print("-" * 95)

    for seq_len in lengths_to_test:
        # ---- MHA (8 Heads) ----
        pref_mha, dec_mha, tot_mha, peak_mem_mha = benchmark_run(
            profile_cached, model_mha, encoded, seq_len, base_config["context_length"], runs=3, warmup=1
        )
        tps_mha = seq_len / tot_mha
        dec_mha_ms = dec_mha * 1000
        theo_mha = (2 * 1 * seq_len * 8 * 8 * 64 * 4) / (1024 * 1024)
        
        print(f"{seq_len:<10} | {'MHA':<10} | {dec_mha_ms:<15.2f} | {tps_mha:<12.1f} | {peak_mem_mha:<12.1f} | {theo_mha:<12.2f}")
        md_content += f"| {seq_len} | MHA | {dec_mha_ms:.2f} | {tps_mha:.1f} | {peak_mem_mha:.1f} | {theo_mha:.2f} |\n"
        
        # ---- GQA (2 Heads) ----
        pref_gqa, dec_gqa, tot_gqa, peak_mem_gqa = benchmark_run(
            profile_cached, model_gqa, encoded, seq_len, base_config["context_length"], runs=3, warmup=1
        )
        tps_gqa = seq_len / tot_gqa
        dec_gqa_ms = dec_gqa * 1000
        theo_gqa = (2 * 1 * seq_len * 8 * 2 * 64 * 4) / (1024 * 1024)
        
        print(f"{seq_len:<10} | {'GQA':<10} | {dec_gqa_ms:<15.2f} | {tps_gqa:<12.1f} | {peak_mem_gqa:<12.1f} | {theo_gqa:<12.2f}")
        print("-" * 95)
        md_content += f"| {seq_len} | GQA | {dec_gqa_ms:.2f} | {tps_gqa:.1f} | {peak_mem_gqa:.1f} | {theo_gqa:.2f} |\n"

    with open("mha_vs_gqa_results.md", "w") as f:
        f.write(md_content)
        
    print("\nBenchmark Complete! Saved to mha_vs_gqa_results.md")

if __name__ == "__main__":
    main()
