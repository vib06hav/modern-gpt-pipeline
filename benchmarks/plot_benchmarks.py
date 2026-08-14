import matplotlib.pyplot as plt
import os

# Hardcoded from benchmark_results.md to avoid fragile markdown parsing
tokens = [128, 256, 512, 1019]

uncached_latency = [9.42, 8.86, 11.78, 22.43]
cached_latency = [9.60, 8.62, 9.38, 8.97]

uncached_memory = [52.5, 101.0, 200.3, 396.1]
cached_memory = [1.8, 3.3, 7.0, 12.3]

# Create output dir if needed
os.makedirs("plots", exist_ok=True)

# 1. Latency vs Sequence Length Plot
plt.figure(figsize=(8, 5))
plt.plot(tokens, uncached_latency, marker='o', color='#e74c3c', linewidth=2.5, label='Uncached (O(N²))')
plt.plot(tokens, cached_latency, marker='o', color='#2ecc71', linewidth=2.5, label='Cached (O(N))')
plt.title('Decode Latency vs. Sequence Length', fontsize=14, fontweight='bold')
plt.xlabel('Sequence Length (Tokens)', fontsize=12)
plt.ylabel('Decode Latency per Token (ms)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('plots/latency_vs_length.png', dpi=300)
plt.close()

# 2. Memory vs Sequence Length Plot
plt.figure(figsize=(8, 5))
plt.plot(tokens, uncached_memory, marker='o', color='#e74c3c', linewidth=2.5, label='Uncached')
plt.plot(tokens, cached_memory, marker='o', color='#3498db', linewidth=2.5, label='GQA Cached')
plt.title('Peak GPU Memory vs. Sequence Length', fontsize=14, fontweight='bold')
plt.xlabel('Sequence Length (Tokens)', fontsize=12)
plt.ylabel('Peak GPU Memory (MB)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('plots/memory_vs_length.png', dpi=300)
plt.close()

print("Successfully generated latency_vs_length.png and memory_vs_length.png in benchmarks/plots/")
