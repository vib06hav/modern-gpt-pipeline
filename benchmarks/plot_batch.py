import matplotlib.pyplot as plt
import os

batch_sizes = [1, 2, 4, 8, 16]

mha_vram = [34.1, 68.5, 134.2, 270.0, 547.5]
gqa_vram = [12.5, 24.1, 51.4, 102.5, 195.9]

mha_tps = [115.8, 250.1, 507.5, 1045.6, 574.7]
gqa_tps = [117.4, 234.1, 481.0, 940.9, 1718.4]

os.makedirs("plots", exist_ok=True)

# 1. VRAM vs Batch Size
plt.figure(figsize=(8, 5))
plt.plot(batch_sizes, mha_vram, marker='o', color='#e74c3c', linewidth=2.5, label='MHA (8 KV Heads)')
plt.plot(batch_sizes, gqa_vram, marker='o', color='#3498db', linewidth=2.5, label='GQA (2 KV Heads)')
plt.title('KV Cache Memory vs. Batch Size (1000 tokens)', fontsize=14, fontweight='bold')
plt.xlabel('Concurrent Sequences (Batch Size)', fontsize=12)
plt.ylabel('Peak GPU Memory (MB)', fontsize=12)
plt.xticks(batch_sizes)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('plots/batch_memory.png', dpi=300)
plt.close()

# 2. Throughput vs Batch Size
plt.figure(figsize=(8, 5))
plt.plot(batch_sizes, mha_tps, marker='o', color='#e74c3c', linewidth=2.5, label='MHA')
plt.plot(batch_sizes, gqa_tps, marker='o', color='#2ecc71', linewidth=2.5, label='GQA')
plt.title('Generation Throughput vs. Batch Size (1000 tokens)', fontsize=14, fontweight='bold')
plt.xlabel('Concurrent Sequences (Batch Size)', fontsize=12)
plt.ylabel('Tokens / Second', fontsize=12)
plt.xticks(batch_sizes)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('plots/batch_throughput.png', dpi=300)
plt.close()

print("Successfully generated batch_memory.png and batch_throughput.png in benchmarks/plots/")
