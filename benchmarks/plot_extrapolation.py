import matplotlib.pyplot as plt
import os

lengths = [1024, 1536, 2048]
ppl = [251.44, 261.63, 273.95]

os.makedirs("plots", exist_ok=True)

plt.figure(figsize=(8, 5))
plt.plot(lengths, ppl, marker='o', color='#8e44ad', linewidth=2.5, label='Modern GPT (RoPE)')
plt.title('RoPE Extrapolation: Perplexity vs Context Length', fontsize=14, fontweight='bold')
plt.xlabel('Sequence Length (Tokens)', fontsize=12)
plt.ylabel('Perplexity (Lower is Better)', fontsize=12)
plt.xticks(lengths, labels=['1024\n(Trained Length)', '1536\n(1.5x Length)', '2048\n(2.0x Length)'])

# Add baseline limit line
plt.axvline(x=1024, color='red', linestyle='--', label='Absolute Embeddings Crash Limit')

plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('plots/extrapolation_ppl.png', dpi=300)
plt.close()
