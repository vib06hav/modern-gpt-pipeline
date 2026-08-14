# Architecture Benchmark: MHA vs GQA

| Tokens | Architecture | Decode/Tok (ms) | Tokens/Sec | Peak VRAM (MB) | Theoretical KV Size (MB) |
|---|---|---|---|---|---|
| 128 | MHA | 9.54 | 104.9 | 4.6 | 4.00 |
| 128 | GQA | 8.45 | 118.3 | 1.8 | 1.00 |
| 512 | MHA | 9.04 | 110.7 | 24.5 | 16.00 |
| 512 | GQA | 8.74 | 114.5 | 7.5 | 4.00 |
| 1000 | MHA | 8.37 | 119.5 | 34.1 | 31.25 |
| 1000 | GQA | 9.09 | 109.9 | 12.5 | 7.81 |
