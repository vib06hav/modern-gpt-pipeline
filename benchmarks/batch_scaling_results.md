# Batched Throughput & Memory Scaling (Context: 1000)

| Batch Size | Architecture | Tokens/Sec | Peak VRAM (MB) |
|---|---|---|---|
| 1 | MHA | 115.8 | 34.1 |
| 1 | GQA | 117.4 | 12.5 |
| 2 | MHA | 250.1 | 68.5 |
| 2 | GQA | 234.1 | 24.1 |
| 4 | MHA | 507.5 | 134.2 |
| 4 | GQA | 481.0 | 51.4 |
| 8 | MHA | 1045.6 | 270.0 |
| 8 | GQA | 940.9 | 102.5 |
| 16 | MHA | 574.7 | 547.5 |
| 16 | GQA | 1718.4 | 195.9 |
