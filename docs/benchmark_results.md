# KV Cache Robust Benchmark Results

Using 2 warmup runs and 5 measurement runs. Timed with perf_counter and CUDA sync.

| Tokens | Mode | Prefill (s) | Decode/Tok (ms) | Tokens/Sec | Peak Mem (MB) |
|---|---|---|---|---|---|
| 128 | Uncached | 0.017 | 9.42 | 106.3 | 52.5 |
| 128 | Cached | 0.011 | 9.60 | 103.4 | 1.8 |
| 256 | Uncached | 0.009 | 8.86 | 112.9 | 101.0 |
| 256 | Cached | 0.009 | 8.62 | 116.0 | 3.3 |
| 512 | Uncached | 0.012 | 11.78 | 84.7 | 200.3 |
| 512 | Cached | 0.018 | 9.38 | 106.6 | 7.0 |
| 1019 | Uncached | 0.010 | 22.43 | 44.6 | 396.1 |
| 1019 | Cached | 0.009 | 8.97 | 111.4 | 12.3 |
