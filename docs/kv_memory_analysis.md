# KV Cache Memory Analysis

## 1. Parameters & Formula
**Model Specifications:**
- Batch Size ($B$): 1
- Sequence Length ($T$): 1019
- Layers ($L$): 8
- Head Dimension ($D_{head}$): 64
- Bytes per float ($P_{bytes}$): 4 (Float32)

**KV Cache Memory Formula:**
`Cache Size (Bytes) = 2 (Keys & Values) × B × T × L × KV_Heads × D_head × P_bytes`

---

## 2. Theoretical Memory Calculation

### Standard Multi-Head Attention (MHA)
In MHA, KV_Heads equals Query_Heads (8).
- `KV_Heads` = 8
- `Bytes` = 2 × 1 × 1019 × 8 × 8 × 64 × 4
- `Bytes` = 33,390,592
- **Theoretical MHA Cache Size:** 31.84 MB

### Grouped-Query Attention (GQA) [Implemented]
In GQA, KV_Heads is explicitly reduced to 2.
- `KV_Heads` = 2
- `Bytes` = 2 × 1 × 1019 × 8 × 2 × 64 × 4
- `Bytes` = 8,347,648
- **Theoretical GQA Cache Size:** 7.96 MB

---

## 3. Measured GPU Memory Comparison (at 1019 tokens)

| Metric | Value (MB) |
| :--- | :--- |
| **Measured Peak GPU Memory (Cached)** | 12.3 MB |
| **Theoretical GQA Cache Size** | 7.96 MB |
| **Difference (Intermediate PyTorch Overhead)** | 4.34 MB |

*Note: The 4.34 MB overhead consists of the intermediate activation tensors required to compute the forward pass during generation (e.g., query projections, attention scores matrix, softmax output, and context vector).*
