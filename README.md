# Modern GPT Architecture & Training Pipeline

A state-of-the-art LLM implementation from scratch in PyTorch, featuring:
- **RoPE** (Rotary Positional Embeddings)
- **RMSNorm** (Root Mean Square Normalization)
- **SwiGLU** (Swish Gated Linear Unit)
- **GQA** (Grouped Query Attention)

## Local Testing
To run a fast sanity check on your CPU with a dummy dataset:
```bash
python test_train.py --dataset dummy
```

## AWS Execution Plan

### Step 1: Trial Run (TinyShakespeare)
**Goal:** Verify the AWS EC2 instance, GPU drivers, and pipeline stability.
1. Spin up an AWS EC2 instance (e.g. g4dn.xlarge or g5.xlarge) using the **AWS Deep Learning AMI**.
2. Clone this repo and install the requirements:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Run on the tiny dataset (1MB) to ensure the model learns and generates properly on the GPU:
   ```bash
   python test_train.py --dataset data/tinyshakespeare.txt --batch-size 8 --epochs 3
   ```

### Step 2: Final Training Run (FineWeb-Edu)
**Goal:** Train a credible, high-quality checkpoint (~20M-40M parameters) to be used for Inference & KV Cache benchmarking.
1. Download a subset of the FineWeb-Edu dataset (exact token count and subset size TBD).
2. Train the model using the optimized CLI arguments:
   ```bash
   python test_train.py --dataset data/fineweb_subset.txt --checkpoint checkpoints/final_model.pth --batch-size 16 --epochs 1
   ```
3. Use the resulting `final_model.pth` for **Stage 3: Inference and KV Caching experiments**.
