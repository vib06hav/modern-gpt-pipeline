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

## Corrected AWS Execution Plan

### Step 1: The AWS GPU Quota Check
1. Go to AWS **Service Quotas** -> Amazon EC2.
2. Ensure your limit for `Running On-Demand G and VT instances` is at least **4 vCPUs** (enough for one `g4dn.xlarge`).

### Step 2: Launching the GPU Server
1. Go to the EC2 Dashboard -> **Launch instance**.
2. **OS (AMI):** Search for `Deep Learning OSS Nvidia Driver AMI GPU PyTorch`. Look at the options and select a current Ubuntu version (e.g., Ubuntu 22.04). Ensure it supports G4dn.
3. **Instance Type:** `g4dn.xlarge` (T4 GPU is plenty for a 30M model).
4. **Key Pair:** Create and download a new `.pem` key pair.
5. **Network settings (Security):** Change SSH traffic from "Anywhere" to **"My IP"**.
6. **Storage:** Set to **30-50 GB gp3** (you can increase this later if needed).

### Step 3: Connecting & Verifying
1. SSH into the server using the appropriate username (usually `ubuntu` for Ubuntu AMIs, but check based on your AMI choice):
   ```bash
   ssh -i your-key.pem ubuntu@YOUR_AWS_IP
   ```
2. Verify the GPU is working:
   ```bash
   nvidia-smi
   ```

### Step 4: The Training Setup
1. Start `tmux` to keep your session alive:
   ```bash
   tmux
   ```
2. Clone your repo:
   ```bash
   git clone https://github.com/vib06hav/modern-gpt-pipeline.git
   cd modern-gpt-pipeline
   ```
3. Set up environment and get data:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   
   mkdir data
   curl -o data/tinyshakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
   ```

### Step 5: Run the Trial (TinyShakespeare)
Run a controlled step-based training (e.g., 1000 steps) rather than blind epochs:
```bash
python test_train.py \
    --dataset data/tinyshakespeare.txt \
    --batch-size 8 \
    --max-steps 1000 \
    --checkpoint checkpoints/aws_trial.pth
```
Verify the loss goes down and a checkpoint is saved.

### Step 6: Retrieve & TERMINATE
1. From your **local machine**, download the checkpoint:
   ```bash
   scp -i your-key.pem ubuntu@YOUR_AWS_IP:~/modern-gpt-pipeline/checkpoints/aws_trial.pth ./
   ```
2. **TERMINATE THE INSTANCE** from the AWS Dashboard immediately to stop consuming credits!
