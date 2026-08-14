import torch
import torch.nn as nn
from model_baseline import BaselineGPTModel
from prepare_fineweb import get_fineweb_dataloader
import time
import os
import argparse

def save_checkpoint(model, optimizer, step, loss, config, filename):
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config
    }
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")

def main():
    parser = argparse.ArgumentParser(description="Train Baseline GPT-2 on FineWeb-Edu")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum number of training steps")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/baseline_model.pth", help="Path to save checkpoint")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Exact same 30M param configuration used for the modernized model
    config = {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 512,
        "n_heads": 8,
        "n_layers": 8,
        "drop_rate": 0.0,
        "qkv_bias": False
    }

    # Initialize the BASELINE model (No RoPE, No RMSNorm, No SwiGLU, No GQA)
    model = BaselineGPTModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
    criterion = nn.CrossEntropyLoss()

    print("Loading FineWeb-Edu dataset stream...")
    train_loader = get_fineweb_dataloader(
        batch_size=args.batch_size, 
        context_length=config["context_length"], 
        max_docs=50000 
    )

    print(f"Starting Baseline Training for {args.max_steps} steps...")
    
    model.train()
    step = 0
    total_loss = 0.0
    start_time = time.time()

    for inputs, targets in train_loader:
        if step >= args.max_steps:
            break

        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        logits = model(inputs)
        
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)
        
        loss = criterion(logits_flat, targets_flat)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        step += 1

        if step % 50 == 0:
            avg_loss = total_loss / 50
            elapsed = time.time() - start_time
            print(f"Step {step}/{args.max_steps} | Loss: {avg_loss:.4f} | Time: {elapsed:.2f}s")
            total_loss = 0.0
            start_time = time.time()

    # Save final checkpoint
    save_checkpoint(model, optimizer, step, avg_loss, config, args.checkpoint)
    print("Baseline Training Complete!")

if __name__ == "__main__":
    main()
