import torch
import tiktoken
import os
import argparse
from config import get_config
from model import GPTModel
from data import create_dataloader
from train import train_model_simple

def main():
    parser = argparse.ArgumentParser(description="Train a modern GPT model")
    parser.add_argument("--dataset", type=str, default="dummy", help="Path to dataset text file, or 'dummy' for testing")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--checkpoint", type=str, default="model.pth")
    parser.add_argument("--seed", type=int, default=1337)
    
    # Architecture overrides
    parser.add_argument("--emb-dim", type=int, default=768)
    parser.add_argument("--n-layers", type=int, default=12)
    parser.add_argument("--n-heads", type=int, default=12)
    parser.add_argument("--n-kv-heads", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=1024)
    
    args = parser.parse_args()

    # 1. Setup device and seed
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Get data
    if args.dataset == "dummy":
        text_data = (
            "Hello, this is a test of the GPT model training. "
            "We are training it to understand language and generate text. "
            "It uses self-attention and transformer blocks to learn patterns. "
        ) * 200
    else:
        with open(args.dataset, "r", encoding="utf-8") as f:
            text_data = f.read()

    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    # 3. Create DataLoaders
    # Using 16 for testing; in production you'd use args.context_length
    max_len = 16 if args.dataset == "dummy" else args.context_length
    print(f"Creating dataloaders with batch_size={args.batch_size}, max_len={max_len}...")
    train_loader = create_dataloader(train_data, batch_size=args.batch_size, max_length=max_len, stride=max_len, shuffle=True)
    val_loader = create_dataloader(val_data, batch_size=args.batch_size, max_length=max_len, stride=max_len, shuffle=False)

    # 4. Instantiate Model
    config = get_config(
        emb_dim=args.emb_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        context_length=args.context_length
    )
    print("Initializing model with config:", config)
    model = GPTModel(config)
    model.to(device)

    # 5. Setup Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    # 6. Train the model
    print("Starting training...")
    tokenizer = tiktoken.get_encoding("gpt2")
    
    train_losses, val_losses, track_tokens_seen = train_model_simple(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=args.epochs,
        eval_freq=5,
        eval_iter=1,
        start_context="Hello, this is",
        tokenizer=tokenizer
    )
    
    print("Training complete!")
    
    # 7. Save proper checkpoint
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "train_loss": train_losses[-1] if train_losses else None,
        "val_loss": val_losses[-1] if val_losses else None,
    }, args.checkpoint)
    print(f"Proper checkpoint saved to {args.checkpoint}")

if __name__ == "__main__":
    main()
