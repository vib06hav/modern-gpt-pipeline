import torch
import torch.nn as nn
from datasets import load_dataset
import tiktoken
import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import GPTModel
from model_baseline import BaselineGPTModel
from rope import precompute_rope_params

def get_held_out_data(tokenizer, context_length, num_docs=100, skip_docs=50000):
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    iterator = iter(dataset)
    for _ in range(skip_docs):
        next(iterator)
        
    encoded_docs = []
    for _ in range(num_docs):
        doc = next(iterator)
        encoded = tokenizer.encode(doc["text"])
        encoded_docs.extend(encoded)
        
    blocks = []
    for i in range(0, len(encoded_docs) - context_length, context_length):
        blocks.append(encoded_docs[i:i+context_length+1]) # +1 for targets
        
    return torch.tensor(blocks)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="../checkpoints/final_model.pth")
    parser.add_argument("--model-type", type=str, choices=["modern", "baseline"], default="modern")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    
    if args.model_type == "modern":
        model = GPTModel(config)
    else:
        model = BaselineGPTModel(config)
        
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    tokenizer = tiktoken.get_encoding("gpt2")
    criterion = nn.CrossEntropyLoss()
    
    # We will test 1.0x (1024), 1.5x (1536), and 2.0x (2048) training context lengths
    test_lengths = [1024, 1536, 2048]
    
    md_content = "# RoPE Extrapolation Test\n\n"
    md_content += "| Context Length | Extrapolation | Cross-Entropy Loss | Perplexity |\n"
    md_content += "|---|---|---|---|\n"
    
    print("\n" + "="*70)
    print("ROPE EXTRAPOLATION TEST")
    print("="*70)
    
    for length in test_lengths:
        if args.model_type == "modern":
            # Dynamically extend RoPE matrices AND the Causal Mask for extrapolation mathematically
            for block in model.trf_blocks:
                head_dim = config["emb_dim"] // config["n_heads"]
                cos, sin = precompute_rope_params(head_dim, length)
                block.att.cos = cos.to(device)
                block.att.sin = sin.to(device)
                # Expand Causal Mask
                block.att.mask = torch.triu(torch.ones(length, length), diagonal=1).to(device)
        else:
            # For Baseline, we only need to expand the causal mask (Absolute embeddings will fail organically)
            for block in model.trf_blocks:
                block.att.mask = torch.triu(torch.ones(length, length), diagonal=1).to(device)
            
        print(f"Fetching held-out data for context length {length}...")
        val_data = get_held_out_data(tokenizer, length, num_docs=100, skip_docs=50000)
        
        batch_size = 2 if length > 1024 else 4
        num_batches = len(val_data) // batch_size
        
        total_loss = 0.0
        with torch.no_grad():
            for i in range(0, len(val_data) - batch_size, batch_size):
                batch = val_data[i:i+batch_size].to(device)
                inputs = batch[:, :-1]
                targets = batch[:, 1:]
                
                logits = model(inputs)
                
                logits_flat = logits.reshape(-1, logits.size(-1))
                targets_flat = targets.reshape(-1)
                
                loss = criterion(logits_flat, targets_flat)
                total_loss += loss.item()
                
        avg_loss = total_loss / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        extrap = f"{length/1024:.1f}x"
        print(f"Context: {length:<5} ({extrap:<4}) | Loss: {avg_loss:.4f} | PPL: {perplexity:.4f}")
        md_content += f"| {length} | {extrap} | {avg_loss:.4f} | {perplexity:.4f} |\n"
        
    with open("extrapolation_results.md", "w") as f:
        f.write(md_content)
        
    print("\nSaved to extrapolation_results.md")

if __name__ == "__main__":
    main()
