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

def get_held_out_data(tokenizer, context_length, num_docs=100, skip_docs=50000):
    # Using streaming to skip exactly the docs we trained on
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    
    iterator = iter(dataset)
    for _ in range(skip_docs):
        next(iterator)
        
    encoded_docs = []
    for _ in range(num_docs):
        doc = next(iterator)
        encoded = tokenizer.encode(doc["text"])
        encoded_docs.extend(encoded)
        
    # Chunk into context_length blocks
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
    
    print("Fetching held-out validation data (Skipping first 50,000 training docs)...")
    val_data = get_held_out_data(tokenizer, config["context_length"], num_docs=200, skip_docs=50000)
    print(f"Constructed {len(val_data)} validation batches.")
    
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    batch_size = 4
    num_batches = len(val_data) // batch_size
    
    print("Evaluating...")
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
    
    print("\n" + "="*40)
    print("QUALITY METRICS (Held-Out FineWeb-Edu)")
    print("="*40)
    print(f"Cross-Entropy Loss: {avg_loss:.4f}")
    print(f"Perplexity:         {perplexity:.4f}")
    print("="*40)
    
    md_content = "# Quality Metrics\n\n"
    md_content += "Evaluated on a strictly held-out slice of FineWeb-Edu (Docs 50,001 - 50,200).\n\n"
    md_content += f"- **Cross-Entropy Loss:** {avg_loss:.4f}\n"
    md_content += f"- **Perplexity:** {perplexity:.4f}\n"
    
    with open("quality_results.md", "w") as f:
        f.write(md_content)
        
    print("Saved to quality_results.md")

if __name__ == "__main__":
    main()
