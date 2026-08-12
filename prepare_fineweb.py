import os
from datasets import load_dataset
import tiktoken

def main():
    print("Downloading FineWeb-Edu subset...")
    # Stream the dataset so we don't download the entire 10TB
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    
    os.makedirs("data", exist_ok=True)
    out_file = "data/fineweb_subset.txt"
    
    # We want roughly 20 million tokens for a solid initial run.
    target_tokens = 20_000_000 
    
    tokenizer = tiktoken.get_encoding("gpt2")
    
    print(f"Streaming text into {out_file} until we hit {target_tokens} tokens...")
    
    with open(out_file, "w", encoding="utf-8") as f:
        tokens_collected = 0
        for i, row in enumerate(dataset):
            text = row["text"]
            f.write(text + "\n\n<|endoftext|>\n\n")
            
            # Count tokens to know when to stop
            tokens_collected += len(tokenizer.encode(text))
            
            if i % 1000 == 0:
                print(f"Progress: {tokens_collected / 1_000_000:.2f}M / 20.0M tokens...")
                
            if tokens_collected >= target_tokens:
                break
                
    print(f"Finished! Saved {tokens_collected} tokens to {out_file}")

if __name__ == "__main__":
    main()
