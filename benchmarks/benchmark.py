import torch
import time
import tiktoken
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import GPT_CONFIG_124M
from model import GPTModel
from generate import generate_text_simple, generate_text_cached, token_ids_to_text

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the ACTUAL trained model!
    checkpoint_path = "../checkpoints/final_model.pth"
    print(f"Loading {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    
    
    print("Initializing model...")
    model = GPTModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    tokenizer = tiktoken.get_encoding("gpt2")
    prompt = "Hello, my name is"
    encoded = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
    
    max_new_tokens = 250

    print("\n--- UNCACHED GENERATION (O(N^2)) ---")
    start_time = time.time()
    out_uncached = generate_text_simple(model, encoded, max_new_tokens, config["context_length"])
    uncached_time = time.time() - start_time
    print(f"Time: {uncached_time:.2f}s | Speed: {max_new_tokens / uncached_time:.1f} tokens/sec")
    # print("Text:", token_ids_to_text(out_uncached, tokenizer).replace("\n", " "))

    print("\n--- CACHED GENERATION (O(N)) ---")
    start_time = time.time()
    out_cached = generate_text_cached(model, encoded, max_new_tokens, config["context_length"])
    cached_time = time.time() - start_time
    print(f"Time: {cached_time:.2f}s | Speed: {max_new_tokens / cached_time:.1f} tokens/sec")
    # print("Text:", token_ids_to_text(out_cached, tokenizer).replace("\n", " "))
    
    speedup = uncached_time / cached_time
    print(f"\nKV Cache Speedup: {speedup:.2f}x faster!")
    
    print("\nUNCACHED:", out_uncached[0, :20].tolist())
    print("CACHED:  ", out_cached[0, :20].tolist())
    
    assert torch.allclose(out_uncached, out_cached), "Outputs do not match! Cache logic has a bug."
    print("SUCCESS: Uncached and Cached outputs match exactly!")

if __name__ == "__main__":
    main()
