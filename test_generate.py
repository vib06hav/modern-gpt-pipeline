import torch
import tiktoken
from config import GPT_CONFIG_124M
from model import GPTModel
from generate import generate, text_to_token_ids, token_ids_to_text

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Initialize model and load weights
    model = GPTModel(GPT_CONFIG_124M)
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.to(device)
    model.eval()

    # 2. Setup tokenizer and starting context
    tokenizer = tiktoken.get_encoding("gpt2")
    start_context = "Hello, this is a test of"
    
    print(f"Starting context: '{start_context}'")
    encoded = text_to_token_ids(start_context, tokenizer).to(device)

    # 3. Generate text using our fancy generate function
    # It supports temperature and top_k!
    context_size = model.pos_emb.weight.shape[0]
    
    with torch.no_grad():
        token_ids = generate(
            model=model,
            idx=encoded,
            max_new_tokens=40,
            context_size=context_size,
            temperature=0.7,
            top_k=10
        )
    
    # 4. Decode and print
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print("\n--- Generated Text ---")
    print(decoded_text)
    print("----------------------")

if __name__ == "__main__":
    main()
