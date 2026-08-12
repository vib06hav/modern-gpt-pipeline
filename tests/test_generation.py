"""
Tests for generate.py
- generate_text_simple produces the right number of new tokens
- generate (with temperature/top_k) produces the right number of tokens
- text_to_token_ids and token_ids_to_text are inverses of each other
- checkpoint save/load produces identical generation output
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import tiktoken
from model import GPTModel
from generate import generate_text_simple, generate, text_to_token_ids, token_ids_to_text


# Small config for fast tests
TEST_CFG = {
    "vocab_size": 50257,
    "context_length": 64,
    "emb_dim": 128,
    "n_heads": 4,
    "n_layers": 2,
    "drop_rate": 0.0,
    "qkv_bias": False
}


def test_generate_simple_token_count():
    """generate_text_simple should return exactly (input_len + max_new_tokens) tokens."""
    model = GPTModel(TEST_CFG)
    model.eval()

    prompt_len = 5
    max_new_tokens = 20
    idx = torch.randint(0, TEST_CFG["vocab_size"], (1, prompt_len))

    with torch.no_grad():
        output = generate_text_simple(model, idx, max_new_tokens, context_size=TEST_CFG["context_length"])

    expected_len = prompt_len + max_new_tokens
    assert output.shape[1] == expected_len, \
        f"Output length {output.shape[1]} != expected {expected_len}"

    print("PASSED: test_generate_simple_token_count")


def test_generate_with_temperature_token_count():
    """generate with temperature should return at most (input_len + max_new_tokens) tokens."""
    model = GPTModel(TEST_CFG)
    model.eval()

    prompt_len = 5
    max_new_tokens = 20
    idx = torch.randint(0, TEST_CFG["vocab_size"], (1, prompt_len))

    with torch.no_grad():
        output = generate(
            model, idx, max_new_tokens,
            context_size=TEST_CFG["context_length"],
            temperature=1.0, top_k=10
        )

    # Could be shorter if eos_id is hit, but without eos_id it should be exact
    expected_len = prompt_len + max_new_tokens
    assert output.shape[1] == expected_len, \
        f"Output length {output.shape[1]} != expected {expected_len}"

    print("PASSED: test_generate_with_temperature_token_count")


def test_tokenizer_roundtrip():
    """Encoding then decoding text should return the original text."""
    tokenizer = tiktoken.get_encoding("gpt2")
    text = "Hello, world! This is a test."

    token_ids = text_to_token_ids(text, tokenizer)
    recovered = token_ids_to_text(token_ids, tokenizer)

    assert recovered == text, f"Roundtrip failed: '{recovered}' != '{text}'"

    print("PASSED: test_tokenizer_roundtrip")


def test_checkpoint_save_load():
    """Loading a saved checkpoint should produce identical generation output."""
    checkpoint_path = os.path.join(os.path.dirname(__file__), "temp_test_model.pth")

    model = GPTModel(TEST_CFG)
    model.eval()

    # Generate with original model
    idx = torch.randint(0, TEST_CFG["vocab_size"], (1, 5))
    with torch.no_grad():
        output_before = generate_text_simple(model, idx, max_new_tokens=10, context_size=TEST_CFG["context_length"])

    # Save checkpoint
    torch.save(model.state_dict(), checkpoint_path)

    # Load into a fresh model
    model2 = GPTModel(TEST_CFG)
    model2.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model2.eval()

    with torch.no_grad():
        output_after = generate_text_simple(model2, idx, max_new_tokens=10, context_size=TEST_CFG["context_length"])

    assert torch.equal(output_before, output_after), \
        "Checkpoint save/load produced different generation output!"

    # Cleanup
    os.remove(checkpoint_path)

    print("PASSED: test_checkpoint_save_load")


if __name__ == "__main__":
    test_tokenizer_roundtrip()
    test_generate_simple_token_count()
    test_generate_with_temperature_token_count()
    test_checkpoint_save_load()
    print("\n=== All generation tests passed! ===")
