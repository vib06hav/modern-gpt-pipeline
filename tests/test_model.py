"""
Tests for model.py
- GPTModel: [B, T] input -> [B, T, vocab_size] output
- TransformerBlock: [B, T, emb_dim] -> [B, T, emb_dim]
- LayerNorm: output has roughly zero mean, unit variance
- GELU: matches expected values
- FeedForward: correct shape
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from config import GPT_CONFIG_124M
from model import GPTModel, TransformerBlock, LayerNorm, GELU, FeedForward


# Use a smaller config for fast testing
TEST_CFG = {
    "vocab_size": 50257,
    "context_length": 64,
    "emb_dim": 128,
    "n_heads": 4,
    "n_layers": 2,
    "drop_rate": 0.0,
    "qkv_bias": False
}


def test_gpt_model_output_shape():
    """GPTModel should map [B, T] integer input to [B, T, vocab_size] logits."""
    B, T = 2, 16
    model = GPTModel(TEST_CFG)
    model.eval()

    x = torch.randint(0, TEST_CFG["vocab_size"], (B, T))
    with torch.no_grad():
        logits = model(x)

    assert logits.shape == (B, T, TEST_CFG["vocab_size"]), \
        f"GPTModel output shape {logits.shape} != expected ({B}, {T}, {TEST_CFG['vocab_size']})"

    print("PASSED: test_gpt_model_output_shape")


def test_transformer_block_shape():
    """TransformerBlock should preserve shape: [B, T, emb_dim] -> [B, T, emb_dim]."""
    B, T = 2, 16
    block = TransformerBlock(TEST_CFG)
    block.eval()

    x = torch.randn(B, T, TEST_CFG["emb_dim"])
    with torch.no_grad():
        out = block(x)

    assert out.shape == (B, T, TEST_CFG["emb_dim"]), \
        f"TransformerBlock output shape {out.shape} != expected ({B}, {T}, {TEST_CFG['emb_dim']})"

    print("PASSED: test_transformer_block_shape")


def test_layernorm():
    """After LayerNorm, each token's features should have ~0 mean and ~1 std."""
    emb_dim = 128
    ln = LayerNorm(emb_dim)

    x = torch.randn(2, 10, emb_dim) * 5 + 3  # shifted and scaled input
    out = ln(x)

    # Check per-token statistics (last dim)
    mean = out.mean(dim=-1)
    std = out.std(dim=-1)

    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-4), \
        f"LayerNorm mean not ~0: {mean}"
    assert torch.allclose(std, torch.ones_like(std), atol=0.1), \
        f"LayerNorm std not ~1: {std}"

    print("PASSED: test_layernorm")


def test_gelu():
    """GELU(0) should be 0, and GELU should be monotonically increasing for large positive values."""
    gelu = GELU()

    # GELU(0) ≈ 0
    zero = gelu(torch.tensor(0.0))
    assert abs(zero.item()) < 1e-6, f"GELU(0) = {zero.item()}, expected ~0"

    # GELU should be positive for positive inputs and roughly equal to x for large x
    large = gelu(torch.tensor(5.0))
    assert large.item() > 4.9, f"GELU(5.0) = {large.item()}, expected ~5.0"

    print("PASSED: test_gelu")


def test_feedforward_shape():
    """FeedForward should preserve shape: [B, T, emb_dim] -> [B, T, emb_dim]."""
    B, T = 2, 16
    ff = FeedForward(TEST_CFG)
    ff.eval()

    x = torch.randn(B, T, TEST_CFG["emb_dim"])
    with torch.no_grad():
        out = ff(x)

    assert out.shape == (B, T, TEST_CFG["emb_dim"]), \
        f"FeedForward output shape {out.shape} != expected ({B}, {T}, {TEST_CFG['emb_dim']})"

    print("PASSED: test_feedforward_shape")


def test_full_config():
    """Smoke test: GPTModel with the real 124M config should produce correct output shape."""
    B, T = 1, 8
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()

    x = torch.randint(0, GPT_CONFIG_124M["vocab_size"], (B, T))
    with torch.no_grad():
        logits = model(x)

    assert logits.shape == (B, T, GPT_CONFIG_124M["vocab_size"]), \
        f"Full config output shape {logits.shape} != expected ({B}, {T}, {GPT_CONFIG_124M['vocab_size']})"

    print("PASSED: test_full_config")


if __name__ == "__main__":
    test_gelu()
    test_layernorm()
    test_feedforward_shape()
    test_transformer_block_shape()
    test_gpt_model_output_shape()
    test_full_config()
    print("\n=== All model tests passed! ===")
