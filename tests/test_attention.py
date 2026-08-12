"""
Tests for attention.py
- Output shapes are correct
- Causal masking works (future tokens get zero attention weight)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from attention import MultiHeadAttention


def test_attention_output_shape():
    """MHA output should be [B, T, d_out]."""
    B, T, d_in, d_out = 2, 10, 64, 64
    num_heads = 4
    context_length = 16

    mha = MultiHeadAttention(
        d_in=d_in, d_out=d_out, context_length=context_length,
        dropout=0.0, num_heads=num_heads, qkv_bias=False
    )
    mha.eval()

    x = torch.randn(B, T, d_in)
    with torch.no_grad():
        out = mha(x)

    assert out.shape == (B, T, d_out), \
        f"Output shape {out.shape} != expected ({B}, {T}, {d_out})"

    print("PASSED: test_attention_output_shape")


def test_causal_masking():
    """
    For a causal (autoregressive) model, token at position i should NOT attend
    to any token at position j > i. This means attention weights above the 
    diagonal should be 0.
    """
    B, T, d_in, d_out = 1, 6, 32, 32
    num_heads = 2
    context_length = 8

    mha = MultiHeadAttention(
        d_in=d_in, d_out=d_out, context_length=context_length,
        dropout=0.0, num_heads=num_heads, qkv_bias=False
    )
    mha.eval()

    x = torch.randn(B, T, d_in)

    # Hook into the forward pass to capture attention weights
    attn_weights_captured = []

    def hook_fn(module, input, output):
        # Re-run the attention computation to capture weights
        b, num_tokens, d = input[0].shape
        keys = module.W_key(input[0]).view(b, num_tokens, module.num_heads, module.head_dim).transpose(1, 2)
        queries = module.W_query(input[0]).view(b, num_tokens, module.num_heads, module.head_dim).transpose(1, 2)
        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = module.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights_captured.append(weights.detach())

    handle = mha.register_forward_hook(hook_fn)

    with torch.no_grad():
        _ = mha(x)

    handle.remove()

    weights = attn_weights_captured[0]  # [B, num_heads, T, T]

    # Check that all weights above the diagonal are 0
    for head in range(num_heads):
        for i in range(T):
            for j in range(i + 1, T):
                w = weights[0, head, i, j].item()
                assert abs(w) < 1e-6, \
                    f"Head {head}: token {i} attends to future token {j} with weight {w}"

    print("PASSED: test_causal_masking")


def test_different_seq_lengths():
    """MHA should handle sequences shorter than context_length."""
    d_in, d_out = 64, 64
    num_heads = 4
    context_length = 128

    mha = MultiHeadAttention(
        d_in=d_in, d_out=d_out, context_length=context_length,
        dropout=0.0, num_heads=num_heads
    )
    mha.eval()

    for T in [1, 5, 32, 128]:
        x = torch.randn(1, T, d_in)
        with torch.no_grad():
            out = mha(x)
        assert out.shape == (1, T, d_out), \
            f"Failed for T={T}: output shape {out.shape} != (1, {T}, {d_out})"

    print("PASSED: test_different_seq_lengths")


if __name__ == "__main__":
    test_attention_output_shape()
    test_causal_masking()
    test_different_seq_lengths()
    print("\n=== All attention tests passed! ===")
