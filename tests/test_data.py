"""
Tests for data.py
- input/target are shifted by exactly 1 position
- dataset length is correct
- dataloader produces correct batch shapes
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import tiktoken
import torch
from data import GPTDatasetV1, create_dataloader


def test_input_target_shift():
    """Target tokens should be the input tokens shifted by 1 position."""
    tokenizer = tiktoken.get_encoding("gpt2")
    text = "Hello world, this is a simple test sentence for the GPT dataset."
    max_length = 8
    stride = 8

    dataset = GPTDatasetV1(text, tokenizer, max_length, stride)
    token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

    # For the first window: input = tokens[0:8], target = tokens[1:9]
    input_ids, target_ids = dataset[0]

    assert input_ids.tolist() == token_ids[0:max_length], \
        f"Input mismatch: {input_ids.tolist()} != {token_ids[0:max_length]}"
    assert target_ids.tolist() == token_ids[1:max_length + 1], \
        f"Target mismatch: {target_ids.tolist()} != {token_ids[1:max_length + 1]}"

    # Verify every position: target[i] == input[i+1] (except the last target token)
    for i in range(len(input_ids) - 1):
        assert target_ids[i].item() == input_ids[i + 1].item(), \
            f"Position {i}: target[{i}]={target_ids[i].item()} != input[{i+1}]={input_ids[i+1].item()}"

    print("PASSED: test_input_target_shift")


def test_dataset_length():
    """Dataset length should match the number of sliding windows."""
    tokenizer = tiktoken.get_encoding("gpt2")
    text = "A B C D E F G H I J " * 50  # repeated text
    max_length = 4
    stride = 2

    dataset = GPTDatasetV1(text, tokenizer, max_length, stride)
    token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

    expected_len = len(range(0, len(token_ids) - max_length, stride))
    assert len(dataset) == expected_len, \
        f"Dataset length {len(dataset)} != expected {expected_len}"

    print("PASSED: test_dataset_length")


def test_dataloader_shapes():
    """DataLoader should produce batches with correct shapes [batch_size, max_length]."""
    text = "The quick brown fox jumps over the lazy dog. " * 100
    batch_size = 4
    max_length = 16

    dataloader = create_dataloader(text, batch_size=batch_size, max_length=max_length, stride=max_length)

    inputs, targets = next(iter(dataloader))

    assert inputs.shape == (batch_size, max_length), \
        f"Input shape {inputs.shape} != expected ({batch_size}, {max_length})"
    assert targets.shape == (batch_size, max_length), \
        f"Target shape {targets.shape} != expected ({batch_size}, {max_length})"
    assert inputs.dtype == torch.long, f"Input dtype {inputs.dtype} != torch.long"
    assert targets.dtype == torch.long, f"Target dtype {targets.dtype} != torch.long"

    print("PASSED: test_dataloader_shapes")


if __name__ == "__main__":
    test_input_target_shift()
    test_dataset_length()
    test_dataloader_shapes()
    print("\n=== All data tests passed! ===")
