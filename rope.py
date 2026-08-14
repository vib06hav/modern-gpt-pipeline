import torch

def precompute_rope_params(head_dim, context_length, theta_base=10000.0):
    half_dim = head_dim // 2
    inv_freq = 1.0 / (theta_base ** (torch.arange(half_dim, dtype=torch.float32) / half_dim))
    positions = torch.arange(context_length, dtype=torch.float32)

    angles = torch.outer(positions, inv_freq)

    angles = torch.cat((angles, angles), dim=-1)
    
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return cos, sin

def apply_rotary_emb(x, cos, sin):
    # x is now shaped [b, num_heads, num_tokens, head_dim]
    b, num_heads, num_tokens, head_dim = x.shape
    
    # cos and sin are passed in already sliced from attention.py, shape [num_tokens, head_dim]
    # We unsqueeze to [1, 1, num_tokens, head_dim] to broadcast across batch and heads
    cos = cos.unsqueeze(0).unsqueeze(1).to(x.device)
    sin = sin.unsqueeze(0).unsqueeze(1).to(x.device)

    half = head_dim // 2
    x_first_half = x[..., :half]
    x_second_half = x[..., half:]

    x_rotated = torch.cat(
        [-x_second_half, x_first_half], dim=-1
    )

    out = (x*cos) + (x_rotated * sin)
    return out