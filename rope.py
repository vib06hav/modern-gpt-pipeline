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
    b, num_tokens, num_heads, head_dim = x.shape
    cos = cos[:num_tokens, :].to(x.device)
    sin = sin[:num_tokens, :].to(x.device)

    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)

    half = head_dim // 2
    x_first_half = x[..., :half]
    x_second_half = x[..., half:]

    x_rotated = torch.cat(
        [-x_second_half, x_first_half], dim=-1
    )

    out = (x*cos) + (x_rotated * sin)
    return out