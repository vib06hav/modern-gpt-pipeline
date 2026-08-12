import torch
import torch.nn as nn
from rope import apply_rotary_emb, precompute_rope_params

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False, use_rope=False, num_kv_heads=None):
        super().__init__()

        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, self.num_kv_heads*self.head_dim, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, self.num_kv_heads*self.head_dim, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))
        self.use_rope = use_rope
        if self.use_rope:
            cos, sin = precompute_rope_params(self.head_dim, context_length)
            self.register_buffer("cos", cos)
            self.register_buffer("sin", sin)

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_kv_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_kv_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        if self.num_queries_per_kv > 1:
            keys = torch.repeat_interleave(keys, self.num_queries_per_kv, dim=2)
            values = torch.repeat_interleave(values, self.num_queries_per_kv, dim=2)
        
        keys = keys.transpose(1,2)
        values = values.transpose(1,2)
        queries = queries.transpose(1,2)

        if self.use_rope:
            queries = apply_rotary_emb(queries, self.cos, self.sin)
            keys = apply_rotary_emb(keys, self.cos, self.sin)
        
        attn_scores = queries @ keys.transpose(2,3)

        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context_vec = (attn_weights @ values).transpose(1,2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        
        return context_vec
    


         
        

        