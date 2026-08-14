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
        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)
        self.ptr_current_pos = 0
    
    def reset_cache(self):
        self.cache_k = None
        self.cache_v = None
        self.ptr_current_pos = 0

    def forward(self, x, use_cache=False):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_kv_heads, self.head_dim).transpose(1,2)
        values = values.view(b, num_tokens, self.num_kv_heads, self.head_dim).transpose(1,2)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1,2)

        if self.use_rope:
            if use_cache:
                cos = self.cos[self.ptr_current_pos : self.ptr_current_pos + num_tokens]
                sin = self.sin[self.ptr_current_pos : self.ptr_current_pos + num_tokens]
            else:
                cos = self.cos[:num_tokens]
                sin = self.sin[:num_tokens]
            queries = apply_rotary_emb(queries, cos, sin)
            keys = apply_rotary_emb(keys, cos, sin)

        if use_cache:
            if self.cache_k is None:
                self.cache_k, self.cache_v = keys, values
            else:
                self.cache_k = torch.cat([self.cache_k, keys], dim=2)
                self.cache_v = torch.cat([self.cache_v, values], dim=2)

            keys_base, values_base = self.cache_k, self.cache_v
            self.ptr_current_pos += num_tokens
        else:
            keys_base, values_base = keys, values
            self.ptr_current_pos = 0

        if self.num_queries_per_kv > 1:
            keys = torch.repeat_interleave(keys_base, self.num_queries_per_kv, dim=1)
            values = torch.repeat_interleave(values_base, self.num_queries_per_kv, dim=1)
        else:
            keys, values = keys_base, values_base
        
        attn_scores = queries @ keys.transpose(2,3)

        num_tokens_Q = queries.shape[2]
        num_tokens_K = keys.shape[2]

        if num_tokens_Q > 1:
            mask_bool = self.mask.bool()[:num_tokens_Q, :num_tokens_K]
            attn_scores.masked_fill_(mask_bool, -torch.inf)
            
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context_vec = (attn_weights @ values).transpose(1,2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        
        return context_vec
    


         
        

        