import torch
import torch.nn as nn
from attention import MultiHeadAttention

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.shift = nn.Parameter(torch.zeros(emb_dim))
        self.scale = nn.Parameter(torch.ones(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        norm_x = (x - mean)/ torch.sqrt(var+self.eps)
        return self.scale * norm_x + self.shift

class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(emb_dim))
    
    def forward(self, x):
        mean_square = x.pow(2).mean(dim=-1, keepdim=True)
        rms = torch.sqrt(mean_square + self.eps)
        return x * (self.weight / rms)

class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))

class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        hidden_dim = int(8.0 * cfg["emb_dim"] / 3.0)

        self.w1 = nn.Linear(cfg["emb_dim"], hidden_dim, bias=False) # Path A (Gate)
        self.w3 = nn.Linear(cfg["emb_dim"], hidden_dim, bias=False) # Path B (Signal)
        self.w2 = nn.Linear(hidden_dim, cfg["emb_dim"], bias=False) # Compression
    
    def forward(self, x):
        gate = torch.nn.functional.silu(self.w1(x))
        signal = self.w3(x)

        return self.w2(gate * signal)

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            qkv_bias=cfg["qkv_bias"],
            dropout=cfg["drop_rate"],
            context_length=cfg["context_length"],
            use_rope=cfg["use_rope"],
            n_kv_heads=cfg["n_kv_heads"])

        if cfg["use_swiglu"]:
            self.ff = SwiGLU(cfg)
        else:
            self.ff = FeedForward(cfg)

        if cfg["use_rmsnorm"]:
            self.norm1 = RMSNorm(cfg["emb_dim"])
            self.norm2 = RMSNorm(cfg["emb_dim"])
        else:
            self.norm1 = LayerNorm(cfg["emb_dim"])
            self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x+shortcut

        return x

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.use_rope = cfg["use_rope"]
        if not self.use_rope:
            self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
    
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        if cfg["use_rmsnorm"]:
            self.final_norm = RMSNorm(cfg["emb_dim"])
        else:
            self.final_norm = LayerNorm(cfg["emb_dim"])
        
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
    
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        if self.use_rope:
            x = tok_embeds
        else:
            pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
            x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

        