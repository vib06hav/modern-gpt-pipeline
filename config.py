# config.py

def get_config(
    emb_dim=768,
    n_layers=12,
    n_heads=12,
    n_kv_heads=4,
    context_length=1024,
):
    return {
        "vocab_size": 50257,
        "context_length": context_length,
        "emb_dim": emb_dim,
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "n_layers": n_layers,
        "drop_rate": 0.1,
        "qkv_bias": False,
        "use_rope": True,
        "use_rmsnorm": True,
        "use_swiglu": True,
    }

# Kept for backward compatibility with tests
GPT_CONFIG_124M = get_config()
