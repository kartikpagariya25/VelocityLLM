"""Per-model settings for the Smart policy, derived from the model folder itself.

Smart estimates GPU memory in tokens, so it needs the KV-cache cost of one token and the size of the
weights. Both come from the model's own config.json and weight files, so any model found under
models/ works without editing a config by hand.
"""
import json
from pathlib import Path

from .models import resolve, weights_mb

# Runtime memory a loaded engine uses besides weights and KV cache (activations, CUDA graphs).
# Measured on four models it ranged from ~0 to ~470 MB; 300 keeps the KV estimate on the safe (low) side.
OVERHEAD_MB = 300
DTYPE_BYTES = 2  # vLLM serves these models in bf16/fp16


def kv_bytes_per_token(model_dir: Path):
    try:
        cfg = json.loads((model_dir / "config.json").read_text())
    except (OSError, ValueError):
        return None
    cfg = cfg.get("text_config", cfg)
    try:
        layers = int(cfg["num_hidden_layers"])
        heads = int(cfg["num_attention_heads"])
        kv_heads = int(cfg.get("num_key_value_heads") or heads)
        head_dim = int(cfg.get("head_dim") or int(cfg["hidden_size"]) // heads)
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
    return 2 * layers * kv_heads * head_dim * DTYPE_BYTES


def smart_settings(model_path: str):
    """Returns {"kv_bytes_per_token", "model_weights_mb"} or None when the model folder cannot be read."""
    folder = resolve(model_path)
    if not folder.is_dir():
        return None
    kv = kv_bytes_per_token(folder)
    size = weights_mb(folder)
    if not kv or not size:
        return None
    return {"kv_bytes_per_token": kv, "model_weights_mb": size + OVERHEAD_MB}


def write_smart_config(model_path: str, directory: Path):
    """Writes smart_config.yaml next to the run's files. Returns (path, settings) or (None, None)."""
    st = smart_settings(model_path)
    if st is None:
        return None, None
    path = Path(directory) / "smart_config.yaml"
    try:
        path.write_text("smart:\n" + "".join(f"  {k}: {v}\n" for k, v in st.items()))
    except OSError:
        return None, None
    return path, st
