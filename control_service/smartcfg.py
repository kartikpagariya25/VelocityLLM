"""Per-model settings for the Smart policy, derived from the model folder itself.

Smart estimates GPU memory in tokens, so it needs the KV-cache cost of one token and the size of the
weights. Both come from the model's own config.json and weight files, so any model found under
models/ works without editing a config by hand.
"""
import json
import os
from pathlib import Path

from scheduler_engine.kv_model import kv_bytes_from_config

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
    return kv_bytes_from_config(cfg, kv_dtype_bytes())


def kv_dtype_bytes() -> int:
    return 1 if os.environ.get("VELOCITY_KV_CACHE_DTYPE", "auto") == "fp8" else DTYPE_BYTES


def env_overrides() -> dict:
    out = {}
    for pair in os.environ.get("VELOCITY_SMART_OVERRIDES", "").split(","):
        key, sep, value = pair.partition("=")
        if sep and key.strip():
            out[key.strip()] = value.strip()
    return out


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
        smart = {**st, **env_overrides()}
        text = "smart:\n" + "".join(f"  {k}: {v}\n" for k, v in smart.items())
        if os.environ.get("VELOCITY_KV_CACHE_DTYPE", "auto") == "fp8":
            text += "kv_cache_dtype: fp8\n"
        path.write_text(text)
    except OSError:
        return None, None
    return path, st
