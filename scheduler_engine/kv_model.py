import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger("velocityllm.kv_model")

_WEIGHT_EXT = (".safetensors", ".bin", ".pt", ".gguf")


@dataclass
class ModelFootprint:
    kv_bytes_per_token: int
    weights_mb: int
    source: str


def _text_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    for key in ("text_config", "language_config", "llm_config"):
        sub = cfg.get(key)
        if isinstance(sub, dict) and sub.get("num_hidden_layers"):
            return sub
    return cfg


def kv_bytes_from_config(cfg: Dict[str, Any], dtype_bytes: int = 2) -> Optional[int]:
    t = _text_cfg(cfg)
    layers = t.get("num_hidden_layers") or t.get("n_layer")
    heads = t.get("num_attention_heads") or t.get("n_head")
    hidden = t.get("hidden_size") or t.get("n_embd")
    kv_heads = t.get("num_key_value_heads") or heads
    head_dim = t.get("head_dim") or (hidden // heads if hidden and heads else None)
    if not (layers and kv_heads and head_dim):
        return None
    return int(2 * layers * kv_heads * head_dim * dtype_bytes)


def weights_mb_from_dir(path: str) -> Optional[int]:
    total = 0
    try:
        for name in os.listdir(path):
            if name.endswith(_WEIGHT_EXT):
                total += os.path.getsize(os.path.join(path, name))
    except OSError:
        return None
    return int(total / (1024 * 1024)) if total else None


def derive_footprint(model_path: str, overhead_mb: int = 600, dtype_bytes: int = 2) -> Optional[ModelFootprint]:
    cfg_path = os.path.join(str(model_path), "config.json")
    try:
        with open(cfg_path, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
    except (OSError, ValueError):
        return None
    kv = kv_bytes_from_config(cfg, dtype_bytes)
    if not kv:
        return None
    weights = weights_mb_from_dir(str(model_path))
    if weights is None:
        return None
    return ModelFootprint(kv, weights + overhead_mb, "config.json")
