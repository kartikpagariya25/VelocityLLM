#!/usr/bin/env python3
"""Print a smart-scheduler YAML for a local model dir, derived from its config.json + weight file sizes.
usage: modelcfg.py <model_dir> [overhead_mb]"""
import glob, json, os, sys

d = sys.argv[1]
overhead = int(sys.argv[2]) if len(sys.argv) > 2 else 700
cfg = json.load(open(os.path.join(d, "config.json")))
cfg = cfg.get("text_config", cfg)
layers = cfg["num_hidden_layers"]
heads = cfg["num_attention_heads"]
kv_heads = cfg.get("num_key_value_heads", heads)
head_dim = cfg.get("head_dim") or cfg["hidden_size"] // heads
dtype_bytes = 2  # bf16/fp16 (vLLM default for these models)
kv_bytes = 2 * layers * kv_heads * head_dim * dtype_bytes
files = glob.glob(os.path.join(d, "*.safetensors")) or glob.glob(os.path.join(d, "*.bin"))
weights_mb = int(sum(os.path.getsize(f) for f in files) / (1024 * 1024))
print("smart:")
print(f"  kv_bytes_per_token: {kv_bytes}    # 2*{layers}L*{kv_heads}kv*{head_dim}hd*{dtype_bytes}B")
print(f"  model_weights_mb: {weights_mb + overhead}    # {weights_mb} MB weights + {overhead} MB overhead guess")
