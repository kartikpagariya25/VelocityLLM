"""Automatic hardware and model detection, so nothing is tied to one machine.

Model lookup order:  --model-path flag  ->  VELOCITY_MODEL_PATH env var
                     ->  a model folder in ./models or ~/velocity-models
                     ->  asking the user in the terminal (interactive only)
"""
import os
import shutil
import subprocess
from pathlib import Path

MODEL_ENV = "VELOCITY_MODEL_PATH"


def detect_gpu():
    """Return {'name', 'memory_mb'} for the first NVIDIA GPU, or None if there is none."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip().splitlines()
        if not out:
            return None
        name, mem = [p.strip() for p in out[0].split(",", 1)]
        return {"name": name, "memory_mb": int(float(mem))}
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def find_local_model(search_dirs):
    """First sub-folder that looks like a model (has config.json), or None."""
    for base in search_dirs:
        base = Path(base).expanduser()
        if not base.is_dir():
            continue
        for sub in sorted(base.iterdir()):
            if sub.is_dir() and (sub / "config.json").is_file():
                return str(sub)
    return None


def default_search_dirs():
    return [Path.cwd() / "models", Path.home() / "velocity-models"]


def resolve_model_path(explicit=None, interactive=False, ask=input):
    """Return a model path, or None if nothing was found and nothing was typed."""
    if explicit:
        return explicit
    env = os.environ.get(MODEL_ENV)
    if env:
        return env
    found = find_local_model(default_search_dirs())
    if found:
        return found
    if interactive:
        typed = ask("No model found. Enter the model folder path (or press Enter to use the simulated engine): ").strip()
        return typed or None
    return None
