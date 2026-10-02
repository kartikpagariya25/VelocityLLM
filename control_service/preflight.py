import importlib.util
import shutil

from .engine import free_port
from .gpu import gpu_info
from .models import describe, discover


def check(cid, label, ok, detail, level="error"):
    return {"id": cid, "label": label, "ok": bool(ok), "level": level, "detail": detail}


async def run_checks(settings, active_run=None, mock=None):
    mock = settings.mock if mock is None else mock
    checks = []
    gpu = await gpu_info()
    if mock:
        checks.append(check("gpu", "GPU", True, "Mock engine selected, no GPU needed", "info"))
    elif gpu:
        free = gpu["total_mb"] - gpu["used_mb"]
        checks.append(check("gpu", "GPU", True, f"{gpu['name']}, {gpu['total_mb']} MB total, {free} MB free"))
        checks.append(check("gpu_free", "GPU memory free", free >= 3000, f"{free} MB free; close other GPU programs if a model fails to load", "warn"))
    else:
        checks.append(check("gpu", "GPU", False, "nvidia-smi found no NVIDIA GPU on this machine. Start the service with --mock to test the pipeline without one."))
    if mock:
        checks.append(check("vllm", "vLLM", True, "Not needed for the mock engine", "info"))
    else:
        has = importlib.util.find_spec("vllm") is not None
        checks.append(check("vllm", "vLLM installed", has, "Found" if has else "Python package vllm is missing in this environment (pip install vllm)"))
    models = discover(settings)
    info = [describe(n, p, mock) for n, p in models.items()]
    usable = [m for m in info if m["available"]]
    checks.append(
        check(
            "models", "Models", bool(usable),
            f"{len(usable)} usable: {', '.join(m['id'] for m in usable)}" if usable
            else "No model folders found. Put them under models/ (each with config.json) or start with --model NAME=PATH",
        )
    )
    try:
        settings.results_dir.mkdir(parents=True, exist_ok=True)
        free_gb = shutil.disk_usage(settings.results_dir).free / 1e9
        checks.append(check("disk", "Results folder", free_gb > 0.5, f"{free_gb:.1f} GB free at {settings.results_dir}"))
    except OSError as e:
        checks.append(check("disk", "Results folder", False, f"Cannot write to {settings.results_dir}: {e}"))
    try:
        checks.append(check("port", "Engine port", True, f"Port {free_port(settings.engine_base_port)} is free", "info"))
    except Exception as e:
        checks.append(check("port", "Engine port", False, str(e)))
    checks.append(check("busy", "No other run active", active_run is None, "Idle" if active_run is None else f"Run {active_run} is in progress"))
    blocking = [c for c in checks if not c["ok"] and c["level"] == "error"]
    return {"ok": not blocking, "checks": checks, "gpu": gpu, "mock": mock, "models": info}
