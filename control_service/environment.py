import asyncio
import importlib.metadata
import platform
import socket
import subprocess
import sys

from .config import ROOT


def _git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


async def _driver():
    try:
        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await asyncio.wait_for(proc.communicate(), 4)
        return out.decode().strip().splitlines()[0] if proc.returncode == 0 and out else None
    except (OSError, asyncio.TimeoutError, IndexError):
        return None


def _version(pkg):
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return None


async def capture():
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "vllm": _version("vllm"),
        "torch": _version("torch"),
        "git_commit": _git_commit(),
        "nvidia_driver": await _driver(),
    }
