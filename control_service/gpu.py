import asyncio

QUERY = ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"]


async def gpu_info(timeout: float = 4.0):
    try:
        proc = await asyncio.create_subprocess_exec(
            *QUERY, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout)
    except (FileNotFoundError, PermissionError, asyncio.TimeoutError, OSError):
        return None
    if proc.returncode != 0:
        return None
    line = out.decode(errors="ignore").strip().splitlines()
    if not line:
        return None
    parts = [p.strip() for p in line[0].split(",")]
    try:
        return {"name": parts[0], "total_mb": int(parts[1]), "used_mb": int(parts[2]), "util_percent": int(parts[3])}
    except (IndexError, ValueError):
        return None
