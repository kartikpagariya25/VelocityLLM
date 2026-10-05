import asyncio
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import aiohttp

from .config import ROOT

sys.path.insert(0, str(ROOT / "load_generator"))
sys.path.insert(0, str(ROOT / "scheduler_engine"))
from generate_load import build_request_plan  # noqa: E402

SCENARIOS = {
    "flood": {"label": "Flood", "pattern": "flood", "mixed": False, "duration": 0.0, "description": "All requests arrive at the same instant"},
    "mixed": {"label": "Mixed", "pattern": "flood", "mixed": True, "duration": 0.0, "description": "Short and long prompts arrive together"},
    "steady": {"label": "Steady", "pattern": "constant", "mixed": False, "duration": 20.0, "description": "Requests spread evenly over 20 s"},
    "burst": {"label": "Burst", "pattern": "burst", "mixed": False, "duration": 20.0, "description": "Light random traffic with two sudden spikes, takes about a minute"},
}
PRESET_TOKENS = {"short": 64, "medium": 128, "long": 256}


def build_plan(scenario, users, seed, preset, text, max_tokens=None, vision=False, image_mix="mixed"):
    spec = SCENARIOS[scenario]
    max_tokens = max_tokens or PRESET_TOKENS.get(preset, 128)
    if max_tokens == PRESET_TOKENS.get(preset, 128) and preset == "custom" and text:
        max_tokens = min(256, max(32, len(text.strip()) // 2))
    args = SimpleNamespace(
        mixed_prompts=spec["mixed"], num_requests=users, seed=seed,
        max_tokens=max_tokens, pattern=spec["pattern"], duration=spec["duration"],
    )
    plan = build_request_plan(args)
    if preset == "custom" and text and not spec["mixed"]:
        for item in plan:
            item["prompt"] = text.strip()
    if vision:
        from .vision_load import attach_images

        attach_images(plan, image_mix, seed)
    return plan


CODE_REASONS = {"sla_risk": "sla_impossible", "memory_risk": "memory", "policy_limit": "policy_limit"}


def reject_reason(detail):
    code = detail.get("reason_code") if isinstance(detail, dict) else None
    if code in CODE_REASONS:
        return CODE_REASONS[code]
    reason = str(detail.get("reason", "") if isinstance(detail, dict) else detail).lower()
    if "queue full" in reason:
        return "queue_full"
    if "predicted latency" in reason:
        return "sla_impossible"
    if "burst" in reason:
        return "burst_shed"
    if "memory" in reason:
        return "memory"
    return "other"


async def run_load(base_url, plan, timeout_s, on_request=None, id_prefix="r"):
    loop = asyncio.get_running_loop()
    rows = []
    t0 = loop.time()
    connector = aiohttp.TCPConnector(limit=max(8, len(plan) * 2))
    timeout = aiohttp.ClientTimeout(total=timeout_s)

    async def one(session, i, item):
        delay = item.get("arrival_time", 0.0) - (loop.time() - t0)
        if delay > 0:
            await asyncio.sleep(delay)
        rid = f"{id_prefix}-{i + 1}"
        sent = loop.time() - t0
        row = {
            "index": i, "request_id": rid, "arrival_s": sent, "start_s": None, "end_s": sent,
            "status": "error", "http_status": None, "tokens": 0, "priority": "NORMAL",
            "reject_reason": None, "retry_after_s": None, "error": None, "is_long": bool(item.get("is_long")),
            "queue_s": None, "exec_s": None, "image_tokens": int(item.get("image_tokens") or 0),
            "device": item.get("device"),
        }
        try:
            payload = {"prompt": item["prompt"], "max_tokens": item["max_tokens"], "temperature": 0.0, "request_id": rid}
            if item.get("image"):
                payload["image"] = item["image"]
            async with session.post(f"{base_url}/generate", json=payload) as resp:
                body = None
                try:
                    body = await resp.json(content_type=None)
                except Exception:
                    body = None
                row["end_s"] = loop.time() - t0
                row["http_status"] = resp.status
                if resp.status == 200 and isinstance(body, dict):
                    row["status"] = "served"
                    row["tokens"] = int(body.get("tokens_generated") or 0)
                    row["queue_s"] = body.get("queue_time_seconds")
                    row["exec_s"] = body.get("execution_time_seconds")
                    if row["exec_s"] is not None:
                        row["start_s"] = max(row["arrival_s"], row["end_s"] - float(row["exec_s"]))
                elif resp.status == 429:
                    detail = body.get("detail") if isinstance(body, dict) else None
                    row["status"] = "rejected"
                    row["reject_reason"] = reject_reason(detail)
                    row["retry_after_s"] = (detail or {}).get("retry_after_seconds") if isinstance(detail, dict) else None
                else:
                    msg = body.get("detail") if isinstance(body, dict) else None
                    row["error"] = f"HTTP {resp.status}: {msg}" if msg else f"HTTP {resp.status}"
                    row["reject_reason"] = "gpu_oom" if resp.status == 503 else "http_error"
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            row["end_s"] = loop.time() - t0
            row["error"] = "request timed out"
            row["reject_reason"] = "timeout"
        except aiohttp.ClientError as e:
            row["end_s"] = loop.time() - t0
            row["error"] = f"{type(e).__name__}: {e}"
            row["reject_reason"] = "connection"
        rows.append(row)
        if on_request:
            await on_request(row)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [asyncio.create_task(one(session, i, item)) for i, item in enumerate(plan)]
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
    rows.sort(key=lambda r: r["index"])
    return rows, loop.time() - t0


def write_csv(path: Path, rows):
    import csv

    fields = ["index", "request_id", "arrival_s", "start_s", "end_s", "status", "http_status", "tokens",
              "reject_reason", "retry_after_s", "queue_s", "exec_s", "is_long", "image_tokens", "device", "error"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def shuffled(seq, seed):
    out = list(seq)
    random.Random(seed).shuffle(out)
    return out
