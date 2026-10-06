"""Vision benchmark core: start `velocityllm serve --vision`, send image requests, measure what comes back.

Uses only the public command and HTTP API of the installed package, so it works from a repo checkout or from pip.
Each request carries a different image, so no cache can hide the cost of reading it.
"""
import asyncio
import importlib.util
import os
import random
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import aiohttp
from PIL import Image, ImageDraw

# Run from a repository checkout (tools/vision_bench/..) as well as from a pip install.
if importlib.util.find_spec("scheduler_engine") is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scheduler_engine import vision  # noqa: E402

CODE_ROOT = str(Path(importlib.util.find_spec("scheduler_engine").submodule_search_locations[0]).parent)


def server_command():
    """`velocityllm serve` when the package is installed, else the same server straight from the repository."""
    try:
        installed = importlib.util.find_spec("velocityllm.cli") is not None
    except ModuleNotFoundError:
        installed = False
    return [sys.executable, "-m", "velocityllm", "serve"] if installed else [sys.executable, "-m", "scheduler_engine.server"]

POLICIES = ("static", "dynamic", "smart")
IMAGE_MIX = ((224, 0.50), (448, 0.35), (896, 0.15))  # same mix as the Arena's "mixed" images
PROMPTS = ("Describe this image in detail.", "What is the main subject of this image?",
           "Summarise what this image shows in two sentences.", "List the key elements you can see.")
COLOURS = ("red", "green", "blue", "orange", "purple", "teal", "gold", "crimson")


def synthetic_image(side, rnd):
    """A unique picture: coloured shapes on a plain background (seeded, so reproducible)."""
    img = Image.new("RGB", (side, side), rnd.choice(("white", "ivory", "lightgray", "lightyellow")))
    draw = ImageDraw.Draw(img)
    for _ in range(rnd.randint(4, 9)):
        x0, y0 = rnd.randint(0, side - 20), rnd.randint(0, side - 20)
        x1, y1 = rnd.randint(x0 + 10, side), rnd.randint(y0 + 10, side)
        shape = draw.ellipse if rnd.random() < 0.5 else draw.rectangle
        shape((x0, y0, x1, y1), fill=rnd.choice(COLOURS))
    return img


def make_plan(rate, duration, seed, sla_ms=8000, max_tokens=48):
    """Poisson arrivals, each with its own image. The same seed gives the same users for every policy."""
    rnd = random.Random(seed)
    sides = [s for s, _ in IMAGE_MIX]
    weights = [w for _, w in IMAGE_MIX]
    plan, t, i = [], 0.0, 0
    while True:
        t += rnd.expovariate(rate)
        if t > duration:
            return plan
        side = rnd.choices(sides, weights)[0]
        plan.append(dict(id=f"v{seed}-{i}", t=t, side=side, image=vision.to_base64(synthetic_image(side, rnd)),
                         image_tokens=vision.tokens_for_size(side, side), prompt=rnd.choice(PROMPTS),
                         max_tokens=max_tokens, sla_ms=sla_ms))
        i += 1


def refusal_label(detail):
    if isinstance(detail, dict):
        code = detail.get("reason_code")
        if code:
            return {"sla_risk": "would be late", "memory_risk": "memory", "queue_overload": "line too long",
                    "policy_limit": "background paused"}.get(code, code)
        text = str(detail.get("reason", "")).lower()
    else:
        text = str(detail).lower()
    if "queue full" in text or "burst" in text:
        return "line too long"
    if "predicted latency" in text:
        return "would be late"
    if "memory" in text:
        return "memory"
    return "other"


def clean_path():
    """Windows folders on a WSL PATH break the GPU engine (nvcc lookup); drop them."""
    return os.pathsep.join(p for p in os.environ.get("PATH", "").split(os.pathsep) if not p.startswith("/mnt/"))


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Engine:
    """The server (`velocityllm serve`, or scheduler_engine.server from a checkout) running as a child process."""

    def __init__(self, policy, *, mock=True, model_path=None, sla_ms=8000, max_concurrency=8, vision_on=True):
        self.policy, self.mock, self.model_path, self.vision_on = policy, mock, model_path, vision_on
        self.sla_ms, self.max_concurrency = sla_ms, max_concurrency
        self.port = free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        self.log = Path(tempfile.mkdtemp(prefix="velocity_vision_")) / f"{policy}.log"
        self.proc = None

    def command(self):
        cmd = server_command() + ["--policy", self.policy, "--port", str(self.port),
                                  "--host", "127.0.0.1", "--sla-ms", str(self.sla_ms), "--max-concurrency", str(self.max_concurrency)]
        if self.vision_on:
            cmd.append("--vision")
        return cmd + (["--mock"] if self.mock else ["--model-path", str(self.model_path)])

    def tail(self, n=12):
        try:
            return "\n".join(self.log.read_text(errors="replace").splitlines()[-n:])
        except OSError:
            return ""

    def __enter__(self):
        import json
        import urllib.request

        env = dict(os.environ, PATH=clean_path(), PYTHONPATH=os.pathsep.join(filter(None, [CODE_ROOT, os.environ.get("PYTHONPATH")])))
        with open(self.log, "wb") as out:
            self.proc = subprocess.Popen(self.command(), stdout=out, stderr=subprocess.STDOUT, env=env)
        deadline = time.time() + (60 if self.mock else 480)
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"The {self.policy} server stopped while starting:\n{self.tail()}")
            try:
                with urllib.request.urlopen(f"{self.base}/health", timeout=2) as r:
                    health = json.load(r)
                if health.get("policy", "").startswith(self.policy[:4]):
                    break
            except Exception:
                time.sleep(0.5)
        else:
            self.__exit__()
            raise RuntimeError(f"The {self.policy} server did not become ready in time:\n{self.tail()}")
        if self.vision_on and not health.get("vision"):
            self.__exit__()
            raise RuntimeError("The server started without vision support (/health says vision: false).")
        if not self.mock:
            text = self.log.read_text(errors="replace")
            if "Falling back to MockBackend" in text or "AsyncLLMEngine initialized successfully" not in text:
                self.__exit__()
                raise RuntimeError('vLLM did not load, so this would be a fake run. Install it with: pip install "velocityllm[gpu]"')
        return self

    def __exit__(self, *exc):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        time.sleep(1 if self.mock else 6)  # let the GPU memory settle before the next policy
        return False


async def _send(session, base, spec, t0, out):
    await asyncio.sleep(max(0.0, spec["t"] - (time.time() - t0)))
    start = time.time()
    status, label, reported, text = "failed", "", None, ""
    try:
        async with session.post(f"{base}/generate", headers={"X-SLA-Ms": str(spec["sla_ms"])},
                                json={"prompt": spec["prompt"], "max_tokens": spec["max_tokens"], "image": spec["image"]}) as r:
            body = await r.json(content_type=None)
            if r.status == 200:
                status, reported, text = "served", body.get("image_tokens"), str(body.get("response", ""))
            elif r.status == 429:
                detail = body.get("detail", body) if isinstance(body, dict) else body
                status, label = "refused", refusal_label(detail)
            else:
                label = f"HTTP {r.status}"
    except asyncio.TimeoutError:
        label = "timeout"
    except aiohttp.ClientError as e:
        label = type(e).__name__
    latency = time.time() - start
    on_time = status == "served" and latency * 1000 <= spec["sla_ms"]
    out.append(dict(id=spec["id"], side=spec["side"], image_tokens=spec["image_tokens"], reported_tokens=reported,
                    sent_at=round(spec["t"], 2), status=status, latency=round(latency, 3), on_time=on_time,
                    outcome="on time" if on_time else ("late" if status == "served" else ("refused" if status == "refused" else "failed")),
                    why=label, answer=text[:200]))


async def _measure(base, plan, warm_plan, progress, label):
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0), timeout=aiohttp.ClientTimeout(total=180)) as session:
        if warm_plan:
            progress(f"{label}: warming up", 0.0)
            t0 = time.time()
            await asyncio.gather(*[_send(session, base, s, t0, []) for s in warm_plan])
            await asyncio.sleep(1)
        out, t0 = [], time.time()
        tasks = [asyncio.create_task(_send(session, base, s, t0, out)) for s in plan]

        async def report():
            while not all(t.done() for t in tasks):
                progress(f"{label}: {len(out)} of {len(plan)} requests finished", len(out) / max(1, len(plan)))
                await asyncio.sleep(0.4)

        reporter = asyncio.create_task(report())
        await asyncio.gather(*tasks)
        wall = time.time() - t0
        reporter.cancel()
        stats = {}
        try:
            async with session.get(f"{base}/stats") as r:
                stats = await r.json()
        except aiohttp.ClientError:
            pass
        return out, wall, stats


def pct(part, whole):
    return round(100 * part / whole) if whole else None


def percentile(values, q):
    v = sorted(values)
    return round(v[min(len(v) - 1, int(len(v) * q))], 2) if v else None


def summarize(rows, wall, stats):
    served = [r for r in rows if r["status"] == "served"]
    by_size = {}
    for side, _ in IMAGE_MIX:
        group = [r for r in rows if r["side"] == side]
        lat = [r["latency"] for r in group if r["status"] == "served"]
        by_size[side] = dict(offered=len(group), answered=len(lat), refused=sum(r["status"] == "refused" for r in group),
                             on_time_pct=pct(sum(r["on_time"] for r in group), len(group)), median_s=percentile(lat, 0.5),
                             tokens=vision.tokens_for_size(side, side))
    planned = sum(r["image_tokens"] for r in served)
    reported = sum(r["reported_tokens"] or 0 for r in served)
    return dict(
        offered=len(rows), answered=len(served), refused=sum(r["status"] == "refused" for r in rows),
        failed=sum(r["status"] == "failed" for r in rows), on_time=sum(r["on_time"] for r in rows),
        on_time_pct=pct(sum(r["on_time"] for r in rows), len(rows)),
        p99_s=percentile([r["latency"] for r in served], 0.99), by_size=by_size,
        image_tokens_sent=planned, image_tokens_reported=reported, server_image_tokens=stats.get("total_image_tokens"),
        tokens_ok=(planned == reported), wall_s=round(wall, 1))


def run_policy(policy, rate, duration, seed, *, mock=True, model_path=None, sla_ms=8000, max_concurrency=8, warmup_s=4,
               progress=lambda text, fraction: None):
    plan = make_plan(rate, duration, seed, sla_ms)
    warm = make_plan(max(rate / 2, 0.5), warmup_s, seed + 1000, sla_ms) if warmup_s else []
    label = policy.capitalize()
    progress(f"{label}: starting the server", 0.0)
    with Engine(policy, mock=mock, model_path=model_path, sla_ms=sla_ms, max_concurrency=max_concurrency) as eng:
        rows, wall, stats = asyncio.run(_measure(eng.base, plan, warm, progress, label))
    return dict(policy=policy, rows=rows, summary=summarize(rows, wall, stats))


def run_all(policies, rate, duration, seed, **kw):
    progress = kw.pop("progress", lambda text, fraction: None)
    results = {}
    for i, p in enumerate(policies):
        results[p] = run_policy(p, rate, duration, seed, progress=lambda text, f, i=i: progress(text, (i + f) / len(policies)), **kw)
    progress("Done", 1.0)
    return results


def smoke(policy="dynamic", *, mock=True, model_path=None, image_path=None, max_concurrency=8):
    """Four requests with pictures whose answer is known, so a human can see the model really looks at the image."""
    cases = []
    if image_path:
        img = Image.open(image_path).convert("RGB")
        cases.append(("your image", vision.to_base64(img), "Describe this image in two sentences."))
    else:
        for colour in ("red", "blue", "green"):
            img = Image.new("RGB", (448, 448), "white")
            ImageDraw.Draw(img).ellipse((100, 100, 348, 348), fill=colour)
            cases.append((f"a {colour} circle on white", vision.to_base64(img), "What colour is the circle? Answer in one word."))
        img = Image.new("RGB", (448, 448), "white")
        ImageDraw.Draw(img).rectangle((100, 100, 348, 348), fill="orange")
        cases.append(("an orange square on white", vision.to_base64(img), "What shape is this and what colour? Answer briefly."))

    async def go(base):
        out = []
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as session:
            for name, b64, question in cases:
                start = time.time()
                async with session.post(f"{base}/generate", json={"prompt": question, "max_tokens": 40, "image": b64}) as r:
                    body = await r.json(content_type=None)
                out.append(dict(picture=name, question=question, status=r.status, answer=str(body.get("response", body))[:200].strip(),
                                image_tokens=body.get("image_tokens"), seconds=round(time.time() - start, 2)))
        return out

    with Engine(policy, mock=mock, model_path=model_path, max_concurrency=max_concurrency) as eng:
        return asyncio.run(go(eng.base))
