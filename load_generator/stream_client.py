"""
Streaming load client shared by the rate sweep and the server comparison tools.

One workload (Poisson arrivals, chat-like prompt and output lengths) can be replayed against the VelocityLLM
server, a vLLM OpenAI-compatible server, or a Triton Inference Server, and scored identically.
"""

import asyncio
import json
import math
import random
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import aiohttp

PROTOCOLS = ("velocity", "openai", "triton-generate", "triton-infer")

VOCABULARY = (
    "system scheduler request latency memory token batch queue model prompt kernel cache decode prefill "
    "throughput device server network stream signal buffer window policy budget limit target answer "
    "question context summary report planning review design vector matrix gradient tensor layer attention "
    "language training dataset metric sample region market weather travel recipe garden history science"
).split()


@dataclass
class Target:
    name: str
    protocol: str
    url: str
    model: str = "model"

    @staticmethod
    def parse(spec: str) -> "Target":
        """NAME=PROTOCOL@URL or NAME=PROTOCOL@URL#MODEL"""
        name, _, rest = spec.partition("=")
        protocol, _, address = rest.partition("@")
        url, _, model = address.partition("#")
        if not name or protocol not in PROTOCOLS or not url:
            raise ValueError(f"target must look like NAME=PROTOCOL@URL[#MODEL] with PROTOCOL in {PROTOCOLS}, got: {spec}")
        return Target(name=name, protocol=protocol, url=url.rstrip("/"), model=model or "model")


@dataclass
class PlannedRequest:
    index: int
    arrival: float
    prompt: str
    max_tokens: int


@dataclass
class Sample:
    index: int
    arrival: float
    status: str
    e2e_seconds: float = 0.0
    ttft_seconds: Optional[float] = None
    tokens: int = 0
    per_token_ms: Optional[float] = None
    ttft_is_estimate: bool = False
    error: str = ""


def make_prompt(rng: random.Random, words: int) -> str:
    return " ".join(rng.choice(VOCABULARY) for _ in range(max(1, words)))


def load_length_pairs(path: str) -> List[Tuple[int, int]]:
    """JSON list of [prompt_tokens, output_tokens] pairs, for example exported from a chat dataset."""
    with open(path) as f:
        return [(int(a), int(b)) for a, b in json.load(f)]


def build_workload(
    count: int,
    rps: float,
    seed: int = 42,
    lengths: Optional[Sequence[Tuple[int, int]]] = None,
    max_output_tokens: int = 256,
) -> List[PlannedRequest]:
    """Poisson arrivals at the given rate; lengths drawn from a log-normal chat-like distribution or a length file."""
    rng = random.Random(seed)
    plan: List[PlannedRequest] = []
    clock = 0.0
    for i in range(count):
        clock += rng.expovariate(rps)
        if lengths:
            prompt_tokens, output_tokens = lengths[rng.randrange(len(lengths))]
        else:
            prompt_tokens = rng.lognormvariate(math.log(40), 0.8)
            output_tokens = rng.lognormvariate(math.log(80), 0.9)
        prompt_tokens = int(min(max(prompt_tokens, 4), 512))
        output_tokens = int(min(max(output_tokens, 8), max_output_tokens))
        plan.append(PlannedRequest(i, clock, make_prompt(rng, int(prompt_tokens * 0.75)), output_tokens))
    return plan


def _sse_payloads(raw_line: bytes):
    line = raw_line.decode("utf-8", "ignore").strip()
    if not line.startswith("data:"):
        return None
    body = line[5:].strip()
    if not body or body == "[DONE]":
        return None
    try:
        return json.loads(body)
    except ValueError:
        return None


async def _stream_tokens(resp, token_of) -> Tuple[Optional[float], int, Optional[str]]:
    """Reads an SSE response; returns (time of first token, token count, error text)."""
    first = None
    tokens = 0
    usage_tokens = None
    async for raw in resp.content:
        payload = _sse_payloads(raw)
        if payload is None:
            continue
        if isinstance(payload, dict) and payload.get("error"):
            return first, tokens, str(payload.get("reason") or payload["error"])
        if isinstance(payload, dict) and payload.get("usage"):
            usage_tokens = payload["usage"].get("completion_tokens")
        if token_of(payload):
            if first is None:
                first = time.perf_counter()
            tokens += 1
    return first, (usage_tokens if usage_tokens is not None else tokens), None


async def send_request(session: aiohttp.ClientSession, target: Target, item: PlannedRequest, headers: Dict[str, str]) -> Sample:
    sample = Sample(index=item.index, arrival=item.arrival, status="error")
    start = time.perf_counter()
    try:
        if target.protocol == "velocity":
            url = f"{target.url}/generate/stream"
            body = {"prompt": item.prompt, "max_tokens": item.max_tokens, "stream": True}
            token_of = lambda p: isinstance(p, dict) and "delta" in p and not p.get("is_finished") and p.get("delta") != ""
        elif target.protocol == "openai":
            url = f"{target.url}/v1/completions"
            body = {
                "model": target.model, "prompt": item.prompt, "max_tokens": item.max_tokens,
                "temperature": 0.0, "stream": True, "stream_options": {"include_usage": True},
            }
            token_of = lambda p: bool(p.get("choices")) and bool(p["choices"][0].get("text"))
        elif target.protocol == "triton-generate":
            url = f"{target.url}/v2/models/{target.model}/generate_stream"
            body = {"text_input": item.prompt, "parameters": {"stream": True, "max_tokens": item.max_tokens, "temperature": 0.0}}
            token_of = lambda p: bool(p.get("text_output"))
        else:
            return await _send_triton_infer(session, target, item, headers)

        async with session.post(url, json=body, headers=headers) as resp:
            if resp.status == 429:
                sample.status = "rejected"
                sample.e2e_seconds = time.perf_counter() - start
                return sample
            if resp.status != 200:
                sample.error = f"HTTP {resp.status}"
                sample.e2e_seconds = time.perf_counter() - start
                return sample
            first, tokens, error = await _stream_tokens(resp, token_of)
        end = time.perf_counter()
        sample.e2e_seconds = end - start
        if error is not None:
            sample.status = "rejected"
            sample.error = error
            return sample
        sample.status = "ok"
        sample.tokens = tokens
        if first is not None:
            sample.ttft_seconds = first - start
            if tokens > 1 and end > first:
                sample.per_token_ms = (end - first) / (tokens - 1) * 1000.0
    except Exception as err:
        sample.error = str(err)[:120]
        sample.e2e_seconds = time.perf_counter() - start
    return sample


async def _send_triton_infer(session, target: Target, item: PlannedRequest, headers: Dict[str, str]) -> Sample:
    sample = Sample(index=item.index, arrival=item.arrival, status="error", ttft_is_estimate=True)
    start = time.perf_counter()
    body = {
        "inputs": [
            {"name": "prompt", "shape": [1, 1], "datatype": "BYTES", "data": [item.prompt]},
            {"name": "max_tokens", "shape": [1, 1], "datatype": "INT32", "data": [item.max_tokens]},
        ]
    }
    try:
        async with session.post(f"{target.url}/v2/models/{target.model}/infer", json=body, headers=headers) as resp:
            sample.e2e_seconds = time.perf_counter() - start
            if resp.status != 200:
                sample.error = f"HTTP {resp.status}"
                return sample
            data = await resp.json()
        outputs = {o["name"]: o["data"] for o in data.get("outputs", [])}
        sample.tokens = int(outputs.get("num_tokens", [0])[0])
        sample.status = "ok"
        sample.ttft_seconds = sample.e2e_seconds
    except Exception as err:
        sample.error = str(err)[:120]
        sample.e2e_seconds = time.perf_counter() - start
    return sample


async def run_workload(target: Target, plan: Sequence[PlannedRequest], api_key: Optional[str] = None, timeout: float = 300.0):
    """Fires every request at its scheduled arrival time. Returns (samples, wall_seconds)."""
    headers = {"X-API-Key": api_key} if api_key else {}
    connector = aiohttp.TCPConnector(limit=0)
    client_timeout = aiohttp.ClientTimeout(total=timeout)
    async with aiohttp.ClientSession(connector=connector, timeout=client_timeout) as session:
        wall_start = time.perf_counter()

        async def fire(item: PlannedRequest) -> Sample:
            delay = item.arrival - (time.perf_counter() - wall_start)
            if delay > 0:
                await asyncio.sleep(delay)
            return await send_request(session, target, item, headers)

        samples = await asyncio.gather(*(fire(item) for item in plan))
        wall = time.perf_counter() - wall_start
    return list(samples), wall


def nearest_rank(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    return sorted_values[min(int(len(sorted_values) * q), len(sorted_values) - 1)]


def summarize(samples: Sequence[Sample], wall_seconds: float, sla_seconds: float, ttft_sla_seconds: Optional[float] = None) -> Dict[str, float]:
    """Scores one run. A request is on time when it completed within the SLA (and the TTFT SLA if one is set)."""
    ok = [s for s in samples if s.status == "ok"]
    e2e = sorted(s.e2e_seconds for s in ok)
    ttft = sorted(s.ttft_seconds for s in ok if s.ttft_seconds is not None)
    per_token = [s.per_token_ms for s in ok if s.per_token_ms]
    on_time = [
        s for s in ok
        if s.e2e_seconds <= sla_seconds and (ttft_sla_seconds is None or s.ttft_seconds is None or s.ttft_seconds <= ttft_sla_seconds)
    ]
    wall = max(wall_seconds, 1e-9)
    offered = len(samples)
    return {
        "offered": offered,
        "served": len(ok),
        "rejected": sum(1 for s in samples if s.status == "rejected"),
        "errors": sum(1 for s in samples if s.status == "error"),
        "wall_seconds": wall,
        "requests_per_s": len(ok) / wall,
        "tokens_per_s": sum(s.tokens for s in ok) / wall,
        "goodput_requests_per_s": len(on_time) / wall,
        "goodput_tokens_per_s": sum(s.tokens for s in on_time) / wall,
        "sla_attainment_percent": 100.0 * len(on_time) / offered if offered else 0.0,
        "e2e_p50": nearest_rank(e2e, 0.50),
        "e2e_p95": nearest_rank(e2e, 0.95),
        "e2e_p99": nearest_rank(e2e, 0.99),
        "ttft_p50": nearest_rank(ttft, 0.50),
        "ttft_p95": nearest_rank(ttft, 0.95),
        "ttft_p99": nearest_rank(ttft, 0.99),
        "per_token_ms_mean": sum(per_token) / len(per_token) if per_token else 0.0,
        "ttft_is_estimate": any(s.ttft_is_estimate for s in ok),
    }


def sample_rows(samples: Sequence[Sample]) -> List[Dict]:
    return [asdict(s) for s in samples]
