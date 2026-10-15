"""
Tests for the streaming load client and the rate sweep analysis helpers.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from load_generator.stream_client import Sample, Target, build_workload, nearest_rank, summarize
from tools.rate_sweep import capacity_table


def test_workload_is_deterministic_for_a_seed():
    a = build_workload(30, 5.0, seed=7)
    b = build_workload(30, 5.0, seed=7)
    assert [(r.arrival, r.prompt, r.max_tokens) for r in a] == [(r.arrival, r.prompt, r.max_tokens) for r in b]
    assert [r.max_tokens for r in a] != [r.max_tokens for r in build_workload(30, 5.0, seed=8)]


def test_workload_arrivals_are_poisson_paced_and_lengths_vary():
    plan = build_workload(400, 10.0, seed=1)
    arrivals = [r.arrival for r in plan]
    assert arrivals == sorted(arrivals)
    assert 30.0 < arrivals[-1] < 55.0
    lengths = [r.max_tokens for r in plan]
    assert min(lengths) >= 8 and max(lengths) <= 256
    assert len(set(lengths)) > 20


def test_workload_uses_supplied_length_pairs():
    plan = build_workload(20, 5.0, seed=1, lengths=[(40, 33)])
    assert {r.max_tokens for r in plan} == {33}


@pytest.mark.parametrize(
    "spec,protocol,url,model",
    [
        ("vllm=openai@http://localhost:8001#my/model", "openai", "http://localhost:8001", "my/model"),
        ("vel=velocity@http://localhost:8000/", "velocity", "http://localhost:8000", "model"),
        ("tri=triton-infer@http://localhost:8002#hf_generate", "triton-infer", "http://localhost:8002", "hf_generate"),
    ],
)
def test_target_parsing(spec, protocol, url, model):
    target = Target.parse(spec)
    assert (target.protocol, target.url, target.model) == (protocol, url, model)


@pytest.mark.parametrize("spec", ["nonsense", "x=ftp@http://h", "x=openai@"])
def test_target_parsing_rejects_bad_specs(spec):
    with pytest.raises(ValueError):
        Target.parse(spec)


def make_samples():
    return [
        Sample(0, 0.0, "ok", e2e_seconds=1.0, ttft_seconds=0.1, tokens=50, per_token_ms=18.0),
        Sample(1, 0.1, "ok", e2e_seconds=9.0, ttft_seconds=4.0, tokens=100, per_token_ms=50.0),
        Sample(2, 0.2, "rejected", e2e_seconds=0.01),
        Sample(3, 0.3, "error", error="boom"),
    ]


def test_summary_counts_goodput_and_rates():
    summary = summarize(make_samples(), wall_seconds=10.0, sla_seconds=5.0)
    assert (summary["offered"], summary["served"], summary["rejected"], summary["errors"]) == (4, 2, 1, 1)
    assert summary["tokens_per_s"] == pytest.approx(15.0)
    assert summary["goodput_tokens_per_s"] == pytest.approx(5.0)
    assert summary["goodput_requests_per_s"] == pytest.approx(0.1)
    assert summary["sla_attainment_percent"] == pytest.approx(25.0)
    assert summary["ttft_p99"] == pytest.approx(4.0)


def test_ttft_sla_tightens_goodput():
    summary = summarize(make_samples(), wall_seconds=10.0, sla_seconds=20.0, ttft_sla_seconds=1.0)
    assert summary["goodput_tokens_per_s"] == pytest.approx(5.0)


def test_nearest_rank_matches_server_percentiles():
    values = [float(i) for i in range(1, 101)]
    assert nearest_rank(values, 0.99) == 100.0
    assert nearest_rank([], 0.5) == 0.0


def row(target, rps, p99, goodput, served=10, offered=10, rejected=0):
    return {
        "target": target, "rps": rps, "e2e_p99": p99, "goodput_tokens_per_s": goodput,
        "served": served, "offered": offered, "rejected": rejected, "tokens_per_s": goodput * 2,
    }


def test_capacity_table_picks_highest_rate_within_sla():
    rows = [
        row("a", 2, 3.0, 50), row("a", 4, 5.0, 90), row("a", 8, 12.0, 40),
        row("b", 2, 3.0, 50), row("b", 4, 9.0, 30),
    ]
    capacity = capacity_table(rows, sla_s=6.0, reject_tolerance=0.01)
    assert capacity["a"]["max_rps_within_p99_sla"] == 4
    assert capacity["a"]["peak_goodput_at_rps"] == 4
    assert capacity["b"]["max_rps_within_p99_sla"] == 2


def test_capacity_table_does_not_credit_load_shedding():
    rows = [row("shed", 8, 2.0, 10, served=4, offered=10, rejected=6)]
    assert capacity_table(rows, sla_s=6.0, reject_tolerance=0.01)["shed"]["max_rps_within_p99_sla"] == 0.0


async def serve_fake(routes):
    from aiohttp import web

    app = web.Application()
    for method, path, handler in routes:
        app.router.add_route(method, path, handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}"


async def sse(request, lines):
    import asyncio
    from aiohttp import web

    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)
    for line in lines:
        await resp.write(f"data: {line}\n\n".encode())
        await asyncio.sleep(0.01)
    return resp


@pytest.mark.asyncio
async def test_openai_protocol_reads_stream_usage_and_ttft():
    import json
    from load_generator.stream_client import run_workload

    async def completions(request):
        chunks = [json.dumps({"choices": [{"text": t}]}) for t in ("a", "b", "c")]
        chunks.append(json.dumps({"choices": [], "usage": {"completion_tokens": 3}}))
        return await sse(request, chunks + ["[DONE]"])

    runner, url = await serve_fake([("POST", "/v1/completions", completions)])
    try:
        samples, _ = await run_workload(Target("vllm", "openai", url), build_workload(2, 50.0, seed=1))
    finally:
        await runner.cleanup()
    assert [s.status for s in samples] == ["ok", "ok"]
    assert all(s.tokens == 3 and s.ttft_seconds is not None and s.ttft_seconds < s.e2e_seconds for s in samples)


@pytest.mark.asyncio
async def test_triton_generate_protocol_counts_streamed_text():
    import json
    from load_generator.stream_client import run_workload

    async def generate_stream(request):
        return await sse(request, [json.dumps({"text_output": t}) for t in ("x", "y", "z", "w")])

    runner, url = await serve_fake([("POST", "/v2/models/vllm_model/generate_stream", generate_stream)])
    try:
        samples, _ = await run_workload(Target("triton", "triton-generate", url, "vllm_model"), build_workload(1, 50.0, seed=1))
    finally:
        await runner.cleanup()
    assert samples[0].status == "ok" and samples[0].tokens == 4


@pytest.mark.asyncio
async def test_triton_infer_protocol_marks_ttft_as_estimate():
    from aiohttp import web
    from load_generator.stream_client import run_workload

    async def infer(request):
        body = await request.json()
        assert {i["name"] for i in body["inputs"]} == {"prompt", "max_tokens"}
        return web.json_response({"outputs": [{"name": "text", "data": ["hi"]}, {"name": "num_tokens", "data": [7]}]})

    runner, url = await serve_fake([("POST", "/v2/models/hf_generate/infer", infer)])
    try:
        samples, wall = await run_workload(Target("triton", "triton-infer", url, "hf_generate"), build_workload(1, 50.0, seed=1))
    finally:
        await runner.cleanup()
    assert samples[0].tokens == 7
    assert samples[0].ttft_is_estimate and samples[0].ttft_seconds == samples[0].e2e_seconds
    assert summarize(samples, wall, 5.0)["ttft_is_estimate"]


@pytest.mark.asyncio
async def test_velocity_protocol_flags_stream_rejections():
    import json
    from load_generator.stream_client import run_workload

    async def stream(request):
        return await sse(request, [json.dumps({"error": "Rate limit / SLA protection shed", "reason": "queue full"})])

    runner, url = await serve_fake([("POST", "/generate/stream", stream)])
    try:
        samples, _ = await run_workload(Target("vel", "velocity", url), build_workload(1, 50.0, seed=1))
    finally:
        await runner.cleanup()
    assert samples[0].status == "rejected"


def test_merge_sweeps_relabels_single_target_and_combines(tmp_path):
    import json
    from tools.merge_sweeps import load_rows

    def write(name, target):
        d = tmp_path / name
        d.mkdir()
        row = {"target": target, "rps": 2.0, "offered": 10, "served": 10, "rejected": 0, "e2e_p99": 1.0,
               "goodput_tokens_per_s": 5.0, "tokens_per_s": 6.0}
        (d / "sweep_summary.json").write_text(json.dumps({"sla_ms": 8000, "rows": [row]}))
        return str(d)

    plain = load_rows(write("a", "dynamic"))
    renamed = load_rows(write("b", "dynamic") + "=dynamic (chunked on)")
    assert plain[0]["target"] == "dynamic"
    assert renamed[0]["target"] == "dynamic (chunked on)"
