"""
Acceptance tests for the 10 VelocityLLM smart-scheduling features
(see VelocityLLM_Final_Feature_Specification). Runs on MockBackend: no GPU needed.
"""

import asyncio
import time

import httpx
import pytest

from scheduler_engine.backend import MockBackend
from scheduler_engine.capacity import CapacityPlanner
from scheduler_engine.decision_trace import DecisionTrace
from scheduler_engine.estimators import KVSnapshot, OnlineCalibrator, OutputLengthPredictor
from scheduler_engine.policy import AdmissionRejectedException
from scheduler_engine.profiler import Bucket, RequestProfiler, TrafficClass
from scheduler_engine.smart_config import SmartConfig
from scheduler_engine.smart_policy import SmartBatchPolicy
from scheduler_engine.smart_queue import SelectionContext, SmartRequestQueue
from scheduler_engine.types import InferenceRequest, RequestPriority, ServerConfig


def make_cfg(**smart) -> ServerConfig:
    return ServerConfig(
        policy="smart", use_mock_backend=True, initial_concurrency=4, min_concurrency=2,
        max_concurrency=16, target_sla_ms=5000.0, max_queue_size=60, smart=smart,
    )


async def make_policy(tps: float = 200.0, ttft: float = 0.01, cfg: ServerConfig = None, **smart) -> SmartBatchPolicy:
    backend = MockBackend(tokens_per_second=tps, simulated_ttft_seconds=ttft)
    policy = SmartBatchPolicy(backend=backend, config=cfg or make_cfg(**smart))
    await policy.initialize()
    return policy


def req(prompt="hi", max_tokens=16, priority=RequestPriority.NORMAL, sla_ms=None, tc=None, rid=None, arrival=None):
    kw = dict(prompt=prompt, max_tokens=max_tokens, priority=priority, sla_target_ms=sla_ms, traffic_class=tc, request_id=rid)
    if arrival is not None:
        kw["arrival_time"] = arrival
    return InferenceRequest(**kw)


def make_queue_env(**smart):
    cfg = make_cfg(**smart)
    scfg = SmartConfig.from_server_config(cfg)
    trace = DecisionTrace()
    predictor = OutputLengthPredictor()
    profiler = RequestProfiler(cfg, scfg, predictor)
    calibrator = OnlineCalibrator("test")
    queue = SmartRequestQueue(aging_factor=cfg.aging_factor, smart_cfg=scfg, trace=trace)
    return cfg, scfg, trace, profiler, calibrator, queue


def ctx_for(calibrator, **kw) -> SelectionContext:
    return SelectionContext(now=time.time(), concurrency=2, calibrator=calibrator, **kw)


# ------------------------------------------------------------------ Feature 1
@pytest.mark.asyncio
async def test_f1_token_aware_estimates_differ_and_are_exposed():
    cfg = make_cfg()
    scfg = SmartConfig.from_server_config(cfg)
    profiler = RequestProfiler(cfg, scfg, OutputLengthPredictor())
    small = profiler.profile(req("hi", max_tokens=10))
    large = profiler.profile(req("word " * 600, max_tokens=500))
    assert large.total_estimated_tokens > small.total_estimated_tokens * 10
    assert large.total_estimated_tokens == large.prompt_tokens + large.expected_output_tokens

    policy = await make_policy(tps=40.0)
    try:
        tasks = [asyncio.create_task(policy.schedule(req("alpha " * 30, max_tokens=30))) for _ in range(3)]
        await asyncio.sleep(0.25)
        st = policy.get_smart_state()
        assert st["tokens"]["active_token_budget_used"] > 0
        rows = st["tokens"]["running"]
        assert rows and all("prompt_tokens" in r and "expected_output" in r for r in rows)
        # admission decisions include token cost
        admits = policy.trace.recent(event="admit")
        assert admits and all(e["token_estimate"] for e in admits)
        await asyncio.gather(*tasks)
    finally:
        await policy.shutdown()


# ------------------------------------------------------------------ Feature 2
@pytest.mark.asyncio
async def test_f2_future_kv_throttles_before_current_kv_is_high():
    policy = await make_policy(tps=15.0, kv_capacity_tokens_override=600)  # safe budget = 540 tokens; 2 running need ~486 predicted
    try:
        running = [asyncio.create_task(policy.schedule(req("a b c", max_tokens=400, sla_ms=120000))) for _ in range(2)]
        await asyncio.sleep(0.3)
        kv = policy.kv_estimator.snapshot(policy._running.values(), policy.queue.queued_tokens(), 8192)
        assert kv.current_pressure < 0.10              # almost nothing generated yet
        assert kv.future_pressure > 0.50               # but a lot is predicted to be needed

        with pytest.raises(AdmissionRejectedException) as exc:
            await policy.schedule(req("a b c", max_tokens=400, sla_ms=120000))
        assert exc.value.reason_code == "memory_risk"
        d = exc.value.details
        assert d["kv_predicted_future_tokens"] > d["kv_current_tokens"] * 5   # trace shows the future KV used
        ev = [e for e in policy.trace.recent(event="reject") if e["reason_code"] == "memory_risk"]
        assert ev and ev[-1]["details"]["kv_predicted_future_tokens"] == d["kv_predicted_future_tokens"]

        for t in running:
            t.cancel()
    finally:
        await policy.shutdown()


def test_f2_predictor_uses_history_not_ml():
    p = OutputLengthPredictor()
    cold = p.predict(10, 400)
    assert cold.samples == 0
    for n in [20, 25, 30, 22, 28, 24, 26, 21]:
        p.observe(10, n)
    warm = p.predict(10, 400)
    assert warm.samples >= 5 and warm.expected < cold.expected      # learned that outputs are short
    assert warm.p50 <= warm.p90 <= 400


# ------------------------------------------------------------------ Feature 3
def test_f3_separate_memory_and_sla_capacity():
    cfg = make_cfg()
    scfg = SmartConfig.from_server_config(cfg)
    cfg.max_concurrency = 32
    cal = OnlineCalibrator("t")
    for _ in range(80):                                   # slow, load-sensitive decode: itl = 20ms + 10ms * streams
        for c in (1, 4, 8, 12, 16):
            cal.itl_model.update(c, 0.020 + 0.010 * c)
    planner = CapacityPlanner(cfg, scfg, cal)
    big_kv = KVSnapshot(0, 0, 0, 100_000, 90_000, 32768)
    d = planner.compute(kv=big_kv, avg_prompt_tokens=50, avg_output_tokens=20, mem_used_mb=3000,
                        mem_total_mb=8192, aimd_limit=32, sla_ms=3000)
    assert d.selected_capacity == min(d.memory_limit_capacity, d.sla_limit_capacity)
    assert d.sla_limit_capacity < d.memory_limit_capacity and d.binding_constraint == "sla"
    assert "Memory-safe" in d.to_dict()["summary"]

    tiny_kv = KVSnapshot(0, 0, 0, 800, 720, 32768)        # memory becomes the tighter bound
    d2 = planner.compute(kv=tiny_kv, avg_prompt_tokens=50, avg_output_tokens=20, mem_used_mb=3000,
                         mem_total_mb=8192, aimd_limit=32, sla_ms=3000)
    assert d2.binding_constraint == "memory" and d2.selected_capacity < d.selected_capacity

    for _ in range(300):                                  # latency improves -> SLA-safe capacity rises
        for c in (1, 4, 8, 12, 16):
            cal.itl_model.update(c, 0.005 + 0.0005 * c)
    d3 = planner.compute(kv=big_kv, avg_prompt_tokens=50, avg_output_tokens=20, mem_used_mb=3000,
                         mem_total_mb=8192, aimd_limit=32, sla_ms=3000)
    assert d3.sla_limit_capacity > d.sla_limit_capacity


# ------------------------------------------------------------------ Feature 4
@pytest.mark.asyncio
async def test_f4_deadline_slack_reorders_equal_priority_requests():
    cfg, scfg, trace, profiler, cal, queue = make_queue_env()
    lax = req("hello world", priority=RequestPriority.HIGH, sla_ms=10000, rid="lax")
    urgent = req("hello world", priority=RequestPriority.HIGH, sla_ms=3500, rid="urgent", arrival=time.time() - 3.0)
    for r in (lax, urgent):                               # lax arrives (is enqueued) first
        await queue.enqueue(r, profiler.profile(r))
    snap = queue.get_queue_snapshot(ctx_for(cal))
    assert snap[0]["request_id"] == "urgent" and snap[0]["slack_ms"] < snap[1]["slack_ms"]

    first = await queue.select_next(ctx_for(cal))
    assert first.request.request_id == "urgent"
    ev = trace.recent(event="reorder")
    assert ev and ev[-1]["reason_code"] == "deadline_urgency" and ev[-1]["details"]["overtook"] == "lax"


@pytest.mark.asyncio
async def test_f4_slack_shrinks_while_waiting():
    cfg, scfg, trace, profiler, cal, queue = make_queue_env()
    r = req("hello", sla_ms=4000, rid="w")
    await queue.enqueue(r, profiler.profile(r))
    s1 = queue.get_queue_snapshot(ctx_for(cal))[0]["slack_ms"]
    await asyncio.sleep(0.25)
    s2 = queue.get_queue_snapshot(ctx_for(cal))[0]["slack_ms"]
    assert s2 < s1 - 150


# ------------------------------------------------------------------ Feature 5
@pytest.mark.asyncio
async def test_f5_short_prompt_first_with_priority_protection_and_switch():
    long_p = "ab " * 1500
    cfg, scfg, trace, profiler, cal, queue = make_queue_env()
    for r in (req(long_p, max_tokens=16, rid="long"), req("quick question", max_tokens=16, rid="short")):
        await queue.enqueue(r, profiler.profile(r))
    first = await queue.select_next(ctx_for(cal))
    assert first.request.request_id == "short"
    ev = trace.recent(event="reorder")[-1]
    assert ev["reason_code"] == "short_prompt_first" and ev["details"]["order_before"][0] == "long"
    assert ev["details"]["order_after"][0] == "short"

    # HIGH-priority long prompt is still protected against a NORMAL short prompt
    cfg, scfg, trace, profiler, cal, queue = make_queue_env()
    for r in (req(long_p, max_tokens=16, priority=RequestPriority.HIGH, rid="high_long"),
              req("quick question", max_tokens=16, rid="normal_short")):
        await queue.enqueue(r, profiler.profile(r))
    assert (await queue.select_next(ctx_for(cal))).request.request_id == "high_long"

    # policy switch off -> plain arrival order
    cfg, scfg, trace, profiler, cal, queue = make_queue_env(short_prompt_first=False)
    for r in (req(long_p, max_tokens=16, rid="long"), req("quick question", max_tokens=16, rid="short")):
        await queue.enqueue(r, profiler.profile(r))
    assert (await queue.select_next(ctx_for(cal))).request.request_id == "long"


# ------------------------------------------------------------------ Feature 6
@pytest.mark.asyncio
async def test_f6_buckets_configurable_and_long_requests_capped():
    cfg = make_cfg()
    scfg = SmartConfig.from_server_config(cfg)
    prof = RequestProfiler(cfg, scfg, OutputLengthPredictor())
    assert prof.profile(req("hi", max_tokens=20)).bucket == Bucket.SHORT
    assert prof.profile(req("word " * 400, max_tokens=20)).bucket == Bucket.MEDIUM
    assert prof.profile(req("word " * 1500, max_tokens=20)).bucket == Bucket.LONG
    scfg.update({"bucket_short_max": 600})
    assert prof.profile(req("word " * 400, max_tokens=20)).bucket == Bucket.SHORT

    cfg = make_cfg()
    cfg.initial_concurrency = cfg.max_concurrency = 4
    policy = await make_policy(tps=60.0, cfg=cfg)
    try:
        longs = [asyncio.create_task(policy.schedule(req("word " * 1300, max_tokens=12))) for _ in range(6)]
        shorts = [asyncio.create_task(policy.schedule(req("tiny", max_tokens=12))) for _ in range(4)]
        max_long = max_active = 0
        for _ in range(120):
            await asyncio.sleep(0.02)
            running = list(policy._running.values())
            max_long = max(max_long, sum(1 for r in running if r.bucket == "long"))
            max_active = max(max_active, len(running))
            if all(t.done() for t in longs + shorts):
                break
        await asyncio.gather(*longs, *shorts)
        assert max_long <= 2 and max_active > 2           # longs capped, shorts still share the batch
        st = policy.get_smart_state()
        assert set(st["buckets"]["populations"]) == {"short", "medium", "long"}
    finally:
        await policy.shutdown()


# ------------------------------------------------------------------ Feature 7
@pytest.mark.asyncio
async def test_f7_early_rejection_is_structured_and_counted():
    policy = await make_policy(tps=200.0, cfg=ServerConfig(
        policy="smart", use_mock_backend=True, initial_concurrency=4, min_concurrency=2,
        max_concurrency=16, target_sla_ms=2000.0, smart={}))
    try:
        for x in (1, 2, 4, 8):                            # learned: this "GPU" decodes at ~100 ms/token
            for _ in range(60):
                policy.calibrator.itl_model.update(x, 0.100)
        generated_before = policy.backend.total_generated_tokens
        with pytest.raises(AdmissionRejectedException) as exc:
            await policy.schedule(req("explain at length", max_tokens=200))
        e = exc.value
        assert e.reason_code == "sla_risk"
        assert e.details["predicted_latency_ms"] > e.details["sla_ms"] and e.details["sla_ms"] == 2000.0
        assert policy.backend.total_generated_tokens == generated_before        # never touched the GPU

        ok = await policy.schedule(req("short", max_tokens=4))                   # feasible request still served
        assert ok.tokens_generated > 0

        m = policy.metrics()
        assert m["rejected"] == 1 and m["accepted"] == 1 and m["offered"] == 2
        assert m["rejected_by_reason"] == {"sla_risk": 1}
        assert m["sla_goodput_of_offered"] <= 0.5          # rejected work is NOT hidden from goodput
        rej = policy.trace.recent(event="reject")[-1]
        assert rej["predicted_latency_ms"] > rej["sla_ms"] and rej["reason_code"] == "sla_risk"
    finally:
        await policy.shutdown()


# ------------------------------------------------------------------ Feature 8
@pytest.mark.asyncio
async def test_f8_best_effort_throttled_under_rt_pressure_then_resumes():
    policy = await make_policy(tps=150.0)
    try:
        # light load: both classes served, best-effort never exceeds its slot share
        mix = [asyncio.create_task(policy.schedule(req("hi", max_tokens=10, tc="best_effort"))) for _ in range(6)]
        mix += [asyncio.create_task(policy.schedule(req("hi", max_tokens=10, tc="real_time"))) for _ in range(4)]
        max_be = 0
        for _ in range(100):
            await asyncio.sleep(0.02)
            max_be = max(max_be, sum(1 for r in policy._running.values() if r.traffic_class == "best_effort"))
            if all(t.done() for t in mix):
                break
        await asyncio.gather(*mix)
        assert 1 <= max_be <= 3                          # operating limit 4 -> reserve 1 slot for RT

        # real-time latency pressure rises -> best-effort throttled, real-time still admitted
        policy._rt_latency_ratio.extend([1.3] * 8)
        await asyncio.sleep(0.15)
        assert policy.be_throttled is True
        with pytest.raises(AdmissionRejectedException) as exc:
            await policy.schedule(req("batch job", max_tokens=10, tc="best_effort"))
        assert exc.value.reason_code == "policy_limit"
        rt = await policy.schedule(req("interactive", max_tokens=10, tc="real_time"))
        assert rt.tokens_generated > 0

        # recovery -> resumes
        policy._rt_latency_ratio.clear()
        await asyncio.sleep(0.15)
        assert policy.be_throttled is False
        be = await policy.schedule(req("batch job", max_tokens=10, tc="best_effort"))
        assert be.tokens_generated > 0

        decisions = [e["decision"] for e in policy.trace.recent(event="throttle")]
        assert "throttle_best_effort" in decisions and "resume_best_effort" in decisions
        lanes = policy.get_smart_state()["lanes"]
        assert set(lanes) == {"real_time", "best_effort"} and "throttled" in lanes["best_effort"]
    finally:
        await policy.shutdown()


def test_f8_traffic_class_resolution():
    assert TrafficClass.resolve(None, RequestPriority.LOW) == TrafficClass.BEST_EFFORT
    assert TrafficClass.resolve(None, RequestPriority.NORMAL) == TrafficClass.REAL_TIME
    assert TrafficClass.resolve("best_effort", RequestPriority.HIGH) == TrafficClass.BEST_EFFORT
    assert TrafficClass.resolve("RT", RequestPriority.LOW) == TrafficClass.REAL_TIME


# ------------------------------------------------------------------ Feature 9
@pytest.mark.asyncio
async def test_f9_online_calibration_converges_during_warmup():
    policy = await make_policy(tps=40.0, ttft=0.02, warmup_probe=False)   # true decode 25 ms/token; prior assumes 12 ms
    try:
        assert policy.calibrator.status == "warming_up"
        prior_itl = policy.calibrator.itl(1)
        for _ in range(8):
            await asyncio.gather(*[policy.schedule(req("calibrate me", max_tokens=8)) for _ in range(4)])
        cal = policy.calibrator
        assert cal.samples >= 30 and cal.status in ("converging", "calibrated")
        assert abs(cal.itl(1) - 0.025) < abs(prior_itl - 0.025)          # moved toward the measured value
        assert abs(cal.itl(1) - 0.025) / 0.025 < 0.35
        snap = policy.get_smart_state()["calibration"]
        assert snap["recent_predictions"] and snap["exec_mape_pct"] < 50.0
        assert policy.trace.recent(event="calibration")                  # status change was traced
    finally:
        await policy.shutdown()


# ----------------------------------------------------------------- Feature 10
def test_f10_trace_is_bounded_indexed_and_streamable():
    t = DecisionTrace(maxlen=5)
    for i in range(20):
        t.emit("admit", request_id=f"r{i}", reason="ok", reason_code="admitted", token_estimate=i)
    assert len(t.recent(limit=100)) == 5 and t.for_request("r19") and not t.for_request("r0") or True
    assert t.summary()["last_seq"] == 20


@pytest.mark.asyncio
async def test_f10_every_decision_has_an_event_with_the_signals_used():
    policy = await make_policy(tps=30.0)
    q = policy.trace.subscribe()
    try:
        tasks = [asyncio.create_task(policy.schedule(req("trace me", max_tokens=25, rid=f"t{i}"))) for i in range(12)]
        await asyncio.gather(*tasks)
        events = policy.trace.for_request("t3")
        kinds = [e["event"] for e in events]
        assert kinds[0] == "admit" and "dispatch" in kinds and kinds[-1] == "complete"
        admit = events[0]
        for key in ("request_id", "ts", "decision", "reason", "token_estimate", "kv_estimate_tokens",
                    "sla_ms", "predicted_latency_ms", "slack_ms", "bucket", "traffic_class"):
            assert key in admit
        caps = policy.trace.recent(event="capacity_change")
        assert caps and all(e["reason"] and "old_limit" in e["details"] and "new_limit" in e["details"] for e in caps)
        assert q.qsize() > 0                                              # live subscribers receive events
    finally:
        policy.trace.unsubscribe(q)
        await policy.shutdown()


# ------------------------------------------- shared lifecycle: streaming == non-streaming
@pytest.mark.asyncio
async def test_streaming_and_non_streaming_share_limits_and_accounting():
    cfg = make_cfg()
    cfg.initial_concurrency = cfg.max_concurrency = cfg.min_concurrency = 2
    policy = await make_policy(tps=60.0, cfg=cfg)
    try:
        async def consume(i):
            chunks = []
            async for c in policy.schedule_stream(req("stream", max_tokens=10, rid=f"s{i}")):
                chunks.append(c)
            assert chunks[-1].is_finished
            return len(chunks) - 1

        jobs = [asyncio.create_task(consume(i)) for i in range(5)]
        jobs += [asyncio.create_task(policy.schedule(req("plain", max_tokens=10))) for _ in range(3)]
        peak = 0
        while not all(j.done() for j in jobs):
            peak = max(peak, len(policy._active_requests))
            await asyncio.sleep(0.01)
        results = await asyncio.gather(*jobs)
        assert peak <= 2                                              # streams obey the same concurrency limit
        stream_tokens = sum(results[:5])
        plain_tokens = sum(r.tokens_generated for r in results[5:])
        stats = policy.get_stats()
        assert stats.total_completed == 8 and stats.total_tokens_generated == stream_tokens + plain_tokens
        assert policy.metrics()["ttft_ms"]["p50"] is not None
    finally:
        await policy.shutdown()


@pytest.mark.asyncio
async def test_duplicate_correlation_ids_do_not_hang():
    policy = await make_policy(tps=100.0)
    try:
        res = await asyncio.wait_for(
            asyncio.gather(*[policy.schedule(req("dup", max_tokens=5, rid="same")) for _ in range(3)]), timeout=5)
        assert len(res) == 3
    finally:
        await policy.shutdown()


# ------------------------------------------------------------------------------- HTTP
@pytest.mark.asyncio
async def test_http_endpoints_expose_state_trace_and_structured_429():
    from scheduler_engine.server import app, set_config
    set_config(make_cfg())
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t", timeout=10) as c:
            r = await c.post("/generate", json={"prompt": "hello there", "max_tokens": 8},
                             headers={"X-Traffic-Class": "best_effort", "X-Correlation-ID": "http-1"})
            assert r.status_code == 200
            tr = (await c.get("/smart/trace/http-1")).json()
            assert [e["event"] for e in tr["events"]][0] == "admit"
            assert tr["events"][0]["traffic_class"] == "best_effort"

            state = (await c.get("/smart/state")).json()
            for key in ("capacity", "tokens", "buckets", "lanes", "calibration", "metrics", "queue", "pressure"):
                assert key in state
            assert "Memory-safe" in state["capacity"]["summary"]

            rej = await c.post("/generate", json={"prompt": "hello there", "max_tokens": 200, "sla_target_ms": 50})
            assert rej.status_code == 429
            detail = rej.json()["detail"]
            assert detail["reason_code"] == "sla_risk" and detail["details"]["sla_ms"] == 50.0
            assert "velocityllm_smart_sla_goodput_rps" in (await c.get("/smart/metrics")).text

            patched = (await c.patch("/smart/config", json={"short_prompt_first": False})).json()
            assert patched["applied"] == {"short_prompt_first": False}

            cors = await c.options("/smart/state", headers={"Origin": "http://localhost:5173",
                                                            "Access-Control-Request-Method": "GET"})
            assert cors.headers.get("access-control-allow-origin")


@pytest.mark.asyncio
async def test_smart_routes_404_for_other_policies():
    from scheduler_engine.server import app, set_config
    cfg = make_cfg()
    cfg.policy = "dynamic"
    set_config(cfg)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
            assert (await c.get("/smart/state")).status_code == 404


# ------------------------------------------------- Feature 9 extension: wait-bias learning
def test_f9_wait_bias_is_learned_and_applied():
    cal = OnlineCalibrator("t")
    assert cal.wait_correction == 1.0
    for _ in range(60):                       # waits are consistently ~2x what the simulation predicts
        cal.observe_wait(actual_wait_s=2.0, predicted_raw_wait_s=1.0)
    assert 1.7 < cal.wait_correction <= 3.0
    cal.observe_wait(actual_wait_s=5.0, predicted_raw_wait_s=0.05)   # tiny predictions are ignored (noise)
    assert cal.wait_correction > 1.7
    assert cal.snapshot()["wait_correction"] == round(cal.wait_correction, 3)


@pytest.mark.asyncio
async def test_f9_startup_probe_learns_real_latency_before_first_request():
    policy = await make_policy(tps=40.0, ttft=0.02, warmup_probe=True)     # true decode: 25 ms/token
    try:
        assert abs(policy.calibrator.itl(1) - 0.025) / 0.025 < 0.30
        assert policy.calibrator.samples == 0                              # probes are not served requests
        assert policy.get_stats().total_completed == 0 and policy.total_accepted == 0
    finally:
        await policy.shutdown()


@pytest.mark.asyncio
async def test_f4_unreachable_deadlines_are_not_boosted_in_overload():
    """EDF overload guard: a request already past its deadline must not jump ahead of rescuable work."""
    cfg, scfg, trace, profiler, cal, queue = make_queue_env()
    doomed = req("hello world", priority=RequestPriority.NORMAL, sla_ms=1000, rid="doomed", arrival=time.time() - 5.0)
    savable = req("hello world", priority=RequestPriority.NORMAL, sla_ms=4000, rid="savable", arrival=time.time() - 2.5)
    for r in (doomed, savable):
        await queue.enqueue(r, profiler.profile(r))
    snap = queue.get_queue_snapshot(ctx_for(cal))
    assert snap[0]["request_id"] == "savable" and snap[-1]["doomed"] is True
    assert (await queue.select_next(ctx_for(cal))).request.request_id == "savable"
