"""
Unit & Integration Tests - Scheduler Policies (Static & Dynamic)
"""

import asyncio
import pytest
from scheduler_engine.backend import MockBackend
from scheduler_engine.policy import DynamicBatchPolicy, StaticBatchPolicy
from scheduler_engine.types import InferenceRequest, RequestPriority, ServerConfig


@pytest.mark.asyncio
async def test_static_batch_policy():
    config = ServerConfig(initial_concurrency=4, target_sla_ms=5000.0)
    backend = MockBackend(tokens_per_second=200.0, simulated_ttft_seconds=0.01)
    policy = StaticBatchPolicy(backend=backend, config=config)
    await policy.initialize()

    req = InferenceRequest(prompt="Test prompt", max_tokens=10)
    res = await policy.schedule(req)

    assert res.prompt == "Test prompt"
    assert res.tokens_generated > 0
    assert res.latency_seconds > 0.0
    assert len(res.response) > 0

    stats = policy.get_stats()
    assert stats.total_completed == 1
    assert stats.total_tokens_generated > 0

    await policy.shutdown()


@pytest.mark.asyncio
async def test_dynamic_batch_policy():
    config = ServerConfig(
        initial_concurrency=4,
        min_concurrency=2,
        max_concurrency=8,
        target_sla_ms=5000.0,
    )
    backend = MockBackend(tokens_per_second=200.0, simulated_ttft_seconds=0.01)
    policy = DynamicBatchPolicy(backend=backend, config=config)
    await policy.initialize()

    # Schedule multiple concurrent requests
    requests = [
        InferenceRequest(
            prompt=f"Prompt {i}",
            max_tokens=10,
            priority=RequestPriority.HIGH if i % 2 == 0 else RequestPriority.NORMAL,
        )
        for i in range(5)
    ]

    tasks = [policy.schedule(r) for r in requests]
    responses = await asyncio.gather(*tasks)

    assert len(responses) == 5
    for r in responses:
        assert r.tokens_generated > 0
        assert r.latency_seconds > 0.0
        assert r.sla_met is True

    stats = policy.get_stats()
    assert stats.total_completed == 5
    assert stats.policy_name == "dynamic_continuous"

    await policy.shutdown()


@pytest.mark.asyncio
async def test_dynamic_policy_streaming():
    config = ServerConfig(target_sla_ms=5000.0)
    backend = MockBackend(tokens_per_second=200.0, simulated_ttft_seconds=0.01)
    policy = DynamicBatchPolicy(backend=backend, config=config)
    await policy.initialize()

    req = InferenceRequest(prompt="Stream prompt", max_tokens=15, stream=True)
    chunks = []
    async for chunk in policy.schedule_stream(req):
        chunks.append(chunk)

    assert len(chunks) > 0
    assert any(c.is_finished for c in chunks)

    await policy.shutdown()


@pytest.mark.asyncio
async def test_dynamic_policy_admits_when_admission_check_breaks():
    config = ServerConfig(initial_concurrency=2, min_concurrency=1, max_concurrency=4, target_sla_ms=5000.0)
    policy = DynamicBatchPolicy(backend=MockBackend(tokens_per_second=300.0, simulated_ttft_seconds=0.01), config=config)
    await policy.initialize()

    def broken(*args, **kwargs):
        raise RuntimeError("estimator failure")

    policy.admission_controller.evaluate = broken
    res = await policy.schedule(InferenceRequest(prompt="hello", max_tokens=8))
    assert res.tokens_generated > 0
    await policy.shutdown()


@pytest.mark.asyncio
async def test_dynamic_policy_keeps_serving_when_controller_breaks():
    config = ServerConfig(initial_concurrency=2, min_concurrency=1, max_concurrency=4, target_sla_ms=5000.0)
    policy = DynamicBatchPolicy(backend=MockBackend(tokens_per_second=300.0, simulated_ttft_seconds=0.01), config=config)
    await policy.initialize()

    def broken(*args, **kwargs):
        raise RuntimeError("controller failure")

    policy.adaptive_controller.evaluate_and_tune = broken
    res = await asyncio.wait_for(policy.schedule(InferenceRequest(prompt="hello", max_tokens=8)), timeout=5)
    assert res.tokens_generated > 0
    await policy.shutdown()


class _CountingBackend(MockBackend):
    def pop_token_count(self, request_id):
        return 77


@pytest.mark.asyncio
async def test_policies_report_the_backend_token_count_not_the_chunk_count():
    config = ServerConfig(initial_concurrency=2, min_concurrency=1, max_concurrency=4, target_sla_ms=5000.0)
    for policy_cls in (StaticBatchPolicy, DynamicBatchPolicy):
        policy = policy_cls(backend=_CountingBackend(tokens_per_second=300.0, simulated_ttft_seconds=0.01), config=config)
        await policy.initialize()
        res = await policy.schedule(InferenceRequest(prompt="hello", max_tokens=8))
        assert res.tokens_generated == 77
        await policy.shutdown()


async def _run_overload(drop):
    config = ServerConfig(
        initial_concurrency=2, min_concurrency=2, max_concurrency=2,
        target_sla_ms=1500.0, max_queue_size=100, drop_hopeless_requests=drop,
    )
    backend = MockBackend(tokens_per_second=50.0, simulated_ttft_seconds=0.01)
    policy = DynamicBatchPolicy(backend=backend, config=config)
    await policy.initialize()
    policy.admission_controller._ema_service_time = 0.01
    policy.admission_controller._ema_token_time = 0.001
    requests = [InferenceRequest(prompt=f"p{i}", max_tokens=20) for i in range(30)]
    results = await asyncio.gather(*(policy.schedule(r) for r in requests), return_exceptions=True)
    await policy.shutdown()
    return policy, results


@pytest.mark.asyncio
async def test_hopeless_queued_requests_are_declined():
    from scheduler_engine.policy import AdmissionRejectedException
    from scheduler_engine.types import AdmissionStatus

    policy, results = await _run_overload(drop=True)
    declined = [r for r in results if isinstance(r, AdmissionRejectedException)]
    served = [r for r in results if not isinstance(r, Exception)]
    assert declined and served
    assert all(d.status == AdmissionStatus.REJECTED_SLA_IMPOSSIBLE for d in declined)
    assert "predicted latency" in declined[0].reason.lower()
    assert len(declined) + len(served) == 30
    assert policy.admission_controller.total_dropped_waiting == len(declined)


@pytest.mark.asyncio
async def test_hopeless_requests_run_when_dropping_is_off():
    policy, results = await _run_overload(drop=False)
    assert all(not isinstance(r, Exception) for r in results)
    assert policy.admission_controller.total_dropped_waiting == 0
