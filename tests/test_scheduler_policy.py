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
