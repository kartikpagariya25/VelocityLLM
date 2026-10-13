"""
Tests for queue ordering, batch token budget, true static batching and latency metrics.
"""

import asyncio
import time

import pytest

from scheduler_engine.backend import MockBackend
from scheduler_engine.metrics import LatencyTracker, percentile, timing_breakdown, token_cost
from scheduler_engine.policy import DynamicBatchPolicy, StaticBatchPolicy
from scheduler_engine.priority_queue import PrioritizedRequestQueue
from scheduler_engine.static_batching import StaticBatchingPolicy
from scheduler_engine.types import InferenceRequest, RequestPriority, ServerConfig


def make_request(i, max_tokens=10, priority=RequestPriority.NORMAL):
    return InferenceRequest(prompt=f"prompt {i}", max_tokens=max_tokens, priority=priority, request_id=f"r{i}")


async def drain(queue):
    order = []
    while queue.size:
        entry = await queue.dequeue(timeout=0.01)
        order.append(entry.request.request_id)
    return order


@pytest.mark.asyncio
async def test_fcfs_order_ignores_priority_and_size():
    queue = PrioritizedRequestQueue(order="fcfs")
    await queue.enqueue(make_request(1, max_tokens=500, priority=RequestPriority.LOW))
    await asyncio.sleep(0.002)
    await queue.enqueue(make_request(2, max_tokens=5, priority=RequestPriority.HIGH))
    await asyncio.sleep(0.002)
    await queue.enqueue(make_request(3, max_tokens=50))
    assert await drain(queue) == ["r1", "r2", "r3"]


@pytest.mark.asyncio
async def test_sjf_order_serves_shortest_first():
    queue = PrioritizedRequestQueue(order="sjf")
    await queue.enqueue(make_request(1, max_tokens=400))
    await queue.enqueue(make_request(2, max_tokens=20))
    await queue.enqueue(make_request(3, max_tokens=120))
    assert await drain(queue) == ["r2", "r3", "r1"]


@pytest.mark.asyncio
async def test_priority_order_is_default():
    queue = PrioritizedRequestQueue()
    await queue.enqueue(make_request(1, priority=RequestPriority.LOW))
    await queue.enqueue(make_request(2, priority=RequestPriority.HIGH))
    await queue.enqueue(make_request(3, priority=RequestPriority.NORMAL))
    assert await drain(queue) == ["r2", "r3", "r1"]


@pytest.mark.asyncio
async def test_sjf_aging_eventually_promotes_a_long_request():
    queue = PrioritizedRequestQueue(aging_factor=1.0, order="sjf")
    await queue.enqueue(make_request(1, max_tokens=500))
    queue._entry_map["r1"].enqueue_time -= 10.0
    await queue.enqueue(make_request(2, max_tokens=300))
    assert await drain(queue) == ["r1", "r2"]


def test_unknown_queue_order_is_rejected():
    with pytest.raises(ValueError):
        PrioritizedRequestQueue(order="random")


@pytest.mark.asyncio
async def test_peek_does_not_remove():
    queue = PrioritizedRequestQueue(order="sjf")
    await queue.enqueue(make_request(1, max_tokens=300))
    await queue.enqueue(make_request(2, max_tokens=10))
    head = await queue.peek()
    assert head.request.request_id == "r2"
    assert queue.size == 2
    assert (await queue.dequeue(timeout=0.01)).request.request_id == "r2"


class PeakBackend(MockBackend):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.peak = 0

    async def generate_stream(self, *args, **kwargs):
        async for chunk in super().generate_stream(*args, **kwargs):
            self.peak = max(self.peak, self.active_inferences)
            yield chunk


async def run_dynamic(max_batch_tokens, count=6):
    config = ServerConfig(
        initial_concurrency=8,
        min_concurrency=8,
        max_concurrency=8,
        target_sla_ms=60000.0,
        max_batch_tokens=max_batch_tokens,
        drop_hopeless_requests=False,
    )
    backend = PeakBackend(tokens_per_second=400.0, simulated_ttft_seconds=0.005)
    policy = DynamicBatchPolicy(backend=backend, config=config)
    await policy.initialize()
    results = await asyncio.gather(*(policy.schedule(make_request(i, max_tokens=40)) for i in range(count)))
    await policy.shutdown()
    return backend.peak, results


@pytest.mark.asyncio
async def test_token_budget_caps_concurrent_requests():
    cost = token_cost("prompt 0", 40)
    peak, results = await run_dynamic(max_batch_tokens=cost * 2 + 1)
    assert len(results) == 6
    assert peak <= 2


@pytest.mark.asyncio
async def test_no_token_budget_allows_full_concurrency():
    peak, _ = await run_dynamic(max_batch_tokens=0)
    assert peak > 2


@pytest.mark.asyncio
async def test_token_budget_still_serves_a_request_larger_than_the_budget():
    peak, results = await run_dynamic(max_batch_tokens=5, count=3)
    assert len(results) == 3
    assert peak == 1


async def make_static_batch(batch_size, timeout_ms, tokens_per_second=400.0):
    config = ServerConfig(
        initial_concurrency=batch_size,
        static_batch_timeout_ms=timeout_ms,
        target_sla_ms=60000.0,
    )
    backend = MockBackend(tokens_per_second=tokens_per_second, simulated_ttft_seconds=0.005)
    policy = StaticBatchingPolicy(backend=backend, config=config)
    await policy.initialize()
    return policy


@pytest.mark.asyncio
async def test_full_batch_dispatches_without_waiting_for_timeout():
    policy = await make_static_batch(batch_size=3, timeout_ms=5000)
    start = time.time()
    results = await asyncio.gather(*(policy.schedule(make_request(i, max_tokens=10)) for i in range(3)))
    assert time.time() - start < 2.0
    assert policy.batches_run == 1
    assert all(r.queue_time_seconds < 0.5 for r in results)
    await policy.shutdown()


@pytest.mark.asyncio
async def test_partial_batch_waits_for_timeout():
    policy = await make_static_batch(batch_size=8, timeout_ms=300)
    results = await asyncio.gather(*(policy.schedule(make_request(i, max_tokens=10)) for i in range(2)))
    assert all(r.queue_time_seconds >= 0.25 for r in results)
    assert policy.batches_run == 1
    await policy.shutdown()


@pytest.mark.asyncio
async def test_short_request_is_held_until_the_longest_in_its_batch_finishes():
    policy = await make_static_batch(batch_size=2, timeout_ms=5000, tokens_per_second=100.0)
    short, long = await asyncio.gather(
        policy.schedule(make_request(1, max_tokens=4)),
        policy.schedule(make_request(2, max_tokens=60)),
    )
    assert short.tokens_generated < long.tokens_generated
    assert short.latency_seconds >= long.latency_seconds * 0.9
    await policy.shutdown()


@pytest.mark.asyncio
async def test_second_batch_starts_only_after_the_first_finishes():
    policy = await make_static_batch(batch_size=2, timeout_ms=5000, tokens_per_second=200.0)
    first = [policy.schedule(make_request(i, max_tokens=30)) for i in range(2)]
    second = [policy.schedule(make_request(i, max_tokens=30)) for i in range(2, 4)]
    results = await asyncio.gather(*first, *second)
    first_end = max(r.latency_seconds for r in results[:2])
    assert all(r.queue_time_seconds >= first_end * 0.8 for r in results[2:])
    assert policy.batches_run == 2
    await policy.shutdown()


@pytest.mark.asyncio
async def test_static_batching_streams_tokens_and_finishes():
    policy = await make_static_batch(batch_size=1, timeout_ms=50)
    chunks = [c async for c in policy.schedule_stream(make_request(1, max_tokens=8))]
    assert chunks[-1].is_finished
    assert sum(1 for c in chunks if not c.is_finished) > 0
    await policy.shutdown()


@pytest.mark.asyncio
async def test_static_batching_cancel_removes_pending_request():
    policy = await make_static_batch(batch_size=8, timeout_ms=5000)
    task = asyncio.create_task(policy.schedule(make_request(1)))
    await asyncio.sleep(0.05)
    assert await policy.cancel_request("r1")
    with pytest.raises(asyncio.CancelledError):
        await task
    assert policy.get_stats().queued_requests == 0
    await policy.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("policy_cls", [StaticBatchPolicy, DynamicBatchPolicy])
async def test_response_and_stats_carry_ttft_and_goodput(policy_cls):
    config = ServerConfig(initial_concurrency=4, target_sla_ms=60000.0)
    backend = MockBackend(tokens_per_second=200.0, simulated_ttft_seconds=0.05)
    policy = policy_cls(backend=backend, config=config)
    await policy.initialize()
    result = await policy.schedule(make_request(1, max_tokens=20))
    assert result.ttft_seconds >= 0.05
    assert result.ttft_seconds < result.latency_seconds
    assert result.per_token_latency_ms > 0
    stats = policy.get_stats().to_dict()
    assert stats["p99_ttft_seconds"] >= 0.05
    assert stats["goodput_requests"] == 1
    assert stats["goodput_tokens"] == result.tokens_generated
    await policy.shutdown()


@pytest.mark.asyncio
async def test_goodput_excludes_requests_that_miss_their_sla():
    config = ServerConfig(initial_concurrency=4, target_sla_ms=50.0, drop_hopeless_requests=False)
    backend = MockBackend(tokens_per_second=100.0, simulated_ttft_seconds=0.01)
    policy = StaticBatchPolicy(backend=backend, config=config)
    await policy.initialize()
    result = await policy.schedule(make_request(1, max_tokens=30))
    assert not result.sla_met
    stats = policy.get_stats().to_dict()
    assert stats["goodput_requests"] == 0
    assert stats["goodput_tokens"] == 0
    await policy.shutdown()


def test_percentile_uses_nearest_rank():
    values = list(range(1, 101))
    assert percentile(values, 0.50) == 51
    assert percentile(values, 0.99) == 100
    assert percentile([], 0.99) == 0.0


def test_timing_breakdown_measures_from_enqueue():
    ttft, per_token = timing_breakdown(10.0, 10.5, 12.5, 5)
    assert ttft == pytest.approx(0.5)
    assert per_token == pytest.approx(500.0)
    assert timing_breakdown(10.0, None, 12.0, 5) == (None, None)
    assert timing_breakdown(10.0, 10.5, 10.6, 1) == (pytest.approx(0.5), None)


def test_latency_tracker_snapshot():
    tracker = LatencyTracker()
    tracker.record(0.1, 20.0, 50, True)
    tracker.record(0.3, 40.0, 70, False)
    snap = tracker.snapshot()
    assert snap["p50_ttft_seconds"] == pytest.approx(0.3)
    assert snap["avg_per_token_latency_ms"] == pytest.approx(30.0)
    assert snap["goodput_requests"] == 1
    assert snap["goodput_tokens"] == 50
