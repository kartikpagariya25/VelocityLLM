import asyncio
import base64
import io

import pytest
from PIL import Image

from scheduler_engine.admission import AdmissionController
from scheduler_engine.backend import MockBackend
from scheduler_engine.policy import DynamicBatchPolicy
from scheduler_engine.types import AdmissionStatus, InferenceRequest, RequestPriority, ServerConfig


def make_image(side):
    buf = io.BytesIO()
    Image.new("RGB", (side, side), (30, 90, 160)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def request(side=None, max_tokens=64, priority=RequestPriority.NORMAL):
    return InferenceRequest(prompt="Describe", max_tokens=max_tokens, priority=priority, image=make_image(side) if side else None)


def test_image_tokens_raise_execution_estimate():
    controller = AdmissionController(ServerConfig())
    plain = controller.estimate_execution_time(64)
    assert controller.estimate_execution_time(64, 256) > plain
    assert controller.estimate_execution_time(64, 1024) > controller.estimate_execution_time(64, 256)


def test_queued_image_tokens_lengthen_the_wait():
    controller = AdmissionController(ServerConfig())
    assert controller.estimate_wait_time(10, 8, None, 8000) > controller.estimate_wait_time(10, 8)


def test_prefill_cost_is_learned_from_completions():
    controller = AdmissionController(ServerConfig())
    before = controller._ema_prefill_time
    for _ in range(30):
        controller.update_completion_stats(1.6, 64, image_tokens=1024)
    assert controller._ema_prefill_time > before
    assert controller._ema_prefill_time <= controller._max_prefill_time


def test_text_completions_leave_prefill_estimate_alone():
    controller = AdmissionController(ServerConfig())
    before = controller._ema_prefill_time
    controller.update_completion_stats(1.0, 64)
    assert controller._ema_prefill_time == before


def test_heavy_images_are_shed_first_under_memory_pressure():
    controller = AdmissionController(ServerConfig(target_sla_ms=30000.0))
    args = dict(current_queue_size=0, active_concurrency=4, gpu_memory_used_mb=7400, gpu_memory_total_mb=8192)
    heavy = controller.evaluate(request(896, priority=RequestPriority.HIGH), **args)
    light = controller.evaluate(request(224, priority=RequestPriority.HIGH), **args)
    assert heavy.admitted is False
    assert heavy.status == AdmissionStatus.REJECTED_OVERLOAD
    assert "heavy image" in heavy.reason
    assert light.admitted is True


async def _peak_active(sides, limit, learned_prefill=None):
    config = ServerConfig(initial_concurrency=limit, min_concurrency=limit, max_concurrency=limit, target_sla_ms=60000.0, vision=True)
    backend = MockBackend(tokens_per_second=400.0, simulated_ttft_seconds=0.01, prefill_seconds_per_image_token=0.0002)
    policy = DynamicBatchPolicy(backend=backend, config=config)
    await policy.initialize()
    if learned_prefill:
        policy.admission_controller._ema_prefill_time = learned_prefill
        policy.admission_controller._ema_service_time = 1.0
    peak = 0

    async def watch():
        nonlocal peak
        while True:
            peak = max(peak, len(policy._active_requests))
            await asyncio.sleep(0.002)

    watcher = asyncio.create_task(watch())
    try:
        await asyncio.gather(*(policy.schedule(request(side, max_tokens=24)) for side in sides))
    finally:
        watcher.cancel()
        stats = policy.get_stats()
        await policy.shutdown()
    return peak, stats


async def test_large_images_occupy_more_slots():
    light_peak, _ = await _peak_active([None] * 8, 4)
    heavy_peak, stats = await _peak_active([896] * 8, 4, learned_prefill=0.002)
    assert light_peak == 4
    assert heavy_peak <= 2
    assert stats.total_image_tokens == 8 * 1024


async def test_mixed_load_completes_and_reports_image_tokens():
    _, stats = await _peak_active([None, 224, 448, 896, 224, None], 4)
    assert stats.total_completed == 6
    assert stats.total_image_tokens == 64 * 2 + 256 + 1024
    assert stats.to_dict()["total_image_tokens"] == stats.total_image_tokens


async def test_telemetry_failure_does_not_stall_dispatch():
    config = ServerConfig(initial_concurrency=2, min_concurrency=2, max_concurrency=2, target_sla_ms=60000.0)
    backend = MockBackend(tokens_per_second=400.0, simulated_ttft_seconds=0.01)

    def broken():
        raise RuntimeError("nvidia-smi timeout")

    backend.get_gpu_telemetry = broken
    policy = DynamicBatchPolicy(backend=backend, config=config)
    await policy.initialize()
    try:
        results = await asyncio.gather(*(policy.schedule(request(None, max_tokens=8)) for _ in range(4)))
    finally:
        await policy.shutdown()
    assert all(r.tokens_generated > 0 for r in results)


async def test_controller_ramps_with_weighted_image_load():
    config = ServerConfig(initial_concurrency=4, min_concurrency=4, max_concurrency=16, target_sla_ms=60000.0, vision=True)
    backend = MockBackend(tokens_per_second=400.0, simulated_ttft_seconds=0.01, prefill_seconds_per_image_token=0.0002)
    policy = DynamicBatchPolicy(backend=backend, config=config)
    await policy.initialize()
    try:
        await asyncio.gather(*(policy.schedule(request(448, max_tokens=24)) for _ in range(40)))
        stats = policy.get_stats()
    finally:
        await policy.shutdown()
    assert stats.total_completed == 40


async def test_smart_policy_forwards_images_and_counts_their_tokens():
    from scheduler_engine.smart_policy import SmartBatchPolicy

    seen = []

    class Recorder(MockBackend):
        async def generate_stream(self, prompt, max_tokens, temperature, request_id, image=None):
            seen.append(image)
            async for chunk in super().generate_stream(prompt, max_tokens, temperature, request_id, image=image):
                yield chunk

    config = ServerConfig(policy="smart", use_mock_backend=True, vision=True, target_sla_ms=60000.0)
    policy = SmartBatchPolicy(backend=Recorder(tokens_per_second=400.0, simulated_ttft_seconds=0.01), config=config)
    await policy.initialize()
    try:
        plain = policy.profiler.profile(request(None, max_tokens=16))
        heavy = policy.profiler.profile(request(896, max_tokens=16))
        await policy.schedule(request(448, max_tokens=8))
    finally:
        await policy.shutdown()
    assert heavy.prompt_tokens - plain.prompt_tokens == 1024
    assert any(item for item in seen)


async def test_cold_start_completions_do_not_poison_estimates():
    config = ServerConfig(initial_concurrency=4, min_concurrency=2, max_concurrency=8, target_sla_ms=60000.0)
    backend = MockBackend(tokens_per_second=400.0, simulated_ttft_seconds=0.01)
    policy = DynamicBatchPolicy(backend=backend, config=config)
    await policy.initialize()
    try:
        before = policy.admission_controller._ema_service_time
        policy.total_completed = 0
        await asyncio.gather(*(policy.schedule(request(None, max_tokens=8)) for _ in range(policy._burn_in_completions)))
        assert policy.exec_times == []
        assert policy.admission_controller._ema_service_time == before
        await policy.schedule(request(None, max_tokens=8))
        assert len(policy.exec_times) == 1
    finally:
        await policy.shutdown()
