"""
Unit Tests - Admission Controller
"""

import pytest
from scheduler_engine.admission import AdmissionController
from scheduler_engine.types import (
    AdmissionStatus,
    InferenceRequest,
    RequestPriority,
    ServerConfig,
)


def test_admission_accepts_under_normal_conditions():
    config = ServerConfig(max_queue_size=10, target_sla_ms=2000.0)
    controller = AdmissionController(config)

    req = InferenceRequest(prompt="Hello", max_tokens=50, priority=RequestPriority.NORMAL)
    result = controller.evaluate(
        request=req,
        current_queue_size=0,
        active_concurrency=4,
        gpu_memory_used_mb=2000,
        gpu_memory_total_mb=8192,
    )

    assert result.admitted is True
    assert result.status == AdmissionStatus.ACCEPTED
    assert controller.total_accepted == 1


def test_admission_rejects_on_queue_overflow():
    config = ServerConfig(max_queue_size=5, target_sla_ms=5000.0)
    controller = AdmissionController(config)

    req = InferenceRequest(prompt="Overflow test", max_tokens=20)
    result = controller.evaluate(
        request=req,
        current_queue_size=5,  # at max capacity
        active_concurrency=4,
    )

    assert result.admitted is False
    assert result.status == AdmissionStatus.REJECTED_OVERLOAD
    assert result.retry_after_seconds > 0
    assert controller.total_rejected_overload == 1


def test_admission_rejects_when_sla_impossible():
    config = ServerConfig(max_queue_size=50, target_sla_ms=200.0)  # very tight 200ms SLA
    controller = AdmissionController(config)

    # With 20 items in queue and low concurrency, estimated wait is several seconds
    req = InferenceRequest(prompt="Slow queue", max_tokens=200, sla_target_ms=150.0)
    result = controller.evaluate(
        request=req,
        current_queue_size=20,
        active_concurrency=2,
    )

    assert result.admitted is False
    assert result.status == AdmissionStatus.REJECTED_SLA_IMPOSSIBLE
    assert controller.total_rejected_sla == 1


def test_admission_high_priority_tolerance():
    config = ServerConfig(max_queue_size=50, target_sla_ms=1000.0)
    controller = AdmissionController(config)

    # Configure controller service time to a known baseline
    controller._ema_service_time = 0.65
    controller._ema_token_time = 0.001

    # A normal request might exceed tolerance, while high priority gets 1.35x headroom
    normal_req = InferenceRequest(prompt="Normal", max_tokens=100, priority=RequestPriority.NORMAL, sla_target_ms=600.0)
    high_req = InferenceRequest(prompt="High", max_tokens=100, priority=RequestPriority.HIGH, sla_target_ms=600.0)

    # Queue 1 waits 0.65s; even at half the usual exec time the total is ~0.73s.
    # Normal limit 0.63s -> rejected, high limit 0.81s -> accepted
    res_normal = controller.evaluate(normal_req, current_queue_size=1, active_concurrency=1)
    res_high = controller.evaluate(high_req, current_queue_size=1, active_concurrency=1)

    assert res_normal.admitted is False
    assert res_high.admitted is True


def test_admission_ema_update():
    config = ServerConfig()
    controller = AdmissionController(config)
    initial_service = controller._ema_service_time

    controller.update_completion_stats(latency_seconds=3.0, tokens_generated=100)
    assert controller._ema_service_time > initial_service


def test_slow_cold_start_does_not_lock_out_traffic():
    config = ServerConfig(max_queue_size=100, target_sla_ms=8000.0)
    controller = AdmissionController(config)
    for _ in range(8):
        controller.update_completion_stats(50.0, 50)

    req = InferenceRequest(prompt="Hello", max_tokens=128, priority=RequestPriority.NORMAL)
    result = controller.evaluate(request=req, current_queue_size=0, active_concurrency=2)
    assert result.admitted is True


def test_idle_server_always_admits_a_probe():
    config = ServerConfig(max_queue_size=100, target_sla_ms=1000.0)
    controller = AdmissionController(config)
    controller._ema_token_time = 1.0

    req = InferenceRequest(prompt="Hello", max_tokens=128, priority=RequestPriority.NORMAL)
    assert controller.evaluate(request=req, current_queue_size=0, active_concurrency=0).admitted is True
    assert controller.evaluate(request=req, current_queue_size=0, active_concurrency=3).admitted is False


def test_idle_probe_uses_real_in_flight_count_not_the_concurrency_limit():
    config = ServerConfig(max_queue_size=100, target_sla_ms=1000.0)
    controller = AdmissionController(config)
    controller._ema_token_time = 1.0

    req = InferenceRequest(prompt="Hello", max_tokens=128, priority=RequestPriority.NORMAL)
    idle = controller.evaluate(request=req, current_queue_size=0, active_concurrency=8, in_flight=0)
    busy = controller.evaluate(request=req, current_queue_size=0, active_concurrency=8, in_flight=2)
    assert idle.admitted is True
    assert busy.admitted is False


def test_stale_estimates_decay_toward_baseline():
    config = ServerConfig(max_queue_size=100, target_sla_ms=8000.0)
    controller = AdmissionController(config)
    controller._ema_token_time = 0.5
    controller._last_completion -= controller._stale_after_seconds + 1

    req = InferenceRequest(prompt="Hello", max_tokens=10)
    controller.evaluate(request=req, current_queue_size=0, active_concurrency=4, in_flight=1)
    assert controller._ema_token_time < 0.5


def _memory_case(used_mb, baseline_mb):
    controller = AdmissionController(ServerConfig(max_queue_size=50, target_sla_ms=5000.0))
    req = InferenceRequest(prompt="x", max_tokens=10, priority=RequestPriority.NORMAL)
    return controller.evaluate(
        request=req,
        current_queue_size=0,
        active_concurrency=2,
        gpu_memory_used_mb=used_mb,
        gpu_memory_total_mb=8192,
        gpu_memory_baseline_mb=baseline_mb,
    )


def test_memory_limits_ignore_a_full_gpu_that_has_not_grown():
    assert _memory_case(7900, 7900).admitted is True


def test_memory_limits_still_apply_when_usage_grows():
    assert _memory_case(7900, 6500).admitted is False


def _flood_decision(queue_size, ceiling):
    controller = AdmissionController(ServerConfig(max_queue_size=256, target_sla_ms=8000.0))
    controller._ema_service_time = 0.86
    controller._ema_token_time = 0.012
    req = InferenceRequest(prompt="x", max_tokens=128, priority=RequestPriority.NORMAL)
    return controller.evaluate(
        request=req, current_queue_size=queue_size, active_concurrency=8, in_flight=8, concurrency_ceiling=ceiling
    )


def test_reachable_concurrency_is_credited_in_the_wait_estimate():
    assert _flood_decision(110, None).admitted is False
    assert _flood_decision(110, 32).admitted is True


def test_wait_estimate_still_rejects_a_hopeless_queue():
    assert _flood_decision(250, 32).admitted is False


def test_estimator_ignores_non_finite_samples():
    controller = AdmissionController(ServerConfig())
    before = (controller._ema_service_time, controller._ema_token_time)
    controller.update_completion_stats(float("nan"), 10)
    controller.update_completion_stats(float("inf"), 10)
    assert (controller._ema_service_time, controller._ema_token_time) == before


def test_no_false_rejection_when_ceiling_has_capacity():
    config = ServerConfig(max_queue_size=200, target_sla_ms=8000.0, max_concurrency=32)
    controller = AdmissionController(config)
    controller._ema_service_time = 2.0
    req = InferenceRequest(prompt="x", max_tokens=128)
    result = controller.evaluate(req, current_queue_size=40, active_concurrency=8, in_flight=8, concurrency_ceiling=32)
    assert result.admitted is True


def test_slot_weight_grows_with_learned_prefill():
    controller = AdmissionController(ServerConfig())
    assert controller.slot_weight(0) == 1.0
    light = controller.slot_weight(1024)
    controller._ema_prefill_time = 0.002
    assert controller.slot_weight(1024) > light
    assert controller.slot_weight(10**9) <= 1.0 + controller._max_extra_slots
