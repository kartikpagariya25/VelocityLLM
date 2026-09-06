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
    controller._ema_service_time = 0.5
    controller._ema_token_time = 0.001

    # A normal request might exceed tolerance, while high priority gets 1.35x headroom
    normal_req = InferenceRequest(prompt="Normal", max_tokens=100, priority=RequestPriority.NORMAL, sla_target_ms=600.0)
    high_req = InferenceRequest(prompt="High", max_tokens=100, priority=RequestPriority.HIGH, sla_target_ms=600.0)

    # With queue size 1, wait is 0.5s / 1 = 0.5s. Exec time is ~0.15s. Total ~0.65s.
    # Normal target 0.6s * 1.05 = 0.63s -> rejected
    # High target 0.6s * 1.35 = 0.81s -> accepted
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
