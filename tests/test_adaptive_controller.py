"""
Unit Tests - Adaptive Batch Size Controller
"""

import time
import pytest
from scheduler_engine.adaptive_controller import AdaptiveBatchController
from scheduler_engine.types import ServerConfig


def test_adaptive_controller_additive_increase():
    config = ServerConfig(initial_concurrency=4, min_concurrency=2, max_concurrency=16)
    controller = AdaptiveBatchController(config)
    controller.cooldown_seconds = 0.0  # disable cooldown for unit testing

    # Queue has work, active is at capacity, memory is low -> should increase
    new_limit = controller.evaluate_and_tune(
        gpu_util_percent=70.0,
        gpu_memory_used_mb=2000,
        gpu_memory_total_mb=8192,
        queue_depth=5,
        active_requests=4,
    )

    assert new_limit == 5
    assert controller.current_concurrency == 5


def test_adaptive_controller_multiplicative_decrease_on_memory_pressure():
    config = ServerConfig(initial_concurrency=8, min_concurrency=2, max_concurrency=16)
    controller = AdaptiveBatchController(config)
    controller.cooldown_seconds = 0.0

    # 93% memory utilization (exceeds hard limit 92%)
    new_limit = controller.evaluate_and_tune(
        gpu_util_percent=98.0,
        gpu_memory_used_mb=7600,
        gpu_memory_total_mb=8192,
        queue_depth=5,
        active_requests=8,
    )

    # 8 * 0.75 = 6
    assert new_limit == 6
    assert controller.current_concurrency == 6


def test_adaptive_controller_multiplicative_decrease_on_sla_breach():
    config = ServerConfig(initial_concurrency=10, min_concurrency=2, max_concurrency=20)
    controller = AdaptiveBatchController(config)
    controller.cooldown_seconds = 0.0

    new_limit = controller.evaluate_and_tune(
        gpu_util_percent=80.0,
        gpu_memory_used_mb=3000,
        gpu_memory_total_mb=8192,
        queue_depth=2,
        active_requests=10,
        recent_sla_breach=True,
    )

    # 10 * 0.75 = 7
    assert new_limit == 7


def test_adaptive_controller_respects_bounds():
    config = ServerConfig(initial_concurrency=2, min_concurrency=2, max_concurrency=4)
    controller = AdaptiveBatchController(config)
    controller.cooldown_seconds = 0.0

    # Try to decrease below min
    controller.evaluate_and_tune(
        gpu_util_percent=99.0,
        gpu_memory_used_mb=8000,
        gpu_memory_total_mb=8192,
        queue_depth=1,
        active_requests=2,
        recent_sla_breach=True,
    )
    assert controller.current_concurrency == 2  # bounded at min_concurrency=2
