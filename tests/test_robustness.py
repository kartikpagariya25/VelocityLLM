"""
VelocityLLM - Phase 3 Robustness & Edge Case Hardening Test Suite
Tests input validation/sanitization, GPU memory limits, emergency OOM recovery,
client-disconnect slot reclamation, burst-shedding backpressure, and structured logging.
"""

import asyncio
import pytest

from scheduler_engine.backend import GPUOutOfMemoryError, MockBackend
from scheduler_engine.policy import (
    AdmissionRejectedException,
    DynamicBatchPolicy,
)
from scheduler_engine.admission import AdmissionController
from scheduler_engine.priority_queue import PrioritizedRequestQueue
from scheduler_engine.types import (
    AdmissionStatus,
    InferenceRequest,
    RequestPriority,
    ServerConfig,
)
from scheduler_engine.validation import (
    InputValidationError,
    sanitize_prompt,
    validate_prompt,
    validate_sampling_params,
)
from scheduler_engine.logging_config import (
    get_correlation_id,
    set_correlation_id,
)


# =========================================================================
# 1. Input Validation and Sanitization Tests
# =========================================================================

def test_sanitize_prompt_removes_null_bytes_and_controls():
    raw_prompt = "Hello\x00World\x07! \x1b[31mRed\x1b[0m \t\nValid text."
    sanitized = sanitize_prompt(raw_prompt)
    assert "\x00" not in sanitized
    assert "\x07" not in sanitized
    assert "HelloWorld" in sanitized
    assert "Valid text." in sanitized


def test_empty_and_whitespace_prompt_rejection():
    with pytest.raises(InputValidationError):
        validate_prompt("   ")

    with pytest.raises(InputValidationError):
        validate_prompt("\x00\x00\t\n")

    with pytest.raises(ValueError):
        InferenceRequest(prompt="   ", max_tokens=10)


def test_prompt_exceeding_context_window_rejection():
    # Model max length is 4096. A prompt with 4000 words + 200 tokens exceeds 4096.
    huge_prompt = "word " * 4000
    with pytest.raises(InputValidationError):
        validate_prompt(huge_prompt, max_tokens=200, max_model_len=4096)

    with pytest.raises(ValueError):
        InferenceRequest(prompt=huge_prompt, max_tokens=200)


def test_sampling_parameters_bounds():
    # Temperature out of bounds
    with pytest.raises(InputValidationError):
        validate_sampling_params(temperature=-0.5, max_tokens=50)

    with pytest.raises(InputValidationError):
        validate_sampling_params(temperature=2.5, max_tokens=50)

    # Max tokens out of bounds
    with pytest.raises(InputValidationError):
        validate_sampling_params(temperature=0.7, max_tokens=0)

    with pytest.raises(InputValidationError):
        validate_sampling_params(temperature=0.7, max_tokens=5000, max_model_len=4096)

    # Penalties out of bounds
    with pytest.raises(InputValidationError):
        validate_sampling_params(presence_penalty=3.0)


# =========================================================================
# 2. GPU Memory Soft-Limit and OOM Emergency Recovery Tests
# =========================================================================

def test_gpu_memory_soft_limit_shedding():
    config = ServerConfig(soft_memory_limit_ratio=0.85, hard_memory_limit_ratio=0.92)
    admission = AdmissionController(config)

    req_low = InferenceRequest(prompt="Low priority test", max_tokens=20, priority=RequestPriority.LOW)
    req_high = InferenceRequest(prompt="High priority test", max_tokens=20, priority=RequestPriority.HIGH)

    # Under 88% memory (7200MB / 8192MB = 87.8%): soft limit sheds LOW priority
    res_low = admission.evaluate(
        req_low,
        current_queue_size=0,
        active_concurrency=2,
        gpu_memory_used_mb=7200,
        gpu_memory_total_mb=8192,
    )
    assert res_low.admitted is False
    assert res_low.status == AdmissionStatus.REJECTED_OVERLOAD
    assert "soft limit" in res_low.reason.lower()

    # Under 88% memory: HIGH priority is protected and admitted
    res_high = admission.evaluate(
        req_high,
        current_queue_size=0,
        active_concurrency=2,
        gpu_memory_used_mb=7200,
        gpu_memory_total_mb=8192,
    )
    assert res_high.admitted is True

    # Under 95% memory (7800MB / 8192MB = 95.2%): hard limit sheds even HIGH priority
    res_hard = admission.evaluate(
        req_high,
        current_queue_size=0,
        active_concurrency=2,
        gpu_memory_used_mb=7800,
        gpu_memory_total_mb=8192,
    )
    assert res_hard.admitted is False
    assert "hard limit" in res_hard.reason.lower()


def test_oom_emergency_recovery_and_throttle():
    async def _run():
        config = ServerConfig(
            initial_concurrency=6,
            min_concurrency=2,
            max_concurrency=12,
            target_sla_ms=5000.0,
        )
        backend = MockBackend(config=config, simulated_ttft_seconds=0.01)
        policy = DynamicBatchPolicy(backend=backend, config=config)
        await policy.initialize()

        # Inject simulated CUDA OOM
        backend.inject_oom(True)

        req = InferenceRequest(prompt="Trigger OOM", max_tokens=10)
        with pytest.raises(GPUOutOfMemoryError):
            await policy.schedule(req)

        # Verify OOM recovery actions:
        # 1. Concurrency limit collapsed to min_concurrency
        assert policy.adaptive_controller.current_concurrency == config.min_concurrency
        # 2. OOM recovery counter incremented
        stats = policy.get_stats()
        assert stats.oom_recoveries_count >= 1
        # 3. Backend memory/trigger cleared
        assert backend._simulate_oom is False

        # Verify system can immediately process subsequent requests normally
        req2 = InferenceRequest(prompt="Normal request post-OOM", max_tokens=5)
        res2 = await policy.schedule(req2)
        assert res2.tokens_generated > 0

        await policy.shutdown()

    asyncio.run(_run())


# =========================================================================
# 3. Client Disconnect & Slot Reclamation Tests
# =========================================================================

def test_client_disconnect_queued_slot_reclamation():
    async def _run():
        config = ServerConfig(initial_concurrency=2, target_sla_ms=5000.0)
        backend = MockBackend(config=config, simulated_ttft_seconds=0.05)
        policy = DynamicBatchPolicy(backend=backend, config=config)
        await policy.initialize()

        # Enqueue a request
        req = InferenceRequest(prompt="Waiting in queue", max_tokens=10, request_id="client-cancel-queued")
        fut = await policy.queue.enqueue(req)

        # Client aborts connection before dequeuing
        cancelled = await policy.cancel_request("client-cancel-queued", reason="Client aborted")
        assert cancelled is True
        assert fut.cancelled() or fut.done()
        assert policy.queue.size == 0
        assert policy.client_disconnects_count >= 1

        await policy.shutdown()

    asyncio.run(_run())


def test_client_disconnect_active_slot_reclamation():
    async def _run():
        config = ServerConfig(initial_concurrency=4, target_sla_ms=5000.0)
        backend = MockBackend(config=config, tokens_per_second=5.0, simulated_ttft_seconds=0.02)
        policy = DynamicBatchPolicy(backend=backend, config=config)
        await policy.initialize()

        req = InferenceRequest(prompt="Long generation prompt", max_tokens=50, request_id="client-cancel-active")
        task = asyncio.create_task(policy.schedule(req))

        # Wait until request enters active execution
        await asyncio.sleep(0.05)
        assert "client-cancel-active" in policy._active_requests

        # Client aborts connection while actively generating
        reclaimed = await policy.cancel_request("client-cancel-active", reason="Client connection dropped")
        assert reclaimed is True

        # Allow task loop to process cancellation
        await asyncio.sleep(0.05)
        assert "client-cancel-active" not in policy._active_requests
        assert policy.client_disconnects_count >= 1

        # Verify task completed with cancellation
        with pytest.raises((asyncio.CancelledError, Exception)):
            await task

        await policy.shutdown()

    asyncio.run(_run())


# =========================================================================
# 4. Burst-Shedding & Differentiated Backpressure Tests
# =========================================================================

def test_differentiated_burst_shedding():
    config = ServerConfig(max_queue_size=10, burst_shed_queue_ratio=0.70, target_sla_ms=5000.0)
    admission = AdmissionController(config)

    req_low = InferenceRequest(prompt="Low priority", max_tokens=10, priority=RequestPriority.LOW)
    req_norm = InferenceRequest(prompt="Normal priority", max_tokens=10, priority=RequestPriority.NORMAL)
    req_high = InferenceRequest(prompt="High priority", max_tokens=10, priority=RequestPriority.HIGH)

    # Queue at 7/10 (70%): LOW priority should be shed by burst protection
    res_low = admission.evaluate(req_low, current_queue_size=7, active_concurrency=4)
    assert res_low.admitted is False
    assert res_low.status == AdmissionStatus.REJECTED_OVERLOAD
    assert "Burst shedding active" in res_low.reason

    # Normal and High priority are admitted at 70%
    res_norm = admission.evaluate(req_norm, current_queue_size=7, active_concurrency=4)
    assert res_norm.admitted is True

    res_high = admission.evaluate(req_high, current_queue_size=7, active_concurrency=4)
    assert res_high.admitted is True

    # Queue at 9/10 (90% severe burst): NORMAL is also shed, only HIGH is protected
    res_norm_severe = admission.evaluate(req_norm, current_queue_size=9, active_concurrency=4)
    assert res_norm_severe.admitted is False
    assert "Severe burst" in res_norm_severe.reason

    res_high_severe = admission.evaluate(req_high, current_queue_size=9, active_concurrency=4)
    assert res_high_severe.admitted is True


# =========================================================================
# 5. Correlation ID & Structured Logging Propagation Tests
# =========================================================================

def test_correlation_id_contextvar_propagation():
    test_cid = "corr-test-12345"
    set_correlation_id(test_cid)
    assert get_correlation_id() == test_cid

    # Verify InferenceRequest picks up correlation_id if provided
    req = InferenceRequest(prompt="CID prompt", max_tokens=5, request_id=test_cid)
    assert req.request_id == test_cid
