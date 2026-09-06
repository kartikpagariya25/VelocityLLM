"""
VelocityLLM - Phase 3 Verification & Edge Case Hardening Demonstration
Demonstrates all 6 novel robustness pillars of Phase 3:
1. Input Validation and Sanitization (empty prompts, context limits, null bytes)
2. GPU Memory Soft/Hard Limit Protection (multi-tier memory preemption)
3. GPU OOM Emergency Recovery (concurrency floor collapse, cache purge, zero crash)
4. Client-Disconnect Detection and Immediate Slot Reclamation (queued & active)
5. Differentiated Burst-Shedding & Backpressure (HTTP 429 with Retry-After)
6. Structured Logging with Request Correlation IDs and Telemetry Export
"""

import asyncio
import json
import logging
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from scheduler_engine.types import (
    AdmissionStatus,
    InferenceRequest,
    RequestPriority,
    ServerConfig,
)
from scheduler_engine.backend import GPUOutOfMemoryError, MockBackend
from scheduler_engine.policy import (
    AdmissionRejectedException,
    DynamicBatchPolicy,
)
from scheduler_engine.admission import AdmissionController
from scheduler_engine.validation import (
    InputValidationError,
    sanitize_prompt,
    validate_prompt,
    validate_sampling_params,
)
from scheduler_engine.logging_config import (
    StructuredJsonFormatter,
    get_correlation_id,
    set_correlation_id,
)


async def verify_input_validation():
    print("\n[1/6] Verifying Input Validation & Adversarial Sanitization...")

    # 1. Null byte & control character sanitization
    adversarial_prompt = "Explain quantum computing\x00\x07 with \x1b[31mmalicious\x1b[0m bytes."
    sanitized = sanitize_prompt(adversarial_prompt)
    assert "\x00" not in sanitized
    assert "\x07" not in sanitized
    print(f"  [OK] Sanitized adversarial prompt (removed null bytes & control codes)")

    # 2. Empty / whitespace rejection
    try:
        validate_prompt("   \t\n   ")
        assert False, "Should have rejected empty prompt"
    except InputValidationError as e:
        print(f"  [OK] Rejected whitespace-only prompt: '{e.message}'")

    # 3. Context window bound enforcement
    huge_prompt = "token " * 4100
    try:
        validate_prompt(huge_prompt, max_tokens=100, max_model_len=4096)
        assert False, "Should have rejected oversized prompt"
    except InputValidationError as e:
        print(f"  [OK] Enforced context window limit: {e.message[:70]}...")

    # 4. Sampling parameter bounds
    try:
        validate_sampling_params(temperature=3.5, max_tokens=50)
        assert False, "Should have rejected temperature 3.5"
    except InputValidationError as e:
        print(f"  [OK] Enforced sampling parameter validation: {e.message}")


async def verify_gpu_memory_limits():
    print("\n[2/6] Verifying GPU Memory Soft/Hard Limit Protection...")
    config = ServerConfig(soft_memory_limit_ratio=0.85, hard_memory_limit_ratio=0.94)
    admission = AdmissionController(config)

    req_low = InferenceRequest(prompt="Low priority query", max_tokens=20, priority=RequestPriority.LOW)
    req_high = InferenceRequest(prompt="High priority query", max_tokens=20, priority=RequestPriority.HIGH)

    # 1. Memory at 88% (soft limit): LOW priority shed, HIGH priority admitted
    res_low = admission.evaluate(
        req_low,
        current_queue_size=0,
        active_concurrency=2,
        gpu_memory_used_mb=7200,
        gpu_memory_total_mb=8192,
    )
    assert res_low.admitted is False
    assert "soft limit" in res_low.reason.lower()
    print(f"  [OK] Soft limit (87.9% VRAM): Shed LOW priority request -> {res_low.reason}")

    res_high = admission.evaluate(
        req_high,
        current_queue_size=0,
        active_concurrency=2,
        gpu_memory_used_mb=7200,
        gpu_memory_total_mb=8192,
    )
    assert res_high.admitted is True
    print("  [OK] Soft limit (87.9% VRAM): Successfully admitted HIGH priority request")

    # 2. Memory at 96% (hard limit): Even HIGH priority shed to prevent CUDA crash
    res_hard = admission.evaluate(
        req_high,
        current_queue_size=0,
        active_concurrency=2,
        gpu_memory_used_mb=7900,
        gpu_memory_total_mb=8192,
    )
    assert res_hard.admitted is False
    assert "hard limit" in res_hard.reason.lower()
    print(f"  [OK] Hard limit (96.4% VRAM): Preempted all traffic -> {res_hard.reason}")


async def verify_oom_recovery():
    print("\n[3/6] Verifying GPU OOM Emergency Recovery & Concurrency Collapse...")
    config = ServerConfig(
        initial_concurrency=8,
        min_concurrency=2,
        max_concurrency=16,
        target_sla_ms=5000.0,
    )
    backend = MockBackend(config=config, simulated_ttft_seconds=0.01)
    policy = DynamicBatchPolicy(backend=backend, config=config)
    await policy.initialize()

    assert policy.adaptive_controller.current_concurrency == 8

    # Inject simulated CUDA OOM
    backend.inject_oom(True)
    req = InferenceRequest(prompt="Provoke CUDA OOM", max_tokens=10, request_id="oom-test-1")

    try:
        await policy.schedule(req)
        assert False, "Should have raised GPUOutOfMemoryError"
    except GPUOutOfMemoryError as e:
        print(f"  [OK] Caught GPUOutOfMemoryError gracefully without process crash: {e}")

    # Verify AIMD emergency concurrency collapse to floor
    current_concurrency = policy.adaptive_controller.current_concurrency
    assert current_concurrency == config.min_concurrency
    print(f"  [OK] AIMD controller emergency collapsed concurrency: 8 -> {current_concurrency}")

    # Verify OOM telemetry counter
    stats = policy.get_stats()
    assert stats.oom_recoveries_count == 1
    print(f"  [OK] Telemetry oom_recoveries_count: {stats.oom_recoveries_count}")

    # Verify system self-healed and immediately serves subsequent requests
    req_healthy = InferenceRequest(prompt="Post-OOM healthy request", max_tokens=8)
    res_healthy = await policy.schedule(req_healthy)
    assert res_healthy.tokens_generated > 0
    print(f"  [OK] Post-OOM recovery confirmed: generated {res_healthy.tokens_generated} tokens")

    await policy.shutdown()


async def verify_client_disconnect_reclamation():
    print("\n[4/6] Verifying Client-Disconnect Detection & Slot Reclamation...")
    config = ServerConfig(initial_concurrency=4, target_sla_ms=5000.0)
    backend = MockBackend(config=config, tokens_per_second=10.0, simulated_ttft_seconds=0.02)
    policy = DynamicBatchPolicy(backend=backend, config=config)
    await policy.initialize()

    # Part A: Client disconnect while waiting in queue
    req_queued = InferenceRequest(prompt="Waiting in queue", max_tokens=10, request_id="disconnect-queued")
    fut = await policy.queue.enqueue(req_queued)
    assert policy.queue.size == 1

    cancelled_queued = await policy.cancel_request("disconnect-queued", reason="Client aborted HTTP connection")
    assert cancelled_queued is True
    assert policy.queue.size == 0
    print("  [OK] Queued request cancelled: removed from queue and future cancelled")

    # Part B: Client disconnect while actively generating
    req_active = InferenceRequest(prompt="Active long generation", max_tokens=50, request_id="disconnect-active")
    active_task = asyncio.create_task(policy.schedule(req_active))

    # Wait until execution starts
    await asyncio.sleep(0.04)
    assert "disconnect-active" in policy._active_requests
    print("  [OK] Request entered active execution on backend")

    # Abort connection
    reclaimed = await policy.cancel_request("disconnect-active", reason="Client closed stream")
    assert reclaimed is True

    await asyncio.sleep(0.04)
    assert "disconnect-active" not in policy._active_requests
    print("  [OK] Active generation aborted mid-stream & execution slot reclaimed immediately")

    stats = policy.get_stats()
    assert stats.client_disconnects_count == 2
    print(f"  [OK] Total client_disconnects_count tracked: {stats.client_disconnects_count}")

    try:
        await active_task
    except (asyncio.CancelledError, Exception):
        pass

    await policy.shutdown()


async def verify_burst_shedding():
    print("\n[5/6] Verifying Differentiated Burst Shedding & Backpressure...")
    config = ServerConfig(
        max_queue_size=10,
        burst_shed_queue_ratio=0.70,
        target_sla_ms=5000.0,
    )
    admission = AdmissionController(config)

    req_low = InferenceRequest(prompt="Low priority task", max_tokens=15, priority=RequestPriority.LOW)
    req_normal = InferenceRequest(prompt="Normal priority task", max_tokens=15, priority=RequestPriority.NORMAL)
    req_high = InferenceRequest(prompt="High priority task", max_tokens=15, priority=RequestPriority.HIGH)

    # 1. At 70% queue depth (7/10): LOW priority is shed, NORMAL and HIGH admitted
    res_low = admission.evaluate(req_low, current_queue_size=7, active_concurrency=4)
    assert res_low.admitted is False
    assert res_low.status == AdmissionStatus.REJECTED_OVERLOAD
    assert res_low.retry_after_seconds > 0
    print(f"  [OK] 70% Queue Depth: Shed LOW priority -> Retry-After: {res_low.retry_after_seconds}s")

    res_norm = admission.evaluate(req_normal, current_queue_size=7, active_concurrency=4)
    assert res_norm.admitted is True
    print("  [OK] 70% Queue Depth: Admitted NORMAL priority request")

    # 2. At 90% queue depth (9/10): NORMAL priority is also shed to protect HIGH priority SLA
    res_norm_severe = admission.evaluate(req_normal, current_queue_size=9, active_concurrency=4)
    assert res_norm_severe.admitted is False
    print(f"  [OK] 90% Queue Depth: Shed NORMAL priority to protect headroom")

    res_high = admission.evaluate(req_high, current_queue_size=9, active_concurrency=4)
    assert res_high.admitted is True
    print("  [OK] 90% Queue Depth: Admitted HIGH priority request")

    # 3. Verify burst shed counter
    stats = admission.get_stats()
    assert stats["total_burst_shed"] == 2
    print(f"  [OK] Admission burst_shed counter verified: {stats['total_burst_shed']}")


async def verify_structured_logging():
    print("\n[6/6] Verifying Structured Logging & Request Correlation IDs...")

    test_corr_id = "req-corr-999a"
    set_correlation_id(test_corr_id)
    assert get_correlation_id() == test_corr_id
    print(f"  [OK] ContextVar correlation ID active: {get_correlation_id()}")

    # Test JSON formatter output
    formatter = StructuredJsonFormatter()
    record = logging.LogRecord(
        name="velocityllm.test",
        level=logging.INFO,
        pathname="test.py",
        lineno=42,
        msg="Inference completed successfully",
        args=(),
        exc_info=None,
    )
    record.correlation_id = test_corr_id
    formatted_json = formatter.format(record)
    parsed = json.loads(formatted_json)

    assert parsed["correlation_id"] == test_corr_id
    assert parsed["message"] == "Inference completed successfully"
    assert "timestamp" in parsed
    print(f"  [OK] Structured JSON log record verified: {formatted_json}")


async def main():
    print("=" * 65)
    print("       VELOCITY-LLM PHASE 3 VERIFICATION SUITE")
    print("       (Robustness, Edge Cases, Fault Recovery, & Logging)")
    print("=" * 65)

    await verify_input_validation()
    await verify_gpu_memory_limits()
    await verify_oom_recovery()
    await verify_client_disconnect_reclamation()
    await verify_burst_shedding()
    await verify_structured_logging()

    print("\n" + "=" * 65)
    print("  [SUCCESS] ALL PHASE 3 REQUIREMENTS & EXIT CRITERIA VERIFIED!")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
