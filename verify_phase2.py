"""
VelocityLLM - Phase 2 Verification & Demonstration Script
Demonstrates all novel components of Phase 2:
1. Swappable Policy Interface (Static vs. Dynamic)
2. SLA-Aware Admission Control with Burst Shedding (HTTP 429)
3. Prioritized Queue with Anti-Starvation Aging
4. Adaptive Batch Size Controller (AIMD Feedback Loop)
5. Telemetry & Metrics Verification
"""

import asyncio
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from scheduler_engine.types import (
    InferenceRequest,
    RequestPriority,
    ServerConfig,
)
from scheduler_engine.backend import MockBackend
from scheduler_engine.policy import (
    AdmissionRejectedException,
    DynamicBatchPolicy,
    StaticBatchPolicy,
)
from scheduler_engine.admission import AdmissionController
from scheduler_engine.priority_queue import PrioritizedRequestQueue
from scheduler_engine.adaptive_controller import AdaptiveBatchController


async def verify_admission_control():
    print("\n[1/5] Verifying SLA-Aware Admission Controller...")
    config = ServerConfig(max_queue_size=5, target_sla_ms=2000.0)
    admission = AdmissionController(config)

    # 1. Low load: should accept
    req_normal = InferenceRequest(prompt="Accept test", max_tokens=30, priority=RequestPriority.NORMAL)
    res1 = admission.evaluate(req_normal, current_queue_size=0, active_concurrency=2)
    assert res1.admitted is True
    print("  [OK] Accepted standard request under low queue depth")

    # 2. Impossible SLA: should reject early to protect promises
    req_impossible = InferenceRequest(
        prompt="Impossible SLA test",
        max_tokens=200,
        sla_target_ms=50.0,  # 50ms is impossible for 200 tokens
    )
    res2 = admission.evaluate(req_impossible, current_queue_size=4, active_concurrency=2)
    assert res2.admitted is False
    assert "exceeds target SLA" in res2.reason
    print(f"  [OK] Gracefully shed impossible SLA request: {res2.reason}")

    # 3. Queue overflow: should shed with HTTP 429 backpressure
    res3 = admission.evaluate(req_normal, current_queue_size=5, active_concurrency=4)
    assert res3.admitted is False
    print("  [OK] Saturated queue shed with backpressure protection")


async def verify_anti_starvation_aging():
    print("\n[2/5] Verifying Prioritized Queue & Anti-Starvation Aging...")
    pq = PrioritizedRequestQueue(aging_factor=2.0)

    # Enqueue a LOW priority request first
    req_low = InferenceRequest(prompt="Low Priority Job", request_id="req-low", priority=RequestPriority.LOW)
    await pq.enqueue(req_low)

    # Let aging accumulate (1.1s * 2.0 = 2.2 deduction -> score drops from 3.0 to 0.8)
    await asyncio.sleep(1.1)

    # Enqueue a newly arrived HIGH priority request (score 1.0)
    req_high = InferenceRequest(prompt="High Priority Job", request_id="req-high", priority=RequestPriority.HIGH)
    await pq.enqueue(req_high)

    # Dequeue: The aged LOW request should be promoted ahead of the new HIGH request!
    entry1 = await pq.dequeue()
    entry2 = await pq.dequeue()

    assert entry1.request.request_id == "req-low"
    assert entry2.request.request_id == "req-high"
    print("  [OK] Anti-starvation aging successfully promoted older low-priority request")


async def verify_adaptive_batch_controller():
    print("\n[3/5] Verifying Adaptive Batch-Size Controller (AIMD)...")
    config = ServerConfig(initial_concurrency=4, min_concurrency=2, max_concurrency=12)
    controller = AdaptiveBatchController(config)
    controller.cooldown_seconds = 0.0

    # Test Additive Increase under pending queue and healthy GPU
    new_concurrency = controller.evaluate_and_tune(
        gpu_util_percent=75.0,
        gpu_memory_used_mb=3000,
        gpu_memory_total_mb=8192,
        queue_depth=6,
        active_requests=4,
    )
    assert new_concurrency == 5
    print(f"  [OK] Additive increase expanded concurrency ceiling: 4 -> {new_concurrency}")

    # Test Multiplicative Decrease under high memory pressure (94% VRAM)
    contracted = controller.evaluate_and_tune(
        gpu_util_percent=98.0,
        gpu_memory_used_mb=7700,
        gpu_memory_total_mb=8192,
        queue_depth=6,
        active_requests=5,
    )
    assert contracted < 5
    print(f"  [OK] Multiplicative decrease contracted concurrency ceiling: 5 -> {contracted} (OOM protection)")


async def verify_static_vs_dynamic_execution():
    print("\n[4/5] Verifying Swappable Policies (Static vs Dynamic execution)...")
    config = ServerConfig(initial_concurrency=4, target_sla_ms=10000.0)
    backend_static = MockBackend(tokens_per_second=250.0, simulated_ttft_seconds=0.01)
    static_policy = StaticBatchPolicy(backend=backend_static, config=config)
    await static_policy.initialize()

    backend_dynamic = MockBackend(tokens_per_second=250.0, simulated_ttft_seconds=0.01)
    dynamic_policy = DynamicBatchPolicy(backend=backend_dynamic, config=config)
    await dynamic_policy.initialize()

    # Run batch of 6 requests through both policies
    prompts = [f"Inference prompt sample #{i}" for i in range(6)]

    print("  Running static baseline batch...")
    t0 = time.time()
    static_tasks = [
        static_policy.schedule(InferenceRequest(prompt=p, max_tokens=15, sla_target_ms=10000.0))
        for p in prompts
    ]
    static_res = await asyncio.gather(*static_tasks)
    static_time = time.time() - t0

    print("  Running dynamic continuous batching...")
    t1 = time.time()
    dynamic_tasks = [
        dynamic_policy.schedule(
            InferenceRequest(
                prompt=p,
                max_tokens=15,
                priority=RequestPriority.HIGH if i % 2 == 0 else RequestPriority.NORMAL,
                sla_target_ms=10000.0,
            )
        )
        for i, p in enumerate(prompts)
    ]
    dynamic_res = await asyncio.gather(*dynamic_tasks)
    dynamic_time = time.time() - t1

    print(f"  [OK] Static completed {len(static_res)} requests in {static_time:.2f}s")
    print(f"  [OK] Dynamic completed {len(dynamic_res)} requests in {dynamic_time:.2f}s")

    await static_policy.shutdown()
    await dynamic_policy.shutdown()


async def verify_telemetry():
    print("\n[5/5] Verifying Telemetry & Statistics Export...")
    config = ServerConfig(initial_concurrency=4)
    backend = MockBackend(tokens_per_second=300.0, simulated_ttft_seconds=0.01)
    policy = DynamicBatchPolicy(backend=backend, config=config)
    await policy.initialize()

    await policy.schedule(InferenceRequest(prompt="Telemetry check", max_tokens=10))
    stats = policy.get_stats().to_dict()

    assert stats["total_completed"] == 1
    assert stats["total_tokens_generated"] > 0
    assert "sla_compliance_rate_percent" in stats
    assert "effective_concurrency_limit" in stats
    print("  [OK] Full telemetry dict validated:")
    for k, v in stats.items():
        print(f"     - {k}: {v}")

    await policy.shutdown()


async def main():
    print("=" * 65)
    print("       VELOCITY-LLM PHASE 2 VERIFICATION SUITE")
    print("=" * 65)
    await verify_admission_control()
    await verify_anti_starvation_aging()
    await verify_adaptive_batch_controller()
    await verify_static_vs_dynamic_execution()
    await verify_telemetry()
    print("\n" + "=" * 65)
    print("  [SUCCESS] ALL PHASE 2 REQUIREMENTS & EXIT CRITERIA VERIFIED!")
    print("=" * 65)


if __name__ == "__main__":
    asyncio.run(main())
