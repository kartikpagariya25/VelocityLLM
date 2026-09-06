"""
VelocityLLM - SLA-Aware Admission Controller
Evaluates incoming inference requests against current system load, queue depth,
and latency SLA deadlines to decide whether to accept, queue, or shed load (HTTP 429).
"""

import logging
import time
from typing import Optional, Tuple

from scheduler_engine.types import (
    AdmissionResult,
    AdmissionStatus,
    InferenceRequest,
    RequestPriority,
    ServerConfig,
)

logger = logging.getLogger("velocityllm.admission")


class AdmissionController:
    """
    SLA-aware Admission Controller for LLM serving.
    Guarantees latency promises by denying requests that would either breach their
    own SLA or cause existing queued/in-flight requests to breach theirs.
    """

    def __init__(self, config: ServerConfig):
        self.config = config
        self.max_queue_size = config.max_queue_size
        self.default_sla_ms = config.target_sla_ms

        # Exponential moving averages for service times
        self._ema_token_time = 0.015  # ~15ms per token initial baseline
        self._ema_service_time = 1.2   # ~1.2s average request service time
        self._alpha = 0.1             # Smoothing factor for EMA updates

        # Metrics counters
        self.total_evaluated = 0
        self.total_accepted = 0
        self.total_rejected_overload = 0
        self.total_rejected_sla = 0
        self.total_burst_shed = 0

    def update_completion_stats(self, latency_seconds: float, tokens_generated: int) -> None:
        """Update service time estimation metrics from completed requests."""
        if latency_seconds <= 0:
            return

        self._ema_service_time = (self._alpha * latency_seconds) + (
            (1.0 - self._alpha) * self._ema_service_time
        )
        if tokens_generated > 0:
            token_rate = latency_seconds / tokens_generated
            self._ema_token_time = (self._alpha * token_rate) + (
                (1.0 - self._alpha) * self._ema_token_time
            )

    def estimate_wait_time(self, current_queue_size: int, active_concurrency: int) -> float:
        """
        Estimate queueing wait time in seconds before an arriving request begins execution.
        E[wait] = (queue_length * avg_service_time) / max(1, concurrency)
        """
        effective_concurrency = max(1, active_concurrency)
        return (current_queue_size * self._ema_service_time) / effective_concurrency

    def estimate_execution_time(self, max_tokens: int) -> float:
        """
        Estimate execution generation duration for a request.
        """
        # Prefill / TTFT base + generation phase
        return 0.05 + (max_tokens * self._ema_token_time)

    def evaluate(
        self,
        request: InferenceRequest,
        current_queue_size: int,
        active_concurrency: int,
        gpu_memory_used_mb: int = 0,
        gpu_memory_total_mb: int = 8192,
    ) -> AdmissionResult:
        """
        Evaluate whether to admit or reject an incoming request.
        Implements multi-tiered differentiated burst shedding, GPU memory limits,
        and SLA headroom prediction.
        """
        self.total_evaluated += 1
        est_wait = self.estimate_wait_time(current_queue_size, active_concurrency)

        # 1. Check Hard Queue Capacity Limit
        if current_queue_size >= self.max_queue_size:
            self.total_rejected_overload += 1
            retry_after = max(0.5, round(est_wait, 2))
            logger.warning(
                "Request %s rejected: Queue full (%d/%d). Retry after %.2fs",
                request.request_id,
                current_queue_size,
                self.max_queue_size,
                retry_after,
            )
            return AdmissionResult(
                status=AdmissionStatus.REJECTED_OVERLOAD,
                admitted=False,
                reason=f"Server queue capacity saturated ({current_queue_size}/{self.max_queue_size}). Shedding load.",
                estimated_delay_seconds=retry_after,
                retry_after_seconds=retry_after,
            )

        # 2. GPU Memory Hard Limit (Preempt all traffic to prevent CUDA OOM crash)
        if gpu_memory_total_mb > 0:
            mem_utilization = gpu_memory_used_mb / gpu_memory_total_mb
            hard_limit = getattr(self.config, "hard_memory_limit_ratio", 0.94)
            if mem_utilization >= hard_limit:
                self.total_rejected_overload += 1
                return AdmissionResult(
                    status=AdmissionStatus.REJECTED_OVERLOAD,
                    admitted=False,
                    reason=f"GPU critical memory hard limit reached ({mem_utilization*100:.1f}% >= {hard_limit*100:.0f}%). Shedding all traffic.",
                    estimated_delay_seconds=1.5,
                    retry_after_seconds=1.5,
                )

        # 3. SLA Target Headroom Verification
        target_sla_sec = (request.sla_target_ms or self.default_sla_ms) / 1000.0
        est_exec = self.estimate_execution_time(request.max_tokens)
        est_total_latency = est_wait + est_exec

        # Allow HIGH priority requests slightly more grace headroom (1.35x)
        tolerance_multiplier = 1.35 if request.priority == RequestPriority.HIGH else 1.05

        if est_total_latency > (target_sla_sec * tolerance_multiplier):
            self.total_rejected_sla += 1
            retry_after = round(est_wait, 2)
            logger.info(
                "Request %s rejected: Predicted latency %.2fs exceeds SLA target %.2fs.",
                request.request_id,
                est_total_latency,
                target_sla_sec,
            )
            return AdmissionResult(
                status=AdmissionStatus.REJECTED_SLA_IMPOSSIBLE,
                admitted=False,
                reason=(
                    f"Predicted latency ({est_total_latency:.2f}s) exceeds target SLA "
                    f"({target_sla_sec:.2f}s) under current load."
                ),
                estimated_delay_seconds=est_total_latency,
                retry_after_seconds=max(0.2, retry_after),
            )

        # 4. Differentiated Burst-Shedding Backpressure
        burst_ratio = getattr(self.config, "burst_shed_queue_ratio", 0.70)
        burst_threshold = int(self.max_queue_size * burst_ratio)
        severe_burst_threshold = int(self.max_queue_size * 0.90)

        # Tier 4.1: Severe burst (>= 90%) -> Shed LOW and NORMAL priority traffic
        if current_queue_size >= severe_burst_threshold and request.priority > RequestPriority.HIGH:
            self.total_rejected_overload += 1
            self.total_burst_shed += 1
            retry_after = max(0.5, round(est_wait, 2))
            logger.warning(
                "Request %s shed by severe burst-shedding (%d/%d queued). Priority: %s",
                request.request_id,
                current_queue_size,
                self.max_queue_size,
                request.priority.name,
            )
            return AdmissionResult(
                status=AdmissionStatus.REJECTED_OVERLOAD,
                admitted=False,
                reason=(
                    f"Severe burst backpressure ({current_queue_size}/{self.max_queue_size} queued). "
                    f"Shedding {request.priority.name} priority traffic to guarantee HIGH priority SLAs."
                ),
                estimated_delay_seconds=retry_after,
                retry_after_seconds=retry_after,
            )

        # Tier 4.2: Moderate burst (>= burst_ratio, e.g. 70%) -> Shed LOW priority traffic
        if current_queue_size >= burst_threshold and request.priority == RequestPriority.LOW:
            self.total_rejected_overload += 1
            self.total_burst_shed += 1
            retry_after = max(0.5, round(est_wait, 2))
            logger.info(
                "Request %s shed by burst-shedding (%d/%d queued). Shedding LOW priority request.",
                request.request_id,
                current_queue_size,
                self.max_queue_size,
            )
            return AdmissionResult(
                status=AdmissionStatus.REJECTED_OVERLOAD,
                admitted=False,
                reason=(
                    f"Burst shedding active ({current_queue_size}/{self.max_queue_size} queued). "
                    f"Shedding LOW priority traffic to preserve system headroom."
                ),
                estimated_delay_seconds=retry_after,
                retry_after_seconds=retry_after,
            )

        # 5. GPU Memory Soft-Limit Protection (Shed non-HIGH traffic)
        if gpu_memory_total_mb > 0:
            mem_utilization = gpu_memory_used_mb / gpu_memory_total_mb
            soft_limit = getattr(self.config, "soft_memory_limit_ratio", 0.88)
            if mem_utilization >= soft_limit and request.priority != RequestPriority.HIGH:
                self.total_rejected_overload += 1
                return AdmissionResult(
                    status=AdmissionStatus.REJECTED_OVERLOAD,
                    admitted=False,
                    reason=f"GPU memory soft limit reached ({mem_utilization*100:.1f}% >= {soft_limit*100:.0f}%). Shedding non-urgent traffic.",
                    estimated_delay_seconds=1.0,
                    retry_after_seconds=1.0,
                )

        # Request Accepted
        self.total_accepted += 1
        status = AdmissionStatus.ACCEPTED if current_queue_size == 0 else AdmissionStatus.QUEUED
        return AdmissionResult(
            status=status,
            admitted=True,
            reason="Admitted successfully.",
            estimated_delay_seconds=est_wait,
            retry_after_seconds=0.0,
        )

    def get_stats(self) -> dict:
        """Return operational counters and metrics for the admission controller."""
        return {
            "total_evaluated": self.total_evaluated,
            "total_accepted": self.total_accepted,
            "total_rejected_overload": self.total_rejected_overload,
            "total_rejected_sla": self.total_rejected_sla,
            "total_burst_shed": self.total_burst_shed,
            "ema_service_time_seconds": round(self._ema_service_time, 4),
            "ema_token_time_seconds": round(self._ema_token_time, 5),
        }
