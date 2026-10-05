"""
VelocityLLM - SLA-Aware Admission Controller
Evaluates incoming inference requests against current system load, queue depth,
and latency SLA deadlines to decide whether to accept, queue, or shed load (HTTP 429).
"""

import logging
import math
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
        self._ema_prefill_time = 0.0002  # seconds of prefill per image token (~5,000 tokens/s)
        self._max_prefill_time = 0.002
        self._alpha = 0.1             # Smoothing factor for EMA updates
        self._max_sample_growth = 2.0  # A single sample may at most double the estimate
        self._baseline_token_time = self._ema_token_time
        self._baseline_service_time = self._ema_service_time
        self._stale_after_seconds = 10.0
        self._last_completion = time.monotonic()

        # Metrics counters
        self.total_evaluated = 0
        self.total_accepted = 0
        self.total_rejected_overload = 0
        self.total_rejected_sla = 0
        self.total_burst_shed = 0
        self.total_dropped_waiting = 0

    def update_completion_stats(self, latency_seconds: float, tokens_generated: int, image_tokens: int = 0) -> None:
        """Update service time estimation metrics from completed requests."""
        if latency_seconds <= 0:
            return
        self._last_completion = time.monotonic()

        if image_tokens > 0:
            decode_estimate = 0.05 + tokens_generated * self._ema_token_time
            sample = min(max((latency_seconds - decode_estimate) / image_tokens, 0.0), self._max_prefill_time)
            self._ema_prefill_time = (self._alpha * sample) + ((1.0 - self._alpha) * self._ema_prefill_time)
            latency_seconds = max(latency_seconds - image_tokens * self._ema_prefill_time, 0.1 * latency_seconds)

        service_sample = min(latency_seconds, self._ema_service_time * self._max_sample_growth)
        self._ema_service_time = (self._alpha * service_sample) + (
            (1.0 - self._alpha) * self._ema_service_time
        )
        if tokens_generated > 0:
            token_rate = min(latency_seconds / tokens_generated, self._ema_token_time * self._max_sample_growth)
            self._ema_token_time = (self._alpha * token_rate) + (
                (1.0 - self._alpha) * self._ema_token_time
            )

    def _decay_stale_estimates(self) -> None:
        """Pull estimates back toward their baseline when no request has completed for a while."""
        now = time.monotonic()
        if now - self._last_completion < self._stale_after_seconds:
            return
        self._ema_token_time = (self._ema_token_time + self._baseline_token_time) / 2
        self._ema_service_time = (self._ema_service_time + self._baseline_service_time) / 2
        self._last_completion = now
        logger.warning("No completions for %.0fs; service-time estimates decayed toward baseline.", self._stale_after_seconds)

    @staticmethod
    def projected_slots(active_concurrency: int, concurrency_ceiling: Optional[int] = None) -> float:
        """
        Effective number of parallel slots to plan with. The adaptive controller can raise the limit
        within about a second, but throughput grows roughly with the square root of the batch size
        (4x the concurrency gave about 1.9x the tokens/s on the reference laptop GPU), so a reachable
        ceiling is credited at that rate instead of fully.
        """
        base = max(1, active_concurrency)
        if concurrency_ceiling and concurrency_ceiling > base:
            return base * math.sqrt(concurrency_ceiling / base)
        return float(base)

    def estimate_wait_time(
        self,
        current_queue_size: int,
        active_concurrency: int,
        concurrency_ceiling: Optional[int] = None,
        queued_image_tokens: int = 0,
    ) -> float:
        """
        Estimate queueing wait time in seconds before an arriving request begins execution.
        E[wait] = (queue_length * avg_service_time + queued image prefill) / projected parallel slots
        """
        slots = self.projected_slots(active_concurrency, concurrency_ceiling)
        work = current_queue_size * self._ema_service_time + queued_image_tokens * self._ema_prefill_time
        return work / slots

    def estimate_execution_time(self, max_tokens: int, image_tokens: int = 0) -> float:
        """
        Estimate execution generation duration for a request.
        """
        # Prefill / TTFT base + image prefill + generation phase
        return 0.05 + image_tokens * self._ema_prefill_time + (max_tokens * self._ema_token_time)

    def sla_limit_seconds(self, request: InferenceRequest) -> float:
        """SLA the request is judged against, including the extra grace HIGH priority gets."""
        target = (request.sla_target_ms or self.default_sla_ms) / 1000.0
        return target * (1.35 if request.priority == RequestPriority.HIGH else 1.05)

    def is_hopeless(self, request: InferenceRequest, waited_seconds: float) -> bool:
        """True when the request would miss its SLA even if it ran at twice the usual speed."""
        optimistic_exec = 0.5 * self.estimate_execution_time(request.max_tokens, request.image_tokens)
        return waited_seconds + optimistic_exec > self.sla_limit_seconds(request)

    def evaluate(
        self,
        request: InferenceRequest,
        current_queue_size: int,
        active_concurrency: int,
        gpu_memory_used_mb: int = 0,
        gpu_memory_total_mb: int = 8192,
        in_flight: Optional[int] = None,
        gpu_memory_baseline_mb: Optional[int] = None,
        concurrency_ceiling: Optional[int] = None,
        queued_image_tokens: int = 0,
    ) -> AdmissionResult:
        """
        Evaluate whether to admit or reject an incoming request.
        Implements multi-tiered differentiated burst shedding, GPU memory limits,
        and SLA headroom prediction.
        """
        self.total_evaluated += 1
        self._decay_stale_estimates()
        memory_grew = (
            gpu_memory_baseline_mb is None
            or gpu_memory_total_mb <= 0
            or gpu_memory_used_mb - gpu_memory_baseline_mb >= 0.03 * gpu_memory_total_mb
        )
        est_wait = self.estimate_wait_time(current_queue_size, active_concurrency, concurrency_ceiling, queued_image_tokens)

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
            if mem_utilization >= hard_limit and memory_grew:
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
        est_exec = self.estimate_execution_time(request.max_tokens, request.image_tokens)
        est_total_latency = est_wait + est_exec

        # Allow HIGH priority requests slightly more grace headroom (1.35x)
        tolerance_multiplier = 1.35 if request.priority == RequestPriority.HIGH else 1.05

        running = active_concurrency if in_flight is None else in_flight
        server_idle = current_queue_size == 0 and running == 0
        if not server_idle and est_total_latency > (target_sla_sec * tolerance_multiplier):
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
            heavy_image = request.image_tokens >= getattr(self.config, "heavy_image_tokens", 768)
            if mem_utilization >= soft_limit and memory_grew and (request.priority != RequestPriority.HIGH or heavy_image):
                self.total_rejected_overload += 1
                return AdmissionResult(
                    status=AdmissionStatus.REJECTED_OVERLOAD,
                    admitted=False,
                    reason=f"GPU memory soft limit reached ({mem_utilization*100:.1f}% >= {soft_limit*100:.0f}%). Shedding {'heavy image' if heavy_image else 'non-urgent'} traffic.",
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
            "total_dropped_waiting": self.total_dropped_waiting,
            "ema_service_time_seconds": round(self._ema_service_time, 4),
            "ema_token_time_seconds": round(self._ema_token_time, 5),
            "ema_prefill_seconds_per_image_token": round(self._ema_prefill_time, 6),
        }
