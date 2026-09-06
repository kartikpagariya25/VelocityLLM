"""
VelocityLLM - Adaptive Batch Size Controller
Implements feedback control loop (AIMD) to dynamically tune the scheduler's
concurrency ceiling based on live GPU memory headroom, utilization, and queue depth.
"""

import logging
import time
from typing import Dict, Tuple

from scheduler_engine.types import ServerConfig

logger = logging.getLogger("velocityllm.adaptive_controller")


class AdaptiveBatchController:
    """
    Dynamically adjusts effective concurrency limit C_eff between [min_concurrency, max_concurrency].
    Uses Additive-Increase / Multiplicative-Decrease (AIMD) control algorithm.
    """

    def __init__(self, config: ServerConfig):
        self.config = config
        self.min_concurrency = config.min_concurrency
        self.max_concurrency = config.max_concurrency
        self.current_concurrency = config.initial_concurrency

        # Target thresholds from ServerConfig
        self.target_gpu_util_percent = 85.0
        self.soft_memory_limit_ratio = getattr(config, "soft_memory_limit_ratio", 0.88)
        self.hard_memory_limit_ratio = getattr(config, "hard_memory_limit_ratio", 0.94)

        # AIMD tuning parameters
        self.increase_step = 1
        self.decrease_factor = 0.75
        self.cooldown_seconds = 0.5
        self._last_adjustment_time = time.time()
        self._emergency_cooldown_until = 0.0

        # Telemetry
        self.adjustment_history = []

    def trigger_emergency_oom_throttle(self, cooldown_seconds: float = 2.0) -> int:
        """
        Immediately collapses concurrency limit to min_concurrency and activates
        an emergency cooldown to allow GPU memory to clear before resuming AIMD.
        """
        old_limit = self.current_concurrency
        self.current_concurrency = self.min_concurrency
        now = time.time()
        self._last_adjustment_time = now
        self._emergency_cooldown_until = now + cooldown_seconds

        reason = f"Emergency OOM Recovery: collapsed concurrency from {old_limit} to {self.min_concurrency}"
        logger.warning(reason)

        self.adjustment_history.append({
            "timestamp": now,
            "old_limit": old_limit,
            "new_limit": self.current_concurrency,
            "reason": reason,
        })
        return self.current_concurrency

    def evaluate_and_tune(
        self,
        gpu_util_percent: float,
        gpu_memory_used_mb: int,
        gpu_memory_total_mb: int,
        queue_depth: int,
        active_requests: int,
        recent_sla_breach: bool = False,
    ) -> int:
        """
        Evaluate current metrics and return updated concurrency limit.
        """
        now = time.time()
        if now < self._emergency_cooldown_until or (now - self._last_adjustment_time < self.cooldown_seconds):
            return self.current_concurrency

        mem_ratio = (
            (gpu_memory_used_mb / gpu_memory_total_mb)
            if gpu_memory_total_mb > 0
            else 0.5
        )

        old_limit = self.current_concurrency
        reason = ""

        # Condition 1: High Memory Pressure or SLA breach -> Multiplicative Decrease
        if mem_ratio >= self.hard_memory_limit_ratio or recent_sla_breach:
            new_limit = max(
                self.min_concurrency,
                int(self.current_concurrency * self.decrease_factor),
            )
            reason = (
                f"Severe backpressure (Memory: {mem_ratio*100:.1f}%, SLA breach: {recent_sla_breach})"
            )
            self.current_concurrency = new_limit

        # Condition 2: Moderate Memory Pressure -> Cautious Decrement
        elif mem_ratio >= self.soft_memory_limit_ratio:
            new_limit = max(self.min_concurrency, self.current_concurrency - 1)
            reason = f"Memory soft limit reached ({mem_ratio*100:.1f}%)"
            self.current_concurrency = new_limit

        # Condition 3: GPU Headroom available + Work waiting in queue -> Additive Increase
        elif (
            queue_depth > 0
            and active_requests >= self.current_concurrency
            and mem_ratio < self.soft_memory_limit_ratio
            and gpu_util_percent < 95.0
        ):
            new_limit = min(self.max_concurrency, self.current_concurrency + self.increase_step)
            reason = (
                f"Queue pending ({queue_depth}) with GPU headroom (Util: {gpu_util_percent}%, Mem: {mem_ratio*100:.1f}%)"
            )
            self.current_concurrency = new_limit

        # Condition 4: Completely idle -> Slowly reset towards initial concurrency
        elif queue_depth == 0 and active_requests == 0 and self.current_concurrency > self.config.initial_concurrency:
            if now - self._last_adjustment_time > 3.0:
                self.current_concurrency = max(self.config.initial_concurrency, self.current_concurrency - 1)
                reason = "Idle cooldown back towards baseline concurrency"

        if self.current_concurrency != old_limit:
            self._last_adjustment_time = now
            logger.info(
                "Adaptive Batch Controller: concurrency %d -> %d (%s)",
                old_limit,
                self.current_concurrency,
                reason,
            )
            self.adjustment_history.append({
                "timestamp": now,
                "old_limit": old_limit,
                "new_limit": self.current_concurrency,
                "reason": reason,
            })
            if len(self.adjustment_history) > 100:
                self.adjustment_history.pop(0)

        return self.current_concurrency

    def get_stats(self) -> Dict:
        """Return controller diagnostic info."""
        return {
            "current_concurrency_limit": self.current_concurrency,
            "min_concurrency": self.min_concurrency,
            "max_concurrency": self.max_concurrency,
            "target_gpu_util_percent": self.target_gpu_util_percent,
            "total_adjustments": len(self.adjustment_history),
        }
