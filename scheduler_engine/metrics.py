"""
VelocityLLM - Latency and goodput tracking shared by every scheduling policy.
"""

from collections import deque
from typing import Deque, Dict, Optional


def percentile(sorted_values, q: float) -> float:
    if not sorted_values:
        return 0.0
    return sorted_values[min(int(len(sorted_values) * q), len(sorted_values) - 1)]


class LatencyTracker:
    """Rolling window of per-request TTFT, per-token latency and on-time (goodput) counts."""

    def __init__(self, window: int = 500):
        self._ttft: Deque[float] = deque(maxlen=window)
        self._per_token_ms: Deque[float] = deque(maxlen=window)
        self.goodput_requests = 0
        self.goodput_tokens = 0

    def record(
        self,
        ttft_seconds: Optional[float],
        per_token_latency_ms: Optional[float],
        tokens: int,
        on_time: bool,
    ) -> None:
        if ttft_seconds is not None:
            self._ttft.append(ttft_seconds)
        if per_token_latency_ms:
            self._per_token_ms.append(per_token_latency_ms)
        if on_time:
            self.goodput_requests += 1
            self.goodput_tokens += tokens

    def snapshot(self) -> Dict[str, float]:
        ttft = sorted(self._ttft)
        per_token = list(self._per_token_ms)
        return {
            "p50_ttft_seconds": percentile(ttft, 0.50),
            "p95_ttft_seconds": percentile(ttft, 0.95),
            "p99_ttft_seconds": percentile(ttft, 0.99),
            "avg_per_token_latency_ms": sum(per_token) / len(per_token) if per_token else 0.0,
            "goodput_requests": self.goodput_requests,
            "goodput_tokens": self.goodput_tokens,
        }


def timing_breakdown(enqueue_time: float, first_token_time: Optional[float], end_time: float, tokens: int):
    """Returns (ttft_seconds, per_token_latency_ms) measured from the moment the request was enqueued."""
    if first_token_time is None:
        return None, None
    ttft = max(0.0, first_token_time - enqueue_time)
    if tokens > 1 and end_time > first_token_time:
        return ttft, (end_time - first_token_time) / (tokens - 1) * 1000.0
    return ttft, None


def token_cost(prompt: str, max_tokens: int, image_tokens: int = 0) -> int:
    """Worst-case tokens a request can occupy in a batch: prompt estimate, generation budget and image tokens."""
    return max(1, len(prompt) // 4) + max_tokens + image_tokens
