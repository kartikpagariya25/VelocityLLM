"""
VelocityLLM - Request Profiler
First stage of the pipeline: turns an InferenceRequest into a RequestProfile
(token demand, bucket, traffic class, deadline) before any admission decision.
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

from scheduler_engine.estimators import OutputLengthPredictor
from scheduler_engine.types import InferenceRequest, RequestPriority
from scheduler_engine.validation import estimate_prompt_tokens


class TrafficClass(str, Enum):
    REAL_TIME = "real_time"
    BEST_EFFORT = "best_effort"

    @classmethod
    def resolve(cls, value: Optional[str], priority: RequestPriority) -> "TrafficClass":
        """Explicit class wins. Otherwise LOW priority is treated as best-effort."""
        if value:
            v = value.strip().lower().replace("-", "_")
            if v in ("real_time", "realtime", "rt"):
                return cls.REAL_TIME
            if v in ("best_effort", "besteffort", "be", "batch"):
                return cls.BEST_EFFORT
        return cls.BEST_EFFORT if priority == RequestPriority.LOW else cls.REAL_TIME


class Bucket(str, Enum):
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


@dataclass
class RequestProfile:
    request_id: str
    priority: RequestPriority
    traffic_class: TrafficClass
    prompt_tokens: int
    max_tokens: int
    expected_output_tokens: int
    p90_output_tokens: int
    total_estimated_tokens: int      # prompt + expected output  (KV footprint estimate)
    bucket: Bucket
    sla_ms: float
    arrival_time: float
    deadline: float                  # absolute epoch seconds
    # filled in by admission (used by the queue for slack math)
    predicted_exec_s: float = 0.0
    predicted_wait_s: float = 0.0       # corrected (what admission used)
    predicted_wait_raw_s: float = 0.0   # uncorrected simulation output (used to learn the bias)
    predicted_latency_s: float = 0.0

    @property
    def sla_s(self) -> float:
        return self.sla_ms / 1000.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "priority": self.priority.name,
            "traffic_class": self.traffic_class.value,
            "prompt_tokens": self.prompt_tokens,
            "max_tokens": self.max_tokens,
            "expected_output_tokens": self.expected_output_tokens,
            "p90_output_tokens": self.p90_output_tokens,
            "total_estimated_tokens": self.total_estimated_tokens,
            "bucket": self.bucket.value,
            "sla_ms": self.sla_ms,
            "predicted_exec_ms": round(self.predicted_exec_s * 1000, 1),
            "predicted_wait_ms": round(self.predicted_wait_s * 1000, 1),
            "predicted_latency_ms": round(self.predicted_latency_s * 1000, 1),
        }


class RequestProfiler:
    def __init__(self, server_cfg: Any, smart_cfg: Any, predictor: OutputLengthPredictor):
        self.server_cfg = server_cfg
        self.smart = smart_cfg
        self.predictor = predictor

    def bucket_for(self, total_tokens: int) -> Bucket:
        if total_tokens <= self.smart.bucket_short_max:
            return Bucket.SHORT
        if total_tokens <= self.smart.bucket_medium_max:
            return Bucket.MEDIUM
        return Bucket.LONG

    def profile(self, request: InferenceRequest) -> RequestProfile:
        prompt_tokens = estimate_prompt_tokens(request.prompt)
        pred = self.predictor.predict(prompt_tokens, request.max_tokens)
        total = prompt_tokens + pred.expected
        sla_ms = float(request.sla_target_ms or self.server_cfg.target_sla_ms)
        arrival = request.arrival_time or time.time()
        return RequestProfile(
            request_id=request.request_id or "",
            priority=request.priority,
            traffic_class=TrafficClass.resolve(getattr(request, "traffic_class", None), request.priority),
            prompt_tokens=prompt_tokens,
            max_tokens=request.max_tokens,
            expected_output_tokens=pred.expected,
            p90_output_tokens=pred.p90,
            total_estimated_tokens=total,
            bucket=self.bucket_for(total),
            sla_ms=sla_ms,
            arrival_time=arrival,
            deadline=arrival + sla_ms / 1000.0,
        )
