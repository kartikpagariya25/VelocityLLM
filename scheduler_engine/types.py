"""
VelocityLLM - Data Types and Schemas
Defines core data models, request/response contracts, and telemetry structures
for the dynamic batching scheduler engine.
"""

from dataclasses import dataclass, field
from enum import Enum, IntEnum
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator

from scheduler_engine.validation import (
    InputValidationError,
    sanitize_prompt,
    validate_prompt,
    validate_sampling_params,
)


class RequestPriority(IntEnum):
    """
    Request priority classes.
    Lower numerical value indicates higher priority (1 is highest, 3 is lowest).
    """
    HIGH = 1
    NORMAL = 2
    LOW = 3

    @classmethod
    def from_str(cls, val: str) -> "RequestPriority":
        val_lower = val.strip().lower()
        if val_lower in ("high", "1"):
            return cls.HIGH
        elif val_lower in ("low", "3"):
            return cls.LOW
        return cls.NORMAL


class AdmissionStatus(str, Enum):
    """Admission decision status for incoming requests."""
    ACCEPTED = "accepted"
    QUEUED = "queued"
    REJECTED_OVERLOAD = "rejected_overload"
    REJECTED_SLA_IMPOSSIBLE = "rejected_sla_impossible"


@dataclass
class AdmissionResult:
    """Result of an admission controller evaluation."""
    status: AdmissionStatus
    admitted: bool
    reason: str = ""
    estimated_delay_seconds: float = 0.0
    retry_after_seconds: float = 0.0


class InferenceRequest(BaseModel):
    """Standard inference request payload with strict validation and sanitization."""
    prompt: str = Field(..., min_length=1, description="The input prompt text")
    max_tokens: int = Field(default=100, ge=1, le=4096, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    priority: RequestPriority = Field(
        default=RequestPriority.NORMAL,
        description="Request priority level: 1 (HIGH), 2 (NORMAL), 3 (LOW)"
    )
    sla_target_ms: Optional[float] = Field(
        default=None,
        ge=50.0,
        description="Optional client-specified latency SLA target in milliseconds"
    )
    stream: bool = Field(default=False, description="Stream back tokens incrementally")
    request_id: Optional[str] = Field(default=None, description="Client or system request correlation ID")
    arrival_time: float = Field(default_factory=time.time, description="Timestamp when request arrived")

    @field_validator("prompt")
    @classmethod
    def sanitize_and_validate_prompt(cls, v: str) -> str:
        clean = sanitize_prompt(v)
        if not clean:
            raise ValueError("Prompt cannot be empty or only whitespace/control characters.")
        return clean

    @model_validator(mode="after")
    def validate_request_bounds(self) -> "InferenceRequest":
        validate_prompt(self.prompt, max_tokens=self.max_tokens, max_model_len=4096)
        validate_sampling_params(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            max_model_len=4096,
        )
        return self


class InferenceResponse(BaseModel):
    """Standard inference response payload."""
    request_id: str
    prompt: str
    response: str
    latency_seconds: float
    queue_time_seconds: float = 0.0
    execution_time_seconds: float = 0.0
    tokens_generated: int = 0
    tokens_per_second: float = 0.0
    sla_met: bool = True
    priority: str = "normal"


class GenerationChunk(BaseModel):
    """Chunk for streaming token output."""
    request_id: str
    delta: str
    token_index: int
    is_finished: bool = False
    finish_reason: Optional[str] = None


# OpenAI-compatible API schemas
class CompletionRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_tokens: int = Field(default=100, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    priority: Optional[str] = "normal"
    stream: bool = False
    model: Optional[str] = "llama-3.2-1b"

    @field_validator("prompt")
    @classmethod
    def sanitize_and_validate_prompt(cls, v: str) -> str:
        clean = sanitize_prompt(v)
        if not clean:
            raise ValueError("Prompt cannot be empty or only whitespace/control characters.")
        return clean

    @model_validator(mode="after")
    def validate_completion_bounds(self) -> "CompletionRequest":
        validate_prompt(self.prompt, max_tokens=self.max_tokens, max_model_len=4096)
        validate_sampling_params(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            max_model_len=4096,
        )
        return self


class CompletionChoice(BaseModel):
    text: str
    index: int = 0
    finish_reason: str = "stop"


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: UsageInfo
    latency_seconds: float
    sla_met: bool = True


@dataclass
class SchedulerStats:
    """Internal telemetry for dynamic and static scheduler policy state."""
    policy_name: str
    active_requests: int = 0
    queued_requests: int = 0
    effective_concurrency_limit: int = 8
    total_accepted: int = 0
    total_rejected: int = 0
    total_completed: int = 0
    total_tokens_generated: int = 0
    sla_breach_count: int = 0
    avg_latency_seconds: float = 0.0
    p50_latency_seconds: float = 0.0
    p95_latency_seconds: float = 0.0
    p99_latency_seconds: float = 0.0
    gpu_utilization_percent: float = 0.0
    gpu_memory_used_mb: int = 0
    gpu_memory_total_mb: int = 0
    avg_queue_wait_seconds: float = 0.0

    # Phase 3 Robustness Telemetry
    client_disconnects_count: int = 0
    burst_shed_count: int = 0
    oom_recoveries_count: int = 0
    validation_errors_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "active_requests": self.active_requests,
            "queued_requests": self.queued_requests,
            "effective_concurrency_limit": self.effective_concurrency_limit,
            "total_accepted": self.total_accepted,
            "total_rejected": self.total_rejected,
            "total_completed": self.total_completed,
            "total_tokens_generated": self.total_tokens_generated,
            "sla_breach_count": self.sla_breach_count,
            "sla_compliance_rate_percent": (
                round((1.0 - (self.sla_breach_count / max(self.total_completed, 1))) * 100.0, 2)
                if self.total_completed > 0 else 100.0
            ),
            "avg_latency_seconds": round(self.avg_latency_seconds, 4),
            "p50_latency_seconds": round(self.p50_latency_seconds, 4),
            "p95_latency_seconds": round(self.p95_latency_seconds, 4),
            "p99_latency_seconds": round(self.p99_latency_seconds, 4),
            "gpu_utilization_percent": self.gpu_utilization_percent,
            "gpu_memory_used_mb": self.gpu_memory_used_mb,
            "gpu_memory_total_mb": self.gpu_memory_total_mb,
            "avg_queue_wait_seconds": round(self.avg_queue_wait_seconds, 4),
            "client_disconnects_count": self.client_disconnects_count,
            "burst_shed_count": self.burst_shed_count,
            "oom_recoveries_count": self.oom_recoveries_count,
            "validation_errors_count": self.validation_errors_count,
        }


@dataclass
class ServerConfig:
    """Master configuration for the scheduler service."""
    policy: str = "dynamic"  # "dynamic" or "static"
    model_path: str = "/home/kartiklin/velocityllm/models/llama-3.2-1b"
    use_mock_backend: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    gpu_memory_utilization: float = 0.80
    max_model_len: int = 4096
    
    # Dynamic Scheduler specific
    min_concurrency: int = 2
    max_concurrency: int = 32
    initial_concurrency: int = 8
    target_sla_ms: float = 3000.0  # 3.0s default target
    max_queue_size: int = 100
    aging_factor: float = 0.25     # Priority promotion rate per second of waiting
    gpu_sample_interval: float = 0.2

    # Phase 3 Robustness additions
    enable_structured_logging: bool = False
    burst_shed_queue_ratio: float = 0.70    # Start shedding LOW priority at 70% queue depth
    soft_memory_limit_ratio: float = 0.88   # Soft memory throttle threshold
    hard_memory_limit_ratio: float = 0.94   # Hard memory preemption threshold

