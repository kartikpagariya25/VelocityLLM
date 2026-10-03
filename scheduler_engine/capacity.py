"""
VelocityLLM - Capacity Planner (Feature 3: Separate Memory Limit and SLA Limit)
Computes, independently:
  memory_limit_capacity : how many concurrent requests fit in the SAFE KV budget
  sla_limit_capacity    : how many concurrent requests keep predicted latency inside the SLA
and selects the tighter one. The binding constraint is reported for the UI/trace.
"""

from dataclasses import dataclass, field
from typing import Any, Dict

from scheduler_engine.estimators import KVSnapshot, OnlineCalibrator


@dataclass
class CapacityDecision:
    memory_limit_capacity: int = 0
    sla_limit_capacity: int = 0
    selected_capacity: int = 0      # min(memory, sla)  -- spec definition
    aimd_limit: int = 0             # reactive AIMD brake
    operating_limit: int = 0        # what the dispatcher actually uses = min(selected, aimd)
    binding_constraint: str = "none"  # "memory" | "sla" | "aimd"
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_limit_capacity": self.memory_limit_capacity,
            "sla_limit_capacity": self.sla_limit_capacity,
            "selected_capacity": self.selected_capacity,
            "aimd_limit": self.aimd_limit,
            "operating_limit": self.operating_limit,
            "binding_constraint": self.binding_constraint,
            "summary": (
                f"Memory-safe: {self.memory_limit_capacity}, SLA-safe: {self.sla_limit_capacity}, "
                f"selected: {self.selected_capacity}, AIMD: {self.aimd_limit}, operating: {self.operating_limit}"
            ),
            "details": self.details,
        }


class CapacityPlanner:
    def __init__(self, server_cfg: Any, smart_cfg: Any, calibrator: OnlineCalibrator):
        self.server_cfg = server_cfg
        self.smart = smart_cfg
        self.calibrator = calibrator

    def compute(
        self,
        *,
        kv: KVSnapshot,
        avg_prompt_tokens: float,
        avg_output_tokens: float,
        mem_used_mb: int,
        mem_total_mb: int,
        aimd_limit: int,
        sla_ms: float,
    ) -> CapacityDecision:
        min_c = self.server_cfg.min_concurrency
        max_c = self.server_cfg.max_concurrency

        # ---- memory-safe capacity (token/KV based) --------------------------------
        footprint = max(1.0, avg_prompt_tokens + avg_output_tokens)
        by_kv = int(kv.safe_tokens // footprint)
        memory_cap = max(min_c, min(max_c, by_kv))
        mem_ratio = (mem_used_mb / mem_total_mb) if mem_total_mb > 0 else 0.0
        hard = getattr(self.server_cfg, "hard_memory_limit_ratio", 0.94)
        emergency = mem_ratio >= hard
        if emergency:
            memory_cap = min_c

        # ---- SLA-safe capacity (calibrated latency model) --------------------------
        budget_s = (sla_ms / 1000.0) * self.smart.sla_safety
        sla_cap = min_c
        for c in range(min_c, max_c + 1):
            est = self.calibrator.exec_time(int(avg_prompt_tokens), int(max(1, avg_output_tokens)), c)
            if est <= budget_s:
                sla_cap = c
            else:
                break
        sla_cap = max(min_c, min(max_c, sla_cap))

        selected = max(min_c, min(memory_cap, sla_cap))
        operating = max(min_c, min(selected, aimd_limit))

        if operating < selected:
            binding = "aimd"
        elif memory_cap <= sla_cap:
            binding = "memory"
        else:
            binding = "sla"

        return CapacityDecision(
            memory_limit_capacity=memory_cap,
            sla_limit_capacity=sla_cap,
            selected_capacity=selected,
            aimd_limit=aimd_limit,
            operating_limit=operating,
            binding_constraint=binding,
            details={
                "avg_footprint_tokens": round(footprint, 1),
                "kv_safe_tokens": kv.safe_tokens,
                "gpu_memory_ratio": round(mem_ratio, 4),
                "memory_emergency": emergency,
                "sla_budget_ms": round(budget_s * 1000, 1),
                "predicted_exec_ms_at_selected": round(
                    self.calibrator.exec_time(int(avg_prompt_tokens), int(max(1, avg_output_tokens)), selected) * 1000, 1
                ),
            },
        )
