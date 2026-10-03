"""
VelocityLLM - Smart Scheduler Configuration
All tunables for the 10 research-derived scheduling features live here, so the
existing ServerConfig stays small. Values can be overridden from YAML (`smart:`
section), or live at runtime via PATCH /smart/config.
"""

import logging
from dataclasses import dataclass, fields
from typing import Any, Dict, Optional

logger = logging.getLogger("velocityllm.smart_config")


@dataclass
class SmartConfig:
    # ---- Feature 1/2: token + KV model -------------------------------------
    # Llama-3.2-1B: 16 layers * 8 KV heads * 64 dim * 2 (K,V) * 2 bytes = 32 KiB/token.
    # Change this for other models (see docs: kv_bytes_per_token = 2*layers*kv_heads*head_dim*dtype_bytes).
    kv_bytes_per_token: int = 32768
    model_weights_mb: int = 2600          # approx VRAM taken by weights + activations
    kv_safe_fraction: float = 0.90        # fraction of KV capacity we treat as "safe"
    kv_capacity_tokens_override: int = 0  # >0 forces KV capacity (tokens); handy for tests/demos

    # ---- Feature 6: token-length buckets -----------------------------------
    bucket_short_max: int = 256           # SHORT  <= 256 estimated tokens
    bucket_medium_max: int = 1024         # MEDIUM <= 1024, LONG > 1024
    max_long_fraction: float = 0.50       # LONG requests may use at most this share of slots

    # ---- Feature 4/5/6: queue scoring (lower score = dispatched first) -----
    deadline_priority: bool = True        # F4: slack-based urgency
    short_prompt_first: bool = True       # F5: short-prompt-first policy switch
    bucket_compat: bool = True            # F6: prefer bucket-compatible batch composition
    w_priority: float = 1.0               # per priority level (HIGH=0, NORMAL=1, LOW=2)
    w_urgency: float = 1.5                # max pull from deadline urgency (urgency in [0,2])
    w_short: float = 0.6                  # short-prompt bonus; kept < w_priority so HIGH stays protected
    w_doomed: float = 1.2                 # penalty for requests that can no longer meet their deadline (EDF overload guard)
    w_bucket: float = 0.25                # penalty for bucket mismatch with active batch
    short_ref_tokens: int = 1024          # prompt length at which short-first penalty saturates

    # ---- Feature 7: early rejection ----------------------------------------
    early_reject_margin: float = 0.05     # reject if predicted > SLA * (1 + margin)
    early_reject_margin_high: float = 0.35  # more grace for HIGH priority / real-time
    memory_risk_pressure: float = 1.00    # future KV pressure (vs safe cap) above which non-HIGH is rejected

    # ---- Feature 3: capacity -----------------------------------------------
    sla_safety: float = 0.85              # SLA-safe capacity targets 85% of the SLA
    aimd_ramp_step: int = 2               # AIMD additive step (faster ramp toward selected capacity)

    # ---- Feature 8: real-time vs best-effort -------------------------------
    rt_reserve_fraction: float = 0.25     # slots kept free of BE work for RT bursts
    be_throttle_pressure: float = 0.80    # pressure above which BE is throttled
    be_resume_pressure: float = 0.60      # pressure below which BE resumes (hysteresis)
    be_max_wait_seconds: float = 20.0     # starvation safeguard for throttled BE work

    # ---- Feature 9: calibration --------------------------------------------
    calibration_warmup_samples: int = 20
    calibration_converged_mape: float = 0.30
    calibration_state_path: Optional[str] = None   # JSON file; warm-start across restarts
    warmup_probe: bool = True             # measure TTFT/ITL at concurrency 1/4/8 at startup (also warms the engine)
    warmup_probe_tokens: int = 10         # tokens generated per probe stream

    # ---- Feature 10: decision trace ----------------------------------------
    trace_buffer_size: int = 2000

    @classmethod
    def from_server_config(cls, server_cfg: Any) -> "SmartConfig":
        cfg = cls()
        overrides = getattr(server_cfg, "smart", None) or {}
        cfg.update(overrides)
        return cfg

    def update(self, overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Apply overrides. Unknown keys are ignored with a warning. Returns the applied changes."""
        valid = {f.name: f for f in fields(self)}
        applied: Dict[str, Any] = {}
        for key, value in (overrides or {}).items():
            if key not in valid:
                logger.warning("Ignoring unknown smart config key: %s", key)
                continue
            current = getattr(self, key)
            try:
                if isinstance(current, bool):
                    value = value if isinstance(value, bool) else str(value).lower() in ("1", "true", "yes", "on")
                elif isinstance(current, int) and not isinstance(current, bool):
                    value = int(value)
                elif isinstance(current, float):
                    value = float(value)
            except (TypeError, ValueError):
                logger.warning("Ignoring invalid value for %s: %r", key, value)
                continue
            setattr(self, key, value)
            applied[key] = value
        return applied

    def to_dict(self) -> Dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}
