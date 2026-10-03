"""
VelocityLLM - Smart Admission Controller
Pipeline position: Request Profiler -> Future Resource Estimator -> THIS -> Queue.

Checks, in order (first failing check wins, deterministic reasons):
  1. queue_overload : hard queue cap + tiered burst shedding
  2. memory_risk    : GPU hard/soft memory limit + PREDICTED future KV pressure   [F2]
  3. policy_limit   : best-effort traffic while BE lane is throttled              [F8]
  4. sla_risk       : token-aware predicted completion > SLA (+margin)            [F1, F7]

Prediction of completion time is a small event simulation over the actual
running + queued requests (token-aware), using the online-calibrated TTFT/ITL
models -- not a single hard-coded constant.
"""

import heapq
import logging
from collections import Counter
from typing import Any, Dict, List, Tuple

from scheduler_engine.admission import AdmissionController
from scheduler_engine.decision_trace import EVT_ADMIT, EVT_REJECT, DecisionTrace
from scheduler_engine.estimators import KVSnapshot, OnlineCalibrator, OutputLengthPredictor, RunningInfo
from scheduler_engine.profiler import RequestProfile, TrafficClass
from scheduler_engine.types import AdmissionResult, AdmissionStatus, RequestPriority

logger = logging.getLogger("velocityllm.smart_admission")

REASON_SLA = "sla_risk"
REASON_MEMORY = "memory_risk"
REASON_QUEUE = "queue_overload"
REASON_POLICY = "policy_limit"


def _lane(tc: TrafficClass) -> int:
    return 0 if tc == TrafficClass.REAL_TIME else 1


class SmartAdmissionController(AdmissionController):
    def __init__(
        self,
        server_cfg: Any,
        smart_cfg: Any,
        calibrator: OnlineCalibrator,
        predictor: OutputLengthPredictor,
        trace: DecisionTrace,
    ):
        super().__init__(server_cfg)
        self.smart = smart_cfg
        self.calibrator = calibrator
        self.predictor = predictor
        self.trace = trace
        self.rejected_by_reason: Counter = Counter()

    # -------------------------------------------------------------- estimation
    def _ahead(self, profile: RequestProfile, queued: List[RequestProfile]) -> List[RequestProfile]:
        mine = _lane(profile.traffic_class)
        out = []
        for q in queued:
            ql = _lane(q.traffic_class)
            if ql < mine or (ql == mine and q.priority.value <= profile.priority.value):
                out.append(q)
        return out

    def _residual_tokens(self, r: RunningInfo) -> int:
        pred = self.predictor.predict(r.prompt_tokens, r.max_tokens)
        if r.generated < pred.expected:
            total_out = pred.expected
        else:
            total_out = min(r.max_tokens, max(pred.p90, int(r.generated * 1.25)))
        return max(0, total_out - r.generated)

    def estimate(
        self,
        profile: RequestProfile,
        queued: List[RequestProfile],
        running: List[RunningInfo],
        limit: int,
    ) -> Tuple[float, float, int]:
        """Returns (predicted_wait_s, predicted_exec_s, expected_concurrency)."""
        limit = max(1, limit)
        ahead = self._ahead(profile, queued)
        c_inst = max(1, min(limit, len(running) + len(ahead) + 1))
        # execution happens under the *regime* load, not just the instantaneous one
        c_run = max(1, min(limit, max(c_inst, int(round(self.calibrator.ema_conc)))))
        itl = self.calibrator.itl(c_run)

        free = max(0, limit - len(running))
        slots = [0.0] * free + [self._residual_tokens(r) * itl for r in running]
        if not slots:
            slots = [0.0]
        heapq.heapify(slots)
        for q in ahead:
            t = heapq.heappop(slots)
            svc = self.calibrator.ttft(q.prompt_tokens) + max(0, q.expected_output_tokens - 1) * itl
            heapq.heappush(slots, t + svc)
        wait = heapq.heappop(slots)
        exec_s = self.calibrator.exec_time(profile.prompt_tokens, profile.expected_output_tokens, c_run)
        return wait, exec_s, c_run

    # ---------------------------------------------------------------- decision
    def evaluate_profile(
        self,
        profile: RequestProfile,
        *,
        queued: List[RequestProfile],
        running: List[RunningInfo],
        limit: int,
        kv: KVSnapshot,
        queue_size: int,
        mem_used_mb: int,
        mem_total_mb: int,
        be_throttled: bool,
    ) -> AdmissionResult:
        self.total_evaluated += 1
        raw_wait, exec_s, c_run = self.estimate(profile, queued, running, limit)
        wait = raw_wait * self.calibrator.wait_correction     # self-calibrated queue-wait bias (Feature 9)
        profile.predicted_wait_raw_s = raw_wait
        predicted = wait + exec_s
        profile.predicted_wait_s, profile.predicted_exec_s, profile.predicted_latency_s = wait, exec_s, predicted

        sla_s = profile.sla_s
        is_high = profile.priority == RequestPriority.HIGH
        is_be = profile.traffic_class == TrafficClass.BEST_EFFORT
        mem_ratio = (mem_used_mb / mem_total_mb) if mem_total_mb > 0 else 0.0

        kv_after = kv.predicted_future_tokens + kv.queued_tokens + profile.total_estimated_tokens
        kv_after_pressure = kv_after / max(1, kv.safe_tokens)
        details: Dict[str, Any] = {
            "predicted_wait_ms": round(wait * 1000, 1),
            "wait_correction": round(self.calibrator.wait_correction, 3),
            "predicted_exec_ms": round(exec_s * 1000, 1),
            "expected_concurrency": c_run,
            "queue_size": queue_size,
            "operating_limit": limit,
            "prompt_tokens": profile.prompt_tokens,
            "expected_output_tokens": profile.expected_output_tokens,
            "p90_output_tokens": profile.p90_output_tokens,
            "kv_current_tokens": kv.current_tokens,
            "kv_predicted_future_tokens": kv.predicted_future_tokens,
            "kv_queued_tokens": kv.queued_tokens,
            "kv_safe_tokens": kv.safe_tokens,
            "kv_pressure_after_admit": round(kv_after_pressure, 4),
            "gpu_memory_ratio": round(mem_ratio, 4),
        }
        retry = max(0.5, round(wait, 2))

        def reject(code: str, status: AdmissionStatus, message: str, retry_after: float = retry) -> AdmissionResult:
            if status == AdmissionStatus.REJECTED_SLA_IMPOSSIBLE:
                self.total_rejected_sla += 1
            else:
                self.total_rejected_overload += 1
            self.rejected_by_reason[code] += 1
            self.trace.emit(
                EVT_REJECT,
                request_id=profile.request_id,
                decision="reject",
                reason=message,
                reason_code=code,
                token_estimate=profile.total_estimated_tokens,
                kv_estimate_tokens=profile.total_estimated_tokens,
                sla_ms=profile.sla_ms,
                predicted_latency_ms=predicted * 1000,
                slack_ms=(sla_s - predicted) * 1000,
                bucket=profile.bucket.value,
                traffic_class=profile.traffic_class.value,
                priority=profile.priority.name,
                **details,
            )
            return AdmissionResult(
                status=status, admitted=False, reason=message,
                estimated_delay_seconds=predicted, retry_after_seconds=retry_after,
                reason_code=code,
                details={"predicted_latency_ms": round(predicted * 1000, 1), "sla_ms": profile.sla_ms, **details},
            )

        # 1) queue overload: hard cap, then tiered burst shedding
        if queue_size >= self.max_queue_size:
            return reject(REASON_QUEUE, AdmissionStatus.REJECTED_OVERLOAD,
                          f"Queue full ({queue_size}/{self.max_queue_size}).")
        burst_ratio = getattr(self.config, "burst_shed_queue_ratio", 0.70)
        if queue_size >= int(self.max_queue_size * 0.90) and not is_high:
            self.total_burst_shed += 1
            return reject(REASON_QUEUE, AdmissionStatus.REJECTED_OVERLOAD,
                          f"Severe burst ({queue_size}/{self.max_queue_size} queued): shedding non-HIGH traffic.")
        if queue_size >= int(self.max_queue_size * burst_ratio) and (profile.priority == RequestPriority.LOW or is_be):
            self.total_burst_shed += 1
            return reject(REASON_QUEUE, AdmissionStatus.REJECTED_OVERLOAD,
                          f"Burst ({queue_size}/{self.max_queue_size} queued): shedding LOW/best-effort traffic.")

        # 2) memory risk: GPU limits + PREDICTED future KV pressure
        hard = getattr(self.config, "hard_memory_limit_ratio", 0.94)
        soft = getattr(self.config, "soft_memory_limit_ratio", 0.88)
        if mem_total_mb > 0 and mem_ratio >= hard:
            return reject(REASON_MEMORY, AdmissionStatus.REJECTED_OVERLOAD,
                          f"GPU memory hard limit ({mem_ratio*100:.1f}% >= {hard*100:.0f}%).", retry_after=1.5)
        if mem_total_mb > 0 and mem_ratio >= soft and not is_high:
            return reject(REASON_MEMORY, AdmissionStatus.REJECTED_OVERLOAD,
                          f"GPU memory soft limit ({mem_ratio*100:.1f}% >= {soft*100:.0f}%).", retry_after=1.0)

        if (kv.predicted_future_tokens + kv.queued_tokens) > 0:      # something already committed
            limit_p = self.smart.memory_risk_pressure
            full_p = limit_p / max(self.smart.kv_safe_fraction, 1e-6)  # up to 100% of raw capacity for HIGH
            if kv_after_pressure > full_p:
                return reject(REASON_MEMORY, AdmissionStatus.REJECTED_OVERLOAD,
                              f"Predicted future KV demand {kv_after_pressure*100:.0f}% of safe budget exceeds capacity "
                              f"(current KV only {kv.current_pressure*100:.0f}%).")
            if kv_after_pressure > limit_p and not is_high:
                return reject(REASON_MEMORY, AdmissionStatus.REJECTED_OVERLOAD,
                              f"Predicted future KV demand {kv_after_pressure*100:.0f}% of safe budget "
                              f"(current KV only {kv.current_pressure*100:.0f}%); only HIGH priority admitted.")
            if kv_after_pressure > 0.85 * limit_p and is_be:
                return reject(REASON_MEMORY, AdmissionStatus.REJECTED_OVERLOAD,
                              f"Predicted future KV demand {kv_after_pressure*100:.0f}% approaching safe limit; "
                              f"best-effort deferred.")

        # 3) policy limit: BE lane throttled to protect real-time traffic
        if is_be and be_throttled:
            return reject(REASON_POLICY, AdmissionStatus.REJECTED_OVERLOAD,
                          "Best-effort lane throttled to protect real-time SLAs.", retry_after=max(1.0, retry))

        # 4) early rejection: predicted completion vs SLA
        margin = self.smart.early_reject_margin_high if is_high else self.smart.early_reject_margin
        if predicted > sla_s * (1.0 + margin):
            return reject(REASON_SLA, AdmissionStatus.REJECTED_SLA_IMPOSSIBLE,
                          f"Predicted latency {predicted:.2f}s exceeds SLA {sla_s:.2f}s "
                          f"(wait {wait:.2f}s + exec {exec_s:.2f}s).",
                          retry_after=max(0.2, round(wait, 2)))

        # accepted
        self.total_accepted += 1
        status = AdmissionStatus.ACCEPTED if queue_size == 0 and len(running) < limit else AdmissionStatus.QUEUED
        self.trace.emit(
            EVT_ADMIT,
            request_id=profile.request_id,
            decision="admit",
            reason=f"Predicted {predicted:.2f}s within SLA {sla_s:.2f}s.",
            reason_code="admitted",
            token_estimate=profile.total_estimated_tokens,
            kv_estimate_tokens=profile.total_estimated_tokens,
            sla_ms=profile.sla_ms,
            predicted_latency_ms=predicted * 1000,
            slack_ms=(sla_s - predicted) * 1000,
            bucket=profile.bucket.value,
            traffic_class=profile.traffic_class.value,
            priority=profile.priority.name,
            **details,
        )
        return AdmissionResult(
            status=status, admitted=True, reason="Admitted successfully.",
            estimated_delay_seconds=wait, retry_after_seconds=0.0, reason_code="admitted",
            details={"predicted_latency_ms": round(predicted * 1000, 1), "sla_ms": profile.sla_ms, **details},
        )

    def get_stats(self) -> dict:
        stats = super().get_stats()
        stats["rejected_by_reason"] = dict(self.rejected_by_reason)
        return stats
