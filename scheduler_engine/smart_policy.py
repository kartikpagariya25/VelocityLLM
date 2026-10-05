"""
VelocityLLM - SmartBatchPolicy  (policy name: "smart")

Closed-loop, SLA-aware scheduler layered on the existing DynamicBatchPolicy:

  Request -> Profiler -> Future Resource Estimator -> Admission (early reject)
          -> Smart queue (slack / short-first / buckets / RT-BE lanes)
          -> Capacity planner (min(memory-safe, SLA-safe)) + AIMD brake
          -> Backend -> Runtime telemetry -> Self-calibration -> back to estimators

Static and Dynamic policies are untouched, so the benchmark ladder can compare
Static vs Dynamic vs Smart on identical traces.

Streaming and non-streaming requests share ONE lifecycle: both go through
admission, the queue, the concurrency limit and the same accounting.
"""

import asyncio
import logging
import math
import os
import time
import uuid
from collections import Counter, deque
from typing import Any, AsyncIterator, Deque, Dict, List, Optional, Tuple

from scheduler_engine.backend import GPUOutOfMemoryError, InferenceBackend
from scheduler_engine.capacity import CapacityDecision, CapacityPlanner
from scheduler_engine.decision_trace import (
    EVT_CALIBRATION,
    EVT_CANCEL,
    EVT_CAPACITY,
    EVT_COMPLETE,
    EVT_DISPATCH,
    EVT_THROTTLE,
    DecisionTrace,
)
from scheduler_engine.estimators import (
    KVEstimator,
    KVSnapshot,
    OnlineCalibrator,
    OutputLengthPredictor,
    RunningInfo,
    load_calibration,
    percentile,
    save_calibration,
)
from scheduler_engine.policy import AdmissionRejectedException, DynamicBatchPolicy
from scheduler_engine.profiler import Bucket, RequestProfile, RequestProfiler, TrafficClass
from scheduler_engine.smart_admission import SmartAdmissionController
from scheduler_engine.smart_config import SmartConfig
from scheduler_engine.smart_queue import SelectionContext, SmartEntry, SmartRequestQueue
from scheduler_engine.types import (
    GenerationChunk,
    InferenceRequest,
    InferenceResponse,
    RequestPriority,
    SchedulerStats,
    ServerConfig,
)

logger = logging.getLogger("velocityllm.smart_policy")


def _swallow(fut: "asyncio.Future") -> None:
    if not fut.cancelled():
        fut.exception()  # mark retrieved; stream consumers get errors via their sink


class SmartBatchPolicy(DynamicBatchPolicy):
    def __init__(self, backend: InferenceBackend, config: ServerConfig):
        super().__init__(backend, config)
        self.smart = SmartConfig.from_server_config(config)
        self.trace = DecisionTrace(maxlen=self.smart.trace_buffer_size)
        self.predictor = OutputLengthPredictor()
        self.calibrator = OnlineCalibrator(
            "uninitialized",
            warmup_samples=self.smart.calibration_warmup_samples,
            converged_mape=self.smart.calibration_converged_mape,
        )
        self.kv_estimator = KVEstimator(config, self.smart, self.predictor)
        self.profiler = RequestProfiler(config, self.smart, self.predictor)
        self.capacity_planner = CapacityPlanner(config, self.smart, self.calibrator)

        # swap in the smart components (base class created the plain ones)
        self.queue = SmartRequestQueue(aging_factor=config.aging_factor, smart_cfg=self.smart, trace=self.trace)
        self.admission_controller = SmartAdmissionController(
            config, self.smart, self.calibrator, self.predictor, self.trace
        )
        self.adaptive_controller.increase_step = max(1, int(self.smart.aimd_ramp_step))

        self._running: Dict[str, RunningInfo] = {}
        self._stream_sinks: Dict[str, asyncio.Queue] = {}

        # control state
        self.capacity = CapacityDecision(
            operating_limit=config.initial_concurrency,
            aimd_limit=config.initial_concurrency,
            selected_capacity=config.initial_concurrency,
        )
        self.kv_snapshot: Optional[KVSnapshot] = None
        self.be_throttled = False
        self.pressure = 0.0
        self.pressure_signals: Dict[str, float] = {}
        self._last_binding = "none"
        self._last_operating = config.initial_concurrency
        self._last_calibration_status = "warming_up"
        self._avg_prompt_ema = 64.0
        self._mem_total_mb = 8192
        # Idle GPU-memory baseline (lowest ratio seen). vLLM pre-allocates most of the VRAM at startup,
        # so the ABSOLUTE ratio is high even when idle; only growth above the baseline is real pressure.
        self._mem_baseline_ratio: Optional[float] = None

        # evaluation / observability metrics
        self._completions: Deque[Tuple[float, int, bool, str]] = deque(maxlen=5000)
        self._ttft: Deque[float] = deque(maxlen=1000)        # enqueue -> first token (user-perceived)
        self._ttft_exec: Deque[float] = deque(maxlen=1000)   # dispatch -> first token
        self._itl: Deque[float] = deque(maxlen=1000)
        # AIMD brake input: was EXECUTION slow? (queue-induced lateness is handled by admission, and
        # shrinking concurrency because of queue wait would only make the backlog worse)
        self._recent_exec_slow: Deque[bool] = deque(maxlen=8)
        self._rt_latency_ratio: Deque[float] = deque(maxlen=20)
        self.class_stats: Dict[str, Counter] = {
            TrafficClass.REAL_TIME.value: Counter(),
            TrafficClass.BEST_EFFORT.value: Counter(),
        }
        self.failed_count = 0
        self._first_accept_time: Optional[float] = None

    # ----------------------------------------------------------------- lifecycle
    async def initialize(self) -> None:
        await super().initialize()
        _, _, total = self.backend.get_gpu_telemetry()
        self._mem_total_mb = total or 8192
        self.calibrator.key = "|".join([
            type(self.backend).__name__,
            os.path.basename(str(self.config.model_path).rstrip("/")) or "model",
            f"gpu{self._mem_total_mb}mb",
        ])
        loaded = False
        if self.smart.calibration_state_path:
            loaded = load_calibration(self.smart.calibration_state_path, self.calibrator, self.predictor)
        if self.smart.warmup_probe and not loaded:
            await self._warmup_probe()
        logger.info("SmartBatchPolicy ready (calibration key: %s).", self.calibrator.key)

    async def _warmup_probe(self) -> None:
        """
        Short startup probe: generate a few tokens at concurrency 1, 4 and 8 and one long-prompt run.
        Seeds the TTFT/ITL models with real measurements of THIS model/GPU (self-calibration, F9)
        and warms the engine, so the first real admission decisions are not made from priors.
        These probe runs are not counted as served requests.
        """
        n_tok = max(4, int(self.smart.warmup_probe_tokens))

        async def stream_once(prompt: str, tag: str):
            first = last = None
            n = 0
            start = time.time()
            try:
                async for _ in self.backend.generate_stream(prompt, n_tok, 0.0, f"__warmup_{tag}"):
                    now = time.time()
                    first = first if first is not None else now
                    last = now
                    n += 1
            except Exception as exc:
                logger.warning("Warm-up probe stream failed (%s); keeping priors.", exc)
                return None
            if first is None:
                return None
            itl = ((last - first) / (n - 1)) if n > 1 else None
            return first - start, itl

        try:
            for level in (1, 4, 8):
                level = min(level, max(1, self.config.max_concurrency))
                results = await asyncio.gather(*[stream_once("warm up the engine", f"c{level}_{i}") for i in range(level)])
                results = [r for r in results if r]
                itls = [r[1] for r in results if r[1]]
                if itls:
                    avg = sum(itls) / len(itls)
                    for _ in range(6):                      # outweigh the optimistic prior pseudo-points
                        self.calibrator.itl_model.update(level, avg)
                if level == 1 and results:
                    for _ in range(4):
                        self.calibrator.ttft_model.update(4, results[0][0])
            long_prompt = "warmup " * 600                      # ~1000 prompt tokens -> learns TTFT slope
            res = await stream_once(long_prompt, "long")
            if res:
                from scheduler_engine.validation import estimate_prompt_tokens
                for _ in range(4):
                    self.calibrator.ttft_model.update(estimate_prompt_tokens(long_prompt), res[0])
            logger.info(
                "Warm-up probe done: ITL@1=%.1fms ITL@8=%.1fms.",
                self.calibrator.itl(1) * 1000, self.calibrator.itl(8) * 1000,
            )
        except Exception as exc:  # never block startup on calibration
            logger.warning("Warm-up probe aborted: %s", exc)

    async def shutdown(self) -> None:
        if self.smart.calibration_state_path:
            save_calibration(self.smart.calibration_state_path, self.calibrator, self.predictor)
        await super().shutdown()

    async def cancel_request(self, request_id: str, reason: str = "Client disconnected") -> bool:
        ok = await super().cancel_request(request_id, reason=reason)
        sink = self._stream_sinks.get(request_id)
        if sink is not None:
            sink.put_nowait(None)
        if ok:
            self.trace.emit(EVT_CANCEL, request_id=request_id, decision="cancel", reason=reason, reason_code="cancelled")
        return ok

    # ---------------------------------------------------------------- admission
    def _op_limit(self) -> int:
        return max(1, self.capacity.operating_limit or self.adaptive_controller.current_concurrency)

    def _admit(self, request: InferenceRequest) -> RequestProfile:
        profile = self.profiler.profile(request)
        _, mem_used, mem_total = self.backend.get_gpu_telemetry()
        running = list(self._running.values())
        kv = self.kv_estimator.snapshot(running, self.queue.queued_tokens(), mem_total)
        result = self.admission_controller.evaluate_profile(
            profile,
            queued=self.queue.profiles(),
            running=running,
            limit=self._op_limit(),
            kv=kv,
            queue_size=self.queue.size,
            mem_used_mb=mem_used,
            mem_total_mb=mem_total,
            be_throttled=self.be_throttled,
        )
        cls = profile.traffic_class.value
        if not result.admitted:
            self.total_rejected += 1
            self.class_stats[cls]["rejected"] += 1
            self.burst_shed_count = self.admission_controller.total_burst_shed
            raise AdmissionRejectedException(
                status=result.status, reason=result.reason, retry_after=result.retry_after_seconds,
                reason_code=result.reason_code, details=result.details,
            )
        self.total_accepted += 1
        self.class_stats[cls]["accepted"] += 1
        if self._first_accept_time is None:
            self._first_accept_time = time.time()
        return profile

    async def schedule(self, request: InferenceRequest) -> InferenceResponse:
        if not request.request_id:
            request.request_id = str(uuid.uuid4())
        profile = self._admit(request)
        future = await self.queue.enqueue(request, profile)
        return await future

    async def schedule_stream(self, request: InferenceRequest) -> AsyncIterator[GenerationChunk]:
        """Streaming goes through the SAME admission/queue/limit/accounting as schedule()."""
        if not request.request_id:
            request.request_id = str(uuid.uuid4())
        profile = self._admit(request)

        sink: asyncio.Queue = asyncio.Queue()
        original_id = request.request_id
        self._stream_sinks[original_id] = sink          # register before the dispatcher can see it
        future = await self.queue.enqueue(request, profile)
        rid = request.request_id
        if rid != original_id:
            self._stream_sinks[rid] = self._stream_sinks.pop(original_id)
        future.add_done_callback(_swallow)

        finished = False
        try:
            while True:
                item = await sink.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    finished = True
                    raise item
                if item.is_finished:
                    finished = True
                yield item
        finally:
            self._stream_sinks.pop(rid, None)
            if not finished:
                asyncio.ensure_future(self.cancel_request(rid, reason="Stream consumer disconnected"))

    # ----------------------------------------------------------- control helpers
    def _recent_breach(self) -> bool:
        if len(self._recent_exec_slow) < 4:
            return False
        slow = sum(1 for x in self._recent_exec_slow if x)
        return slow / len(self._recent_exec_slow) >= 0.25

    def _avg_tokens(self) -> Tuple[float, float]:
        profs = self.queue.profiles()
        running = list(self._running.values())
        n = len(profs) + len(running)
        if n == 0:
            return self._avg_prompt_ema, max(1.0, self.predictor.avg_output())
        prompt = sum(p.prompt_tokens for p in profs) + sum(r.prompt_tokens for r in running)
        out = sum(p.expected_output_tokens for p in profs) + sum(r.expected_output for r in running)
        return prompt / n, max(1.0, out / n)

    def _update_throttle(self, kv: KVSnapshot, mem_used: int, mem_total: int) -> None:
        soft = getattr(self.config, "soft_memory_limit_ratio", 0.88)
        mem_ratio = (mem_used / mem_total) if mem_total else 0.0
        if mem_total and mem_used > 0:
            self._mem_baseline_ratio = (
                mem_ratio if self._mem_baseline_ratio is None else min(self._mem_baseline_ratio, mem_ratio)
            )
        base = self._mem_baseline_ratio or 0.0
        mem_signal = max(0.0, mem_ratio - base) / max(soft - base, 0.03)   # 1.0 == at the soft limit
        lanes = self.queue.lane_counts()
        op = max(1, self.capacity.operating_limit)
        rt_ratios = sorted(self._rt_latency_ratio)
        signals = {
            "future_kv": kv.future_pressure,
            "queue_depth": self.queue.size / max(1, self.config.max_queue_size),
            "gpu_memory": mem_signal,
            "rt_latency": (percentile(rt_ratios, 0.95) if len(rt_ratios) >= 5 else 0.0),
            "rt_waiting": lanes[TrafficClass.REAL_TIME.value] / op,
        }
        self.pressure_signals = {k: round(v, 3) for k, v in signals.items()}
        self.pressure = max(signals.values())
        dominant = max(signals, key=signals.get)

        if not self.be_throttled and self.pressure >= self.smart.be_throttle_pressure:
            self.be_throttled = True
            self.trace.emit(
                EVT_THROTTLE, decision="throttle_best_effort", reason_code="rt_protection",
                reason=f"Pressure {self.pressure:.2f} >= {self.smart.be_throttle_pressure:.2f} "
                       f"(dominant: {dominant}); best-effort lane throttled.",
                pressure=round(self.pressure, 3), signals=self.pressure_signals,
            )
        elif self.be_throttled and self.pressure <= self.smart.be_resume_pressure:
            self.be_throttled = False
            self.trace.emit(
                EVT_THROTTLE, decision="resume_best_effort", reason_code="recovered",
                reason=f"Pressure {self.pressure:.2f} <= {self.smart.be_resume_pressure:.2f}; best-effort resumed.",
                pressure=round(self.pressure, 3), signals=self.pressure_signals,
            )

    def _preferred_bucket(self) -> Optional[str]:
        if not self._running:
            return None
        counts = Counter(r.bucket for r in self._running.values())
        top = counts.most_common(2)
        if len(top) > 1 and top[0][1] == top[1][1]:
            return None
        return top[0][0]

    def _make_ctx(self, operating: int, kv: KVSnapshot) -> SelectionContext:
        lanes = self.queue.lane_counts()
        rt_waiting = lanes[TrafficClass.REAL_TIME.value]
        active_n = len(self._active_requests)
        free = operating - active_n
        reserve = math.ceil(self.smart.rt_reserve_fraction * operating)
        be_active = sum(1 for r in self._running.values() if r.traffic_class == TrafficClass.BEST_EFFORT.value)
        be_cap = max(1, operating - reserve)
        allow_be = (not self.be_throttled) and (free - rt_waiting >= 1) and (be_active < be_cap)
        long_active = sum(1 for r in self._running.values() if r.bucket == Bucket.LONG.value)
        long_cap = max(1, math.ceil(self.smart.max_long_fraction * operating))

        def fits(entry: SmartEntry) -> Tuple[bool, str]:
            if not self._running:
                return True, ""          # never deadlock on a single large request
            p = entry.profile
            snap = self.kv_estimator.snapshot(self._running.values(), 0, self._mem_total_mb)
            cap = snap.capacity_tokens if p.priority == RequestPriority.HIGH else snap.safe_tokens
            if snap.predicted_future_tokens + p.total_estimated_tokens > cap:
                return False, "token_budget"
            if p.bucket == Bucket.LONG:
                cur_long = sum(1 for r in self._running.values() if r.bucket == Bucket.LONG.value)
                if cur_long >= long_cap:
                    return False, "long_bucket_cap"
            return True, ""

        return SelectionContext(
            now=time.time(),
            concurrency=max(1, active_n),
            calibrator=self.calibrator,
            preferred_bucket=self._preferred_bucket(),
            allow_best_effort=allow_be,
            hard_pressure=(self.capacity.details.get("memory_emergency", False)),
            fits=fits,
        )

    # ------------------------------------------------------------ dispatch loop
    async def _dispatch_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                util, mem_used, mem_total = self.backend.get_gpu_telemetry()
                self._mem_total_mb = mem_total or self._mem_total_mb
                running = list(self._running.values())
                kv = self.kv_estimator.snapshot(running, self.queue.queued_tokens(), mem_total)
                self.kv_snapshot = kv
                avg_prompt, avg_out = self._avg_tokens()
                sla_ms = self.config.target_sla_ms

                # capacity (F3): memory-safe vs SLA-safe, AIMD as reactive brake
                pre = self.capacity_planner.compute(
                    kv=kv, avg_prompt_tokens=avg_prompt, avg_output_tokens=avg_out,
                    mem_used_mb=mem_used, mem_total_mb=mem_total,
                    aimd_limit=self.adaptive_controller.current_concurrency, sla_ms=sla_ms,
                )
                ctrl = self.adaptive_controller
                ctrl.max_concurrency = max(self.config.min_concurrency, pre.selected_capacity)
                if ctrl.current_concurrency > ctrl.max_concurrency:
                    ctrl.current_concurrency = ctrl.max_concurrency
                n_hist = len(ctrl.adjustment_history)
                aimd_limit = ctrl.evaluate_and_tune(
                    gpu_util_percent=float(util), gpu_memory_used_mb=mem_used, gpu_memory_total_mb=mem_total,
                    queue_depth=self.queue.size, active_requests=len(self._active_requests),
                    recent_sla_breach=self._recent_breach(),
                )
                cap = self.capacity_planner.compute(
                    kv=kv, avg_prompt_tokens=avg_prompt, avg_output_tokens=avg_out,
                    mem_used_mb=mem_used, mem_total_mb=mem_total, aimd_limit=aimd_limit, sla_ms=sla_ms,
                )
                self.capacity = cap
                self._trace_capacity(cap, ctrl.adjustment_history[n_hist:])

                self._update_throttle(kv, mem_used, mem_total)

                # dispatch: pick best-fitting entries until slots are full
                for _ in range(max(0, cap.operating_limit - len(self._active_requests))):
                    if self.queue.size == 0:
                        break
                    ctx = self._make_ctx(cap.operating_limit, kv)
                    entry = await self.queue.select_next(ctx)
                    if entry is None:
                        break
                    self._start(entry)
                await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Error in SmartBatchPolicy dispatch loop: %s", exc, exc_info=True)
                await asyncio.sleep(0.1)

    def _start(self, entry: SmartEntry) -> None:
        p = entry.profile
        rid = entry.request.request_id
        conc_at_dispatch = len(self._active_requests) + 1
        self._running[rid] = RunningInfo(
            request_id=rid, prompt_tokens=p.prompt_tokens, max_tokens=p.max_tokens,
            expected_output=p.expected_output_tokens, p90_output=p.p90_output_tokens,
            bucket=p.bucket.value, traffic_class=p.traffic_class.value, start_time=time.time(),
            predicted_exec_s=self.calibrator.exec_time(p.prompt_tokens, p.expected_output_tokens, conc_at_dispatch),
        )
        task = asyncio.create_task(self._execute_request(entry))
        self._active_requests[rid] = task

    def _trace_capacity(self, cap: CapacityDecision, new_aimd_events: List[Dict[str, Any]]) -> None:
        if cap.operating_limit == self._last_operating and cap.binding_constraint == self._last_binding:
            return
        aimd_reason = new_aimd_events[-1]["reason"] if new_aimd_events else ""
        reason = (
            f"Capacity {self._last_operating} -> {cap.operating_limit}; binding={cap.binding_constraint} "
            f"(memory-safe {cap.memory_limit_capacity}, SLA-safe {cap.sla_limit_capacity}, AIMD {cap.aimd_limit})"
        )
        if aimd_reason:
            reason += f"; AIMD: {aimd_reason}"
        self.trace.emit(
            EVT_CAPACITY, decision="capacity_change", reason=reason, reason_code=cap.binding_constraint,
            old_limit=self._last_operating, new_limit=cap.operating_limit, **cap.to_dict(),
        )
        self._last_operating = cap.operating_limit
        self._last_binding = cap.binding_constraint

    # --------------------------------------------------------------- execution
    async def _execute_request(self, entry: SmartEntry) -> None:
        request, profile = entry.request, entry.profile
        rid = request.request_id
        sink = self._stream_sinks.get(rid)
        info = self._running.get(rid)
        start_exec = time.time()
        queue_time = start_exec - entry.enqueue_time
        ctx = self.queue._last_ctx
        dispatch_slack = None
        if ctx is not None:
            dispatch_slack = (profile.deadline - start_exec - profile.predicted_exec_s) * 1000

        self.trace.emit(
            EVT_DISPATCH, request_id=rid, decision="dispatch",
            reason=f"Dispatched after {queue_time*1000:.0f} ms in queue.", reason_code="dispatched",
            token_estimate=profile.total_estimated_tokens, kv_estimate_tokens=profile.total_estimated_tokens,
            sla_ms=profile.sla_ms, slack_ms=dispatch_slack, bucket=profile.bucket.value,
            traffic_class=profile.traffic_class.value, queue_ms=round(queue_time * 1000, 1),
            active_after=len(self._active_requests),
        )

        chunks: List[str] = []
        first_t: Optional[float] = None
        last_t: Optional[float] = None
        conc_sum = 0
        idx = 0
        try:
            async for chunk in self.backend.generate_stream(
                request.prompt, request.max_tokens, request.temperature, rid, image=request.image
            ):
                now = time.time()
                if first_t is None:
                    first_t = now
                last_t = now
                chunks.append(chunk)
                conc_sum += len(self._active_requests)
                if info is not None:
                    info.generated += 1
                if sink is not None:
                    sink.put_nowait(GenerationChunk(request_id=rid, delta=chunk, token_index=idx, is_finished=False))
                    idx += 1

            end = time.time()
            total_latency = end - entry.enqueue_time
            exec_time = end - start_exec
            n = len(chunks)
            sla_s = profile.sla_s
            sla_met = total_latency <= sla_s
            if not sla_met:
                self.sla_breaches += 1

            ttft_user = (first_t - entry.enqueue_time) if first_t else None
            ttft_exec = (first_t - start_exec) if first_t else None
            itl = ((last_t - first_t) / (n - 1)) if (first_t and last_t and n > 1) else None
            avg_conc = (conc_sum / n) if n else float(len(self._active_requests))

            # accounting shared by streaming and non-streaming
            self.total_completed += 1
            self.total_tokens += n
            self.latencies.append(total_latency)
            self.queue_times.append(queue_time)
            if len(self.latencies) > 500:
                self.latencies.pop(0)
            if len(self.queue_times) > 500:
                self.queue_times.pop(0)
            self._completions.append((end, n, sla_met, profile.traffic_class.value))
            self._recent_exec_slow.append(exec_time > sla_s * self.smart.sla_safety)
            if profile.traffic_class == TrafficClass.REAL_TIME:
                self._rt_latency_ratio.append(total_latency / max(sla_s, 1e-3))
            self.class_stats[profile.traffic_class.value]["completed"] += 1
            if sla_met:
                self.class_stats[profile.traffic_class.value]["within_sla"] += 1
            if ttft_user is not None:
                self._ttft.append(ttft_user)
            if ttft_exec is not None:
                self._ttft_exec.append(ttft_exec)
            if itl is not None:
                self._itl.append(itl)

            # learning: predictor + calibrator (F2, F9)
            self.admission_controller.update_completion_stats(exec_time, n)
            self.predictor.observe(profile.prompt_tokens, n)
            self._avg_prompt_ema = 0.9 * self._avg_prompt_ema + 0.1 * profile.prompt_tokens
            self.calibrator.observe_wait(queue_time, profile.predicted_wait_raw_s)
            pair = self.calibrator.observe(
                prompt_tokens=profile.prompt_tokens, output_tokens=n, ttft_s=ttft_exec, itl_s=itl,
                avg_concurrency=avg_conc,
                predicted_exec_s=((info.predicted_exec_s if info else 0.0) or profile.predicted_exec_s or exec_time),
                actual_exec_s=exec_time,
                predicted_total_s=profile.predicted_latency_s or total_latency, actual_total_s=total_latency,
                bucket=profile.bucket.value,
            )
            self._after_calibration()

            self.trace.emit(
                EVT_COMPLETE, request_id=rid, decision="complete",
                reason=("Within SLA." if sla_met else f"SLA breached by {(total_latency - sla_s)*1000:.0f} ms."),
                reason_code=("sla_met" if sla_met else "sla_breach"),
                token_estimate=profile.total_estimated_tokens, kv_estimate_tokens=profile.total_estimated_tokens,
                sla_ms=profile.sla_ms, predicted_latency_ms=profile.predicted_latency_s * 1000,
                slack_ms=(sla_s - total_latency) * 1000, bucket=profile.bucket.value,
                traffic_class=profile.traffic_class.value, actual_latency_ms=round(total_latency * 1000, 1),
                ttft_ms=(round(ttft_user * 1000, 1) if ttft_user else None),
                itl_ms=(round(itl * 1000, 2) if itl else None), tokens=n,
                prediction_error_pct=pair["error_pct"],
            )

            tps = (n / exec_time) if exec_time > 0 else 0.0
            result = InferenceResponse(
                request_id=rid, prompt=request.prompt, response="".join(chunks),
                latency_seconds=round(total_latency, 4), queue_time_seconds=round(queue_time, 4),
                execution_time_seconds=round(exec_time, 4), tokens_generated=n,
                tokens_per_second=round(tps, 2), sla_met=sla_met, priority=request.priority.name.lower(),
            )
            if sink is not None:
                sink.put_nowait(GenerationChunk(
                    request_id=rid, delta="", token_index=idx, is_finished=True, finish_reason="stop"))
            if not entry.future.done():
                entry.future.set_result(result)
        except asyncio.CancelledError:
            logger.info("Execution task for request %s was cancelled.", rid)
            if not entry.future.done():
                entry.future.cancel()
            raise
        except GPUOutOfMemoryError as oom_err:
            self.oom_recoveries_count += 1
            self.failed_count += 1
            logger.error("GPU OOM during request %s: %s", rid, oom_err)
            self.adaptive_controller.trigger_emergency_oom_throttle(cooldown_seconds=2.0)
            self.backend.handle_oom()
            self.trace.emit(EVT_CAPACITY, request_id=rid, decision="oom_recovery",
                            reason=f"GPU OOM: concurrency collapsed to {self.config.min_concurrency}.",
                            reason_code="oom")
            if sink is not None:
                sink.put_nowait(oom_err)
            if not entry.future.done():
                entry.future.set_exception(oom_err)
        except Exception as err:
            self.failed_count += 1
            if sink is not None:
                sink.put_nowait(err)
            if not entry.future.done():
                entry.future.set_exception(err)
        finally:
            self._active_requests.pop(rid, None)
            self._running.pop(rid, None)
            if sink is not None:
                sink.put_nowait(None)

    def _after_calibration(self) -> None:
        status = self.calibrator.status
        if status != self._last_calibration_status:
            self.trace.emit(
                EVT_CALIBRATION, decision="calibration_status", reason_code=status,
                reason=f"Calibration {self._last_calibration_status} -> {status} "
                       f"({self.calibrator.samples} samples, exec MAPE {self.calibrator.exec_mape*100:.0f}%).",
                **self.calibrator.snapshot(),
            )
            self._last_calibration_status = status
        path = self.smart.calibration_state_path
        if path and self.calibrator.samples % 25 == 0:
            save_calibration(path, self.calibrator, self.predictor)

    # -------------------------------------------------------------- observability
    def get_stats(self) -> SchedulerStats:
        stats = super().get_stats()
        stats.policy_name = "smart_continuous"
        return stats

    def _window(self, seconds: float) -> Tuple[int, int, int, float]:
        now = time.time()
        recent = [c for c in self._completions if c[0] >= now - seconds]
        tokens = sum(c[1] for c in recent)
        ok = sum(1 for c in recent if c[2])
        started = self._first_accept_time or now
        elapsed = max(1.0, min(seconds, now - started))
        return len(recent), ok, tokens, elapsed

    def metrics(self) -> Dict[str, Any]:
        done, ok, tokens, elapsed = self._window(60.0)
        offered = self.total_accepted + self.total_rejected
        within_total = sum(1 for c in self._completions if c[2])

        def pct(values: Deque[float]) -> Dict[str, Optional[float]]:
            o = sorted(values)
            if not o:
                return {"p50": None, "p95": None, "p99": None}
            return {q: round(percentile(o, v) * 1000, 2) for q, v in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99))}

        return {
            "offered": offered,
            "accepted": self.total_accepted,
            "rejected": self.total_rejected,
            "acceptance_rate": round(self.total_accepted / offered, 4) if offered else None,
            "completed": self.total_completed,
            "failed": self.failed_count,
            "within_sla": within_total,
            "sla_goodput_of_offered": round(within_total / offered, 4) if offered else None,
            "sla_goodput_of_completed": (round(within_total / self.total_completed, 4) if self.total_completed else None),
            "sla_goodput_rps_60s": round(ok / elapsed, 3),
            "completed_rps_60s": round(done / elapsed, 3),
            "token_throughput_tps_60s": round(tokens / elapsed, 2),
            "ttft_ms": pct(self._ttft),
            "ttft_exec_ms": pct(self._ttft_exec),
            "itl_ms": pct(self._itl),
            "rejected_by_reason": dict(self.admission_controller.rejected_by_reason),
            "gpu_utilization_percent": self.backend.get_gpu_telemetry()[0],
        }

    def get_smart_state(self) -> Dict[str, Any]:
        _, mem_used, mem_total = self.backend.get_gpu_telemetry()
        kv = self.kv_snapshot or self.kv_estimator.snapshot(self._running.values(), self.queue.queued_tokens(), mem_total)
        running_rows = [
            {
                "request_id": r.request_id, "prompt_tokens": r.prompt_tokens, "generated": r.generated,
                "expected_output": r.expected_output, "max_tokens": r.max_tokens, "bucket": r.bucket,
                "traffic_class": r.traffic_class, "running_seconds": round(time.time() - r.start_time, 3),
            }
            for r in self._running.values()
        ]
        buckets = {b.value: {"queued": 0, "active": 0} for b in Bucket}
        for k, v in self.queue.bucket_counts().items():
            buckets[k]["queued"] = v
        for r in self._running.values():
            buckets[r.bucket]["active"] += 1
        lanes = self.queue.lane_counts()
        lane_view: Dict[str, Any] = {}
        for tc in (TrafficClass.REAL_TIME.value, TrafficClass.BEST_EFFORT.value):
            cs = self.class_stats[tc]
            lane_view[tc] = {
                "queued": lanes[tc],
                "active": sum(1 for r in self._running.values() if r.traffic_class == tc),
                "accepted": cs["accepted"], "rejected": cs["rejected"],
                "completed": cs["completed"], "within_sla": cs["within_sla"],
                "sla_compliance": (round(cs["within_sla"] / cs["completed"], 4) if cs["completed"] else None),
            }
        lane_view[TrafficClass.BEST_EFFORT.value]["throttled"] = self.be_throttled
        return {
            "policy": "smart",
            "sla_ms": self.config.target_sla_ms,
            "features": {
                "token_aware": True,
                "future_kv_prediction": True,
                "memory_vs_sla_capacity": True,
                "deadline_priority": self.smart.deadline_priority,
                "short_prompt_first": self.smart.short_prompt_first,
                "token_buckets": self.smart.bucket_compat,
                "early_rejection": True,
                "rt_vs_be": True,
                "self_calibration": self.calibrator.status,
                "decision_trace": True,
            },
            "capacity": self.capacity.to_dict(),
            "tokens": {
                "active_token_budget_used": kv.current_tokens,
                "kv": kv.to_dict(),
                "running": running_rows,
                "queued_tokens": self.queue.queued_tokens(),
            },
            "buckets": {
                "boundaries": {"short_max": self.smart.bucket_short_max, "medium_max": self.smart.bucket_medium_max},
                "populations": buckets,
            },
            "lanes": lane_view,
            "pressure": {
                "value": round(self.pressure, 3), "signals": self.pressure_signals,
                "throttle_at": self.smart.be_throttle_pressure, "resume_at": self.smart.be_resume_pressure,
                "gpu_memory_baseline_ratio": (round(self._mem_baseline_ratio, 4) if self._mem_baseline_ratio else None),
                "be_throttled": self.be_throttled,
            },
            "queue": self.queue.get_queue_snapshot(),
            "calibration": self.calibrator.snapshot(),
            "predictor": {
                "samples": self.predictor.samples,
                "avg_output_tokens": round(self.predictor.avg_output(), 1),
            },
            "metrics": self.metrics(),
            "trace": self.trace.summary(),
            "config": self.smart.to_dict(),
            "gpu": {"memory_used_mb": mem_used, "memory_total_mb": mem_total},
        }
