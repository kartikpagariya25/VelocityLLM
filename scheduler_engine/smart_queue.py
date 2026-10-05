"""
VelocityLLM - Smart Request Queue
Replaces the static-priority heap with a bounded scoring function (lower score
= dispatched first):

    score = w_priority*(prio-1)        user priority (HIGH/NORMAL/LOW)          [F4]
          - w_urgency*urgency          deadline/slack urgency                   [F4]
          + w_short*min(1,prompt/ref)  short-prompt-first (prefill queueing)    [F5]
          + w_bucket*[bucket mismatch] prefer bucket-compatible batch mix       [F6]
          - aging*wait                 anti-starvation aging

Two logical lanes, REAL_TIME before BEST_EFFORT                                 [F8]
Every pick that differs from plain priority+FIFO emits a `reorder` trace event
naming the dominant reason.                                                     [F10]
"""

import asyncio
import itertools
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from scheduler_engine.decision_trace import EVT_REORDER, DecisionTrace
from scheduler_engine.estimators import OnlineCalibrator
from scheduler_engine.priority_queue import PrioritizedRequestQueue, QueueEntry
from scheduler_engine.profiler import Bucket, RequestProfile, TrafficClass
from scheduler_engine.types import InferenceRequest


class SmartEntry(QueueEntry):
    __slots__ = ("profile",)

    def __init__(self, request: InferenceRequest, entry_id: int, future: asyncio.Future, profile: RequestProfile):
        super().__init__(request, entry_id, future)
        self.profile = profile


@dataclass
class SelectionContext:
    now: float
    concurrency: float
    calibrator: OnlineCalibrator
    preferred_bucket: Optional[str] = None      # dominant bucket of the active batch
    allow_best_effort: bool = True              # policy decides (throttle / RT reserve)
    hard_pressure: bool = False
    # returns (fits, reason_if_not). Policy supplies token-budget + LONG-cap rules.
    fits: Optional[Callable[[SmartEntry], Tuple[bool, str]]] = None


_TERM_REASON = {
    "doomed": "deadline_unreachable_deprioritized",
    "urgency": "deadline_urgency",
    "short": "short_prompt_first",
    "bucket": "bucket_compat",
    "aging": "aging",
}


class SmartRequestQueue(PrioritizedRequestQueue):
    def __init__(self, aging_factor: float, smart_cfg: Any, trace: Optional[DecisionTrace] = None):
        super().__init__(aging_factor=aging_factor)
        self.smart = smart_cfg
        self.trace = trace
        self._last_ctx: Optional[SelectionContext] = None

    # ------------------------------------------------------------------ enqueue
    async def enqueue(self, request: InferenceRequest, profile: Optional[RequestProfile] = None) -> asyncio.Future:
        if profile is None:
            raise ValueError("SmartRequestQueue.enqueue requires a RequestProfile")
        async with self._lock:
            loop = asyncio.get_running_loop()
            future = loop.create_future()
            entry_id = next(self._counter)
            req_id = request.request_id or f"req-{entry_id}"
            if req_id in self._entry_map:        # keep ids unique even if a client re-sends one
                req_id = f"{req_id}#{entry_id}"
            request.request_id = req_id
            profile.request_id = req_id
            entry = SmartEntry(request=request, entry_id=entry_id, future=future, profile=profile)
            self._entry_map[req_id] = entry
            self._notify_event.set()
            return future

    # ------------------------------------------------------------------ scoring
    def score_entry(self, entry: SmartEntry, ctx: SelectionContext) -> Dict[str, float]:
        cfg = self.smart
        p = entry.profile
        wait = max(0.0, ctx.now - entry.enqueue_time)
        est_exec = ctx.calibrator.exec_time(p.prompt_tokens, p.expected_output_tokens, ctx.concurrency)
        slack_s = p.deadline - ctx.now - est_exec
        sla = max(p.sla_s, 0.05)
        used = 1.0 - slack_s / sla                      # fraction of SLA consumed/threatened
        onset = 0.4
        doomed = slack_s < 0.0                          # can no longer meet its deadline
        # Feasible-deadline urgency: only requests that can still be rescued get boosted. Boosting
        # already-late requests (plain EDF) wastes capacity exactly when overloaded.
        urgency = 0.0 if doomed else min(1.0, max(0.0, (used - onset) / (1.0 - onset))) * 2.0

        prio = cfg.w_priority * (int(p.priority.value) - 1)
        urg = -cfg.w_urgency * urgency if cfg.deadline_priority else 0.0
        if cfg.deadline_priority and doomed:
            urg = cfg.w_doomed                          # served after feasible work of the same priority
        short = cfg.w_short * min(1.0, p.prompt_tokens / max(1, cfg.short_ref_tokens)) if cfg.short_prompt_first else 0.0
        bucket = (
            cfg.w_bucket
            if (cfg.bucket_compat and ctx.preferred_bucket and p.bucket.value != ctx.preferred_bucket)
            else 0.0
        )
        aging = -self.aging_factor * wait
        return {
            "score": prio + urg + short + bucket + aging,
            "priority": prio,
            "urgency": urg,
            "short": short,
            "bucket": bucket,
            "aging": aging,
            "urgency_value": urgency,
            "doomed": float(doomed),
            "slack_s": slack_s,
            "wait_s": wait,
            "est_exec_s": est_exec,
        }

    @staticmethod
    def _lane_rank(entry: SmartEntry) -> int:
        return 0 if entry.profile.traffic_class == TrafficClass.REAL_TIME else 1

    def _eligible(self, candidates: List[SmartEntry], ctx: SelectionContext) -> List[SmartEntry]:
        out = []
        for e in candidates:
            if self._lane_rank(e) == 0 or ctx.allow_best_effort:
                out.append(e)
            elif (not ctx.hard_pressure) and (ctx.now - e.enqueue_time) >= self.smart.be_max_wait_seconds:
                out.append(e)  # starvation safeguard: throttled BE work is eventually released
        return out

    # ---------------------------------------------------------------- selection
    async def select_next(self, ctx: SelectionContext) -> Optional[SmartEntry]:
        """Pick, remove and return the best entry that fits; None if nothing can run now."""
        self._last_ctx = ctx
        async with self._lock:
            candidates = [e for e in self._entry_map.values() if not e.cancelled]
            if not candidates:
                self._notify_event.clear()
                return None
            eligible = self._eligible(candidates, ctx)
            if not eligible:
                return None

            scored = [(self.score_entry(e, ctx), e) for e in eligible]
            scored.sort(key=lambda se: (self._lane_rank(se[1]), se[0]["score"], se[1].enqueue_time, se[1].entry_id))
            baseline = sorted(
                eligible,
                key=lambda e: (self._lane_rank(e), int(e.profile.priority.value), e.enqueue_time, e.entry_id),
            )

            chosen: Optional[Tuple[Dict[str, float], SmartEntry]] = None
            skip_reasons: Dict[str, str] = {}
            for terms, e in scored:
                if ctx.fits is not None:
                    ok, why = ctx.fits(e)
                    if not ok:
                        skip_reasons[e.request.request_id] = why
                        continue
                chosen = (terms, e)
                break
            if chosen is None:
                return None

            terms_c, entry = chosen
            if baseline and baseline[0] is not entry:
                self._emit_reorder(entry, terms_c, baseline, scored, skip_reasons)

            del self._entry_map[entry.request.request_id]
            return entry

    def _emit_reorder(self, entry, terms_c, baseline, scored, skip_reasons) -> None:
        if self.trace is None:
            return
        b = baseline[0]
        terms_b = next((t for t, e in scored if e is b), None)
        reason_code = "priority_order"
        if b.request.request_id in skip_reasons:
            reason_code = "capacity_fit"
            human = f"{b.request.request_id} skipped: {skip_reasons[b.request.request_id]}"
        else:
            best_term, best_adv = None, 1e-9
            if terms_b is not None:
                for name in ("urgency", "short", "bucket", "aging"):
                    adv = terms_b[name] - terms_c[name]   # lower term is better -> positive = helped `entry`
                    if adv > best_adv:
                        best_term, best_adv = name, adv
            if best_term:
                reason_code = _TERM_REASON[best_term]
            human = f"moved ahead of {b.request.request_id} due to {reason_code} (score {terms_c['score']:.2f} vs " \
                    f"{(terms_b or {}).get('score', float('nan')):.2f})"

        p = entry.profile
        self.trace.emit(
            EVT_REORDER,
            request_id=entry.request.request_id,
            decision="reorder",
            reason=human,
            reason_code=reason_code,
            token_estimate=p.total_estimated_tokens,
            sla_ms=p.sla_ms,
            slack_ms=terms_c["slack_s"] * 1000,
            bucket=p.bucket.value,
            traffic_class=p.traffic_class.value,
            overtook=b.request.request_id,
            order_before=[e.request.request_id for e in baseline[:5]],
            order_after=[e.request.request_id for _, e in scored[:5]],
            terms={k: round(v, 3) for k, v in terms_c.items()},
        )

    async def dequeue(self, timeout: Optional[float] = None):
        """API-compat with PrioritizedRequestQueue (uses the last selection context)."""
        while True:
            if self._entry_map and self._last_ctx is not None:
                entry = await self.select_next(self._last_ctx)
                if entry is not None:
                    return entry
            try:
                await asyncio.wait_for(self._notify_event.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                return None

    # ------------------------------------------------------------------ views
    def entries(self) -> List[SmartEntry]:
        return [e for e in self._entry_map.values() if not e.cancelled]

    def profiles(self) -> List[RequestProfile]:
        return [e.profile for e in self.entries()]

    def queued_tokens(self) -> int:
        return sum(e.profile.total_estimated_tokens for e in self.entries())

    def lane_counts(self) -> Dict[str, int]:
        out = {TrafficClass.REAL_TIME.value: 0, TrafficClass.BEST_EFFORT.value: 0}
        for e in self.entries():
            out[e.profile.traffic_class.value] += 1
        return out

    def bucket_counts(self) -> Dict[str, int]:
        out = {b.value: 0 for b in Bucket}
        for e in self.entries():
            out[e.profile.bucket.value] += 1
        return out

    def get_queue_snapshot(self, ctx: Optional[SelectionContext] = None) -> List[Dict[str, Any]]:  # type: ignore[override]
        ctx = ctx or self._last_ctx
        entries = self.entries()
        if ctx is None:
            return [{"request_id": e.request.request_id, **e.profile.to_dict()} for e in entries]
        ctx = SelectionContext(
            now=time.time(), concurrency=ctx.concurrency, calibrator=ctx.calibrator,
            preferred_bucket=ctx.preferred_bucket, allow_best_effort=ctx.allow_best_effort,
        )
        rows = []
        for e in entries:
            t = self.score_entry(e, ctx)
            rows.append((self._lane_rank(e), t["score"], e, t))
        rows.sort(key=lambda r: (r[0], r[1], r[2].enqueue_time))
        out = []
        for rank, (_, _, e, t) in enumerate(rows, start=1):
            out.append({
                "position": rank,
                "request_id": e.request.request_id,
                "priority": e.profile.priority.name,
                "traffic_class": e.profile.traffic_class.value,
                "bucket": e.profile.bucket.value,
                "prompt_tokens": e.profile.prompt_tokens,
                "expected_output_tokens": e.profile.expected_output_tokens,
                "total_estimated_tokens": e.profile.total_estimated_tokens,
                "wait_seconds": round(t["wait_s"], 3),
                "slack_ms": round(t["slack_s"] * 1000, 1),
                "urgency": round(t["urgency_value"], 3),
                "doomed": bool(t["doomed"]),
                "score": round(t["score"], 3),
                "score_terms": {k: round(t[k], 3) for k in ("priority", "urgency", "short", "bucket", "aging")},
                "sla_ms": e.profile.sla_ms,
            })
        return out
