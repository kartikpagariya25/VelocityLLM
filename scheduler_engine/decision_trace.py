"""
VelocityLLM - Decision Trace (Feature 10: Explainable Scheduling)
Every accept / reject / reorder / throttle / capacity-change is emitted as a
structured JSON-able event carrying the exact signals the scheduler used.
The UI must only show decisions that exist in this stream.
"""

import asyncio
import itertools
import time
from collections import Counter, OrderedDict, deque
from typing import Any, Deque, Dict, List, Optional

# Canonical event names (keep stable: the frontend keys off these).
EVT_ADMIT = "admit"
EVT_REJECT = "reject"
EVT_REORDER = "reorder"
EVT_DISPATCH = "dispatch"
EVT_COMPLETE = "complete"
EVT_CANCEL = "cancel"
EVT_CAPACITY = "capacity_change"
EVT_THROTTLE = "throttle"
EVT_CALIBRATION = "calibration"


class DecisionTrace:
    """Bounded in-memory event buffer + per-request index + live subscribers."""

    def __init__(self, maxlen: int = 2000, per_request_cap: int = 64, max_requests: int = 1500):
        self._events: Deque[Dict[str, Any]] = deque(maxlen=maxlen)
        self._by_request: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
        self._per_request_cap = per_request_cap
        self._max_requests = max_requests
        self._seq = itertools.count(1)
        self._subscribers: List[asyncio.Queue] = []
        self.counts_by_event: Counter = Counter()
        self.counts_by_reason: Counter = Counter()

    # ------------------------------------------------------------------ emit
    def emit(
        self,
        event: str,
        *,
        request_id: Optional[str] = None,
        decision: Optional[str] = None,
        reason: str = "",
        reason_code: Optional[str] = None,
        token_estimate: Optional[int] = None,
        kv_estimate_tokens: Optional[int] = None,
        sla_ms: Optional[float] = None,
        predicted_latency_ms: Optional[float] = None,
        slack_ms: Optional[float] = None,
        bucket: Optional[str] = None,
        traffic_class: Optional[str] = None,
        **extra: Any,
    ) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "seq": next(self._seq),
            "ts": round(time.time(), 4),
            "event": event,
            "request_id": request_id,
            "decision": decision or event,
            "reason": reason,
            "reason_code": reason_code,
            "token_estimate": token_estimate,
            "kv_estimate_tokens": kv_estimate_tokens,
            "sla_ms": _r(sla_ms),
            "predicted_latency_ms": _r(predicted_latency_ms),
            "slack_ms": _r(slack_ms),
            "bucket": bucket,
            "traffic_class": traffic_class,
        }
        if extra:
            record["details"] = extra

        self._events.append(record)
        self.counts_by_event[event] += 1
        if reason_code:
            self.counts_by_reason[reason_code] += 1

        if request_id:
            bucket_list = self._by_request.get(request_id)
            if bucket_list is None:
                bucket_list = []
                self._by_request[request_id] = bucket_list
                if len(self._by_request) > self._max_requests:
                    self._by_request.popitem(last=False)
            else:
                self._by_request.move_to_end(request_id)
            if len(bucket_list) < self._per_request_cap:
                bucket_list.append(record)

        for q in list(self._subscribers):
            try:
                q.put_nowait(record)
            except asyncio.QueueFull:
                try:  # drop oldest, keep newest
                    q.get_nowait()
                    q.put_nowait(record)
                except Exception:
                    pass
        return record

    # ----------------------------------------------------------------- query
    def recent(
        self,
        limit: int = 200,
        event: Optional[str] = None,
        request_id: Optional[str] = None,
        since_seq: int = 0,
    ) -> List[Dict[str, Any]]:
        if request_id:
            items = list(self._by_request.get(request_id, []))
        else:
            items = list(self._events)
        if event:
            items = [e for e in items if e["event"] == event]
        if since_seq:
            items = [e for e in items if e["seq"] > since_seq]
        return items[-max(1, limit):]

    def for_request(self, request_id: str) -> List[Dict[str, Any]]:
        return list(self._by_request.get(request_id, []))

    def summary(self) -> Dict[str, Any]:
        return {
            "buffered_events": len(self._events),
            "last_seq": (self._events[-1]["seq"] if self._events else 0),
            "by_event": dict(self.counts_by_event),
            "by_reason_code": dict(self.counts_by_reason),
        }

    # ------------------------------------------------------------ subscribers
    def subscribe(self, maxsize: int = 1000) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        if q in self._subscribers:
            self._subscribers.remove(q)


def _r(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(float(value), 2)
