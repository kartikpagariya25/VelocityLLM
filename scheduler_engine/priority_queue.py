"""
VelocityLLM - Prioritized Request Queue with Anti-Starvation Aging
Maintains queued inference requests ordered by priority with dynamic aging
to guarantee low-priority requests are promoted over time and never starved.
"""

import asyncio
import heapq
import itertools
import time
from typing import Dict, List, Optional, Tuple

from scheduler_engine.types import InferenceRequest, RequestPriority

QUEUE_ORDERS = ("priority", "fcfs", "sjf")
SJF_AGING_TOKENS_PER_UNIT = 100.0


class QueueEntry:
    """Represents a queued inference request alongside its priority entry metadata."""
    __slots__ = ("request", "enqueue_time", "entry_id", "cancelled", "future")

    def __init__(self, request: InferenceRequest, entry_id: int, future: asyncio.Future):
        self.request = request
        self.enqueue_time = time.time()
        self.entry_id = entry_id
        self.cancelled = False
        self.future = future

    def calculate_effective_score(self, aging_factor: float, current_time: float, order: str = "priority") -> float:
        """
        Lower score = higher dispatch urgency.
        priority: base_priority - wait_time * aging_factor
        fcfs:     arrival timestamp
        sjf:      requested tokens (plus image tokens) - wait_time * aging_factor * SJF_AGING_TOKENS_PER_UNIT
        """
        wait_time = max(0.0, current_time - self.enqueue_time)
        if order == "fcfs":
            return self.enqueue_time
        if order == "sjf":
            size = self.request.max_tokens + self.request.image_tokens
            return float(size) - wait_time * aging_factor * SJF_AGING_TOKENS_PER_UNIT
        return float(self.request.priority.value) - (wait_time * aging_factor)


class PrioritizedRequestQueue:
    """
    Async prioritized request queue with anti-starvation aging.
    Thread-safe and async-compatible.
    """

    def __init__(self, aging_factor: float = 0.25, order: str = "priority"):
        if order not in QUEUE_ORDERS:
            raise ValueError(f"queue order must be one of {QUEUE_ORDERS}, got {order!r}")
        self.aging_factor = aging_factor
        self.order = order
        self._lock = asyncio.Lock()
        self._entries: List[Tuple[float, int, QueueEntry]] = []  # Min-heap of (score, entry_id, entry)
        self._counter = itertools.count()
        self._entry_map: Dict[str, QueueEntry] = {}  # request_id -> QueueEntry
        self._notify_event = asyncio.Event()

    def __len__(self) -> int:
        return len(self._entry_map)

    @property
    def size(self) -> int:
        return len(self._entry_map)

    @property
    def image_tokens_waiting(self) -> int:
        return sum(entry.request.image_tokens for entry in self._entry_map.values())

    async def enqueue(self, request: InferenceRequest) -> asyncio.Future:
        """
        Enqueue an inference request. Returns a Future that will be resolved
        when the request completes or is cancelled.
        """
        async with self._lock:
            loop = asyncio.get_running_loop()
            future = loop.create_future()
            entry_id = next(self._counter)
            entry = QueueEntry(request=request, entry_id=entry_id, future=future)

            req_id = request.request_id or f"req-{entry_id}"
            request.request_id = req_id

            score = entry.calculate_effective_score(self.aging_factor, time.time(), self.order)
            heapq.heappush(self._entries, (score, entry_id, entry))
            self._entry_map[req_id] = entry
            self._notify_event.set()
            return future

    def _refresh(self) -> None:
        now = time.time()
        live = [
            (entry.calculate_effective_score(self.aging_factor, now, self.order), entry_id, entry)
            for _, entry_id, entry in self._entries
            if not entry.cancelled
        ]
        heapq.heapify(live)
        self._entries = live

    async def peek(self) -> Optional[QueueEntry]:
        """The entry dequeue would return next, without removing it."""
        async with self._lock:
            self._refresh()
            while self._entries:
                _, _, entry = self._entries[0]
                if entry.request.request_id in self._entry_map and not entry.cancelled:
                    return entry
                heapq.heappop(self._entries)
            return None

    async def dequeue(self, timeout: Optional[float] = None) -> Optional[QueueEntry]:
        """Dequeue the most urgent request under the configured order, refreshing aging scores first."""
        while True:
            async with self._lock:
                if not self._entry_map:
                    self._notify_event.clear()
                else:
                    self._refresh()
                    while self._entries:
                        _, _, entry = heapq.heappop(self._entries)
                        req_id = entry.request.request_id
                        if req_id in self._entry_map and not entry.cancelled:
                            del self._entry_map[req_id]
                            return entry

            try:
                await asyncio.wait_for(self._notify_event.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                return None

    async def drop_where(self, predicate) -> List[QueueEntry]:
        """Remove and return every waiting entry for which predicate(entry) is true."""
        async with self._lock:
            matched = [entry for entry in self._entry_map.values() if predicate(entry)]
            for entry in matched:
                self._entry_map.pop(entry.request.request_id, None)
                entry.cancelled = True
            return matched

    def is_queued(self, request_id: str) -> bool:
        """Check if request is currently pending in queue."""
        return request_id in self._entry_map

    async def cancel(self, request_id: str, reason: str = "Cancelled by client") -> bool:
        """Cancel a queued request if it hasn't been dequeued yet."""
        async with self._lock:
            entry = self._entry_map.pop(request_id, None)
            if entry:
                entry.cancelled = True
                if not entry.future.done():
                    entry.future.cancel()
                return True
            return False

    def get_queue_snapshot(self) -> List[Dict]:
        """Return diagnostic snapshot of currently queued items."""
        now = time.time()
        snapshot = []
        for req_id, entry in list(self._entry_map.items()):
            wait_time = now - entry.enqueue_time
            score = entry.calculate_effective_score(self.aging_factor, now, self.order)
            snapshot.append({
                "request_id": req_id,
                "priority": entry.request.priority.name,
                "wait_time_seconds": round(wait_time, 3),
                "effective_score": round(score, 3),
            })
        snapshot.sort(key=lambda x: x["effective_score"])
        return snapshot
