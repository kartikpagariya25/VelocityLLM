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

class QueueEntry:
    """Represents a queued inference request alongside its priority entry metadata."""
    __slots__ = ("request", "enqueue_time", "entry_id", "cancelled", "future")

    def __init__(self, request: InferenceRequest, entry_id: int, future: asyncio.Future):
        self.request = request
        self.enqueue_time = time.time()
        self.entry_id = entry_id
        self.cancelled = False
        self.future = future

    def calculate_effective_score(self, aging_factor: float, current_time: float) -> float:
        """
        Calculates priority score with aging promotion.
        Lower score = higher dispatch urgency.
        score = base_priority - (wait_time_seconds * aging_factor)
        """
        wait_time = max(0.0, current_time - self.enqueue_time)
        return float(self.request.priority.value) - (wait_time * aging_factor)


class PrioritizedRequestQueue:
    """
    Async prioritized request queue with anti-starvation aging.
    Thread-safe and async-compatible.
    """

    def __init__(self, aging_factor: float = 0.25):
        self.aging_factor = aging_factor
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

            score = entry.calculate_effective_score(self.aging_factor, time.time())
            heapq.heappush(self._entries, (score, entry_id, entry))
            self._entry_map[req_id] = entry
            self._notify_event.set()
            return future

    async def dequeue(self, timeout: Optional[float] = None) -> Optional[QueueEntry]:
        """
        Dequeue the highest-urgency request, re-evaluating aging scores across active entries.
        """
        while True:
            async with self._lock:
                if not self._entry_map:
                    self._notify_event.clear()
                else:
                    # Re-heapify with refreshed aging scores
                    now = time.time()
                    active_entries = []
                    for _, entry_id, entry in self._entries:
                        if not entry.cancelled:
                            refreshed_score = entry.calculate_effective_score(self.aging_factor, now)
                            active_entries.append((refreshed_score, entry_id, entry))

                    heapq.heapify(active_entries)
                    self._entries = active_entries

                    while self._entries:
                        _, _, entry = heapq.heappop(self._entries)
                        req_id = entry.request.request_id
                        if req_id in self._entry_map and not entry.cancelled:
                            del self._entry_map[req_id]
                            return entry

            # If queue is empty, wait for an enqueue event or timeout
            try:
                await asyncio.wait_for(self._notify_event.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                return None

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
            score = entry.calculate_effective_score(self.aging_factor, now)
            snapshot.append({
                "request_id": req_id,
                "priority": entry.request.priority.name,
                "wait_time_seconds": round(wait_time, 3),
                "effective_score": round(score, 3),
            })
        snapshot.sort(key=lambda x: x["effective_score"])
        return snapshot
