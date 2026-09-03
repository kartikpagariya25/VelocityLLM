"""
Unit Tests - Prioritized Request Queue with Anti-Starvation Aging
"""

import asyncio
import time
import pytest
from scheduler_engine.priority_queue import PrioritizedRequestQueue
from scheduler_engine.types import InferenceRequest, RequestPriority


@pytest.mark.asyncio
async def test_priority_ordering():
    pq = PrioritizedRequestQueue(aging_factor=0.0)  # No aging for pure priority test

    req_low = InferenceRequest(prompt="Low", request_id="low-1", priority=RequestPriority.LOW)
    req_high = InferenceRequest(prompt="High", request_id="high-1", priority=RequestPriority.HIGH)
    req_norm = InferenceRequest(prompt="Normal", request_id="norm-1", priority=RequestPriority.NORMAL)

    # Enqueue low first, then high, then normal
    await pq.enqueue(req_low)
    await pq.enqueue(req_high)
    await pq.enqueue(req_norm)

    assert len(pq) == 3

    # Dequeue order should be HIGH, NORMAL, LOW
    e1 = await pq.dequeue()
    assert e1.request.request_id == "high-1"

    e2 = await pq.dequeue()
    assert e2.request.request_id == "norm-1"

    e3 = await pq.dequeue()
    assert e3.request.request_id == "low-1"


@pytest.mark.asyncio
async def test_anti_starvation_aging():
    # Aging factor = 2.0 per second.
    # LOW has base score 3.0. HIGH has base score 1.0.
    # After waiting 1.5 seconds, LOW's effective score = 3.0 - (1.5 * 2.0) = 0.0 < 1.0!
    # Therefore, the aged LOW request must be dequeued before a newly arrived HIGH request!
    pq = PrioritizedRequestQueue(aging_factor=2.0)

    req_low = InferenceRequest(prompt="Old Low", request_id="old-low", priority=RequestPriority.LOW)
    await pq.enqueue(req_low)

    # Wait 1.1 seconds so aging reduces its score from 3.0 to 3.0 - 2.2 = 0.8 (< 1.0)
    await asyncio.sleep(1.1)

    req_high = InferenceRequest(prompt="New High", request_id="new-high", priority=RequestPriority.HIGH)
    await pq.enqueue(req_high)

    # Dequeue: old_low should win due to aging promotion!
    e1 = await pq.dequeue()
    assert e1.request.request_id == "old-low"

    e2 = await pq.dequeue()
    assert e2.request.request_id == "new-high"


@pytest.mark.asyncio
async def test_queue_cancellation():
    pq = PrioritizedRequestQueue()

    req = InferenceRequest(prompt="Cancel me", request_id="cancel-1")
    fut = await pq.enqueue(req)

    assert len(pq) == 1
    cancelled = await pq.cancel("cancel-1")
    assert cancelled is True
    assert fut.cancelled()
    assert len(pq) == 0
