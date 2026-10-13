"""
VelocityLLM - True static batching baseline.
Requests are grouped into a batch of N (or whatever has arrived once the timeout passes), the whole batch runs
to completion, and only then does the next batch start. Short requests wait for the longest one in their batch.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import AsyncIterator, List, Optional

from scheduler_engine.backend import InferenceBackend
from scheduler_engine.policy import StaticBatchPolicy
from scheduler_engine.types import GenerationChunk, InferenceRequest, InferenceResponse, ServerConfig

logger = logging.getLogger("velocityllm.static_batching")


class _Batch:
    """Barrier: no member of the batch is released until every member has finished generating."""

    def __init__(self, size: int):
        self._outstanding = set(range(size))
        self._done = asyncio.Event()

    def arrive(self, member: int) -> None:
        self._outstanding.discard(member)
        if not self._outstanding:
            self._done.set()

    async def finish(self, member: int) -> None:
        self.arrive(member)
        await self._done.wait()


@dataclass
class _Pending:
    request: InferenceRequest
    enqueue_time: float
    future: asyncio.Future
    sink: Optional[asyncio.Queue] = field(default=None)


class StaticBatchingPolicy(StaticBatchPolicy):
    def __init__(self, backend: InferenceBackend, config: ServerConfig):
        super().__init__(backend, config)
        self.policy_name = "static_batching"
        self.batch_size = max(1, config.initial_concurrency)
        self.batch_timeout = max(0.0, config.static_batch_timeout_ms / 1000.0)
        self.batches_run = 0
        self._pending: List[_Pending] = []
        self._wake = asyncio.Event()
        self._stopping = False
        self._runner: Optional[asyncio.Task] = None

    async def initialize(self) -> None:
        await self.backend.initialize()
        self._runner = asyncio.create_task(self._batch_loop())
        logger.info(
            "StaticBatchingPolicy initialized (batch size %d, timeout %.0f ms).",
            self.batch_size,
            self.batch_timeout * 1000,
        )

    async def _batch_loop(self) -> None:
        while not self._stopping:
            if not self._pending:
                self._wake.clear()
                await self._wake.wait()
                continue

            remaining = self.batch_timeout - (time.time() - self._pending[0].enqueue_time)
            if len(self._pending) < self.batch_size and remaining > 0:
                self._wake.clear()
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    pass
                continue

            batch = self._pending[: self.batch_size]
            del self._pending[: self.batch_size]
            self._queued_count -= len(batch)
            self.batches_run += 1
            barrier = _Batch(len(batch))
            await asyncio.gather(*(self._run(item, barrier, i) for i, item in enumerate(batch)))

    async def _run(self, item: _Pending, barrier: _Batch, member: int) -> None:
        try:
            result = await self._execute(
                item.request,
                item.enqueue_time,
                sink=item.sink,
                after_generation=lambda: barrier.finish(member),
            )
            if item.sink is not None:
                await item.sink.put(
                    GenerationChunk(
                        request_id=item.request.request_id,
                        delta="",
                        token_index=result.tokens_generated,
                        is_finished=True,
                        finish_reason="stop",
                    )
                )
            if not item.future.done():
                item.future.set_result(result)
        except asyncio.CancelledError:
            barrier.arrive(member)
            if not item.future.done():
                item.future.cancel()
            raise
        except Exception as err:
            barrier.arrive(member)
            if not item.future.done():
                item.future.set_exception(err)
            if item.sink is not None:
                await item.sink.put(err)
        finally:
            if item.sink is not None:
                await item.sink.put(None)

    def _submit(self, request: InferenceRequest, sink: Optional[asyncio.Queue] = None) -> asyncio.Future:
        if not request.request_id:
            request.request_id = str(uuid.uuid4())
        future = asyncio.get_running_loop().create_future()
        self.total_accepted += 1
        self._queued_count += 1
        self._pending.append(_Pending(request, time.time(), future, sink))
        self._wake.set()
        return future

    async def schedule(self, request: InferenceRequest) -> InferenceResponse:
        return await self._submit(request)

    async def schedule_stream(self, request: InferenceRequest) -> AsyncIterator[GenerationChunk]:
        sink: asyncio.Queue = asyncio.Queue()
        future = self._submit(request, sink)
        future.add_done_callback(lambda f: f.exception() if not f.cancelled() else None)
        while True:
            item = await sink.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item

    async def cancel_request(self, request_id: str, reason: str = "Cancelled by client") -> bool:
        for item in self._pending:
            if item.request.request_id == request_id:
                self._pending.remove(item)
                self._queued_count -= 1
                item.future.cancel()
                return True
        return False

    async def shutdown(self) -> None:
        self._stopping = True
        self._wake.set()
        if self._runner:
            self._runner.cancel()
        await self.backend.shutdown()
