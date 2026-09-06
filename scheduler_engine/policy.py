"""
VelocityLLM - Scheduler Policy Implementations
Defines the SchedulerPolicy abstraction and two swappable implementations:
1. StaticBatchPolicy: Fixed-capacity baseline serving as the Phase 1 comparison anchor.
2. DynamicBatchPolicy: Core novel dynamic batching engine with SLA-aware admission,
   adaptive concurrency sizing, and priority scheduling with anti-starvation aging.
"""

from abc import ABC, abstractmethod
import asyncio
import logging
import time
from typing import AsyncIterator, Dict, List, Optional
import uuid

from scheduler_engine.adaptive_controller import AdaptiveBatchController
from scheduler_engine.admission import AdmissionController
from scheduler_engine.backend import GPUOutOfMemoryError, InferenceBackend
from scheduler_engine.priority_queue import PrioritizedRequestQueue
from scheduler_engine.types import (
    AdmissionStatus,
    GenerationChunk,
    InferenceRequest,
    InferenceResponse,
    SchedulerStats,
    ServerConfig,
)

logger = logging.getLogger("velocityllm.policy")


class AdmissionRejectedException(Exception):
    """Exception raised when admission controller rejects a request."""
    def __init__(self, status: AdmissionStatus, reason: str, retry_after: float):
        super().__init__(reason)
        self.status = status
        self.reason = reason
        self.retry_after = retry_after


class SchedulerPolicy(ABC):
    """Abstract interface for LLM scheduler execution policies."""

    @abstractmethod
    async def initialize(self) -> None:
        pass

    @abstractmethod
    async def schedule(self, request: InferenceRequest) -> InferenceResponse:
        pass

    @abstractmethod
    async def schedule_stream(self, request: InferenceRequest) -> AsyncIterator[GenerationChunk]:
        pass

    async def cancel_request(self, request_id: str, reason: str = "Cancelled by client") -> bool:
        """Cancel a request and reclaim resources if supported by the policy."""
        return False

    @abstractmethod
    def get_stats(self) -> SchedulerStats:
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        pass


class StaticBatchPolicy(SchedulerPolicy):
    """
    Static Batching Baseline Policy.
    Uses fixed concurrency limit (default 8), standard FIFO queuing,
    no adaptive sizing, and no SLA-aware rejection.
    """

    def __init__(self, backend: InferenceBackend, config: ServerConfig):
        self.backend = backend
        self.config = config
        self.max_concurrency = config.initial_concurrency  # fixed baseline
        self.semaphore = asyncio.Semaphore(self.max_concurrency)

        self.total_accepted = 0
        self.total_completed = 0
        self.total_tokens = 0
        self.sla_breaches = 0
        self.latencies: List[float] = []
        self._active_count = 0
        self._queued_count = 0

    async def initialize(self) -> None:
        await self.backend.initialize()
        logger.info("StaticBatchPolicy initialized (fixed concurrency: %d).", self.max_concurrency)

    async def schedule(self, request: InferenceRequest) -> InferenceResponse:
        if not request.request_id:
            request.request_id = str(uuid.uuid4())

        self.total_accepted += 1
        self._queued_count += 1
        enqueue_time = time.time()

        async with self.semaphore:
            self._queued_count -= 1
            self._active_count += 1
            start_exec = time.time()
            queue_time = start_exec - enqueue_time

            generated_chunks = []
            try:
                async for chunk in self.backend.generate_stream(
                    request.prompt, request.max_tokens, request.temperature, request.request_id
                ):
                    generated_chunks.append(chunk)
            finally:
                self._active_count -= 1

        exec_time = time.time() - start_exec
        total_latency = time.time() - enqueue_time
        full_text = "".join(generated_chunks)
        token_count = len(generated_chunks)

        self.total_completed += 1
        self.total_tokens += token_count
        self.latencies.append(total_latency)
        if len(self.latencies) > 500:
            self.latencies.pop(0)

        target_sla_sec = (request.sla_target_ms or self.config.target_sla_ms) / 1000.0
        sla_met = total_latency <= target_sla_sec
        if not sla_met:
            self.sla_breaches += 1

        tps = (token_count / exec_time) if exec_time > 0 else 0.0

        return InferenceResponse(
            request_id=request.request_id,
            prompt=request.prompt,
            response=full_text,
            latency_seconds=round(total_latency, 4),
            queue_time_seconds=round(queue_time, 4),
            execution_time_seconds=round(exec_time, 4),
            tokens_generated=token_count,
            tokens_per_second=round(tps, 2),
            sla_met=sla_met,
            priority=request.priority.name.lower(),
        )

    async def schedule_stream(self, request: InferenceRequest) -> AsyncIterator[GenerationChunk]:
        if not request.request_id:
            request.request_id = str(uuid.uuid4())

        self.total_accepted += 1
        self._queued_count += 1
        enqueue_time = time.time()

        async with self.semaphore:
            self._queued_count -= 1
            self._active_count += 1
            try:
                idx = 0
                async for delta in self.backend.generate_stream(
                    request.prompt, request.max_tokens, request.temperature, request.request_id
                ):
                    yield GenerationChunk(
                        request_id=request.request_id,
                        delta=delta,
                        token_index=idx,
                        is_finished=False,
                    )
                    idx += 1
                yield GenerationChunk(
                    request_id=request.request_id,
                    delta="",
                    token_index=idx,
                    is_finished=True,
                    finish_reason="stop",
                )
            finally:
                self._active_count -= 1
                total_latency = time.time() - enqueue_time
                self.latencies.append(total_latency)
                self.total_completed += 1

    def get_stats(self) -> SchedulerStats:
        util, mem_used, mem_total = self.backend.get_gpu_telemetry()
        sorted_latencies = sorted(self.latencies) if self.latencies else [0.0]
        n = len(sorted_latencies)

        return SchedulerStats(
            policy_name="static_baseline",
            active_requests=self._active_count,
            queued_requests=self._queued_count,
            effective_concurrency_limit=self.max_concurrency,
            total_accepted=self.total_accepted,
            total_rejected=0,
            total_completed=self.total_completed,
            total_tokens_generated=self.total_tokens,
            sla_breach_count=self.sla_breaches,
            avg_latency_seconds=sum(sorted_latencies) / n if n else 0.0,
            p50_latency_seconds=sorted_latencies[int(n * 0.50)],
            p95_latency_seconds=sorted_latencies[min(int(n * 0.95), n - 1)],
            p99_latency_seconds=sorted_latencies[min(int(n * 0.99), n - 1)],
            gpu_utilization_percent=float(util),
            gpu_memory_used_mb=mem_used,
            gpu_memory_total_mb=mem_total,
        )

    async def shutdown(self) -> None:
        await self.backend.shutdown()


class DynamicBatchPolicy(SchedulerPolicy):
    """
    Dynamic Continuous Batching Scheduler Engine.
    Combines:
    - SLA-Aware Admission Controller
    - Prioritized Queue with Anti-Starvation Aging
    - Adaptive Batch Sizing Controller (AIMD Feedback Loop)
    - Continuous iteration-level async dispatch loop
    """

    def __init__(self, backend: InferenceBackend, config: ServerConfig):
        self.backend = backend
        self.config = config
        self.admission_controller = AdmissionController(config)
        self.adaptive_controller = AdaptiveBatchController(config)
        self.queue = PrioritizedRequestQueue(aging_factor=config.aging_factor)

        # Concurrency state
        self._active_requests: Dict[str, asyncio.Task] = {}
        self._dispatcher_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

        # Telemetry
        self.total_accepted = 0
        self.total_rejected = 0
        self.total_completed = 0
        self.total_tokens = 0
        self.sla_breaches = 0
        self.latencies: List[float] = []
        self.queue_times: List[float] = []

        # Phase 3 Telemetry
        self.client_disconnects_count = 0
        self.burst_shed_count = 0
        self.oom_recoveries_count = 0
        self.validation_errors_count = 0

    async def cancel_request(self, request_id: str, reason: str = "Client disconnected") -> bool:
        """
        Cancels a request whether it is waiting in the priority queue or actively generating on GPU.
        Immediately reclaims the execution slot and updates telemetry.
        """
        # 1. Check if request is waiting in queue
        if await self.queue.cancel(request_id, reason=reason):
            self.client_disconnects_count += 1
            logger.info("Request %s cancelled while waiting in queue. Reason: %s", request_id, reason)
            return True

        # 2. Check if request is currently executing on backend
        task = self._active_requests.pop(request_id, None)
        if task and not task.done():
            task.cancel()
            self.client_disconnects_count += 1
            logger.info("Active request %s aborted mid-generation (%s). Slot reclaimed.", request_id, reason)
            return True

        return False

    async def initialize(self) -> None:
        await self.backend.initialize()
        self._stop_event.clear()
        self._dispatcher_task = asyncio.create_task(self._dispatch_loop())
        logger.info(
            "DynamicBatchPolicy initialized with initial concurrency %d (bounds: [%d, %d]).",
            self.config.initial_concurrency,
            self.config.min_concurrency,
            self.config.max_concurrency,
        )

    async def _dispatch_loop(self) -> None:
        """
        Continuous dispatch loop:
        1. Queries GPU telemetry
        2. Adjusts adaptive concurrency limit
        3. Dequeues highest priority/aged requests up to capacity limit
        4. Dispatches execution tasks concurrently
        """
        while not self._stop_event.is_set():
            try:
                util, mem_used, mem_total = self.backend.get_gpu_telemetry()
                recent_breach = bool(self.latencies and self.latencies[-1] > (self.config.target_sla_ms / 1000.0))

                effective_limit = self.adaptive_controller.evaluate_and_tune(
                    gpu_util_percent=float(util),
                    gpu_memory_used_mb=mem_used,
                    gpu_memory_total_mb=mem_total,
                    queue_depth=self.queue.size,
                    active_requests=len(self._active_requests),
                    recent_sla_breach=recent_breach,
                )

                available_slots = max(0, effective_limit - len(self._active_requests))
                for _ in range(available_slots):
                    if self.queue.size == 0:
                        break

                    entry = await self.queue.dequeue(timeout=0.01)
                    if entry and not entry.cancelled:
                        task = asyncio.create_task(self._execute_request(entry))
                        self._active_requests[entry.request.request_id] = task

                await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in DynamicBatchPolicy dispatch loop: %s", e, exc_info=True)
                await asyncio.sleep(0.1)

    async def _execute_request(self, entry) -> None:
        """Executes a dequeued request on the backend and resolves its future."""
        request = entry.request
        start_exec = time.time()
        queue_time = start_exec - entry.enqueue_time
        chunks: List[str] = []

        try:
            async for chunk in self.backend.generate_stream(
                request.prompt, request.max_tokens, request.temperature, request.request_id
            ):
                chunks.append(chunk)

            total_latency = time.time() - entry.enqueue_time
            exec_time = time.time() - start_exec
            full_response = "".join(chunks)
            token_count = len(chunks)

            target_sla_sec = (request.sla_target_ms or self.config.target_sla_ms) / 1000.0
            sla_met = total_latency <= target_sla_sec
            if not sla_met:
                self.sla_breaches += 1

            self.total_completed += 1
            self.total_tokens += token_count
            self.latencies.append(total_latency)
            self.queue_times.append(queue_time)
            if len(self.latencies) > 500:
                self.latencies.pop(0)
            if len(self.queue_times) > 500:
                self.queue_times.pop(0)

            # Update admission controller service-time estimate
            self.admission_controller.update_completion_stats(total_latency, token_count)

            tps = (token_count / exec_time) if exec_time > 0 else 0.0

            result = InferenceResponse(
                request_id=request.request_id,
                prompt=request.prompt,
                response=full_response,
                latency_seconds=round(total_latency, 4),
                queue_time_seconds=round(queue_time, 4),
                execution_time_seconds=round(exec_time, 4),
                tokens_generated=token_count,
                tokens_per_second=round(tps, 2),
                sla_met=sla_met,
                priority=request.priority.name.lower(),
            )

            if not entry.future.done():
                entry.future.set_result(result)
        except asyncio.CancelledError:
            logger.info("Execution task for request %s was cancelled.", request.request_id)
            if not entry.future.done():
                entry.future.cancel()
            raise
        except GPUOutOfMemoryError as oom_err:
            self.oom_recoveries_count += 1
            logger.error("GPU OOM encountered during execution of request %s: %s", request.request_id, oom_err)
            self.adaptive_controller.trigger_emergency_oom_throttle(cooldown_seconds=2.0)
            self.backend.handle_oom()
            if not entry.future.done():
                entry.future.set_exception(oom_err)
        except Exception as err:
            if not entry.future.done():
                entry.future.set_exception(err)
        finally:
            self._active_requests.pop(request.request_id, None)

    async def schedule(self, request: InferenceRequest) -> InferenceResponse:
        if not request.request_id:
            request.request_id = str(uuid.uuid4())

        # 1. Evaluate Admission
        _, mem_used, mem_total = self.backend.get_gpu_telemetry()
        admission = self.admission_controller.evaluate(
            request=request,
            current_queue_size=self.queue.size,
            active_concurrency=len(self._active_requests),
            gpu_memory_used_mb=mem_used,
            gpu_memory_total_mb=mem_total,
        )

        if not admission.admitted:
            self.total_rejected += 1
            self.burst_shed_count = self.admission_controller.total_burst_shed
            raise AdmissionRejectedException(
                status=admission.status,
                reason=admission.reason,
                retry_after=admission.retry_after_seconds,
            )

        self.total_accepted += 1
        future = await self.queue.enqueue(request)
        return await future

    async def schedule_stream(self, request: InferenceRequest) -> AsyncIterator[GenerationChunk]:
        """Streaming token dispatch."""
        if not request.request_id:
            request.request_id = str(uuid.uuid4())

        _, mem_used, mem_total = self.backend.get_gpu_telemetry()
        admission = self.admission_controller.evaluate(
            request=request,
            current_queue_size=self.queue.size,
            active_concurrency=len(self._active_requests),
            gpu_memory_used_mb=mem_used,
            gpu_memory_total_mb=mem_total,
        )

        if not admission.admitted:
            self.total_rejected += 1
            self.burst_shed_count = self.admission_controller.total_burst_shed
            raise AdmissionRejectedException(
                status=admission.status,
                reason=admission.reason,
                retry_after=admission.retry_after_seconds,
            )

        self.total_accepted += 1
        queue = asyncio.Queue()

        async def _stream_producer():
            idx = 0
            try:
                async for chunk in self.backend.generate_stream(
                    request.prompt, request.max_tokens, request.temperature, request.request_id
                ):
                    await queue.put(
                        GenerationChunk(
                            request_id=request.request_id,
                            delta=chunk,
                            token_index=idx,
                            is_finished=False,
                        )
                    )
                    idx += 1
                await queue.put(
                    GenerationChunk(
                        request_id=request.request_id,
                        delta="",
                        token_index=idx,
                        is_finished=True,
                        finish_reason="stop",
                    )
                )
            except Exception as e:
                await queue.put(e)
            finally:
                await queue.put(None)  # Sentinel

        producer_task = asyncio.create_task(_stream_producer())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            if not producer_task.done():
                producer_task.cancel()

    def get_stats(self) -> SchedulerStats:
        util, mem_used, mem_total = self.backend.get_gpu_telemetry()
        sorted_latencies = sorted(self.latencies) if self.latencies else [0.0]
        n = len(sorted_latencies)

        avg_q = sum(self.queue_times) / len(self.queue_times) if self.queue_times else 0.0

        return SchedulerStats(
            policy_name="dynamic_continuous",
            active_requests=len(self._active_requests),
            queued_requests=self.queue.size,
            effective_concurrency_limit=self.adaptive_controller.current_concurrency,
            total_accepted=self.total_accepted,
            total_rejected=self.total_rejected,
            total_completed=self.total_completed,
            total_tokens_generated=self.total_tokens,
            sla_breach_count=self.sla_breaches,
            avg_latency_seconds=sum(sorted_latencies) / n if n else 0.0,
            p50_latency_seconds=sorted_latencies[int(n * 0.50)],
            p95_latency_seconds=sorted_latencies[min(int(n * 0.95), n - 1)],
            p99_latency_seconds=sorted_latencies[min(int(n * 0.99), n - 1)],
            gpu_utilization_percent=float(util),
            gpu_memory_used_mb=mem_used,
            gpu_memory_total_mb=mem_total,
            avg_queue_wait_seconds=avg_q,
            client_disconnects_count=self.client_disconnects_count,
            burst_shed_count=self.admission_controller.total_burst_shed,
            oom_recoveries_count=self.oom_recoveries_count,
            validation_errors_count=self.validation_errors_count,
        )

    async def shutdown(self) -> None:
        self._stop_event.set()
        if self._dispatcher_task:
            self._dispatcher_task.cancel()
        for task in self._active_requests.values():
            task.cancel()
        await self.backend.shutdown()
        logger.info("DynamicBatchPolicy shutdown cleanly.")
