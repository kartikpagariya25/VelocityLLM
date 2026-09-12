"""
VelocityLLM - Production Inference Server
Unified FastAPI service implementing the Dynamic Batching Scheduler Engine
with swappable static/dynamic policies and OpenAI-compatible endpoints.
"""

import argparse
import yaml
import asyncio
from contextlib import asynccontextmanager
import logging
import os
import time
from typing import Optional
import uuid

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
import uvicorn

from scheduler_engine.backend import GPUOutOfMemoryError, create_backend
from scheduler_engine.logging_config import (
    configure_logging,
    get_correlation_id,
    set_correlation_id,
)
from scheduler_engine.policy import (
    AdmissionRejectedException,
    DynamicBatchPolicy,
    SchedulerPolicy,
    StaticBatchPolicy,
)
from scheduler_engine.types import (
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    InferenceRequest,
    InferenceResponse,
    RequestPriority,
    ServerConfig,
    UsageInfo,
)

# Initialize logging
configure_logging(level=logging.INFO, json_format=False)
logger = logging.getLogger("velocityllm.server")

# Global runtime state
_config: ServerConfig = ServerConfig()
_policy: Optional[SchedulerPolicy] = None
_start_time: float = time.time()


def get_config() -> ServerConfig:
    global _config
    return _config


def set_config(config: ServerConfig) -> None:
    global _config
    _config = config


def get_policy() -> SchedulerPolicy:
    global _policy
    if _policy is None:
        raise RuntimeError("Scheduler policy is not initialized.")
    return _policy


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes backend and scheduler policy on startup; ensures graceful shutdown."""
    global _policy, _config, _start_time
    _start_time = time.time()

    # Allow environment variables to override config
    if os.getenv("VELOCITY_POLICY"):
        _config.policy = os.getenv("VELOCITY_POLICY", "dynamic").lower()
    if os.getenv("VELOCITY_MOCK"):
        _config.use_mock_backend = os.getenv("VELOCITY_MOCK", "0") in ("1", "true", "True")
    if os.getenv("VELOCITY_MODEL_PATH"):
        _config.model_path = os.getenv("VELOCITY_MODEL_PATH", _config.model_path)
    if os.getenv("VELOCITY_SLA_MS"):
        _config.target_sla_ms = float(os.getenv("VELOCITY_SLA_MS", "3000.0"))
    if os.getenv("VELOCITY_JSON_LOGS"):
        _config.enable_structured_logging = os.getenv("VELOCITY_JSON_LOGS", "0") in ("1", "true", "True")

    if _config.enable_structured_logging:
        configure_logging(level=logging.INFO, json_format=True)

    logger.info(
        "Starting VelocityLLM Server [Policy: %s, Mock: %s, Model: %s]",
        _config.policy,
        _config.use_mock_backend,
        _config.model_path,
    )

    backend = create_backend(_config)

    if _config.policy == "static":
        _policy = StaticBatchPolicy(backend=backend, config=_config)
    else:
        _policy = DynamicBatchPolicy(backend=backend, config=_config)

    await _policy.initialize()
    logger.info("VelocityLLM Server initialized and ready to accept inference requests.")

    yield

    logger.info("Shutting down VelocityLLM Server...")
    if _policy:
        await _policy.shutdown()
        _policy = None
    logger.info("VelocityLLM Server shutdown complete.")


app = FastAPI(
    title="VelocityLLM Dynamic Batching Serving Engine",
    description="High-throughput continuous batching LLM serving engine with SLA-aware admission control.",
    version="3.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):
    """
    Extracts or generates correlation ID for every HTTP transaction.
    Propagates across contextvars and response headers.
    """
    corr_id = (
        request.headers.get("X-Correlation-ID")
        or request.headers.get("X-Request-ID")
        or f"req-{uuid.uuid4().hex[:12]}"
    )
    set_correlation_id(corr_id)
    request.state.correlation_id = corr_id

    response: Response = await call_next(request)
    response.headers["X-Correlation-ID"] = corr_id
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Custom structured response for input validation errors."""
    corr_id = getattr(request.state, "correlation_id", get_correlation_id())
    logger.warning("Input validation error on %s: %s", request.url.path, exc.errors())
    if _policy and hasattr(_policy, "validation_errors_count"):
        _policy.validation_errors_count += 1
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Input Validation Error",
            "correlation_id": corr_id,
            "details": exc.errors(),
        },
        headers={"X-Correlation-ID": corr_id},
    )


@app.post("/generate", response_model=InferenceResponse)
async def generate(req: InferenceRequest, request: Request):
    """
    VelocityLLM Native Generation Endpoint.
    Monitors client disconnect and automatically reclaims execution slots.
    """
    policy = get_policy()
    corr_id = getattr(request.state, "correlation_id", req.request_id or f"req-{uuid.uuid4().hex[:12]}")
    req.request_id = corr_id

    # Create asynchronous scheduling task
    schedule_task = asyncio.create_task(policy.schedule(req))

    try:
        # Monitor client disconnect while request executes
        while not schedule_task.done():
            if await request.is_disconnected():
                logger.warning("Client disconnected from request %s. Cancelling and reclaiming slot.", req.request_id)
                await policy.cancel_request(req.request_id, reason="Client HTTP connection disconnected")
                schedule_task.cancel()
                raise HTTPException(status_code=499, detail="Client Closed Request")
            await asyncio.sleep(0.02)

        return await schedule_task
    except AdmissionRejectedException as e:
        headers = {
            "Retry-After": str(max(1, int(e.retry_after))),
            "X-Correlation-ID": corr_id,
        }
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": "Rate limit / SLA protection shed",
                "correlation_id": corr_id,
                "status": e.status.value,
                "reason": e.reason,
                "retry_after_seconds": e.retry_after,
            },
            headers=headers,
        )
    except GPUOutOfMemoryError as e:
        headers = {"Retry-After": "2", "X-Correlation-ID": corr_id}
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "GPU Out of Memory",
                "correlation_id": corr_id,
                "reason": str(e),
                "retry_after_seconds": 2.0,
            },
            headers=headers,
        )


@app.post("/v1/completions", response_model=CompletionResponse)
async def v1_completions(req: CompletionRequest, request: Request):
    """
    OpenAI-compatible text completions endpoint with disconnect detection.
    """
    policy = get_policy()
    corr_id = getattr(request.state, "correlation_id", f"cmpl-{uuid.uuid4().hex[:12]}")
    priority = RequestPriority.from_str(req.priority or "normal")

    inf_req = InferenceRequest(
        prompt=req.prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        priority=priority,
        request_id=corr_id,
    )

    schedule_task = asyncio.create_task(policy.schedule(inf_req))

    try:
        while not schedule_task.done():
            if await request.is_disconnected():
                logger.warning("Client disconnected from completion %s. Reclaiming slot.", corr_id)
                await policy.cancel_request(corr_id, reason="Client HTTP connection disconnected")
                schedule_task.cancel()
                raise HTTPException(status_code=499, detail="Client Closed Request")
            await asyncio.sleep(0.02)

        inf_res = await schedule_task
        return CompletionResponse(
            id=f"cmpl-{inf_res.request_id}",
            created=int(time.time()),
            model=req.model or "llama-3.2-1b",
            choices=[
                CompletionChoice(
                    text=inf_res.response,
                    index=0,
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=max(1, len(req.prompt.split())),
                completion_tokens=inf_res.tokens_generated,
                total_tokens=max(1, len(req.prompt.split())) + inf_res.tokens_generated,
            ),
            latency_seconds=inf_res.latency_seconds,
            sla_met=inf_res.sla_met,
        )
    except AdmissionRejectedException as e:
        headers = {
            "Retry-After": str(max(1, int(e.retry_after))),
            "X-Correlation-ID": corr_id,
        }
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": "SLA admission shed",
                "correlation_id": corr_id,
                "status": e.status.value,
                "reason": e.reason,
                "retry_after_seconds": e.retry_after,
            },
            headers=headers,
        )
    except GPUOutOfMemoryError as e:
        headers = {"Retry-After": "2", "X-Correlation-ID": corr_id}
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "GPU Out of Memory",
                "correlation_id": corr_id,
                "reason": str(e),
                "retry_after_seconds": 2.0,
            },
            headers=headers,
        )


@app.get("/stats")
async def get_stats():
    """
    Telemetry and performance statistics for the active scheduler engine.
    """
    policy = get_policy()
    stats = policy.get_stats()
    result = stats.to_dict()
    result["uptime_seconds"] = round(time.time() - _start_time, 2)
    return JSONResponse(content=result)


@app.get("/metrics", response_class=PlainTextResponse)
async def get_prometheus_metrics():
    """
    Exposes metrics in standard Prometheus exposition format for scrapers/Grafana.
    Includes Phase 3 robustness counters.
    """
    policy = get_policy()
    stats = policy.get_stats()

    lines = [
        "# HELP velocityllm_active_requests Currently executing requests",
        "# TYPE velocityllm_active_requests gauge",
        f"velocityllm_active_requests {stats.active_requests}",
        "",
        "# HELP velocityllm_queued_requests Requests waiting in admission queue",
        "# TYPE velocityllm_queued_requests gauge",
        f"velocityllm_queued_requests {stats.queued_requests}",
        "",
        "# HELP velocityllm_effective_concurrency Current adaptive batch concurrency limit",
        "# TYPE velocityllm_effective_concurrency gauge",
        f"velocityllm_effective_concurrency {stats.effective_concurrency_limit}",
        "",
        "# HELP velocityllm_total_completed Total successfully completed inference requests",
        "# TYPE velocityllm_total_completed counter",
        f"velocityllm_total_completed {stats.total_completed}",
        "",
        "# HELP velocityllm_total_tokens_generated Cumulative generated tokens",
        "# TYPE velocityllm_total_tokens_generated counter",
        f"velocityllm_total_tokens_generated {stats.total_tokens_generated}",
        "",
        "# HELP velocityllm_sla_breaches Cumulative requests exceeding target SLA",
        "# TYPE velocityllm_sla_breaches counter",
        f"velocityllm_sla_breaches {stats.sla_breach_count}",
        "",
        "# HELP velocityllm_total_rejected Requests rejected by admission controller (HTTP 429)",
        "# TYPE velocityllm_total_rejected counter",
        f"velocityllm_total_rejected {stats.total_rejected}",
        "",
        "# HELP velocityllm_burst_shed_count Requests dropped specifically by burst-shedding backpressure",
        "# TYPE velocityllm_burst_shed_count counter",
        f"velocityllm_burst_shed_count {stats.burst_shed_count}",
        "",
        "# HELP velocityllm_client_disconnects Requests aborted due to client disconnect with reclaimed slot",
        "# TYPE velocityllm_client_disconnects counter",
        f"velocityllm_client_disconnects {stats.client_disconnects_count}",
        "",
        "# HELP velocityllm_oom_recoveries Count of GPU OOM events recovered without crash",
        "# TYPE velocityllm_oom_recoveries counter",
        f"velocityllm_oom_recoveries {stats.oom_recoveries_count}",
        "",
        "# HELP velocityllm_validation_errors Count of malformed/adversarial requests rejected",
        "# TYPE velocityllm_validation_errors counter",
        f"velocityllm_validation_errors {stats.validation_errors_count}",
        "",
        "# HELP velocityllm_p99_latency_seconds Observed 99th percentile end-to-end latency",
        "# TYPE velocityllm_p99_latency_seconds gauge",
        f"velocityllm_p99_latency_seconds {stats.p99_latency_seconds:.4f}",
        "",
        "# HELP velocityllm_gpu_utilization_percent GPU SM utilization percentage",
        "# TYPE velocityllm_gpu_utilization_percent gauge",
        f"velocityllm_gpu_utilization_percent {stats.gpu_utilization_percent:.1f}",
        "",
        "# HELP velocityllm_gpu_memory_used_mb GPU memory currently in use (MB)",
        "# TYPE velocityllm_gpu_memory_used_mb gauge",
        f"velocityllm_gpu_memory_used_mb {stats.gpu_memory_used_mb}",
        "",
    ]
    return "\n".join(lines)


@app.get("/health")
async def health():
    """Health and readiness check endpoint."""
    return {
        "status": "healthy",
        "policy": _config.policy,
        "backend": "mock" if _config.use_mock_backend else "vllm",
        "model": _config.model_path,
        "uptime_seconds": round(time.time() - _start_time, 2),
    }


def create_app(config: Optional[ServerConfig] = None) -> FastAPI:
    """Factory to create and configure the FastAPI application."""
    if config:
        set_config(config)
    return app


def load_config_file(path):
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def load_config_file(path):
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def main():
    parser = argparse.ArgumentParser(description="VelocityLLM Dynamic Serving Engine")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML config file. CLI flags override values from this file.",
    )
    parser.add_argument(
        "--policy",
        choices=["dynamic", "static"],
        default="dynamic",
        help="Scheduler policy to run (default: dynamic)",
    )
    parser.add_argument(
        "--model-path",
        default="/home/kartiklin/velocityllm/models/llama-3.2-1b",
        help="Path or HuggingFace identifier for LLM weights",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host address to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use simulated MockBackend (for testing/development without GPU)",
    )
    parser.add_argument(
        "--sla-ms",
        type=float,
        default=3000.0,
        help="Target SLA deadline in milliseconds (default: 3000ms)",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=32,
        help="Maximum concurrency ceiling for adaptive controller",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Maximum model context length (must not exceed the model's own max_position_embeddings)",
    )
    parser.add_argument(
        "--burst-shed-ratio",
        type=float,
        default=0.70,
        help="Queue depth ratio to begin shedding LOW priority traffic (default: 0.70)",
    )
    parser.add_argument(
        "--structured-logs",
        action="store_true",
        help="Format logs as structured JSON lines for production ingestion",
    )
    args = parser.parse_args()

    if args.config:
        file_defaults = load_config_file(args.config)
        for key, value in file_defaults.items():
            arg_key = key.replace("-", "_")
            if hasattr(args, arg_key):
                default_val = parser.get_default(arg_key)
                if getattr(args, arg_key) == default_val:
                    setattr(args, arg_key, value)

    cfg = ServerConfig(
        policy=args.policy,
        model_path=args.model_path,
        use_mock_backend=args.mock,
        host=args.host,
        port=args.port,
        target_sla_ms=args.sla_ms,
        max_concurrency=args.max_concurrency,
        max_model_len=args.max_model_len,
        burst_shed_queue_ratio=args.burst_shed_ratio,
        enable_structured_logging=args.structured_logs,
    )
    set_config(cfg)

    uvicorn.run(app, host=cfg.host, port=cfg.port)


if __name__ == "__main__":
    main()
