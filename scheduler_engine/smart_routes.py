"""
VelocityLLM - Smart Scheduler HTTP Routes
Everything the dashboard needs to show the 10 features, backed only by data the
scheduler actually produced:

  GET   /smart/state             full snapshot (capacity bars, tokens/KV, buckets, lanes, calibration, metrics)
  GET   /smart/queue             queue with live slack / score terms
  GET   /smart/trace             recent decision events  (?limit&event&request_id&since_seq)
  GET   /smart/trace/{id}        every decision made about one request
  GET   /smart/events            Server-Sent Events: live trace (+ periodic state with ?state=1)
  GET   /smart/metrics           Prometheus text (goodput, TTFT/ITL, rejections by reason, ...)
  GET   /smart/config            current tunables
  PATCH /smart/config            change tunables live (e.g. {"short_prompt_first": false})
"""

import asyncio
import json
import time
from typing import Any, Callable, Dict

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from scheduler_engine.smart_policy import SmartBatchPolicy


def build_smart_router(get_policy: Callable[[], Any]) -> APIRouter:
    router = APIRouter(prefix="/smart", tags=["smart-scheduler"])

    def smart() -> SmartBatchPolicy:
        policy = get_policy()
        if not isinstance(policy, SmartBatchPolicy):
            raise HTTPException(
                status_code=404,
                detail="Smart scheduler endpoints require the server to run with --policy smart.",
            )
        return policy

    @router.get("/state")
    async def state():
        return JSONResponse(content=smart().get_smart_state())

    @router.get("/queue")
    async def queue():
        return {"queue": smart().queue.get_queue_snapshot()}

    @router.get("/trace")
    async def trace(limit: int = 200, event: str = "", request_id: str = "", since_seq: int = 0):
        p = smart()
        return {
            "events": p.trace.recent(limit=limit, event=event or None, request_id=request_id or None, since_seq=since_seq),
            "summary": p.trace.summary(),
        }

    @router.get("/trace/{request_id}")
    async def trace_for_request(request_id: str):
        events = smart().trace.for_request(request_id)
        if not events:
            raise HTTPException(status_code=404, detail=f"No decision events recorded for {request_id}.")
        return {"request_id": request_id, "events": events}

    @router.get("/events")
    async def events(request: Request, state: int = 0, state_interval: float = 1.0):
        p = smart()

        async def gen():
            q = p.trace.subscribe()
            last_state = 0.0
            try:
                yield "retry: 2000\n\n"
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        ev = await asyncio.wait_for(q.get(), timeout=0.5)
                        yield f"event: trace\ndata: {json.dumps(ev)}\n\n"
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
                    if state and time.time() - last_state >= state_interval:
                        last_state = time.time()
                        yield f"event: state\ndata: {json.dumps(p.get_smart_state())}\n\n"
            finally:
                p.trace.unsubscribe(q)

        return StreamingResponse(
            gen(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @router.get("/config")
    async def get_cfg():
        return smart().smart.to_dict()

    @router.patch("/config")
    async def patch_cfg(changes: Dict[str, Any]):
        p = smart()
        applied = p.smart.update(changes)
        if "aimd_ramp_step" in applied:
            p.adaptive_controller.increase_step = max(1, int(p.smart.aimd_ramp_step))
        p.trace.emit(
            "config_change", decision="config_change", reason_code="operator",
            reason=f"Runtime config changed: {applied}", applied=applied,
        )
        return {"applied": applied, "config": p.smart.to_dict()}

    @router.get("/metrics", response_class=PlainTextResponse)
    async def prom():
        p = smart()
        m = p.metrics()
        cap = p.capacity
        kv = p.kv_snapshot
        lines = []

        def g(name: str, help_: str, value: Any, typ: str = "gauge", labels: str = "") -> None:
            if value is None:
                return
            lines.extend([f"# HELP {name} {help_}", f"# TYPE {name} {typ}", f"{name}{labels} {value}", ""])

        g("velocityllm_smart_offered_total", "Requests offered (accepted + rejected)", m["offered"], "counter")
        g("velocityllm_smart_rejected_total", "Requests rejected by early rejection / policy", m["rejected"], "counter")
        g("velocityllm_smart_within_sla_total", "Completed requests that met their SLA", m["within_sla"], "counter")
        g("velocityllm_smart_sla_goodput_rps", "SLA-compliant completions per second (60s)", m["sla_goodput_rps_60s"])
        g("velocityllm_smart_token_throughput_tps", "Generated tokens per second (60s)", m["token_throughput_tps_60s"])
        g("velocityllm_smart_sla_goodput_of_offered", "within_sla / offered", m["sla_goodput_of_offered"])
        for name, key in (("ttft", "ttft_ms"), ("itl", "itl_ms")):
            for q in ("p50", "p95", "p99"):
                g(f"velocityllm_smart_{name}_{q}_ms", f"{name.upper()} {q} in ms", m[key][q])
        for reason, n in m["rejected_by_reason"].items():
            lines.extend([f'velocityllm_smart_rejected_by_reason_total{{reason="{reason}"}} {n}'])
        lines.append("")
        g("velocityllm_smart_memory_limit_capacity", "Memory-safe concurrency", cap.memory_limit_capacity)
        g("velocityllm_smart_sla_limit_capacity", "SLA-safe concurrency", cap.sla_limit_capacity)
        g("velocityllm_smart_operating_limit", "Dispatcher concurrency ceiling", cap.operating_limit)
        if kv:
            g("velocityllm_smart_kv_current_tokens", "Estimated KV tokens in use", kv.current_tokens)
            g("velocityllm_smart_kv_predicted_future_tokens", "Predicted future KV tokens", kv.predicted_future_tokens)
            g("velocityllm_smart_kv_future_pressure", "Predicted future KV pressure vs safe budget", round(kv.future_pressure, 4))
        g("velocityllm_smart_be_throttled", "1 if best-effort lane is throttled", int(p.be_throttled))
        g("velocityllm_smart_calibration_exec_mape", "Exec-latency prediction error (EMA)", round(p.calibrator.exec_mape, 4))
        return "\n".join(lines)

    return router
