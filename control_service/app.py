import asyncio
import csv
import io
import json
import time
import zipfile
from contextlib import asynccontextmanager
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from .config import MAX_USERS, Settings
from .gpu import gpu_info
from .live import MAX_PER_DEVICE, DeviceSpec, LiveConfig, LiveError, LiveHub, choose_model, lan_addresses
from .loadgen import SCENARIOS
from .vision_load import IMAGE_MIXES
from .models import describe, discover
from .preflight import run_checks
from .runner import Orchestrator
from .runs import RunStore

VERSION = "1.0.0"
ROWS = ("p50_s", "p95_s", "p99_s", "tokens_per_s", "served", "offered", "rejected", "errors",
        "within_sla_served", "within_sla_offered", "zero_token_share")


class JoinRequest(BaseModel):
    name: str = Field(min_length=1, max_length=24)


class SpecRequest(BaseModel):
    device_id: str = Field(min_length=1, max_length=64)
    spec: DeviceSpec


class DeviceRef(BaseModel):
    device_id: str = Field(min_length=1, max_length=64)


class RunRequest(BaseModel):
    model: str = Field(min_length=1, max_length=120)
    scenario: Literal["flood", "mixed", "steady", "burst"] = "flood"
    sla_ms: float = Field(default=8000, ge=1000, le=60000)
    requests: int = Field(default=30, ge=1, le=MAX_USERS)
    repeats: int = Field(default=1, ge=1, le=5)
    mode: Literal["sequential", "side_by_side"] = "sequential"
    prompt_preset: Literal["short", "medium", "long", "custom"] = "medium"
    prompt_text: Optional[str] = Field(default=None, max_length=500)
    image_mix: Literal["small", "mixed", "large"] = "mixed"
    seed: int = Field(default=42, ge=0, le=1_000_000)
    levels: Optional[list[int]] = Field(default=None, max_length=8)
    mock: Optional[bool] = None

    @field_validator("levels")
    @classmethod
    def check_levels(cls, v):
        if v is None:
            return v
        if any(n < 1 or n > MAX_USERS for n in v):
            raise ValueError(f"each level must be between 1 and {MAX_USERS}")
        return sorted(set(v))

    @field_validator("prompt_text")
    @classmethod
    def clean(cls, v):
        return v.strip() if v else v


def sse(record):
    return f"id: {record['id']}\nevent: {record['event']}\ndata: {json.dumps(record['data'])}\n\n"


class SiteFiles(StaticFiles):
    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        if path.startswith("assets/"):
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        else:
            response.headers["Cache-Control"] = "no-cache"
        return response


def create_app(settings: Settings) -> FastAPI:
    store = RunStore(settings.results_dir)
    orch = Orchestrator(settings, store)
    hub = LiveHub()

    @asynccontextmanager
    async def lifespan(_):
        try:
            store.load_disk()
        except OSError:
            pass
        yield
        if hub.timer and not hub.timer.done():
            hub.timer.cancel()
        run = store.runs.get(store.active) if store.active else None
        if run and run.task:
            run.task.cancel()
            await asyncio.gather(run.task, return_exceptions=True)

    app = FastAPI(title="VelocityLLM Control Service", version=VERSION, lifespan=lifespan)

    origins = list(settings.cors_origins)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if "*" in origins else origins,
        allow_origin_regex=None if "*" in origins else r"https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$",
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def auth(request: Request, call_next):
        if settings.api_key and request.url.path.startswith("/api") and request.url.path != "/api/health" and request.method != "OPTIONS":
            if request.headers.get("x-api-key") != settings.api_key and request.query_params.get("key") != settings.api_key:
                return JSONResponse({"detail": "Invalid or missing API key"}, status_code=401)
        return await call_next(request)

    @app.exception_handler(RequestValidationError)
    async def invalid(request: Request, exc: RequestValidationError):
        parts = [f"{'.'.join(str(x) for x in e['loc'][1:])}: {e['msg']}" for e in exc.errors()]
        return JSONResponse({"detail": "; ".join(parts) or "Invalid request"}, status_code=422)

    @app.exception_handler(Exception)
    async def crash(request: Request, exc: Exception):
        return JSONResponse({"detail": f"Control service error: {type(exc).__name__}: {exc}"}, status_code=500)

    def get_run(run_id):
        run = store.runs.get(run_id)
        if not run:
            raise HTTPException(404, f"Run '{run_id}' not found")
        return run

    @app.get("/api/health")
    async def health():
        return {"ok": True, "version": VERSION, "mock": settings.mock, "busy": store.active}

    @app.get("/api/preflight")
    async def preflight():
        return await run_checks(settings, store.active)

    @app.get("/api/models")
    async def models():
        return [describe(n, p, settings.mock) for n, p in discover(settings).items()]

    @app.get("/api/scenarios")
    async def scenarios():
        return [{"id": k, "label": v["label"], "description": v["description"]} for k, v in SCENARIOS.items()]

    @app.get("/api/image-mixes")
    async def image_mixes():
        return [{"id": k, "label": v["label"], "description": v["description"]} for k, v in IMAGE_MIXES.items()]

    @app.post("/api/runs", status_code=202)
    async def create_run(req: RunRequest):
        if req.mode != "sequential":
            raise HTTPException(422, "Side by side needs two GPUs. Use the sequential mode on a single GPU.")
        if req.prompt_preset == "custom" and not req.prompt_text:
            raise HTTPException(422, "A custom prompt needs prompt_text")
        if orch.busy():
            raise HTTPException(409, f"Run {store.active} is already in progress")
        mock = settings.mock if req.mock is None else req.mock
        report = await run_checks(settings, None, mock)
        models_ok = {m["id"] for m in report["models"] if m["available"]}
        if req.model not in models_ok:
            raise HTTPException(422, f"Model '{req.model}' is not available. Usable models: {', '.join(sorted(models_ok)) or 'none'}")
        blocking = [c for c in report["checks"] if not c["ok"] and c["level"] == "error"]
        if blocking:
            raise HTTPException(412, "Pre-flight failed: " + "; ".join(f"{c['label']}: {c['detail']}" for c in blocking))
        cfg = req.model_dump()
        cfg["mock"] = mock
        try:
            run = await orch.start(cfg)
        except RuntimeError as e:
            raise HTTPException(409, str(e))
        return {"run_id": run.id}

    @app.get("/api/runs")
    async def list_runs():
        runs = sorted(store.runs.values(), key=lambda r: r.created, reverse=True)
        return [r.summary() for r in runs]

    @app.get("/api/runs/{run_id}")
    async def run_state(run_id: str):
        run = get_run(run_id)
        events = store.events_of(run)
        phases = [e for e in events if e["event"] == "phase"]
        return {**run.summary(), "phase": phases[-1]["data"]["phase"] if phases else None,
                "event_count": len(events), "results": run.results, "warning": run.warning}

    @app.get("/api/runs/{run_id}/events")
    async def events(run_id: str, request: Request, last_event_id: Optional[int] = None):
        run = get_run(run_id)
        header = request.headers.get("last-event-id")
        try:
            start = int(header) if header else (last_event_id or 0)
        except ValueError:
            start = 0
        all_events = store.events_of(run)

        async def gen():
            i = start
            yield "retry: 1500\n\n"
            while True:
                while i < len(all_events):
                    yield sse(all_events[i])
                    i += 1
                if run.finished and i >= len(all_events):
                    return
                if run.cond is None:
                    return
                async with run.cond:
                    if i >= len(all_events) and not run.finished:
                        try:
                            await asyncio.wait_for(run.cond.wait(), 15)
                        except asyncio.TimeoutError:
                            pass
                if await request.is_disconnected():
                    return
                if i >= len(all_events) and not run.finished:
                    yield ": keepalive\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/api/runs/{run_id}/cancel")
    async def cancel(run_id: str):
        run = get_run(run_id)
        if run.finished or not run.task:
            raise HTTPException(409, "Run already finished")
        run.task.cancel()
        return {"run_id": run.id, "status": "cancelling"}

    @app.get("/api/runs/{run_id}/recording")
    async def recording(run_id: str):
        run = get_run(run_id)
        if not run.results:
            raise HTTPException(409, "Only completed runs have a recording")
        body = {"format": "velocityllm-recording-1", "id": run.id, "config": run.config, "results": run.results, "events": store.events_of(run)}
        return Response(json.dumps(body), media_type="application/json",
                        headers={"Content-Disposition": f'attachment; filename="{run.id}.recording.json"'})

    @app.get("/api/benchmarks/export")
    async def export_all():
        done = [r for r in store.runs.values() if r.results]
        if not done:
            raise HTTPException(404, "No completed runs to export yet")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            summary = io.StringIO()
            w = csv.writer(summary)
            w.writerow(["run_id", "date", "model", "gpu", "users", "scenario", "sla_s", "policy", *ROWS])
            for r in sorted(done, key=lambda x: x.created):
                gpu = (r.results.get("gpu") or {}).get("name", "mock" if r.config.get("mock") else "")
                for pol in ("static", "dynamic", "smart"):
                    res = r.results["results"].get(pol)
                    if not res:
                        continue
                    w.writerow([r.id, time.strftime("%Y-%m-%d %H:%M", time.localtime(r.created)), r.config["model"], gpu,
                                max(r.results.get("levels") or [r.config["requests"]]), r.config["scenario"], r.config["sla_ms"] / 1000, pol,
                                *[res[k] for k in ROWS]])
            z.writestr("benchmarks_summary.csv", summary.getvalue())
            for r in done:
                for f in sorted(r.dir.iterdir()):
                    if f.is_file():
                        z.write(f, f"{r.id}/{f.name}")
        return Response(buf.getvalue(), media_type="application/zip",
                        headers={"Content-Disposition": 'attachment; filename="velocityllm-benchmarks.zip"'})

    @app.get("/api/runs/{run_id}/export")
    async def export(run_id: str, format: Literal["csv", "json"] = "json"):
        run = get_run(run_id)
        if not run.results:
            raise HTTPException(409, "This run has no results yet")
        if format == "json":
            return Response(json.dumps(run.results, indent=2), media_type="application/json",
                            headers={"Content-Disposition": f'attachment; filename="{run.id}.json"'})
        buf = io.StringIO()
        w = csv.writer(buf)
        pols = [p for p in ("static", "dynamic", "smart") if p in run.results["results"]]
        w.writerow(["metric", *pols])
        for k in ROWS:
            w.writerow([k, *[run.results["results"][p][k] for p in pols]])
        return Response(buf.getvalue(), media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="{run.id}.csv"'})

    @app.post("/api/replay/{run_id}")
    async def replay(run_id: str, speed: float = 1.0):
        source = get_run(run_id)
        if not source.results:
            raise HTTPException(409, "Only completed runs can be replayed")
        try:
            run = await orch.start_replay(source, min(max(speed, 0.5), 20.0))
        except RuntimeError as e:
            raise HTTPException(409, str(e))
        except LookupError as e:
            raise HTTPException(404, str(e))
        return {"run_id": run.id}

    LIVE_COLUMNS = ("offered", "served", "rejected", "errors", "p50_s", "p95_s", "p99_s", "tokens", "tokens_per_s", "within_sla_served")

    async def launch_live():
        if orch.busy():
            raise LiveError(409, "The engine is busy with a run, try again when it finishes")
        devices = hub.ready_devices()
        if not devices:
            raise LiveError(422, "No device has sent its traffic yet")
        total = hub.total_requests(devices)
        if total > MAX_USERS:
            raise LiveError(422, f"The devices ask for {total} requests together, the limit is {MAX_USERS}")
        conf = hub.config
        mock = settings.mock
        report = await run_checks(settings, None, mock)
        usable = [m for m in report["models"] if m["available"]]
        model = choose_model(usable, conf.model)
        if not model:
            raise LiveError(422, "No usable model was found on this machine")
        blocking = [c for c in report["checks"] if not c["ok"] and c["level"] == "error"]
        if blocking:
            raise LiveError(412, "Pre-flight failed: " + "; ".join(f"{c['label']}: {c['detail']}" for c in blocking))
        cfg = {
            "model": model, "scenario": "live", "sla_ms": conf.sla_ms, "requests": total, "repeats": conf.repeats,
            "mode": "sequential", "prompt_preset": "medium", "prompt_text": None, "image_mix": "mixed", "seed": conf.seed,
            "levels": None, "mock": mock, "policies": list(conf.policies),
            "live": {"devices": [{"name": d.name, "spec": dict(d.spec)} for d in devices]},
        }
        try:
            run = await orch.start(cfg)
        except RuntimeError as e:
            raise LiveError(409, str(e))
        hub.last_run_id = run.id
        hub.last_error = None
        hub.reset()
        return run, [d["name"] for d in cfg["live"]["devices"]], total

    def raise_http(e: LiveError):
        raise HTTPException(e.status, str(e))

    async def auto_start(delay):
        try:
            await asyncio.sleep(delay)
            if orch.busy():
                hub.fire_at = None
                hub.waiting = True
                while orch.busy():
                    await asyncio.sleep(0.5)
                hub.waiting = False
                hub.fire_at = time.time() + hub.config.window_s
                await asyncio.sleep(hub.config.window_s)
            hub.fire_at = None
            await launch_live()
        except asyncio.CancelledError:
            raise
        except LiveError as e:
            hub.last_error = str(e)
        except Exception as e:
            hub.last_error = f"{type(e).__name__}: {e}"
        finally:
            hub.fire_at = None
            hub.waiting = False

    def arm():
        if hub.timer and not hub.timer.done():
            hub.timer.cancel()
        delay = hub.config.window_s
        hub.fire_at = None if orch.busy() else time.time() + delay
        hub.last_error = None
        hub.timer = asyncio.create_task(auto_start(delay))

    @app.get("/api/live/state")
    async def live_state(device_id: Optional[str] = None):
        me = hub.get(device_id) if device_id else None
        active = store.runs.get(store.active) if store.active else None
        last = store.runs.get(hub.last_run_id) if hub.last_run_id else None
        current = active or last
        models = [m for m in (describe(n, p, settings.mock) for n, p in discover(settings).items()) if m["available"]]
        gpu = None if settings.mock else await gpu_info(1.5)
        return {
            "devices": hub.snapshot(),
            "me": me.public() if me else None,
            "joined": bool(me) if device_id else None,
            "models": models,
            "model": choose_model(models, hub.config.model),
            "gpu": (gpu or {}).get("name") or ("Mock engine" if settings.mock else "PC GPU"),
            "config": hub.config.model_dump(),
            "countdown_s": hub.countdown(),
            "waiting": hub.waiting,
            "error": hub.last_error,
            "run": {"id": current.id, "status": current.status, "config": {k: current.config.get(k) for k in ("model", "sla_ms", "policies", "repeats", "requests")}} if current and current.config.get("live") else None,
            "busy": bool(active),
            "hosts": lan_addresses(),
            "port": settings.port,
            "mock": settings.mock,
            "limits": {"per_device": MAX_PER_DEVICE, "total": MAX_USERS},
        }

    @app.post("/api/live/join")
    async def live_join(req: JoinRequest):
        try:
            device = hub.join(req.name)
        except ValueError as e:
            raise HTTPException(422, str(e))
        except OverflowError as e:
            raise HTTPException(409, str(e))
        return {"device_id": device.id, "name": device.name}

    def store_spec(req: SpecRequest):
        device = hub.get(req.device_id)
        if not device:
            raise HTTPException(404, "This device is not part of the session, join again")
        if req.spec.prompt_preset == "custom" and not (req.spec.prompt_text or "").strip():
            raise HTTPException(422, "A custom prompt needs text")
        device.spec = req.spec.model_dump()
        return device

    @app.post("/api/live/spec")
    async def live_spec(req: SpecRequest):
        return store_spec(req).public()

    @app.post("/api/live/send")
    async def live_send(req: SpecRequest):
        device = store_spec(req)
        armed = hub.config.auto_start
        if armed:
            arm()
        return {"device": device.public(), "armed": armed, "window_s": hub.config.window_s, "queued": orch.busy()}

    @app.post("/api/live/config")
    async def live_config(conf: LiveConfig):
        if conf.model and conf.model not in {n for n in discover(settings)}:
            raise HTTPException(422, f"Model '{conf.model}' is not available")
        hub.config = conf
        if not conf.auto_start and hub.timer and not hub.timer.done():
            hub.timer.cancel()
            hub.fire_at = None
        return conf.model_dump()

    @app.post("/api/live/leave")
    async def live_leave(req: DeviceRef):
        hub.leave(req.device_id)
        return {"ok": True}

    @app.post("/api/live/reset")
    async def live_reset():
        if store.active:
            raise HTTPException(409, "A run is in progress")
        if hub.timer and not hub.timer.done():
            hub.timer.cancel()
        hub.fire_at = None
        hub.reset()
        return {"ok": True}

    @app.post("/api/live/start", status_code=202)
    async def live_start():
        if hub.timer and not hub.timer.done():
            hub.timer.cancel()
            hub.fire_at = None
        try:
            run, names, total = await launch_live()
        except LiveError as e:
            raise_http(e)
        return {"run_id": run.id, "devices": names, "total": total}

    @app.get("/api/live/runs")
    async def live_runs():
        runs = sorted((r for r in store.runs.values() if r.config.get("live")), key=lambda r: r.created, reverse=True)
        return [r.summary() for r in runs]

    @app.get("/api/live/runs/{run_id}/report.csv")
    async def live_report(run_id: str):
        run = get_run(run_id)
        if not run.results or not run.config.get("live"):
            raise HTTPException(409, "This is not a finished live run")
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["policy", "device", *LIVE_COLUMNS])
        for pol, overall in run.results["results"].items():
            w.writerow([pol, "ALL", *[overall.get(k, "") for k in LIVE_COLUMNS]])
            for dev, res in (run.results.get("per_device", {}).get(pol) or {}).items():
                w.writerow([pol, dev, *[res.get(k, "") for k in LIVE_COLUMNS]])
        return Response(buf.getvalue(), media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="{run.id}-live.csv"'})

    if settings.frontend_dir.is_dir():
        app.mount("/", SiteFiles(directory=settings.frontend_dir, html=True), name="site")
    return app
