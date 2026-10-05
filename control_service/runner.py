import asyncio
import json
import secrets
import time

from .config import MAX_USERS
from .environment import capture
from .engine import Engine, EngineError, free_port
from .gpu import gpu_info
from .loadgen import build_plan, run_load, write_csv
from .logparse import classify, is_noise
from .models import describe, discover
from .preflight import run_checks
from .runs import Run
from .scoring import diagnose, median_of, score
from .smartcfg import write_smart_config

POLICIES = ("static", "dynamic", "smart")

# Smart explains every decision in its trace; the Console shows them like the other engines' log lines.
TRACE_CATEGORY = {
    "admit": "admission", "reject": "admission", "reorder": "admission",
    "dispatch": "lifecycle", "complete": "lifecycle", "cancel": "lifecycle",
    "capacity_change": "controller", "throttle": "controller",
    "calibration": "system", "config_change": "system",
}


def trace_line(ev):
    """(level, category, request_id, message) for one Smart decision-trace event."""
    kind = ev.get("event")
    rid = ev.get("request_id")
    reason = ev.get("reason") or ""
    code = ev.get("reason_code")
    d = ev.get("details") or {}
    level = "INFO"
    if kind == "admit":
        slack = ev.get("slack_ms")
        msg = f"Request {rid} admitted [{ev.get('traffic_class')}, {ev.get('bucket')}, ~{ev.get('token_estimate')} tok" + (f", slack {slack / 1000:.2f}s" if slack is not None else "") + f"]: {reason}"
    elif kind == "reject":
        level = "WARNING"
        msg = f"Request {rid} rejected [{code}]: {reason}"
    elif kind == "reorder":
        msg = f"Request {rid} moved ahead of {d.get('overtook')} [{code}]"
    elif kind == "dispatch":
        msg = f"Request {rid} started after {d.get('queue_ms', 0):.0f} ms in queue"
    elif kind == "complete":
        msg = f"Request {rid} completed in {d.get('actual_latency_ms', 0) / 1000:.2f}s ({d.get('tokens', 0)} tokens): {reason}"
    elif kind == "throttle":
        level = "WARNING" if ev.get("decision") == "throttle_best_effort" else "INFO"
        msg = f"Smart pressure control: {reason}"
    elif kind == "capacity_change":
        msg = f"Smart capacity planner: {reason}"
    else:
        msg = f"Smart {kind}: {reason}"
    return level, TRACE_CATEGORY.get(kind, "system"), rid, msg


class RunFailed(Exception):
    def __init__(self, message, detail=None):
        super().__init__(message)
        self.detail = [str(x) for x in (detail or [])]


def new_run_id():
    return f"run-{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}"


class Orchestrator:
    def __init__(self, settings, store):
        self.settings = settings
        self.store = store

    def busy(self):
        return self.store.active is not None

    async def start(self, cfg):
        if self.busy():
            raise RuntimeError(f"Run {self.store.active} is already in progress")
        rid = new_run_id()
        run = Run(rid, cfg, self.settings.results_dir / rid)
        self.store.runs[run.id] = run
        self.store.active = run.id
        run.task = asyncio.create_task(self._guard(run, self._execute(run)))
        return run

    async def start_replay(self, source, speed):
        if self.busy():
            raise RuntimeError(f"Run {self.store.active} is already in progress")
        events = self.store.events_of(source)
        if not events:
            raise LookupError("The recorded run has no stored events")
        cfg = dict(source.config, recorded=True, source_run=source.id)
        rid = new_run_id()
        run = Run(rid, cfg, self.settings.results_dir / rid)
        self.store.runs[run.id] = run
        self.store.active = run.id
        run.task = asyncio.create_task(self._guard(run, self._replay(run, events, speed)))
        return run

    async def _guard(self, run, coro):
        run.status = "running"
        try:
            await coro
            run.status = "completed"
        except asyncio.CancelledError:
            run.status = "cancelled"
            await run.emit("log", self._line(run, None, "WARNING", "cancel", "Run cancelled by the user. Engine stopped."))
            await run.emit("done", {"run_id": run.id, "status": "cancelled"})
        except RunFailed as e:
            run.status = "failed"
            await run.emit("error", {"message": str(e), "detail": e.detail})
            await run.emit("done", {"run_id": run.id, "status": "failed"})
        except Exception as e:
            run.status = "failed"
            await run.emit("error", {"message": f"Unexpected error in the control service: {type(e).__name__}: {e}", "detail": []})
            await run.emit("done", {"run_id": run.id, "status": "failed"})
        finally:
            run.close()
            if self.store.active == run.id:
                self.store.active = None

    def _line(self, run, policy, level, category, message, rid=None):
        return {
            "policy": policy, "ts": time.time(), "t": round(time.monotonic() - run.t0, 3),
            "level": level, "category": category, "request_id": rid, "message": message,
        }

    async def log(self, run, policy, message, category="system", level="INFO", rid=None):
        await run.emit("log", self._line(run, policy, level, category, message, rid))

    async def phase(self, run, name, policy=None, repeat=1):
        await run.emit("phase", {"phase": name, "policy": policy, "repeat": repeat, "ts": time.time()})

    async def _replay(self, run, events, speed):
        prev = 0.0
        for rec in events:
            gap = max(0.0, (rec["at"] - prev) / max(speed, 0.1))
            prev = rec["at"]
            if gap:
                await asyncio.sleep(min(gap, 2.0))
            data = dict(rec["data"])
            if rec["event"] == "info":
                data["recorded"] = True
            if rec["event"] == "done":
                data["run_id"] = run.id
            await run.emit(rec["event"], data)

    async def _settle(self, seconds):
        await asyncio.sleep(0.3 if self.settings.mock else seconds)

    async def _execute(self, run):
        cfg = run.config
        mock = cfg["mock"]
        sla_s = cfg["sla_ms"] / 1000
        await self.phase(run, "preflight")
        report = await run_checks(self.settings, None, mock)
        for c in report["checks"]:
            level = "INFO" if c["ok"] else ("ERROR" if c["level"] == "error" else "WARNING")
            await self.log(run, None, f"Pre-flight: {c['label']}: {c['detail']}", "system", level)
        models = discover(self.settings)
        model_path = models.get(cfg["model"])
        if model_path is None:
            raise RunFailed(f"Unknown model '{cfg['model']}'. Available: {', '.join(models) or 'none'}")
        info = describe(cfg["model"], model_path, mock)
        if not info["available"]:
            raise RunFailed(f"Model folder for '{cfg['model']}' was not found at {model_path}")
        if not report["ok"]:
            raise RunFailed("Pre-flight failed", [f"{c['label']}: {c['detail']}" for c in report["checks"] if not c["ok"] and c["level"] == "error"])
        env = await capture()
        levels_cfg = cfg.get("levels") or [cfg["requests"]]
        await run.emit("info", {
            "gpu": report["gpu"], "model": cfg["model"], "mock": mock, "sla_s": sla_s, "users": max(levels_cfg), "levels": levels_cfg,
            "environment": env,
            "scenario": cfg["scenario"], "repeats": cfg["repeats"], "seed": cfg["seed"], "recorded": False,
        })
        if run.warning:
            await self.log(run, None, run.warning, "system", "WARNING")

        levels = cfg.get("levels") or [cfg["requests"]]
        primary = max(levels)
        collected = {p: {n: [] for n in levels} for p in POLICIES}
        for rnd in range(1, cfg["repeats"] + 1):
            order = POLICIES if rnd % 2 else POLICIES[::-1]
            for policy in order:
                legs = await self._leg(run, cfg, policy, rnd, model_path, mock, sla_s, levels)
                for n, (rows, wall) in legs.items():
                    collected[policy][n].append(score(rows, sla_s, wall))
                    name = f"raw_{policy}_r{rnd}.csv" if len(levels) == 1 else f"raw_{policy}_n{n}_r{rnd}.csv"
                    try:
                        write_csv(run.dir / name, rows)
                    except OSError as e:
                        await self.log(run, policy, f"Could not save raw CSV: {e}", "system", "WARNING")

        await self.phase(run, "scoring")
        per_level = {p: {n: median_of(collected[p][n]) for n in levels} for p in POLICIES}
        if len(levels) > 1 and cfg["repeats"] > 1:
            for p in POLICIES:
                for n in levels:
                    await run.emit("level_result", {"policy": p, "level": n, "final": True, **per_level[p][n]})
        results = {p: per_level[p][primary] for p in POLICIES}
        for p in POLICIES:
            await run.emit("result", {"policy": p, "level": primary, "warnings": diagnose(results[p]), **results[p]})
        payload = {
            "config": cfg, "status": "completed", "created": run.created, "gpu": report["gpu"],
            "model_path": model_path, "levels": levels, "results": results,
            "warnings": {p: diagnose(results[p]) for p in POLICIES},
            "per_level": {p: {str(n): v for n, v in per_level[p].items()} for p in POLICIES},
            "per_repeat": {p: {str(n): v for n, v in collected[p].items()} for p in POLICIES},
            "environment": env,
        }
        run.results = payload
        try:
            (run.dir / "results.json").write_text(json.dumps(payload, indent=2))
        except OSError as e:
            await self.log(run, None, f"Could not save results.json: {e}", "system", "WARNING")
        await self.phase(run, "done")
        await run.emit("done", {"run_id": run.id, "status": "completed"})

    async def _leg(self, run, cfg, policy, rnd, model_path, mock, sla_s, levels):
        s = self.settings
        await self.phase(run, f"{policy}_start", policy, rnd)
        port = free_port(s.engine_base_port)

        async def on_line(line):
            if is_noise(line):
                return
            level, category, rid, message = classify(line)
            await run.emit("log", self._line(run, policy, level, category, message, rid))

        smart_cfg = None
        if policy == "smart":
            smart_cfg, st = write_smart_config(model_path, run.dir)
            if st:
                await self.log(run, policy, f"Smart settings for this model: {st['kv_bytes_per_token']} bytes of KV cache per token, {st['model_weights_mb']} MB for weights and runtime")
            elif not mock:
                await self.log(run, policy, "Smart settings could not be derived from the model folder, using defaults (memory estimates may be off)", "system", "WARNING")

        engine = Engine(s, policy, model_path, cfg["sla_ms"], port, mock, on_line, run.dir / f"server_{policy}_r{rnd}.log", config_path=smart_cfg)
        await self.log(run, policy, "python3 -m scheduler_engine.server " + " ".join(engine.command()[3:]))
        poller = watcher = None
        out = {}
        try:
            try:
                await engine.start()
            except EngineError as e:
                raise RunFailed(f"{policy.capitalize()} engine failed to start: {e}", e.tail)
            await self.log(run, policy, f"Engine healthy on port {port}")
            await self.phase(run, f"{policy}_warmup", policy, rnd)
            warm = build_plan("flood", 8, 1, "short", None, max_tokens=50)
            for _ in range(2):
                warm_rows, _wall = await run_load(engine.base, warm, s.request_timeout, None, "warm")
                if warm_rows and all(r["status"] == "error" for r in warm_rows):
                    raise RunFailed(f"Warm-up failed on the {policy} engine: {warm_rows[0]['error']}", engine.tail)
            await self.log(run, policy, "Warm-up finished, results discarded from scoring")
            if policy == "smart":
                first = await engine.trace(0, 1)
                engine.trace_cursor = ((first or {}).get("summary") or {}).get("last_seq", 0)
            watcher = asyncio.create_task(engine.wait_exit())
            for n in levels:
                await self._settle(s.settle_seconds)
                base = await engine.stats() or {}
                plan = build_plan(cfg["scenario"], n, cfg["seed"] + rnd, cfg["prompt_preset"], cfg["prompt_text"])
                await run.emit("phase", {"phase": f"{policy}_load", "policy": policy, "repeat": rnd, "level": n, "ts": time.time()})
                if len(levels) > 1:
                    await self.log(run, policy, f"Load level: {n} users")
                loop = asyncio.get_running_loop()
                load_start = loop.time()

                async def on_request(row):
                    await run.emit("request", {
                        "policy": policy, "request_id": row["request_id"], "arrival_s": round(row["arrival_s"], 3),
                        "start_s": None if row["start_s"] is None else round(row["start_s"], 3),
                        "end_s": round(row["end_s"], 3), "status": row["status"], "http_status": row["http_status"],
                        "tokens": row["tokens"], "priority": row["priority"], "reject_reason": row["reject_reason"],
                        "retry_after_s": row["retry_after_s"], "error": row["error"],
                    })

                poller = asyncio.create_task(self._poll(run, engine, policy, base, load_start))
                load = asyncio.create_task(run_load(engine.base, plan, s.request_timeout, on_request, "r"))
                done, _ = await asyncio.wait({load, watcher}, return_when=asyncio.FIRST_COMPLETED)
                if load not in done:
                    load.cancel()
                    await asyncio.gather(load, return_exceptions=True)
                    raise RunFailed(f"The {policy} engine crashed during the load phase (exit code {watcher.result()})", engine.tail)
                rows, wall = load.result()
                poller.cancel()
                await asyncio.gather(poller, return_exceptions=True)
                poller = None
                await self._poll_once(run, engine, policy, base, load_start)
                errors = [r for r in rows if r["status"] == "error"]
                if errors:
                    await self.log(run, policy, f"{len(errors)} of {len(rows)} requests failed, first error: {errors[0]['error']}", "error", "WARNING")
                result = score(rows, sla_s, wall)
                notes = diagnose(result)
                for note in notes:
                    await self.log(run, policy, f"Level {n}: {note}", "error", "WARNING")
                await run.emit("level_result", {"policy": policy, "level": n, "repeat": rnd, "warnings": notes, **result})
                out[n] = (rows, wall)
            return out
        finally:
            for t in (poller, watcher):
                if t:
                    t.cancel()
            await asyncio.gather(*(t for t in (poller, watcher) if t), return_exceptions=True)
            await engine.stop()
            await self.log(run, policy, "Engine stopped")
            await self._settle(s.settle_seconds)

    async def _poll(self, run, engine, policy, base, load_start):
        while True:
            await self._poll_once(run, engine, policy, base, load_start)
            await asyncio.sleep(0.5)

    async def _forward_trace(self, run, engine, policy):
        body = await engine.trace(engine.trace_cursor)
        for ev in (body or {}).get("events", []):
            engine.trace_cursor = max(engine.trace_cursor, ev.get("seq", 0))
            level, category, rid, message = trace_line(ev)
            await run.emit("log", self._line(run, policy, level, category, message, rid))

    async def _poll_once(self, run, engine, policy, base, load_start):
        if policy == "smart":
            await self._forward_trace(run, engine, policy)
        st = await engine.stats()
        if not st:
            return
        used, total = st.get("gpu_memory_used_mb") or 0, st.get("gpu_memory_total_mb") or 0
        if not total:
            g = await gpu_info(2.0)
            used, total = (g["used_mb"], g["total_mb"]) if g else (0, 0)
        await run.emit("metrics", {
            "policy": policy, "ts": time.time(), "t": round(asyncio.get_running_loop().time() - load_start, 3),
            "active": st.get("active_requests", 0), "queued": st.get("queued_requests", 0),
            "concurrency_limit": st.get("effective_concurrency_limit", 0),
            "completed": st.get("total_completed", 0) - base.get("total_completed", 0),
            "rejected": st.get("total_rejected", 0) - base.get("total_rejected", 0),
            "tokens": st.get("total_tokens_generated", 0) - base.get("total_tokens_generated", 0),
            "gpu_mem_used_mb": used, "gpu_mem_total_mb": total,
        })
