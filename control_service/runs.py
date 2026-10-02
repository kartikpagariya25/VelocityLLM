import asyncio
import json
import time
from pathlib import Path


class Run:
    def __init__(self, run_id, config, directory: Path):
        self.id = run_id
        self.config = config
        self.dir = directory
        self.status = "queued"
        self.events = []
        self.cond = asyncio.Condition()
        self.finished = False
        self.created = time.time()
        self.t0 = time.monotonic()
        self.task = None
        self.results = None
        self.warning = None
        self._file = None
        try:
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "config.json").write_text(json.dumps(config, indent=2))
            self._file = open(directory / "events.ndjson", "a", buffering=1)
        except OSError as e:
            self.warning = f"Run history cannot be saved: {e}"

    async def emit(self, event, data):
        record = {"id": len(self.events) + 1, "event": event, "data": data, "at": round(time.monotonic() - self.t0, 3)}
        if self._file:
            try:
                self._file.write(json.dumps(record, default=str) + "\n")
            except (OSError, ValueError, TypeError):
                self._file = None
        async with self.cond:
            self.events.append(record)
            if event == "done":
                self.finished = True
            self.cond.notify_all()

    def close(self):
        if self._file:
            try:
                self._file.close()
            except OSError:
                pass
            self._file = None

    def summary(self):
        return {
            "id": self.id,
            "status": self.status,
            "created": self.created,
            "config": self.config,
            "recorded": bool(self.config.get("recorded")),
            "has_results": self.results is not None,
        }


class RunStore:
    def __init__(self, root: Path):
        self.root = root
        self.runs = {}
        self.active = None

    def load_disk(self):
        if not self.root.is_dir():
            return
        for d in sorted(self.root.iterdir()):
            rj = d / "results.json"
            if not rj.exists() or d.name in self.runs:
                continue
            try:
                data = json.loads(rj.read_text())
                run = Run.__new__(Run)
                run.id, run.config, run.dir = d.name, data["config"], d
                run.status, run.results = data.get("status", "completed"), data
                run.events, run.finished, run.created = None, True, data.get("created", d.stat().st_mtime)
                run.cond, run.task, run.warning, run._file = None, None, None, None
                self.runs[d.name] = run
            except (OSError, ValueError, KeyError):
                continue

    def events_of(self, run):
        if run.events is None:
            events = []
            try:
                for line in (run.dir / "events.ndjson").read_text().splitlines():
                    if line.strip():
                        events.append(json.loads(line))
            except (OSError, ValueError):
                pass
            run.events = events
        return run.events
