"""Terminal benchmark: the same traffic against each scheduling policy, on every model you give it.

    velocityllm bench --models-dir ./models --requests 100 --repeats 3

    import velocityllm
    rows = velocityllm.benchmark({"qwen": "/models/qwen2.5-1.5b"}, requests=100, repeats=3)
"""
import argparse
import csv
import importlib.util
import json
import os
import random
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

SCENARIOS = {"flood": [], "mixed": ["--mixed-prompts"]}
KNOWN_POLICIES = ("static", "dynamic", "smart")
GREEN, RED, DIM, BOLD, RESET = "\033[92m", "\033[91m", "\033[2m", "\033[1m", "\033[0m"


def paint(text, color):
    return f"{color}{text}{RESET}" if sys.stdout.isatty() else text


def package_dir(name):
    spec = importlib.util.find_spec(name)
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit(f"velocityllm: the '{name}' package is missing from this install")
    return Path(list(spec.submodule_search_locations)[0])


def find_models(models_dir, only=None, limit=None):
    base = Path(models_dir).expanduser()
    if not base.is_dir():
        raise SystemExit(f"velocityllm bench: models folder not found: {base}")
    found = {p.name: str(p) for p in sorted(base.iterdir()) if (p / "config.json").is_file()}
    if only:
        missing = [n for n in only if n not in found]
        if missing:
            raise SystemExit(f"velocityllm bench: not in {base}: {', '.join(missing)} (available: {', '.join(found) or 'none'})")
        found = {n: found[n] for n in only}
    if limit:
        found = dict(list(found.items())[:limit])
    if not found:
        raise SystemExit(f"velocityllm bench: no model folders (with config.json) in {base}")
    return found


def context_limit(model_path, ceiling):
    try:
        native = int(json.loads((Path(model_path) / "config.json").read_text())["max_position_embeddings"])
    except (OSError, KeyError, ValueError):
        return ceiling
    return min(ceiling, native)


class Bench:
    def __init__(self, models, policies, scenarios, sla_ms, repeats, requests, seed, port, max_model_len,
                 settle, startup_timeout, out_dir, mock, quiet):
        self.models, self.policies, self.scenarios = models, tuple(policies), tuple(scenarios)
        self.sla_ms, self.repeats, self.requests, self.seed = sla_ms, repeats, requests, seed
        self.port, self.max_model_len, self.settle, self.startup_timeout = port, max_model_len, settle, startup_timeout
        self.out_dir, self.mock, self.quiet = Path(out_dir), mock, quiet
        self.raw = self.out_dir / "raw"
        root = str(package_dir("scheduler_engine").parent)
        self.env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [root, os.environ.get("PYTHONPATH")]))}

    def say(self, text):
        if not self.quiet:
            print(text, flush=True)

    def start_server(self, policy, model_path):
        log = open(self.out_dir / "server.log", "ab")
        cmd = [
            sys.executable, "-m", "scheduler_engine.server",
            "--policy", policy, "--sla-ms", str(self.sla_ms), "--port", str(self.port),
            "--model-path", model_path, "--max-model-len", str(context_limit(model_path, self.max_model_len)),
        ]
        if self.mock:
            cmd.append("--mock")
        proc = subprocess.Popen(cmd, cwd=self.out_dir, env=self.env, stdout=log, stderr=log)
        deadline = time.time() + self.startup_timeout
        while time.time() < deadline:
            if proc.poll() is not None:
                raise SystemExit(f"velocityllm bench: server exited early, see {self.out_dir / 'server.log'}")
            try:
                with urllib.request.urlopen(f"http://localhost:{self.port}/health", timeout=2):
                    return proc
            except OSError:
                time.sleep(1 if self.mock else 2)
        proc.terminate()
        raise SystemExit("velocityllm bench: server did not become healthy in time")

    def stop_server(self, proc):
        proc.terminate()
        try:
            proc.wait(timeout=40)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        time.sleep(self.settle)

    def run_load(self, scenario, output, seed, count):
        script = package_dir("load_generator") / "generate_load.py"
        cmd = [
            sys.executable, str(script), "--url", f"http://localhost:{self.port}/generate",
            "--num-requests", str(count), "--concurrency", str(count), "--pattern", "flood",
            "--seed", str(seed), "--output", str(output), "--label", output.stem, *SCENARIOS[scenario],
        ]
        subprocess.run(cmd, cwd=self.out_dir, env=self.env, check=True, stdout=subprocess.DEVNULL)

    def collect(self):
        self.raw.mkdir(parents=True, exist_ok=True)
        for name, path in self.models.items():
            for rnd in range(1, self.repeats + 1):
                order = list(self.policies)
                random.Random(f"{name}-{rnd}").shuffle(order)
                for policy in order:
                    todo = [s for s in self.scenarios if not (self.raw / f"{name}_{policy}_{s}_r{rnd}.csv").exists()]
                    if not todo:
                        continue
                    self.say(f"{name}  round {rnd}/{self.repeats}  {policy}: starting server")
                    proc = self.start_server(policy, path)
                    try:
                        self.run_load("flood", self.raw / "warmup.csv", 1, 8)
                        time.sleep(self.settle)
                        for scenario in todo:
                            out = self.raw / f"{name}_{policy}_{scenario}_r{rnd}.csv"
                            self.run_load(scenario, out, self.seed + rnd, self.requests)
                            self.say(f"    {scenario}: saved {out.name}")
                            time.sleep(self.settle)
                    finally:
                        self.stop_server(proc)

    def aggregate(self):
        sla_s = self.sla_ms / 1000
        rows = []
        for name in self.models:
            for scenario in self.scenarios:
                for policy in self.policies:
                    runs = [score(p, sla_s) for p in sorted(self.raw.glob(f"{name}_{policy}_{scenario}_r*.csv"))]
                    if not runs:
                        continue
                    row = {"model": name, "scenario": scenario, "policy": policy, "runs": len(runs)}
                    row.update({k: statistics.median(r[k] for r in runs) for k in runs[0]})
                    row["p99_min"] = min(r["p99"] for r in runs)
                    row["p99_max"] = max(r["p99"] for r in runs)
                    rows.append(row)
        return rows

    def write(self, rows):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if rows:
            with open(self.out_dir / "summary.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for r in rows)
        with open(self.out_dir / "summary.json", "w") as f:
            json.dump({"sla_ms": self.sla_ms, "repeats": self.repeats, "requests": self.requests, "rows": rows}, f, indent=2)

    def show(self, rows):
        print("\n" + paint(f"VelocityLLM benchmark  |  SLA {self.sla_ms / 1000:g} s  |  {self.requests} requests  |  median of {self.repeats} runs", BOLD))
        head = f"{'model':<18}{'load':<7}{'policy':<9}{'served':>8}{'p50 s':>8}{'p95 s':>8}{'p99 s':>8}{'on time':>9}{'tok/s':>8}"
        print(paint(head, DIM))
        index = {(r["model"], r["scenario"], r["policy"]): r for r in rows}
        for r in rows:
            print(f"{r['model']:<18}{r['scenario']:<7}{r['policy']:<9}{r['served']:>4.0f}/{r['offered']:<3.0f}"
                  f"{r['p50']:>8.2f}{r['p95']:>8.2f}{r['p99']:>8.2f}{r['within_sla_of_offered']:>8.0f}%{r['tokens_per_sec']:>8.0f}")
            base = index.get((r["model"], r["scenario"], "static"))
            if r["policy"] != "static" and base:
                print("  " + delta_line(r, base))
            if r["zero_token_share"] > 50:
                print("  " + paint(f"warning: {r['zero_token_share']:.0f}% of replies had zero tokens", RED))
        print(paint("on time = share of all offered requests answered within the SLA (a rejected request counts as a miss)", DIM))
        print(f"\nSaved: {self.out_dir / 'summary.csv'}")


def pct(new, old):
    return 100 * (new - old) / old if old else 0.0


def delta_line(row, base):
    p50, p95, p99 = (pct(row[k], base[k]) for k in ("p50", "p95", "p99"))
    tput = pct(row["tokens_per_sec"], base["tokens_per_sec"])
    good = p99 < 0 and tput > 0
    text = f"vs static:  p50 {p50:+.0f}%   p95 {p95:+.0f}%   p99 {p99:+.0f}%   tokens/s {tput:+.0f}%   rejected {row['rejected']:.0f}"
    return paint(text, GREEN if good else RED)


def nearest_rank(values, q):
    return values[min(int(len(values) * q), len(values) - 1)] if values else 0.0


def score(path, sla_s):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    served = [r for r in rows if r["status"] == "200"]
    lat = sorted(float(r["client_latency_seconds"]) for r in served)
    tokens = sum(int(float(r["tokens_generated"] or 0)) for r in served)
    span = max((float(r["client_latency_seconds"]) for r in rows), default=0.0)
    within = sum(1 for v in lat if v <= sla_s)
    zero = sum(1 for r in served if int(float(r["tokens_generated"] or 0)) == 0)
    return {
        "offered": len(rows),
        "served": len(served),
        "rejected": sum(1 for r in rows if r["status"] == "429"),
        "p50": nearest_rank(lat, 0.5),
        "p95": nearest_rank(lat, 0.95),
        "p99": nearest_rank(lat, 0.99),
        "within_sla_of_served": 100 * within / len(served) if served else 0.0,
        "within_sla_of_offered": 100 * within / len(rows) if rows else 0.0,
        "tokens_per_sec": tokens / span if span else 0.0,
        "zero_token_share": 100 * zero / len(served) if served else 0.0,
    }


def resolve_models(models, models_dir):
    if isinstance(models, dict):
        return dict(models)
    if isinstance(models, str):
        models = [models]
    names = list(models) if models else None
    if names and all(os.sep in n or n.startswith(("~", ".")) for n in names):
        return {Path(n).expanduser().name: str(Path(n).expanduser()) for n in names}
    return find_models(models_dir or os.path.join(os.getcwd(), "models"), names)


def benchmark(models=None, policies=("static", "dynamic"), scenarios=("flood",), sla_ms=8000.0, repeats=3,
              requests=100, seed=42, port=8000, max_model_len=4096, settle=5.0, startup_timeout=600.0,
              out_dir=None, mock=False, quiet=False, report_only=False, models_dir=None):
    """Run the benchmark and return one dict per (model, scenario, policy) with median p50/p95/p99, tokens/s and on-time share.

    models: {name: path}, a list of model folder names (looked up in models_dir), a list of paths, or None for every model in models_dir.
    """
    bad = [p for p in policies if p not in KNOWN_POLICIES]
    if bad:
        raise ValueError(f"unknown policy: {', '.join(bad)} (choose from {', '.join(KNOWN_POLICIES)})")
    out = (Path(out_dir).expanduser() if out_dir else Path.cwd() / "velocity_bench" / "latest").resolve()
    out.mkdir(parents=True, exist_ok=True)
    bench = Bench(resolve_models(models, models_dir), policies, scenarios, sla_ms, repeats, requests, seed, port, max_model_len,
                  settle, startup_timeout, out, mock, quiet)
    if not report_only:
        bench.collect()
    rows = bench.aggregate()
    bench.write(rows)
    if not quiet:
        bench.show(rows)
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(prog="velocityllm bench", description="Compare scheduling policies on one or more models")
    p.add_argument("--models-dir", default=None, help="folder with one sub-folder per model (default: ./models)")
    p.add_argument("--model", action="append", default=[], help="NAME=PATH, repeat for each model (alternative to --models-dir)")
    p.add_argument("--only", default="", help="comma-separated model folder names to use from --models-dir")
    p.add_argument("--limit", type=int, default=0, help="use at most this many models from --models-dir")
    p.add_argument("--policies", default="static,dynamic", help="comma-separated: static, dynamic, smart")
    p.add_argument("--scenario", default="flood", help="comma-separated: flood, mixed")
    p.add_argument("--sla-ms", type=float, default=8000.0)
    p.add_argument("--requests", type=int, default=100, help="simultaneous users per run")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--settle", type=float, default=5.0, help="seconds to let the GPU rest between runs")
    p.add_argument("--startup-timeout", type=float, default=600.0)
    p.add_argument("--out-dir", default=None, help="where to save results (default: ./velocity_bench/latest)")
    p.add_argument("--mock", action="store_true", help="simulated engine, no GPU needed")
    p.add_argument("--report-only", action="store_true", help="re-print the table from saved runs")
    args = p.parse_args(argv)

    models = {}
    for item in args.model:
        name, _, path = item.partition("=")
        if not name or not path:
            raise SystemExit(f"--model expects NAME=PATH, got: {item}")
        models[name] = path
    if not models:
        only = [n for n in args.only.split(",") if n]
        models = find_models(args.models_dir or os.path.join(os.getcwd(), "models"), only, args.limit)

    benchmark(
        models, policies=[x for x in args.policies.split(",") if x], scenarios=[x for x in args.scenario.split(",") if x],
        sla_ms=args.sla_ms, repeats=args.repeats, requests=args.requests, seed=args.seed, port=args.port,
        max_model_len=args.max_model_len, settle=args.settle, startup_timeout=args.startup_timeout,
        out_dir=args.out_dir, mock=args.mock, report_only=args.report_only,
    )
    return 0
