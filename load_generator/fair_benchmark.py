import argparse
import csv
import json
import random
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "fair"
SCENARIOS = {"flood": [], "mixed": ["--mixed-prompts"]}
POLICIES = ("static", "dynamic")
GREEN, RED, DIM, BOLD, RESET = "\033[92m", "\033[91m", "\033[2m", "\033[1m", "\033[0m"


def paint(text, color):
    return f"{color}{text}{RESET}" if sys.stdout.isatty() else text


def parse_models(items):
    models = {}
    for item in items:
        name, _, path = item.partition("=")
        if not name or not path:
            sys.exit(f"--model expects NAME=PATH, got: {item}")
        models[name] = path
    return models


def context_limit(args, model_path):
    config = Path(model_path)
    if not config.is_absolute():
        config = ROOT / config
    try:
        native = int(json.loads((config / "config.json").read_text())["max_position_embeddings"])
    except (OSError, KeyError, ValueError):
        return args.max_model_len
    return min(args.max_model_len, native)


def start_server(args, policy, model_path):
    log = open(OUT_DIR / "server.log", "ab")
    cmd = [
        sys.executable, "-m", "scheduler_engine.server",
        "--policy", policy, "--sla-ms", str(args.sla_ms),
        "--port", str(args.port), "--model-path", model_path,
        "--max-model-len", str(context_limit(args, model_path)),
    ]
    if args.mock:
        cmd.append("--mock")
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=log)
    deadline = time.time() + args.startup_timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            sys.exit(f"server exited early, see {OUT_DIR / 'server.log'}")
        try:
            with urllib.request.urlopen(f"http://localhost:{args.port}/health", timeout=2):
                return proc
        except OSError:
            time.sleep(2)
    proc.terminate()
    sys.exit("server did not become healthy in time")


def stop_server(proc, settle):
    proc.terminate()
    try:
        proc.wait(timeout=40)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    time.sleep(settle)


def run_load(args, scenario, output, seed, count):
    cmd = [
        sys.executable, str(ROOT / "load_generator" / "generate_load.py"),
        "--url", f"http://localhost:{args.port}/generate",
        "--num-requests", str(count), "--concurrency", str(count),
        "--pattern", "flood", "--seed", str(seed), "--output", str(output),
        "--label", output.stem, *SCENARIOS[scenario],
    ]
    subprocess.run(cmd, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)


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


def collect(args, models):
    raw = OUT_DIR / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for name, path in models.items():
        for rnd in range(1, args.repeats + 1):
            order = list(POLICIES)
            random.Random(f"{name}-{rnd}").shuffle(order)
            pending = [(p, s) for p in order for s in SCENARIOS if not (raw / f"{name}_{p}_{s}_r{rnd}.csv").exists()]
            for policy in order:
                todo = [s for p, s in pending if p == policy]
                if not todo:
                    continue
                print(f"{name}  round {rnd}/{args.repeats}  {policy}: starting server")
                proc = start_server(args, policy, path)
                try:
                    run_load(args, "flood", raw / "warmup.csv", 1, 8)
                    time.sleep(args.settle)
                    for scenario in todo:
                        out = raw / f"{name}_{policy}_{scenario}_r{rnd}.csv"
                        run_load(args, scenario, out, args.seed + rnd, args.requests)
                        print(f"    {scenario}: saved {out.name}")
                        time.sleep(args.settle)
                finally:
                    stop_server(proc, args.settle)


def aggregate(args, models):
    raw = OUT_DIR / "raw"
    sla_s = args.sla_ms / 1000
    table = {}
    for name in models:
        for scenario in SCENARIOS:
            for policy in POLICIES:
                runs = [score(p, sla_s) for p in sorted(raw.glob(f"{name}_{policy}_{scenario}_r*.csv"))]
                if not runs:
                    continue
                table[(name, scenario, policy)] = {
                    "runs": len(runs),
                    **{k: statistics.median(r[k] for r in runs) for k in runs[0]},
                    "p99_min": min(r["p99"] for r in runs),
                    "p99_max": max(r["p99"] for r in runs),
                }
    return table


def report(args, table):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = ["model", "scenario", "policy", "runs", "offered", "served", "rejected", "p50", "p95", "p99", "p99_min", "p99_max",
              "within_sla_of_served", "within_sla_of_offered", "tokens_per_sec", "zero_token_share"]
    with open(OUT_DIR / "fair_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for (model, scenario, policy), v in table.items():
            w.writerow([model, scenario, policy] + [round(v[k], 3) for k in fields[3:]])
    with open(OUT_DIR / "fair_summary.json", "w") as f:
        json.dump({"sla_ms": args.sla_ms, "repeats": args.repeats, "requests": args.requests,
                   "rows": [{"model": m, "scenario": s, "policy": p, **v} for (m, s, p), v in table.items()]}, f, indent=2)

    head = f"{'model':<11}{'load':<7}{'policy':<9}{'served':>7}{'p50 s':>8}{'p99 s':>8}{'in SLA*':>9}{'in SLA**':>10}{'tok/s':>8}"
    print("\n" + paint(f"Fair benchmark  |  SLA {args.sla_ms / 1000:g} s for both policies  |  median of {args.repeats} runs", BOLD))
    print(paint(head, DIM))
    for (model, scenario, policy), v in table.items():
        print(f"{model:<11}{scenario:<7}{policy:<9}{v['served']:>4.0f}/{v['offered']:<2.0f}{v['p50']:>8.2f}{v['p99']:>8.2f}"
              f"{v['within_sla_of_served']:>8.0f}%{v['within_sla_of_offered']:>9.0f}%{v['tokens_per_sec']:>8.0f}")
        if policy == "dynamic" and (model, scenario, "static") in table:
            s = table[(model, scenario, "static")]
            change = 100 * (v["p99"] - s["p99"]) / s["p99"] if s["p99"] else 0.0
            tput = 100 * (v["tokens_per_sec"] - s["tokens_per_sec"]) / s["tokens_per_sec"] if s["tokens_per_sec"] else 0.0
            color = GREEN if change < 0 else RED
            print("  " + paint(f"p99 {change:+.0f}%   tokens/s {tput:+.0f}%   rejected {v['rejected']:.0f}", color))
        if v["zero_token_share"] > 50:
            print("  " + paint(f"warning: {v['zero_token_share']:.0f}% of replies had zero tokens", RED))
    print(paint("* share of served requests within SLA    ** share of all offered requests within SLA (rejected = miss)", DIM))
    print(f"\nSaved: {OUT_DIR / 'fair_summary.csv'}")


def main():
    p = argparse.ArgumentParser(description="Fair static vs dynamic benchmark")
    p.add_argument("--model", action="append", required=True, help="NAME=PATH, repeat for each model")
    p.add_argument("--sla-ms", type=float, default=8000.0)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--requests", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--settle", type=float, default=5.0)
    p.add_argument("--startup-timeout", type=float, default=600.0)
    p.add_argument("--out-name", default="fair")
    p.add_argument("--mock", action="store_true")
    p.add_argument("--report-only", action="store_true")
    args = p.parse_args()

    global OUT_DIR
    OUT_DIR = ROOT / "results" / args.out_name
    models = parse_models(args.model)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not args.report_only:
        collect(args, models)
    report(args, aggregate(args, models))


if __name__ == "__main__":
    main()
