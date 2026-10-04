import argparse
import csv
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "probe"


def start_server(args, limit):
    OUT.mkdir(parents=True, exist_ok=True)
    log = open(OUT / "server.log", "ab")
    cmd = [
        sys.executable, "-m", "scheduler_engine.server", "--policy", "static",
        "--initial-concurrency", str(limit), "--max-concurrency", str(max(limit, 32)),
        "--sla-ms", str(args.sla * 1000), "--port", str(args.port), "--model-path", args.model_path,
        "--max-model-len", str(args.max_model_len),
    ]
    if args.mock:
        cmd.append("--mock")
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=log)
    deadline = time.time() + args.startup_timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            sys.exit(f"server exited early, see {OUT / 'server.log'}")
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


def load(args, users, name):
    out = OUT / f"{name}.csv"
    subprocess.run(
        [sys.executable, str(ROOT / "load_generator" / "generate_load.py"),
         "--url", f"http://localhost:{args.port}/generate", "--num-requests", str(users),
         "--concurrency", str(users), "--pattern", "flood", "--seed", str(args.seed),
         "--output", str(out), "--label", name],
        cwd=ROOT, check=True, stdout=subprocess.DEVNULL,
    )
    summary = json.loads(out.with_name(out.stem + "_summary.json").read_text())
    with open(out, newline="") as f:
        tokens = sum(int(float(r["tokens_generated"] or 0)) for r in csv.DictReader(f) if r["status"] == "200")
    wall = summary["wall_time_seconds"]
    return {
        "tokens": tokens,
        "wall": wall,
        "tok_s": tokens / wall if wall else 0.0,
        "p50": summary["p50_latency_seconds"],
        "p99": summary["p99_latency_seconds"],
        "gpu": summary["avg_gpu_util_percent"],
    }


def main():
    p = argparse.ArgumentParser(description="Measure static-scheduler throughput at different fixed limits")
    p.add_argument("--model-path", required=True)
    p.add_argument("--limits", type=int, nargs="+", default=[8, 16, 32])
    p.add_argument("--users", type=int, nargs="+", default=[30, 150])
    p.add_argument("--sla", type=float, default=8.0)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--startup-timeout", type=int, default=240)
    p.add_argument("--settle", type=float, default=5.0)
    p.add_argument("--mock", action="store_true")
    args = p.parse_args()

    table = []
    for limit in args.limits:
        print(f"limit {limit}: starting static server")
        proc = start_server(args, limit)
        try:
            load(args, 8, "warmup")
            for users in args.users:
                time.sleep(args.settle)
                result = load(args, users, f"limit{limit}_users{users}")
                table.append((limit, users, result))
                print(f"  {users} users done: {result['tok_s']:.0f} tokens/s")
        finally:
            stop_server(proc, args.settle)

    print(f"\n{'limit':>6}{'users':>7}{'wall s':>9}{'tokens':>8}{'tok/s':>8}{'p50 s':>8}{'p99 s':>8}{'gpu %':>7}")
    for limit, users, r in table:
        gpu = "-" if r["gpu"] is None else f"{r['gpu']:.0f}"
        print(f"{limit:>6}{users:>7}{r['wall']:>9.1f}{r['tokens']:>8}{r['tok_s']:>8.0f}{r['p50']:>8.2f}{r['p99']:>8.2f}{gpu:>7}")
    (OUT / "probe_summary.json").write_text(json.dumps(
        [{"limit": l, "users": u, **r} for l, u, r in table], indent=2))
    print(f"\nSaved: {OUT / 'probe_summary.json'}")


if __name__ == "__main__":
    main()
