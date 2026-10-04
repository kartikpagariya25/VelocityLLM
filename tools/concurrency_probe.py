import argparse
import asyncio
import json
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from control_service.loadgen import build_plan, run_load  # noqa: E402
from control_service.scoring import score  # noqa: E402

OUT = ROOT / "results" / "probe"


def start_server(args, policy, limit, name):
    OUT.mkdir(parents=True, exist_ok=True)
    log_path = OUT / f"server_{name.replace(' ', '_')}.log"
    log = open(log_path, "wb")
    cmd = [
        sys.executable, "-m", "scheduler_engine.server", "--policy", policy,
        "--sla-ms", str(args.sla * 1000), "--port", str(args.port), "--model-path", args.model_path,
        "--max-model-len", str(args.max_model_len),
    ]
    if policy == "static":
        cmd += ["--initial-concurrency", str(limit), "--max-concurrency", str(max(limit, 32))]
    if args.mock:
        cmd.append("--mock")
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=log)
    deadline = time.time() + args.startup_timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            sys.exit(f"server exited early, see {log_path}")
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


def load(args, users, scenario="flood", preset="medium", max_tokens=None):
    plan = build_plan(scenario, users, args.seed, preset, None, max_tokens=max_tokens)
    rows, wall = asyncio.run(run_load(f"http://localhost:{args.port}", plan, 600.0))
    reasons = Counter(r.get("reject_reason") or "other" for r in rows if r["status"] == "rejected")
    return score(rows, args.sla, wall), wall, dict(reasons)


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
    p.add_argument("--scenario", choices=["flood", "mixed"], default="flood")
    p.add_argument("--preset", choices=["short", "medium", "long"], default="medium")
    p.add_argument("--dynamic", action="store_true", help="also run the dynamic policy on the same traffic")
    p.add_argument("--mock", action="store_true")
    args = p.parse_args()

    configs = [("static", limit) for limit in args.limits] + ([("dynamic", None)] if args.dynamic else [])
    table = []
    for policy, limit in configs:
        name = f"static limit {limit}" if policy == "static" else "dynamic"
        print(f"{name}: starting server")
        proc = start_server(args, policy, limit, name)
        try:
            load(args, 8, "flood", "short", 50)
            for users in args.users:
                time.sleep(args.settle)
                result, wall, reasons = load(args, users, args.scenario, args.preset)
                result["decline_reasons"] = reasons
                table.append((name, users, wall, result))
                print(f"  {users} users done: {result['tokens_per_s']:.0f} tokens/s")
        finally:
            stop_server(proc, args.settle)

    print(f"\n{'setup':<16}{'users':>6}{'wall s':>8}{'served':>8}{'declined':>9}{'on time':>9}{'tok/s':>7}{'goodput':>9}{'p50 s':>7}{'p99 s':>7}")
    for name, users, wall, r in table:
        on_time = round(r["within_sla_offered"] * r["offered"])
        print(f"{name:<16}{users:>6}{wall:>8.1f}{r['served']:>8}{r['rejected']:>9}{on_time:>9}{r['tokens_per_s']:>7.0f}"
              f"{r['goodput_tokens_per_s']:>9.0f}{r['p50_s']:>7.2f}{r['p99_s']:>7.2f}")
    for name, users, _wall, r in table:
        if r["decline_reasons"]:
            print(f"  {name}, {users} users, declined because: " + ", ".join(f"{k} {v}" for k, v in sorted(r["decline_reasons"].items())))
    (OUT / "probe_summary.json").write_text(json.dumps(
        [{"setup": n, "users": u, "wall_s": w, **r} for n, u, w, r in table], indent=2))
    print(f"\nSaved: {OUT / 'probe_summary.json'}")


if __name__ == "__main__":
    main()
