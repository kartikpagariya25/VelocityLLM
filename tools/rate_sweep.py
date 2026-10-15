"""
Request-rate sweep: replays one Poisson workload at increasing rates against VelocityLLM policies and any
external servers (vLLM, Triton), then writes CSV/JSON results, throughput-latency curves and a capacity table.

Local policies are started and stopped by this tool:
    python tools/rate_sweep.py --policies static,static-batch,dynamic --mock --mock-contention 0.5

External servers are started by you and listed as NAME=PROTOCOL@URL[#MODEL]:
    python tools/rate_sweep.py --policies dynamic --target vllm=openai@http://localhost:8001#meta-llama/Llama-3.2-1B
"""

import argparse
import asyncio
import csv
import json
import shlex
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from load_generator.stream_client import Target, build_workload, load_length_pairs, run_workload, sample_rows, summarize

GREEN, RED, CYAN, DIM, BOLD, RESET = "\033[92m", "\033[91m", "\033[96m", "\033[2m", "\033[1m", "\033[0m"
PALETTE = ["#1f6feb", "#d1453b", "#2a9d5c", "#b8860b", "#8250df", "#0e8a8a", "#6e7781"]
FIELDS = [
    "target", "rps", "offered", "served", "rejected", "errors", "requests_per_s", "tokens_per_s",
    "goodput_requests_per_s", "goodput_tokens_per_s", "sla_attainment_percent",
    "e2e_p50", "e2e_p95", "e2e_p99", "ttft_p50", "ttft_p95", "ttft_p99", "per_token_ms_mean", "ttft_is_estimate",
]


def paint(text: str, color: str) -> str:
    return f"{color}{text}{RESET}" if sys.stdout.isatty() else text


def start_server(args, policy: str, queue_order: str) -> subprocess.Popen:
    out = Path(args.out_dir)
    cmd = [
        sys.executable, "-m", "scheduler_engine.server",
        "--policy", policy, "--port", str(args.port), "--sla-ms", str(args.sla_ms),
        "--max-concurrency", str(args.max_concurrency), "--initial-concurrency", str(args.batch_size),
        "--queue-order", queue_order, "--max-batch-tokens", str(args.max_batch_tokens),
        "--static-batch-timeout-ms", str(args.static_batch_timeout_ms),
        "--model-path", args.model_path, "--max-model-len", str(args.max_model_len),
    ]
    if args.mock:
        cmd += ["--mock", "--mock-tps", str(args.mock_tps), "--mock-contention", str(args.mock_contention)]
    if args.chunked_prefill:
        cmd.append("--chunked-prefill")
    cmd += shlex.split(args.server_args)
    log = open(out / "server.log", "ab")
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=log)
    deadline = time.time() + args.startup_timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            sys.exit(f"server exited early, see {out / 'server.log'}")
        try:
            with urllib.request.urlopen(f"http://localhost:{args.port}/health", timeout=2):
                return proc
        except OSError:
            time.sleep(1)
    proc.terminate()
    sys.exit("server did not become healthy in time")


def stop_server(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    time.sleep(1)


def run_level(args, target: Target, rps: float, lengths) -> Dict:
    runs = [run_once(args, target, rps, lengths, args.seed + k, k) for k in range(args.repeats)]
    if len(runs) == 1:
        return runs[0]
    merged = dict(runs[0])
    for key, value in runs[0].items():
        if key not in ("target", "rps", "offered") and isinstance(value, (int, float)) and not isinstance(value, bool):
            merged[key] = statistics.median(r[key] for r in runs)
    merged["e2e_p99_min"] = min(r["e2e_p99"] for r in runs)
    merged["e2e_p99_max"] = max(r["e2e_p99"] for r in runs)
    return merged


def run_once(args, target: Target, rps: float, lengths, seed: int, repeat: int) -> Dict:
    count = max(args.min_requests, int(round(rps * args.duration)))
    plan = build_workload(count, rps, seed=seed, lengths=lengths, max_output_tokens=args.max_output_tokens)
    samples, wall = asyncio.run(run_workload(target, plan, api_key=args.api_key))
    raw = Path(args.out_dir) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    rows = sample_rows(samples)
    suffix = f"_r{repeat + 1}" if args.repeats > 1 else ""
    with open(raw / f"{target.name.replace(":", "-")}_rps{rps:g}{suffix}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(samples, wall, args.sla_ms / 1000.0, args.ttft_sla_ms / 1000.0 if args.ttft_sla_ms else None)
    return {"target": target.name, "rps": rps, **summary}


def capacity_table(rows: List[Dict], sla_s: float, reject_tolerance: float) -> Dict[str, Dict]:
    """Per target: highest offered rate whose p99 stays within the SLA with few rejections, and peak goodput."""
    result: Dict[str, Dict] = {}
    for name in dict.fromkeys(r["target"] for r in rows):
        mine = sorted((r for r in rows if r["target"] == name), key=lambda r: r["rps"])
        ok = [r for r in mine if r["served"] and r["e2e_p99"] <= sla_s and r["rejected"] <= reject_tolerance * r["offered"]]
        best = max(mine, key=lambda r: r["goodput_tokens_per_s"])
        result[name] = {
            "max_rps_within_p99_sla": max((r["rps"] for r in ok), default=0.0),
            "peak_goodput_tokens_per_s": best["goodput_tokens_per_s"],
            "peak_goodput_at_rps": best["rps"],
            "peak_tokens_per_s": max(r["tokens_per_s"] for r in mine),
        }
    return result


def write_results(args, rows: List[Dict], capacity: Dict[str, Dict]) -> None:
    out = Path(args.out_dir)
    with open(out / "sweep_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(FIELDS)
        for r in rows:
            writer.writerow([round(r[k], 4) if isinstance(r[k], float) else r[k] for k in FIELDS])
    with open(out / "sweep_summary.json", "w") as f:
        json.dump({"sla_ms": args.sla_ms, "seed": args.seed, "rows": rows, "capacity": capacity}, f, indent=2)


def plot(args, rows: List[Dict]) -> Optional[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping plots.")
        return None

    names = list(dict.fromkeys(r["target"] for r in rows))
    colors = {n: PALETTE[i % len(PALETTE)] for i, n in enumerate(names)}
    panels = [
        ("e2e_p99", "p99 end-to-end latency (s)", True),
        ("ttft_p99", "p99 time to first token (s)", False),
        ("tokens_per_s", "Throughput (tokens/s)", False),
        ("goodput_tokens_per_s", "Goodput: on-time tokens/s", False),
        ("sla_attainment_percent", "Offered requests answered within SLA (%)", False),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    flat = axes.flatten()
    for ax, (key, title, sla_line) in zip(flat, panels):
        for n in names:
            pts = sorted((r for r in rows if r["target"] == n), key=lambda r: r["rps"])
            ax.plot([p["rps"] for p in pts], [p[key] for p in pts], marker="o", color=colors[n], label=n, linewidth=2)
        if sla_line:
            ax.axhline(args.sla_ms / 1000.0, color="#6e7781", linestyle="--", linewidth=1, label="SLA")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Offered load (requests/s)")
        ax.grid(alpha=0.25)
    curve = flat[5]
    for n in names:
        pts = sorted((r for r in rows if r["target"] == n), key=lambda r: r["rps"])
        curve.plot([p["tokens_per_s"] for p in pts], [p["e2e_p99"] for p in pts], marker="o", color=colors[n], label=n, linewidth=2)
    curve.axhline(args.sla_ms / 1000.0, color="#6e7781", linestyle="--", linewidth=1)
    curve.set_title("Throughput-latency curve", fontsize=11)
    curve.set_xlabel("Throughput (tokens/s)")
    curve.set_ylabel("p99 end-to-end latency (s)")
    curve.grid(alpha=0.25)
    flat[0].legend(frameon=False)
    fig.suptitle(f"Request-rate sweep, SLA {args.sla_ms / 1000:g} s" + ("  (simulated mock backend)" if args.mock else ""), fontsize=13)
    fig.tight_layout()
    path = Path(args.out_dir) / "throughput_latency_curves.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def print_report(args, rows: List[Dict], capacity: Dict[str, Dict]) -> None:
    sla_s = args.sla_ms / 1000.0
    simulated = "  [simulated mock backend]" if args.mock else ""
    print("\n" + paint(f"Rate sweep  |  SLA {sla_s:g} s  |  seed {args.seed}{simulated}", BOLD))
    head = f"{'target':<18}{'rps':>6}{'served':>10}{'rej':>6}{'tok/s':>9}{'good tok/s':>12}{'p50 s':>8}{'p99 s':>8}{'ttft p99':>10}{'in SLA':>8}"
    print(paint(head, DIM))
    for r in rows:
        late = r["e2e_p99"] > sla_s
        served = f"{r['served']:.0f}/{r['offered']:.0f}"
        line = (f"{r['target']:<18}{r['rps']:>6g}{served:>10}{r['rejected']:>6.0f}{r['tokens_per_s']:>9.0f}"
                f"{r['goodput_tokens_per_s']:>12.0f}{r['e2e_p50']:>8.2f}{r['e2e_p99']:>8.2f}{r['ttft_p99']:>10.2f}{r['sla_attainment_percent']:>7.0f}%")
        print(paint(line, RED) if late else line)
    print("\n" + paint("Capacity at equal p99 latency (p99 <= SLA, rejections <= %.0f%%)" % (args.reject_tolerance * 100), BOLD))
    print(paint(f"{'target':<18}{'max rps':>9}{'peak good tok/s':>18}{'at rps':>8}{'peak tok/s':>12}", DIM))
    best = max((c["max_rps_within_p99_sla"] for c in capacity.values()), default=0.0)
    for name, c in capacity.items():
        line = (f"{name:<18}{c['max_rps_within_p99_sla']:>9g}{c['peak_goodput_tokens_per_s']:>18.0f}"
                f"{c['peak_goodput_at_rps']:>8g}{c['peak_tokens_per_s']:>12.0f}")
        print(paint(line, GREEN) if c["max_rps_within_p99_sla"] == best and best > 0 else line)
    estimated = [r["target"] for r in rows if r["ttft_is_estimate"]]
    if estimated:
        print(paint(f"note: {', '.join(dict.fromkeys(estimated))} is non-streaming, so its TTFT equals its total latency.", DIM))
    print(f"\nSaved: {Path(args.out_dir) / 'sweep_summary.csv'}\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Request-rate sweep and throughput-latency analysis")
    p.add_argument("--policies", default="static,static-batch,dynamic", help="Comma-separated local policies to start, optionally POLICY:ORDER such as dynamic:sjf; empty for none")
    p.add_argument("--target", action="append", default=[], help="External server NAME=PROTOCOL@URL[#MODEL], repeatable")
    p.add_argument("--rps", default="1,2,4,6,8,12,16", help="Comma-separated offered request rates")
    p.add_argument("--duration", type=float, default=20.0, help="Seconds of traffic per rate level")
    p.add_argument("--min-requests", type=int, default=12)
    p.add_argument("--sla-ms", type=float, default=8000.0)
    p.add_argument("--ttft-sla-ms", type=float, default=0.0, help="Optional time-to-first-token SLA applied to goodput")
    p.add_argument("--reject-tolerance", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--repeats", type=int, default=1, help="Runs per rate level with consecutive seeds; the median is reported")
    p.add_argument("--lengths-file", default=None, help="JSON list of [prompt_tokens, output_tokens] pairs from a chat dataset")
    p.add_argument("--max-output-tokens", type=int, default=256)
    p.add_argument("--out-dir", default=str(ROOT / "results" / "sweep"))
    p.add_argument("--api-key", default=None)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--model-path", default="/home/kartiklin/velocityllm/models/llama-3.2-1b")
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=8, help="Static policies: batch size / concurrency cap; dynamic: starting concurrency")
    p.add_argument("--max-concurrency", type=int, default=32)
    p.add_argument("--queue-order", default="priority", choices=["priority", "fcfs", "sjf"])
    p.add_argument("--max-batch-tokens", type=int, default=0)
    p.add_argument("--static-batch-timeout-ms", type=float, default=200.0)
    p.add_argument("--chunked-prefill", action="store_true")
    p.add_argument("--server-args", default="", help="Extra flags passed to every local server, as one quoted string")
    p.add_argument("--mock", action="store_true", help="Use the simulated backend (no GPU)")
    p.add_argument("--mock-tps", type=float, default=80.0)
    p.add_argument("--mock-contention", type=float, default=0.5)
    p.add_argument("--startup-timeout", type=float, default=600.0)
    args = p.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    levels = [float(x) for x in args.rps.split(",") if x.strip()]
    lengths = load_length_pairs(args.lengths_file) if args.lengths_file else None
    policies = [x.strip() for x in args.policies.split(",") if x.strip()]
    rows: List[Dict] = []

    for entry in policies:
        policy, _, order = entry.partition(":")
        print(paint(f"{entry}: starting server", CYAN))
        proc = start_server(args, policy, order or args.queue_order)
        try:
            target = Target(name=entry, protocol="velocity", url=f"http://localhost:{args.port}")
            asyncio.run(run_workload(target, build_workload(6, 6.0, seed=1, max_output_tokens=24), api_key=args.api_key))
            for rps in levels:
                row = run_level(args, target, rps, lengths)
                rows.append(row)
                print(f"  {entry:<14}{rps:>5g} rps  p99 {row['e2e_p99']:.2f}s  tok/s {row['tokens_per_s']:.0f}  rejected {row['rejected']}")
        finally:
            stop_server(proc)

    for spec in args.target:
        target = Target.parse(spec)
        print(paint(f"{target.name}: sweeping external server", CYAN))
        for rps in levels:
            row = run_level(args, target, rps, lengths)
            rows.append(row)
            print(f"  {target.name:<14}{rps:>5g} rps  p99 {row['e2e_p99']:.2f}s  tok/s {row['tokens_per_s']:.0f}  rejected {row['rejected']}")

    if not rows:
        sys.exit("nothing to run: pass --policies and/or --target")
    capacity = capacity_table(rows, args.sla_ms / 1000.0, args.reject_tolerance)
    write_results(args, rows, capacity)
    plot(args, rows)
    print_report(args, rows, capacity)


if __name__ == "__main__":
    main()
