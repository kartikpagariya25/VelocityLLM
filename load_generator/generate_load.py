import argparse
import asyncio
import csv
import json
import sys
import time
import random

import aiohttp

sys.path.append("/home/kartiklin/velocityllm/scheduler_engine")
from gpu_monitor import GPUMonitor
from traffic_patterns import generate_arrival_times, generate_mixed_prompts

DEFAULT_PROMPTS = [
    "What is the capital of France?",
    "Explain photosynthesis in one sentence.",
    "What is the capital of Japan?",
    "Name three programming languages.",
    "What is the boiling point of water?",
    "Tell me a fact about space.",
    "What is the capital of Italy?",
    "Describe a cat in one sentence.",
]


async def send_request(session, url, prompt, max_tokens, results, request_index, is_long=None):
    payload = {"prompt": prompt, "max_tokens": max_tokens}
    start = time.time()
    try:
        async with session.post(url, json=payload) as resp:
            elapsed = time.time() - start
            row = {
                "request_index": request_index,
                "status": resp.status,
                "client_latency_seconds": elapsed,
                "is_long": is_long,
            }
            if resp.status == 200:
                data = await resp.json()
                row.update({
                    "server_latency_seconds": data.get("latency_seconds"),
                    "queue_time_seconds": data.get("queue_time_seconds"),
                    "execution_time_seconds": data.get("execution_time_seconds"),
                    "tokens_generated": data.get("tokens_generated"),
                    "sla_met": data.get("sla_met"),
                })
            else:
                row.update({
                    "server_latency_seconds": None, "queue_time_seconds": None,
                    "execution_time_seconds": None, "tokens_generated": None, "sla_met": None,
                })
            results.append(row)
    except Exception as e:
        elapsed = time.time() - start
        results.append({
            "request_index": request_index, "status": "error",
            "client_latency_seconds": elapsed, "is_long": is_long,
            "server_latency_seconds": None, "queue_time_seconds": None,
            "execution_time_seconds": None, "tokens_generated": None,
            "sla_met": None, "error": str(e),
        })


def build_request_plan(args):
    if args.mixed_prompts:
        base = generate_mixed_prompts(args.num_requests, seed=args.seed)
    else:
        rng = random.Random(args.seed)
        base = [{"prompt": rng.choice(DEFAULT_PROMPTS), "max_tokens": args.max_tokens, "is_long": False}
                for _ in range(args.num_requests)]

    if args.pattern == "flood":
        arrivals = [0.0] * args.num_requests
    else:
        arrivals = generate_arrival_times(args.pattern, args.num_requests, args.duration, seed=args.seed)

    for item, t in zip(base, arrivals):
        item["arrival_time"] = t
    return base


async def scheduled_sender(session, url, item, index, results, concurrency_gate):
    await asyncio.sleep(item["arrival_time"])
    async with concurrency_gate:
        await send_request(session, url, item["prompt"], item["max_tokens"], results, index, item.get("is_long"))


async def run_load(args):
    results = []
    plan = build_request_plan(args)
    monitor = GPUMonitor(sample_interval=0.2)
    concurrency_gate = asyncio.Semaphore(args.concurrency)

    connector = aiohttp.TCPConnector(limit=args.concurrency * 2)
    monitor.start()
    wall_start = time.time()
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            scheduled_sender(session, args.url, item, i, results, concurrency_gate)
            for i, item in enumerate(plan)
        ]
        await asyncio.gather(*tasks)
    wall_elapsed = time.time() - wall_start
    monitor.stop()

    fieldnames = [
        "request_index", "status", "client_latency_seconds", "server_latency_seconds",
        "queue_time_seconds", "execution_time_seconds", "tokens_generated", "sla_met",
        "is_long", "error",
    ]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    successful = [r for r in results if r["status"] == 200]
    rejected_429 = [r for r in results if r["status"] == 429]
    errors = [r for r in results if r["status"] not in (200, 429)]

    latencies = sorted(r["client_latency_seconds"] for r in successful)
    n = len(latencies)
    p50 = latencies[int(n * 0.50)] if n else 0.0
    p95 = latencies[min(int(n * 0.95), n - 1)] if n else 0.0
    p99 = latencies[min(int(n * 0.99), n - 1)] if n else 0.0
    throughput = n / wall_elapsed if wall_elapsed > 0 else 0.0
    sla_met_count = sum(1 for r in successful if r.get("sla_met"))
    sla_rate = (sla_met_count / n * 100) if n else 0.0

    long_reqs = [r for r in successful if r.get("is_long")]
    short_reqs = [r for r in successful if not r.get("is_long")]
    long_p99 = sorted(r["client_latency_seconds"] for r in long_reqs)
    short_p99 = sorted(r["client_latency_seconds"] for r in short_reqs)

    gpu_summary = monitor.summary()

    summary = {
        "label": args.label,
        "pattern": args.pattern,
        "total_requests": args.num_requests,
        "successful": n,
        "rejected_429": len(rejected_429),
        "errors": len(errors),
        "wall_time_seconds": round(wall_elapsed, 3),
        "throughput_req_per_sec": round(throughput, 2),
        "p50_latency_seconds": round(p50, 3),
        "p95_latency_seconds": round(p95, 3),
        "p99_latency_seconds": round(p99, 3),
        "sla_compliance_rate_percent": round(sla_rate, 1),
        "long_prompt_p99_seconds": round(long_p99[-1], 3) if long_p99 else None,
        "short_prompt_p99_seconds": round(short_p99[-1], 3) if short_p99 else None,
        "avg_gpu_util_percent": round(gpu_summary["avg_gpu_util_percent"], 1) if gpu_summary else None,
        "max_gpu_util_percent": round(gpu_summary["max_gpu_util_percent"], 1) if gpu_summary else None,
        "avg_gpu_memory_mb": round(gpu_summary["avg_mem_used_mb"]) if gpu_summary else None,
    }

    summary_path = args.output.replace(".csv", "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print_report(summary)
    print(f"\n  Raw results : {args.output}")
    print(f"  Summary JSON: {summary_path}\n")


def print_report(s):
    W = 58
    print()
    print("=" * W)
    print(f"  BENCHMARK REPORT — {s['label']}".ljust(W))
    print(f"  Pattern: {s['pattern']}".ljust(W))
    print("=" * W)
    print(f"  {'Total requests sent':<30}: {s['total_requests']}")
    print(f"  {'Successful (HTTP 200)':<30}: {s['successful']}")
    print(f"  {'Rejected (HTTP 429 - shed)':<30}: {s['rejected_429']}")
    print(f"  {'Errors':<30}: {s['errors']}")
    print(f"  {'Wall clock time':<30}: {s['wall_time_seconds']}s")
    print("-" * W)
    print("  THROUGHPUT & LATENCY")
    print("-" * W)
    print(f"  {'Throughput':<30}: {s['throughput_req_per_sec']} req/sec")
    print(f"  {'p50 latency':<30}: {s['p50_latency_seconds']}s")
    print(f"  {'p95 latency':<30}: {s['p95_latency_seconds']}s")
    print(f"  {'p99 latency':<30}: {s['p99_latency_seconds']}s")
    print(f"  {'SLA compliance rate':<30}: {s['sla_compliance_rate_percent']}%")
    if s.get("long_prompt_p99_seconds") is not None:
        print("-" * W)
        print("  MIXED PROMPT BREAKDOWN")
        print("-" * W)
        print(f"  {'Long-prompt p99 latency':<30}: {s['long_prompt_p99_seconds']}s")
        print(f"  {'Short-prompt p99 latency':<30}: {s['short_prompt_p99_seconds']}s")
    print("-" * W)
    print("  GPU UTILIZATION")
    print("-" * W)
    print(f"  >>> {'Avg GPU utilization':<26}: {s['avg_gpu_util_percent']}%")
    print(f"  >>> {'Max GPU utilization':<26}: {s['max_gpu_util_percent']}%")
    print(f"      {'Avg GPU memory used':<26}: {s['avg_gpu_memory_mb']} MB")
    print("=" * W)


def main():
    parser = argparse.ArgumentParser(description="VelocityLLM Load Generator")
    parser.add_argument("--url", default="http://localhost:8000/generate")
    parser.add_argument("--num-requests", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--output", default="results.csv")
    parser.add_argument("--label", default="Run")
    parser.add_argument("--pattern", choices=["flood", "constant", "poisson", "burst"], default="flood",
                         help="flood = all at once (saturation test); constant/poisson/burst = spread over --duration seconds")
    parser.add_argument("--duration", type=float, default=10.0, help="Time window in seconds for non-flood patterns")
    parser.add_argument("--mixed-prompts", action="store_true", help="Use mixed short/long prompts (headline E10 scenario)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    asyncio.run(run_load(args))


if __name__ == "__main__":
    main()
