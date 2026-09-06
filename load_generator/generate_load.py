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

PROMPTS = [
    "What is the capital of France?",
    "Explain photosynthesis in one sentence.",
    "What is the capital of Japan?",
    "Name three programming languages.",
    "What is the boiling point of water?",
    "Tell me a fact about space.",
    "What is the capital of Italy?",
    "Describe a cat in one sentence.",
]


async def send_request(session, url, prompt, max_tokens, results, request_index):
    payload = {"prompt": prompt, "max_tokens": max_tokens}
    start = time.time()
    try:
        async with session.post(url, json=payload) as resp:
            elapsed = time.time() - start
            if resp.status == 200:
                data = await resp.json()
                results.append({
                    "request_index": request_index,
                    "status": resp.status,
                    "client_latency_seconds": elapsed,
                    "server_latency_seconds": data.get("latency_seconds"),
                    "queue_time_seconds": data.get("queue_time_seconds"),
                    "execution_time_seconds": data.get("execution_time_seconds"),
                    "tokens_generated": data.get("tokens_generated"),
                    "sla_met": data.get("sla_met"),
                })
            else:
                results.append({
                    "request_index": request_index,
                    "status": resp.status,
                    "client_latency_seconds": elapsed,
                    "server_latency_seconds": None,
                    "queue_time_seconds": None,
                    "execution_time_seconds": None,
                    "tokens_generated": None,
                    "sla_met": None,
                })
    except Exception as e:
        elapsed = time.time() - start
        results.append({
            "request_index": request_index,
            "status": "error",
            "client_latency_seconds": elapsed,
            "server_latency_seconds": None,
            "queue_time_seconds": None,
            "execution_time_seconds": None,
            "tokens_generated": None,
            "sla_met": None,
            "error": str(e),
        })


async def run_load(url, num_requests, concurrency, max_tokens, output_csv, label):
    results = []
    monitor = GPUMonitor(sample_interval=0.2)

    connector = aiohttp.TCPConnector(limit=concurrency)
    monitor.start()
    wall_start = time.time()
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for i in range(num_requests):
            prompt = random.choice(PROMPTS)
            tasks.append(send_request(session, url, prompt, max_tokens, results, i))
        await asyncio.gather(*tasks)
    wall_elapsed = time.time() - wall_start
    monitor.stop()

    with open(output_csv, "w", newline="") as f:
        fieldnames = [
            "request_index", "status", "client_latency_seconds", "server_latency_seconds",
            "queue_time_seconds", "execution_time_seconds", "tokens_generated", "sla_met", "error",
        ]
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

    gpu_summary = monitor.summary()

    summary = {
        "label": label,
        "total_requests": num_requests,
        "successful": n,
        "rejected_429": len(rejected_429),
        "errors": len(errors),
        "wall_time_seconds": round(wall_elapsed, 3),
        "throughput_req_per_sec": round(throughput, 2),
        "p50_latency_seconds": round(p50, 3),
        "p95_latency_seconds": round(p95, 3),
        "p99_latency_seconds": round(p99, 3),
        "sla_compliance_rate_percent": round(sla_rate, 1),
        "avg_gpu_util_percent": round(gpu_summary["avg_gpu_util_percent"], 1) if gpu_summary else None,
        "max_gpu_util_percent": round(gpu_summary["max_gpu_util_percent"], 1) if gpu_summary else None,
        "avg_gpu_memory_mb": round(gpu_summary["avg_mem_used_mb"]) if gpu_summary else None,
    }

    summary_path = output_csv.replace(".csv", "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print_report(summary)
    print(f"\n  Raw results : {output_csv}")
    print(f"  Summary JSON: {summary_path}\n")


def print_report(s):
    W = 58
    print()
    print("=" * W)
    print(f"  BENCHMARK REPORT — {s['label']}".ljust(W))
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
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--output", default="results.csv")
    parser.add_argument("--label", default="Run")
    args = parser.parse_args()

    asyncio.run(run_load(args.url, args.num_requests, args.concurrency, args.max_tokens, args.output, args.label))


if __name__ == "__main__":
    main()
