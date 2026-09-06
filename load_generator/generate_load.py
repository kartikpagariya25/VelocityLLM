"""
VelocityLLM - Advanced Asynchronous Load Generator
Simulates synthetic LLM inference traffic across priority classes,
evaluates SLA compliance, tracks burst-shedding (HTTP 429s), and measures
GPU utilization and latency distributions.
"""

import argparse
import asyncio
import csv
import logging
import os
import random
import sys
import time
from typing import Dict, List, Optional

import aiohttp

# Ensure scheduler_engine is importable
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from scheduler_engine.gpu_monitor import GPUMonitor
except ImportError:
    GPUMonitor = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("velocityllm.load_generator")

PROMPTS = [
    "What is the capital of France?",
    "Explain photosynthesis in one concise sentence.",
    "What is the capital of Japan?",
    "Name three modern programming languages and their primary use cases.",
    "What is the boiling point of water at standard atmospheric pressure?",
    "Tell me a fascinating fact about astrophysics and neutron stars.",
    "What is the capital of Italy and what is its most famous historical monument?",
    "Describe why asynchronous continuous batching improves GPU compute efficiency.",
]


async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    prompt: str,
    max_tokens: int,
    priority: int,
    sla_target_ms: float,
    results: List[Dict],
    request_index: int,
):
    """Sends a single inference request and records client- and server-side metrics."""
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "priority": priority,
        "sla_target_ms": sla_target_ms,
    }
    start = time.time()
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as resp:
            elapsed = time.time() - start
            data = {}
            try:
                data = await resp.json()
            except Exception:
                pass

            if resp.status == 200:
                results.append({
                    "request_index": request_index,
                    "status": 200,
                    "priority": priority,
                    "max_tokens": max_tokens,
                    "client_latency_seconds": elapsed,
                    "server_latency_seconds": data.get("latency_seconds"),
                    "sla_met": data.get("sla_met", True),
                    "tokens_generated": data.get("tokens_generated", 0),
                    "error": None,
                })
            elif resp.status == 429:
                results.append({
                    "request_index": request_index,
                    "status": 429,
                    "priority": priority,
                    "max_tokens": max_tokens,
                    "client_latency_seconds": elapsed,
                    "server_latency_seconds": None,
                    "sla_met": False,
                    "tokens_generated": 0,
                    "error": "SLA_SHED_429",
                })
            else:
                results.append({
                    "request_index": request_index,
                    "status": resp.status,
                    "priority": priority,
                    "max_tokens": max_tokens,
                    "client_latency_seconds": elapsed,
                    "server_latency_seconds": None,
                    "sla_met": False,
                    "tokens_generated": 0,
                    "error": f"HTTP_{resp.status}",
                })
    except Exception as e:
        elapsed = time.time() - start
        results.append({
            "request_index": request_index,
            "status": "error",
            "priority": priority,
            "max_tokens": max_tokens,
            "client_latency_seconds": elapsed,
            "server_latency_seconds": None,
            "sla_met": False,
            "tokens_generated": 0,
            "error": str(e),
        })


async def run_load(
    url: str,
    num_requests: int,
    concurrency: int,
    max_tokens: int,
    priority_mode: str,
    mixed_tokens: bool,
    sla_target_ms: float,
    output_csv: str,
):
    results: List[Dict] = []
    monitor = GPUMonitor(sample_interval=0.2) if GPUMonitor else None

    connector = aiohttp.TCPConnector(limit=concurrency)
    if monitor:
        monitor.start()

    wall_start = time.time()
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for i in range(num_requests):
            prompt = random.choice(PROMPTS)

            # Assign priority based on mode
            if priority_mode == "high":
                prio = 1
            elif priority_mode == "low":
                prio = 3
            elif priority_mode == "mixed":
                # 20% high, 60% normal, 20% low
                prio = random.choices([1, 2, 3], weights=[0.20, 0.60, 0.20])[0]
            else:
                prio = 2  # normal

            # Assign max tokens (mixed head-of-line test or fixed)
            if mixed_tokens:
                req_tokens = random.choice([25, 50, 100, 200])
            else:
                req_tokens = max_tokens

            tasks.append(
                send_request(
                    session=session,
                    url=url,
                    prompt=prompt,
                    max_tokens=req_tokens,
                    priority=prio,
                    sla_target_ms=sla_target_ms,
                    results=results,
                    request_index=i,
                )
            )
        await asyncio.gather(*tasks)

    wall_elapsed = time.time() - wall_start
    if monitor:
        monitor.stop()

    # Save to CSV
    if output_csv:
        with open(output_csv, "w", newline="") as f:
            fieldnames = [
                "request_index",
                "status",
                "priority",
                "max_tokens",
                "client_latency_seconds",
                "server_latency_seconds",
                "sla_met",
                "tokens_generated",
                "error",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)

    # Compute statistics
    successful = [r for r in results if r["status"] == 200]
    shed_429 = [r for r in results if r["status"] == 429]
    errors = [r for r in results if r["status"] not in (200, 429)]
    latencies = [r["client_latency_seconds"] for r in successful]

    total_tokens = sum(r.get("tokens_generated", 0) for r in successful)
    sla_met_count = sum(1 for r in successful if r.get("sla_met") is True)

    print()
    print("=" * 60)
    print("               VELOCITY-LLM BENCHMARK RESULTS")
    print("=" * 60)
    print(f"  Total Requests Submitted    : {num_requests}")
    print(f"  Completed Successfully (200): {len(successful)} ({len(successful)/num_requests*100:.1f}%)")
    print(f"  Shed under SLA Policy (429) : {len(shed_429)} ({len(shed_429)/num_requests*100:.1f}%)")
    print(f"  System/Transport Errors     : {len(errors)}")
    print(f"  Wall Elapsed Time           : {wall_elapsed:.2f}s")

    if latencies:
        latencies.sort()
        p50 = latencies[int(len(latencies) * 0.50)]
        p95 = latencies[min(int(len(latencies) * 0.95), len(latencies) - 1)]
        p99 = latencies[min(int(len(latencies) * 0.99), len(latencies) - 1)]
        req_throughput = len(successful) / wall_elapsed
        token_throughput = total_tokens / wall_elapsed

        print("-" * 60)
        print("  LATENCY & THROUGHPUT (SUCCESSFUL REQUESTS)")
        print("-" * 60)
        print(f"  Request Throughput          : {req_throughput:.2f} req/sec")
        print(f"  Token Throughput            : {token_throughput:.2f} tokens/sec")
        print(f"  p50 Latency                 : {p50:.3f}s")
        print(f"  p95 Latency                 : {p95:.3f}s")
        print(f"  p99 Latency                 : {p99:.3f}s")
        print(f"  SLA Compliance Rate         : {sla_met_count / len(successful) * 100:.1f}%")

    if monitor:
        gpu_summary = monitor.summary()
        if gpu_summary:
            print("-" * 60)
            print("  GPU HARDWARE TELEMETRY")
            print("-" * 60)
            print(f"  >>> AVG GPU UTILIZATION     : {gpu_summary['avg_gpu_util_percent']:.1f}%  <<<")
            print(f"  >>> MAX GPU UTILIZATION     : {gpu_summary['max_gpu_util_percent']:.1f}%  <<<")
            print(f"  MIN GPU Utilization         : {gpu_summary['min_gpu_util_percent']:.1f}%")
            print(f"  AVG GPU Memory Used         : {gpu_summary['avg_mem_used_mb']:.0f} MB")
            print(f"  MAX GPU Memory Used         : {gpu_summary['max_mem_used_mb']:.0f} MB")

    print("=" * 60)
    if output_csv:
        print(f"  Detailed metrics saved to: {output_csv}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="VelocityLLM Load Generator & Benchmarking Tool")
    parser.add_argument("--url", default="http://localhost:8000/generate", help="Target API endpoint")
    parser.add_argument("--num-requests", type=int, default=20, help="Total number of requests")
    parser.add_argument("--concurrency", type=int, default=5, help="Concurrent connection limit")
    parser.add_argument("--max-tokens", type=int, default=50, help="Default max generation tokens")
    parser.add_argument(
        "--priority",
        choices=["mixed", "high", "normal", "low"],
        default="normal",
        help="Priority distribution pattern",
    )
    parser.add_argument(
        "--mixed-tokens",
        action="store_true",
        help="Simulate mixed short/long prompt generation (anti-head-of-line test)",
    )
    parser.add_argument(
        "--sla-ms",
        type=float,
        default=3000.0,
        help="Target SLA deadline in milliseconds",
    )
    parser.add_argument("--output", default="results.csv", help="Output CSV path for request metrics")
    args = parser.parse_args()

    asyncio.run(
        run_load(
            url=args.url,
            num_requests=args.num_requests,
            concurrency=args.concurrency,
            max_tokens=args.max_tokens,
            priority_mode=args.priority,
            mixed_tokens=args.mixed_tokens,
            sla_target_ms=args.sla_ms,
            output_csv=args.output,
        )
    )


if __name__ == "__main__":
    main()
