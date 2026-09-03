import argparse
import asyncio
import csv
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
            data = await resp.json()
            elapsed = time.time() - start
            results.append({
                "request_index": request_index,
                "status": resp.status,
                "client_latency_seconds": elapsed,
                "server_latency_seconds": data.get("latency_seconds"),
            })
    except Exception as e:
        elapsed = time.time() - start
        results.append({
            "request_index": request_index,
            "status": "error",
            "client_latency_seconds": elapsed,
            "server_latency_seconds": None,
            "error": str(e),
        })


async def run_load(url, num_requests, concurrency, max_tokens, output_csv):
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
        fieldnames = ["request_index", "status", "client_latency_seconds", "server_latency_seconds", "error"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    latencies = [r["client_latency_seconds"] for r in results if r["status"] == 200]
    gpu_summary = monitor.summary()

    print()
    print("=" * 50)
    print("  BENCHMARK RESULTS")
    print("=" * 50)
    if latencies:
        latencies.sort()
        p50 = latencies[int(len(latencies) * 0.50)]
        p99 = latencies[min(int(len(latencies) * 0.99), len(latencies) - 1)]
        throughput = len(latencies) / wall_elapsed
        print(f"  Total requests      : {num_requests}")
        print(f"  Successful          : {len(latencies)}")
        print(f"  Wall time           : {wall_elapsed:.2f}s")
        print(f"  Throughput          : {throughput:.2f} req/sec")
        print(f"  p50 latency         : {p50:.3f}s")
        print(f"  p99 latency         : {p99:.3f}s")
    else:
        print("  No successful requests.")

    print("-" * 50)
    if gpu_summary:
        print(f"  >>> AVG GPU UTIL     : {gpu_summary['avg_gpu_util_percent']:.1f}%  <<<")
        print(f"  >>> MAX GPU UTIL     : {gpu_summary['max_gpu_util_percent']:.1f}%  <<<")
        print(f"  MIN GPU util        : {gpu_summary['min_gpu_util_percent']:.1f}%")
        print(f"  AVG GPU memory used : {gpu_summary['avg_mem_used_mb']:.0f} MB")
        print(f"  MAX GPU memory used : {gpu_summary['max_mem_used_mb']:.0f} MB")
        print(f"  GPU samples taken   : {gpu_summary['sample_count']}")
    else:
        print("  No GPU samples captured.")
    print("=" * 50)
    print(f"  Full results saved to: {output_csv}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="VelocityLLM Load Generator")
    parser.add_argument("--url", default="http://localhost:8000/generate")
    parser.add_argument("--num-requests", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--output", default="results.csv")
    args = parser.parse_args()

    asyncio.run(run_load(args.url, args.num_requests, args.concurrency, args.max_tokens, args.output))


if __name__ == "__main__":
    main()
