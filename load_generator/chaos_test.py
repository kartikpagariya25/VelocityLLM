import argparse
import asyncio
import json
import random
import time

import aiohttp

MALFORMED_PAYLOADS = [
    {},
    {"prompt": ""},
    {"prompt": "   "},
    {"prompt": 12345},
    {"prompt": "test", "max_tokens": -5},
    {"prompt": "test", "max_tokens": 999999999},
    {"prompt": "test", "temperature": 50.0},
    {"prompt": "\x00\x01\x02 null bytes test"},
    {"not_a_prompt_field": "test"},
    "not even a json object",
]

GOOD_PROMPTS = [
    "What is the capital of France?",
    "Name three programming languages.",
    "What is the boiling point of water?",
]


async def send_malformed(session, url, api_key, results):
    payload = random.choice(MALFORMED_PAYLOADS)
    headers = {"Content-Type": "application/json", "X-API-Key": api_key}
    try:
        async with session.post(url, json=payload if isinstance(payload, dict) else None,
                                 data=payload if isinstance(payload, str) else None,
                                 headers=headers) as resp:
            results["malformed_handled"] += 1
            results["malformed_status_codes"][resp.status] = results["malformed_status_codes"].get(resp.status, 0) + 1
    except Exception as e:
        results["malformed_connection_errors"] += 1


async def send_good_request(session, url, api_key, results):
    payload = {"prompt": random.choice(GOOD_PROMPTS), "max_tokens": 30}
    headers = {"Content-Type": "application/json", "X-API-Key": api_key}
    try:
        async with session.post(url, json=payload, headers=headers) as resp:
            if resp.status == 200:
                results["good_success"] += 1
            elif resp.status == 429:
                results["good_shed"] += 1
            else:
                results["good_unexpected_status"] += 1
    except Exception:
        results["good_connection_errors"] += 1


async def send_and_abort(session, url, api_key, results):
    payload = {"prompt": random.choice(GOOD_PROMPTS), "max_tokens": 300}
    headers = {"Content-Type": "application/json", "X-API-Key": api_key}
    try:
        async with asyncio.timeout(0.05):
            async with session.post(url, json=payload, headers=headers) as resp:
                await resp.read()
    except (asyncio.TimeoutError, Exception):
        results["aborted_connections"] += 1


async def health_check(session, base_url, api_key):
    try:
        async with session.get(f"{base_url}/health", headers={"X-API-Key": api_key}, timeout=aiohttp.ClientTimeout(total=3)) as resp:
            return resp.status == 200
    except Exception:
        return False


async def run_chaos(base_url, api_key, duration_seconds):
    generate_url = f"{base_url}/generate"
    results = {
        "malformed_handled": 0,
        "malformed_status_codes": {},
        "malformed_connection_errors": 0,
        "good_success": 0,
        "good_shed": 0,
        "good_unexpected_status": 0,
        "good_connection_errors": 0,
        "aborted_connections": 0,
        "health_checks_passed": 0,
        "health_checks_failed": 0,
    }

    async with aiohttp.ClientSession() as session:
        end_time = time.time() + duration_seconds
        tasks = []

        while time.time() < end_time:
            action = random.choices(
                ["malformed", "good", "abort"],
                weights=[0.3, 0.5, 0.2],
            )[0]

            if action == "malformed":
                tasks.append(asyncio.create_task(send_malformed(session, generate_url, api_key, results)))
            elif action == "good":
                tasks.append(asyncio.create_task(send_good_request(session, generate_url, api_key, results)))
            else:
                tasks.append(asyncio.create_task(send_and_abort(session, generate_url, api_key, results)))

            if len(tasks) >= 20:
                await asyncio.gather(*tasks, return_exceptions=True)
                tasks = []
                healthy = await health_check(session, base_url, api_key)
                if healthy:
                    results["health_checks_passed"] += 1
                else:
                    results["health_checks_failed"] += 1

            await asyncio.sleep(0.02)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        final_health = await health_check(session, base_url, api_key)

    print_report(results, final_health)


def print_report(r, final_health):
    W = 60
    print()
    print("=" * W)
    print("  CHAOS TEST REPORT")
    print("=" * W)
    print(f"  {'Well-formed requests succeeded':<38}: {r['good_success']}")
    print(f"  {'Well-formed requests shed (429)':<38}: {r['good_shed']}")
    print(f"  {'Well-formed unexpected status':<38}: {r['good_unexpected_status']}")
    print(f"  {'Well-formed connection errors':<38}: {r['good_connection_errors']}")
    print("-" * W)
    print(f"  {'Malformed requests handled':<38}: {r['malformed_handled']}")
    print(f"  {'Malformed status code breakdown':<38}: {r['malformed_status_codes']}")
    print(f"  {'Malformed connection errors':<38}: {r['malformed_connection_errors']}")
    print("-" * W)
    print(f"  {'Aborted mid-request connections':<38}: {r['aborted_connections']}")
    print("-" * W)
    print(f"  {'Mid-test health checks passed':<38}: {r['health_checks_passed']}")
    print(f"  {'Mid-test health checks failed':<38}: {r['health_checks_failed']}")
    print(f"  {'Final health check':<38}: {'PASS' if final_health else 'FAIL'}")
    print("=" * W)

    if r["good_unexpected_status"] == 0 and r["health_checks_failed"] == 0 and final_health:
        print("  >>> RESULT: Server survived chaos without crashing. <<<")
    else:
        print("  >>> RESULT: Issues detected — review above. <<<")
    print("=" * W)


def main():
    parser = argparse.ArgumentParser(description="VelocityLLM Chaos Test Suite")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--api-key", default="velocityllm-secret-123")
    parser.add_argument("--duration", type=int, default=20, help="Duration in seconds")
    args = parser.parse_args()

    asyncio.run(run_chaos(args.url, args.api_key, args.duration))


if __name__ == "__main__":
    main()
