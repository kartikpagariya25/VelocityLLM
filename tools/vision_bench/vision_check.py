#!/usr/bin/env python3
"""Test a vision-language model through VelocityLLM.

    python vision_check.py --smoke                      # 4 pictures with known answers (simulated engine)
    python vision_check.py                              # image traffic on static, dynamic and smart (simulated engine)
    python vision_check.py --real --model-path models/qwen2-vl-2b-awq --smoke
    python vision_check.py --real --model-path models/qwen2-vl-2b-awq --rate 2 --duration 20
"""
import argparse
import os
import sys

import bench_core as core


def smoke(a):
    print(f"Smoke test: {'real model ' + a.model_path if a.real else 'SIMULATED engine (answers are canned text)'} | policy {a.policy}\n")
    for row in core.smoke(a.policy, mock=not a.real, model_path=a.model_path, image_path=a.image, max_concurrency=a.max_concurrency):
        print(f"Picture : {row['picture']}\nAsked   : {row['question']}\nAnswer  : {row['answer']}\n"
              f"HTTP {row['status']} | vision tokens charged: {row['image_tokens']} | {row['seconds']} s\n")
    if not a.real:
        print("The simulated engine cannot look at pictures. Use --real with a vision model to see real answers.")


def traffic(a):
    print(f"Image traffic: {a.rate:g} users/s for {a.duration:g} s | images 224 px (50%), 448 px (35%), 896 px (15%) | promise {a.sla_ms / 1000:g} s")
    print(f"Engine: {'real model ' + a.model_path if a.real else 'SIMULATED (made-up speed; image cost = 0.4 ms per vision token)'} | ceiling for all: {a.max_concurrency} at once\n")
    seen = {"text": ""}

    def progress(text, fraction):
        if text != seen["text"] and "finished" not in text:
            print(f"  {text} ...", flush=True)
        seen["text"] = text

    results = core.run_all([p.strip() for p in a.policies.split(",")], a.rate, a.duration, a.seed, mock=not a.real,
                           model_path=a.model_path, sla_ms=a.sla_ms, max_concurrency=a.max_concurrency, progress=progress)
    sizes = [s for s, _ in core.IMAGE_MIX]
    head = ["Policy", "Users", "Refused", "On time %", "Slowest s"] + [f"{s}px on time %" for s in sizes] + [f"{s}px median s" for s in sizes] + ["Vision tokens ok"]
    widths = [max(len(h), 8) + 2 for h in head]
    print("\n" + "".join(h.ljust(w) for h, w in zip(head, widths)))
    print("-" * sum(widths))
    problems = []
    for p, r in results.items():
        s = r["summary"]
        cells = [p.capitalize(), s["offered"], s["refused"], s["on_time_pct"], s["p99_s"]]
        cells += [s["by_size"][z]["on_time_pct"] for z in sizes] + [s["by_size"][z]["median_s"] for z in sizes]
        cells.append("yes" if s["tokens_ok"] else f"NO ({s['image_tokens_reported']} vs {s['image_tokens_sent']})")
        print("".join(("-" if c is None else str(c)).ljust(w) for c, w in zip(cells, widths)))
        if not s["tokens_ok"]:
            problems.append(f"{p}: the server charged {s['image_tokens_reported']} vision tokens, the requests carried {s['image_tokens_sent']}")
        if s["failed"]:
            problems.append(f"{p}: {s['failed']} requests failed (not refused, failed)")
    print("\n'On time %' = answered inside the promise, out of ALL users (refused users count against it).")
    print("'Vision tokens ok' = the vision tokens the server reported add up to what the images cost.")
    if not a.real:
        print("Simulated engine: speeds are made up. Use --real on a GPU for real numbers.")
    for line in problems:
        print("PROBLEM:", line)
    return 1 if problems else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="send a few pictures with known answers and print the replies")
    ap.add_argument("--image", help="with --smoke: use your own picture")
    ap.add_argument("--policy", default="dynamic", help="policy for --smoke")
    ap.add_argument("--policies", default=",".join(core.POLICIES))
    ap.add_argument("--rate", type=float, default=None, help="new users per second (default: 40 simulated, 2 real)")
    ap.add_argument("--duration", type=float, default=12)
    ap.add_argument("--sla-ms", type=int, default=8000)
    ap.add_argument("--max-concurrency", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--real", action="store_true", help="use a real vision-language model (needs velocityllm[gpu])")
    ap.add_argument("--model-path", default=os.environ.get("VELOCITY_MODEL_PATH"))
    a = ap.parse_args()
    if a.real and not a.model_path:
        sys.exit("--real needs --model-path (or set VELOCITY_MODEL_PATH)")
    if a.rate is None:
        a.rate = 2 if a.real else 40
    if a.smoke:
        return smoke(a) or 0
    return traffic(a)


if __name__ == "__main__":
    sys.exit(main())
