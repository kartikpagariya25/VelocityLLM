import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV_DIR = ROOT / "results" / "csv"
OUT = ROOT / "frontend" / "src" / "data" / "traces.json"

RUNS = {
    "llama": {
        "flood": ("static_results", "dynamic_results_v3"),
        "mixed": ("static_mixed", "dynamic_mixed"),
    },
    "qwen": {
        "flood": ("qwen_static_flood", "qwen_dynamic_flood"),
        "mixed": ("qwen_static_mixed", "qwen_dynamic_mixed"),
    },
    "stablelm": {
        "flood": ("stablelm_static_flood", "stablelm_dynamic_flood"),
        "mixed": ("stablelm_static_mixed", "stablelm_dynamic_mixed"),
    },
    "tinyllama": {
        "flood": ("tinyllama_static_flood", "tinyllama_dynamic_flood"),
        "mixed": ("tinyllama_static_mixed", "tinyllama_dynamic_mixed"),
    },
}


def percentile(values, q):
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def load(name):
    requests = []
    with open(CSV_DIR / f"{name}.csv", newline="") as f:
        for row in csv.DictReader(f):
            latency = float(row["client_latency_seconds"])
            served = int(row["status"]) == 200
            execution = float(row["execution_time_seconds"] or 0) if served else 0.0
            start = max(0.0, latency - execution) if served else 0.0
            requests.append(
                {
                    "id": int(row["request_index"]),
                    "start": round(start, 3),
                    "end": round(latency, 3),
                    "shed": not served,
                    "tokens": int(row["tokens_generated"] or 0) if served else 0,
                    "long": row.get("is_long", "") == "True",
                }
            )
    requests.sort(key=lambda r: (r["end"], r["id"]))
    served = [r["end"] for r in requests if not r["shed"]]
    return {
        "requests": requests,
        "total": len(requests),
        "served": len(served),
        "shed": len(requests) - len(served),
        "p50": round(percentile(served, 0.5), 2),
        "p99": round(percentile(served, 0.99), 2),
        "slowest": round(max(served), 2),
    }


def main():
    data = {}
    for model, scenarios in RUNS.items():
        data[model] = {}
        for scenario, (static_name, dynamic_name) in scenarios.items():
            data[model][scenario] = {"static": load(static_name), "dynamic": load(dynamic_name)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, separators=(",", ":")))


if __name__ == "__main__":
    main()
