import argparse
import json


def load_summary(path):
    with open(path) as f:
        return json.load(f)


def pct_change(old, new, higher_is_better=True):
    if old == 0:
        return "N/A"
    change = ((new - old) / old) * 100
    sign = "+" if change >= 0 else ""
    tag = "better" if (change > 0) == higher_is_better else "worse"
    return f"{sign}{change:.1f}% ({tag})"


def main():
    parser = argparse.ArgumentParser(description="Compare static vs dynamic benchmark results")
    parser.add_argument("--static", required=True)
    parser.add_argument("--dynamic", required=True)
    args = parser.parse_args()

    s = load_summary(args.static)
    d = load_summary(args.dynamic)

    W = 72
    print()
    print("=" * W)
    print("  STATIC vs DYNAMIC — COMPARISON REPORT".ljust(W))
    print("=" * W)
    print(f"  {'Metric':<28}{'Static':<16}{'Dynamic':<16}{'Change'}")
    print("-" * W)

    rows = [
        ("Throughput (req/sec)", "throughput_req_per_sec", True),
        ("p50 latency (s)", "p50_latency_seconds", False),
        ("p99 latency (s)", "p99_latency_seconds", False),
        ("SLA compliance (%)", "sla_compliance_rate_percent", True),
        ("Avg GPU utilization (%)", "avg_gpu_util_percent", True),
        ("Max GPU utilization (%)", "max_gpu_util_percent", True),
        ("Rejected (429)", "rejected_429", False),
    ]

    for label, key, higher_better in rows:
        sv, dv = s.get(key), d.get(key)
        change = pct_change(sv, dv, higher_better) if isinstance(sv, (int, float)) else "N/A"
        print(f"  {label:<28}{str(sv):<16}{str(dv):<16}{change}")

    print("=" * W)
    print()


if __name__ == "__main__":
    main()
