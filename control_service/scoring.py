import statistics

FIELDS = (
    "p50_s", "p95_s", "p99_s", "tokens_per_s", "served", "offered", "rejected", "errors",
    "within_sla_served", "within_sla_offered", "zero_token_share",
)


def nearest_rank(values, q):
    return values[min(int(len(values) * q), len(values) - 1)] if values else 0.0


def score(rows, sla_s, wall_s):
    served = [r for r in rows if r["status"] == "served"]
    lat = sorted(r["end_s"] - r["arrival_s"] for r in served)
    tokens = sum(r["tokens"] or 0 for r in served)
    within = sum(1 for v in lat if v <= sla_s)
    zero = sum(1 for r in served if not r["tokens"])
    return {
        "p50_s": nearest_rank(lat, 0.5),
        "p95_s": nearest_rank(lat, 0.95),
        "p99_s": nearest_rank(lat, 0.99),
        "tokens_per_s": tokens / wall_s if wall_s > 0 else 0.0,
        "served": len(served),
        "offered": len(rows),
        "rejected": sum(1 for r in rows if r["status"] == "rejected"),
        "errors": sum(1 for r in rows if r["status"] == "error"),
        "within_sla_served": within / len(served) if served else 0.0,
        "within_sla_offered": within / len(rows) if rows else 0.0,
        "zero_token_share": zero / len(served) if served else 0.0,
    }


def median_of(results):
    if len(results) == 1:
        return dict(results[0])
    return {k: statistics.median(r[k] for r in results) for k in FIELDS}
