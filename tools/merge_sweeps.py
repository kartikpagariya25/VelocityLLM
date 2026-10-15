"""
Combine several sweep_summary.json files into one comparison: merged summary, capacity table and plot.

    python tools/merge_sweeps.py --out-dir results/gpu/comparison \
        --input results/gpu/velocity --input results/gpu/vllm --input results/gpu/chunked_on=dynamic+chunked
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.rate_sweep import capacity_table, plot, print_report, write_results


def load_rows(spec: str) -> List[Dict]:
    path, _, label = spec.partition("=")
    source = Path(path)
    source = source / "sweep_summary.json" if source.is_dir() else source
    data = json.loads(source.read_text())
    rows = data["rows"]
    if label:
        if len({r["target"] for r in rows}) != 1:
            raise SystemExit(f"{spec}: a label can only rename a sweep that has one target")
        rows = [{**r, "target": label} for r in rows]
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Merge sweep summaries into one comparison")
    p.add_argument("--input", action="append", required=True, help="Sweep directory or json file, optionally DIR=LABEL to rename its single target")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--sla-ms", type=float, default=None)
    p.add_argument("--reject-tolerance", type=float, default=0.01)
    p.add_argument("--mock", action="store_true")
    args = p.parse_args()

    rows: List[Dict] = []
    first_sla = None
    for spec in args.input:
        rows.extend(load_rows(spec))
        if first_sla is None:
            source = Path(spec.partition("=")[0])
            source = source / "sweep_summary.json" if source.is_dir() else source
            first_sla = json.loads(source.read_text()).get("sla_ms")
    args.sla_ms = args.sla_ms or first_sla or 8000.0
    args.seed = 42
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    capacity = capacity_table(rows, args.sla_ms / 1000.0, args.reject_tolerance)
    write_results(args, rows, capacity)
    plot(args, rows)
    print_report(args, rows, capacity)


if __name__ == "__main__":
    main()
