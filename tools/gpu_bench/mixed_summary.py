import csv, json, sys, statistics as st
from collections import OrderedDict
rows = list(csv.DictReader(open(sys.argv[1])))
g = OrderedDict()
for r in rows: g.setdefault((r['model'], r['policy']), []).append(r)
def cell(vals, nd=0):
    vals = [v for v in vals if v is not None]
    return "-" if not vals else f"{st.mean(vals):.{nd}f} [{min(vals):.{nd}f}-{max(vals):.{nd}f}]"
def get(d, *k):
    for x in k:
        d = d.get(x) if isinstance(d, dict) else None
    return d
LIGHT = []
metrics = [("overall goodput%", ("all", "goodput")), ("interactive goodput%", ("interactive", "goodput")),
           ("HIGH-prio goodput%", ("high", "goodput")), ("standard goodput%", ("standard", "goodput")),
           ("batch served%", ("batch", "served_pct")), ("rejected (count)", ("all", "rejected")),
           ("interactive p99 (s)", ("interactive", "p99")), ("tok/s", ("tok_s",))]
for (m, p), rs in g.items():
    ok = [json.loads(r['json'].replace(';', ',')) for r in rs if r['status'] == 'OK' and r['json']]
    if not ok: print(f"\n{m} / {p}: {rs[0]['status']}"); continue
    print(f"\n{m} / {p}   ({len(ok)} runs, mean [min-max])")
    for label, path in metrics:
        print(f"  {label:24}{cell([get(d, *path) for d in ok], 2 if 'p99' in label else 0)}")
    allg = [get(d, "all", "goodput") for d in ok if get(d, "all", "goodput") is not None]
    LIGHT.append(min(allg) if allg else 0)

if LIGHT and min(LIGHT) >= 98:
    print("\nNOTE: every policy served ~100% -> the load was too light to separate them. Re-run with a higher RATE.")
