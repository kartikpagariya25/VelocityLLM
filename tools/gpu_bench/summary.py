import csv, sys, statistics as st
from collections import OrderedDict
rows = list(csv.DictReader(open(sys.argv[1])))
groups = OrderedDict()
for r in rows: groups.setdefault((r['model'], r['policy']), []).append(r)
def num(r, k):
    try: return float(r[k])
    except: return None
print(f"{'model':16}{'policy':9}{'runs':>5}{'ok':>6}{'rej':>6}{'p50 s':>7}{'p99 s':>7}{'tok/s':>7}{'goodput% mean [min-max]':>26}")
for (m, p), rs in groups.items():
    good = [r for r in rs if r['status'] == 'OK']
    if not good:
        print(f"{m[:15]:16}{p:9}{'':>5}  {rs[0]['status']}"); continue
    g = [num(r, 'goodput_pct') for r in good]
    mean = lambda k: st.mean(v for v in (num(r, k) for r in good) if v is not None)
    rng = f"{st.mean(g):.0f} [{min(g):.0f}-{max(g):.0f}]"
    print(f"{m[:15]:16}{p:9}{len(good):>5}{mean('ok'):>6.0f}{mean('rejected'):>6.0f}{mean('p50_s'):>7.2f}{mean('p99_s'):>7.2f}{mean('tok_s'):>7.0f}{rng:>26}")
print("\nKV capacity check (smart estimate vs vLLM actual):")
seen = set()
for r in rows:
    if r['kv_est_tokens'] and r['kv_vllm_tokens'] and r['model'] not in seen:
        seen.add(r['model']); e, v = int(r['kv_est_tokens']), int(r['kv_vllm_tokens'])
        print(f"  {r['model'][:16]:17} estimate {e:>8,} | vLLM {v:>8,} | off by {100*(e-v)/v:+.0f}%")
bad = [r for r in rows if r['status'] != 'OK']
if bad:
    print("\nDid not run:")
    for r in bad: print(f"  {r['model']} / {r['policy']}: {r['status']} {r['note']}")
