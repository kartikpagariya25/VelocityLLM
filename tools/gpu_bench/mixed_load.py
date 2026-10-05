#!/usr/bin/env python3
"""Open-loop MIXED traffic client (policy-neutral scoring).

Three request classes arrive as a Poisson stream:
  interactive : short prompt, 30% HIGH / 70% NORMAL, real_time, SLA 3 s
  standard    : ~250-word prompt, NORMAL,           real_time, SLA 6 s
  batch       : ~1200-word prompt, LOW,             best_effort, SLA 20 s
Same seed => identical requests and arrival times for every policy (paired comparison).
A request counts as 'within SLA' only if it was served AND its client-measured latency <= its class SLA.

usage: mixed_load.py RATE_PER_S DURATION_S SEED [PORT]   -> prints one RESULT_JSON line
"""
import asyncio, aiohttp, json, random, sys, time

RATE = float(sys.argv[1]); DUR = float(sys.argv[2]); SEED = int(sys.argv[3])
PORT = sys.argv[4] if len(sys.argv) > 4 else "8000"
URL = f"http://localhost:{PORT}/generate"
CLIENT_TIMEOUT = 90
WORDS = ("system request token batch memory cache latency queue model server stream window layer "
         "kernel tensor vector prompt answer scheduler policy budget deadline signal buffer device "
         "thread network storage compute context length output input metric region sample").split()
CLASSES = {
    "interactive": dict(words=(12, 25), max_tokens=32, sla_ms=3000, tc="real_time", weight=0.55),
    "standard":    dict(words=(220, 280), max_tokens=64, sla_ms=6000, tc="real_time", weight=0.30),
    "batch":       dict(words=(1100, 1300), max_tokens=128, sla_ms=20000, tc="best_effort", weight=0.15),
}

def make_plan():
    rnd = random.Random(SEED); t = 0.0; plan = []; i = 0
    names = list(CLASSES); weights = [CLASSES[n]["weight"] for n in names]
    while True:
        t += rnd.expovariate(RATE)
        if t > DUR: break
        name = rnd.choices(names, weights)[0]; c = CLASSES[name]
        prio = 3 if name == "batch" else (1 if (name == "interactive" and rnd.random() < 0.30) else 2)
        n = rnd.randint(*c["words"])
        prompt = f"[req {SEED}-{i}] " + " ".join(rnd.choice(WORDS) for _ in range(n))   # unique prefix: defeats prefix cache
        plan.append(dict(t=t, cls=name, prio=prio, prompt=prompt, max_tokens=c["max_tokens"], sla_ms=c["sla_ms"], tc=c["tc"]))
        i += 1
    return plan

async def one(session, spec, t0, out):
    await asyncio.sleep(max(0.0, spec["t"] - (time.time() - t0)))
    start = time.time(); status = "err"; toks = 0; why = ""
    try:
        async with session.post(URL, headers={"X-SLA-Ms": str(spec["sla_ms"]), "X-Traffic-Class": spec["tc"]},
                json={"prompt": spec["prompt"], "max_tokens": spec["max_tokens"], "priority": spec["prio"]}) as r:
            body = await r.json(content_type=None)
            if r.status == 200:
                status = "ok"; toks = body.get("tokens_generated", 0)
            elif r.status == 429:
                status = "rej"; d = body.get("detail", {}) if isinstance(body, dict) else {}
                why = (d.get("reason_code") or "rejected") if isinstance(d, dict) else "rejected"
            else:
                why = f"HTTP {r.status}"
    except asyncio.TimeoutError:
        status = "timeout"
    except Exception as e:
        why = repr(e)[:40]
    lat = time.time() - start
    out.append(dict(cls=spec["cls"], prio=spec["prio"], status=status, lat=lat, toks=toks,
                    within=(status == "ok" and lat * 1000 <= spec["sla_ms"]), why=why))

def pct(v, q):
    v = sorted(v); return round(v[min(len(v) - 1, int(len(v) * q))], 2) if v else None

async def main():
    plan = make_plan(); out = []; t0 = time.time()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=CLIENT_TIMEOUT)) as s:
        await asyncio.gather(*[one(s, p, t0, out) for p in plan])
    wall = time.time() - t0
    def stats(rows):
        ok = [r for r in rows if r["status"] == "ok"]
        return dict(offered=len(rows), served=len(ok), within=sum(r["within"] for r in rows),
                    rejected=sum(r["status"] == "rej" for r in rows),
                    failed=sum(r["status"] in ("err", "timeout") for r in rows),
                    goodput=round(100 * sum(r["within"] for r in rows) / len(rows)) if rows else None,
                    served_pct=round(100 * len(ok) / len(rows)) if rows else None,
                    p50=pct([r["lat"] for r in ok], .5), p99=pct([r["lat"] for r in ok], .99))
    res = dict(rate=RATE, duration=DUR, seed=SEED, wall_s=round(wall, 1),
               tok_s=round(sum(r["toks"] for r in out) / wall) if wall else 0,
               all=stats(out), high=stats([r for r in out if r["cls"] == "interactive" and r["prio"] == 1]),
               **{c: stats([r for r in out if r["cls"] == c]) for c in CLASSES})
    reasons = {}
    for r in out:
        if r["status"] != "ok": reasons[r["why"] or r["status"]] = reasons.get(r["why"] or r["status"], 0) + 1
    res["not_ok"] = reasons
    print("RESULT_JSON " + json.dumps(res))
asyncio.run(main())
