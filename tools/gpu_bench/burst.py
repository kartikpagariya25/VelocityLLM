import asyncio, aiohttp, sys, time, json
from collections import Counter
N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
PORT = sys.argv[2] if len(sys.argv) > 2 else "8000"
URL = f"http://localhost:{PORT}/generate"
async def one(s, i, res):
    t = time.time()
    try:
        async with s.post(URL, json={"prompt": f"Write a short story about robot number {i}.", "max_tokens": 80}) as r:
            body = await r.json(content_type=None)
            if r.status == 200:
                res.append(("ok", time.time() - t, body.get("tokens_generated", 0), body.get("sla_met"), ""))
            else:
                d = body.get("detail", body) if isinstance(body, dict) else body
                why = (d.get("reason_code") or str(d.get("reason", ""))[:60]) if isinstance(d, dict) else str(d)[:60]
                res.append(("rej", time.time() - t, 0, None, f"HTTP {r.status}: {why}"))
    except Exception as e:
        res.append(("err", time.time() - t, 0, None, repr(e)[:60]))
async def main():
    res = []; t0 = time.time()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as s:
        await asyncio.gather(*[one(s, i, res) for i in range(N)])
    wall = time.time() - t0
    ok = [r for r in res if r[0] == "ok"]; lat = sorted(r[1] for r in ok)
    toks = sum(r[2] for r in ok); within = sum(1 for r in ok if r[3])
    out = dict(sent=N, ok=len(ok), rejected=sum(r[0] == "rej" for r in res), errors=sum(r[0] == "err" for r in res),
               wall_s=round(wall, 1), tok_s=round(toks / wall) if wall else 0,
               p50=round(lat[len(lat)//2], 2) if lat else None,
               p99=round(lat[min(len(lat)-1, int(len(lat)*.99))], 2) if lat else None,
               within_sla=within, goodput_pct=round(100*within/N),
               why=dict(Counter(r[4] for r in res if r[0] != "ok")))
    print("RESULT_JSON " + json.dumps(out))
asyncio.run(main())
