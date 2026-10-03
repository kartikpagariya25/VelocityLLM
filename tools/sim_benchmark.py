"""
SIMULATED-GPU benchmark: Static vs Dynamic vs Smart.
The backend here is a MOCK whose per-token latency grows with concurrent streams. It is NOT a GPU
measurement. Use it to compare scheduling behaviour; re-run on real vLLM before quoting numbers.

    PYTHONPATH=. python tools/sim_benchmark.py mod|over|tight
"""
import asyncio, random, time, sys, statistics as st
from scheduler_engine.backend import MockBackend
from scheduler_engine.policy import StaticBatchPolicy, DynamicBatchPolicy, AdmissionRejectedException
from scheduler_engine.smart_policy import SmartBatchPolicy
from scheduler_engine.types import InferenceRequest, RequestPriority, ServerConfig

class ContendedMock(MockBackend):
    """SIMULATED GPU: per-token latency grows with concurrent streams (shared decode bandwidth)."""
    async def generate_stream(self, prompt, max_tokens, temperature, request_id):
        self.active_inferences += 1
        try:
            n_prompt = max(len(prompt.split()), len(prompt)//4)
            await asyncio.sleep(0.02 + n_prompt*0.00012)            # prefill cost
            target = max(1, min(max_tokens, int(max_tokens*random.uniform(0.5,1.0))))
            for i in range(target):
                yield "tok"
                await asyncio.sleep(0.010 + 0.0035*self.active_inferences)   # contention
        finally:
            self.active_inferences -= 1
    def get_gpu_telemetry(self):
        return min(98, 20+self.active_inferences*6), 3200+self.active_inferences*60, 8192

def workload(n, seed):
    rnd = random.Random(seed); out=[]
    for i in range(n):
        r = rnd.random()
        if r < 0.55:   p, mt, pr, tc = "short question "*rnd.randint(1,4), rnd.choice([16,24,32]), RequestPriority.HIGH if rnd.random()<0.3 else RequestPriority.NORMAL, "real_time"
        elif r < 0.85: p, mt, pr, tc = "medium context "*rnd.randint(40,120), rnd.choice([48,64]), RequestPriority.NORMAL, "real_time"
        else:          p, mt, pr, tc = "long document "*rnd.randint(300,500), rnd.choice([96,128]), RequestPriority.LOW, "best_effort"
        out.append((p,mt,pr,tc))
    return out

async def run(name, sla_ms, rate, seed, n=120):
    cfg = ServerConfig(policy=name, use_mock_backend=True, initial_concurrency=8, min_concurrency=2, max_concurrency=24, target_sla_ms=sla_ms, max_queue_size=100)
    P = {"static":StaticBatchPolicy,"dynamic":DynamicBatchPolicy,"smart":SmartBatchPolicy}[name]
    pol = P(ContendedMock(), cfg); await pol.initialize(); random.seed(seed)
    reqs = workload(n, seed); out=[]; toks=0
    async def one(p,mt,pr,tc):
        nonlocal toks
        req = InferenceRequest(prompt=p,max_tokens=mt,priority=pr,traffic_class=tc if name=="smart" else None)
        try:
            r = await pol.schedule(req); toks += r.tokens_generated; out.append((tc,"ok" if r.sla_met else "late",r.latency_seconds))
        except AdmissionRejectedException: out.append((tc,"rej",0))
    ts=[]; t0=time.time()
    for (p,mt,pr,tc) in reqs:
        ts.append(asyncio.create_task(one(p,mt,pr,tc))); await asyncio.sleep(1.0/rate)
    await asyncio.gather(*ts); dur=time.time()-t0; await pol.shutdown()
    rt=[o for o in out if o[0]=="real_time"]; be=[o for o in out if o[0]=="best_effort"]
    lat=sorted(o[2] for o in rt if o[1]!="rej")
    return dict(
        goodput=100*sum(o[1]=="ok" for o in out)/len(out),
        rt_goodput=100*sum(o[1]=="ok" for o in rt)/max(1,len(rt)),
        rt_ok_served=100*sum(o[1]=="ok" for o in rt)/max(1,sum(o[1]!="rej" for o in rt)),
        rt_p99=lat[min(len(lat)-1,int(len(lat)*.99))] if lat else 0,
        be_done=100*sum(o[1]!="rej" for o in be)/max(1,len(be)),
        rej=100*sum(o[1]=="rej" for o in out)/len(out), tok_s=toks/dur)

import sys
SC={"mod":(6000,8,"MODERATE load"),"over":(6000,40,"OVERLOAD"),"tight":(3000,40,"OVERLOAD + tight SLA")}
async def main():
    for sla,rate,label in [SC[sys.argv[1]]]:
        print(f"\n== {label}: SLA {sla/1000:.0f}s, {rate} req/s, 120 req x 2 seeds, SIMULATED contended GPU (mean of 2)")
        print(f"{'policy':8}{'goodput%':>9}{'RT goodput%':>12}{'RT ok/served%':>14}{'RT p99 s':>9}{'BE served%':>11}{'rejected%':>10}{'tok/s':>7}")
        for pn in ("static","dynamic","smart"):
            rs=[await run(pn,sla,rate,sd) for sd in (11,22)]
            m=lambda k: st.mean(r[k] for r in rs)
            print(f"{pn:8}{m('goodput'):>9.1f}{m('rt_goodput'):>12.1f}{m('rt_ok_served'):>14.1f}{m('rt_p99'):>9.2f}{m('be_done'):>11.1f}{m('rej'):>10.1f}{m('tok_s'):>7.0f}")
asyncio.run(main())
