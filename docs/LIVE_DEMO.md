# Velocity Live demo

One PC runs the VelocityLLM engine (real GPU + model). Phones open a simple page, pick requests / traffic / prompt type and press **Send prompts**. The engine floods the model with the combined traffic, first with Static, then with Dynamic (same requests, same GPU, fair A/B). The PC dashboard shows everything.

## Run

```
cd frontend && npm ci && npm run build && cd ..
python3 -m control_service --host 0.0.0.0 --port 9000
```

- Dashboard (PC): `http://localhost:9000/#/live`
- Phones: `http://<PC-IP>:9000/#/join` (the dashboard prints it)

`--host 0.0.0.0` is required, otherwise phones cannot connect.

## Network

1. Turn on the phone hotspot and connect the PC to it (or use any Wi-Fi both share).
2. Find the PC's address on that network (Windows: `ipconfig`) and type it into the dashboard's "PC address" box if it was not detected.
3. Both phones connect to the same hotspot.

### Service running inside WSL2

WSL2 has its own internal IP that phones cannot reach. Pick one:

- Mirrored networking: in `%UserProfile%\.wslconfig` add
  ```
  [wsl2]
  networkingMode=mirrored
  ```
  then `wsl --shutdown` and start again.
- Port forward (PowerShell as Administrator):
  ```
  $wsl = (wsl hostname -I).Trim().Split(' ')[0]
  netsh interface portproxy add v4tov4 listenport=9000 listenaddress=0.0.0.0 connectport=9000 connectaddress=$wsl
  New-NetFirewallRule -DisplayName "VelocityLLM 9000" -Direction Inbound -Protocol TCP -LocalPort 9000 -Action Allow
  ```
  The WSL address changes after a restart, so repeat the first two lines.

Test from the phone browser: `http://<PC-IP>:9000/api/preflight` should return JSON.

## Flow

1. Phones open `/#/join`, choose requests, traffic type, prompt type, press **Send prompts**.
2. The first send starts a short countdown (default 10 s, "Join window"). Every further send re-arms it, so both phones land in one run. The host can also press **Start now**, or turn auto-start off.
3. The engine runs Static then Dynamic (optionally Smart) on the same merged flood. Requests are fired by the server, so the browser connection limit does not matter.
4. The dashboard shows live panels, concurrency chart, log, results, per-device table and Saved benchmarks. A CSV report per run: `/api/live/runs/<id>/report.csv`.

## Tips for a clean demo

- Use the real model, not `--mock`; run one warm-up run first.
- Keep SLA realistic for the GPU (dashboard SLA box).
- Repeats = 3 gives median numbers; run-to-run variance on laptop GPUs is large.
- Maximum 6 phones, 100 requests per phone.

## Termux (optional)

Termux is not needed: the phone browser is enough. If you want traffic to originate from the phone itself, it would need a stable gateway port on the PC and is not part of this version.
