#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
[ -d .venv ] && . .venv/bin/activate

STAGE="${1:-all}"
MODEL="${MODEL_PATH:-$HOME/velocityllm/models/llama-3.2-1b}"
SLA_MS="${SLA_MS:-8000}"
RPS="${RPS:-2,4,8,12,16,24}"
DURATION="${DURATION:-30}"
OUT="${OUT_DIR:-results/gpu}"
VLLM_PORT="${VLLM_PORT:-8001}"
COMMON=(--model-path "$MODEL" --sla-ms "$SLA_MS" --rps "$RPS" --duration "$DURATION" --max-concurrency 32)

sweep() { python tools/rate_sweep.py "${COMMON[@]}" "$@"; }

velocity() { sweep --policies "${POLICIES:-static,static-batch,dynamic}" --out-dir "$OUT/velocity"; }
batchcap() { sweep --policies static --batch-size 32 --out-dir "$OUT/static_cap32"; }
orders()   { sweep --policies dynamic:fcfs,dynamic:sjf,dynamic:priority --repeats 3 --out-dir "$OUT/policy_order"; }
chunked()  {
  sweep --policies dynamic --chunked-prefill --max-num-batched-tokens 2048 --out-dir "$OUT/chunked_on"
  sweep --policies dynamic --out-dir "$OUT/chunked_off"
}
tokens()   { sweep --policies dynamic --max-batch-tokens 2048 --out-dir "$OUT/token_budget"; }

vllm_stage() {
  mkdir -p "$OUT/vllm"
  vllm serve "$MODEL" --port "$VLLM_PORT" --max-model-len 4096 --gpu-memory-utilization 0.80 --max-num-seqs 32 > "$OUT/vllm/server.log" 2>&1 &
  local pid=$!
  trap 'kill $pid 2>/dev/null || true; wait $pid 2>/dev/null || true' EXIT
  for _ in $(seq 1 300); do
    curl -sf "http://localhost:$VLLM_PORT/v1/models" > /dev/null && break
    kill -0 $pid 2>/dev/null || { echo "vLLM exited early, see $OUT/vllm/server.log"; exit 1; }
    sleep 2
  done
  sweep --policies "" --target "vllm=openai@http://localhost:$VLLM_PORT#$MODEL" --out-dir "$OUT/vllm"
  kill $pid 2>/dev/null || true; wait $pid 2>/dev/null || true; trap - EXIT
}

merge() {
  python tools/merge_sweeps.py --out-dir "$OUT/comparison" \
    --input "$OUT/velocity" --input "$OUT/vllm" --input "$OUT/static_cap32=static (cap 32)"
  python tools/merge_sweeps.py --out-dir "$OUT/chunked_comparison" \
    --input "$OUT/chunked_off=dynamic (chunked off)" --input "$OUT/chunked_on=dynamic (chunked on)"
}

triton_note() {
  command -v docker > /dev/null || { echo "Docker not found: enable Docker Desktop WSL integration, then follow docs/COMPARISON.md section 3."; return 0; }
  echo "Docker found: follow docs/COMPARISON.md section 3 and sweep the Triton targets."
}

case "$STAGE" in
  velocity) velocity ;;
  batchcap) batchcap ;;
  orders) orders ;;
  chunked) chunked ;;
  tokens) tokens ;;
  vllm) vllm_stage ;;
  merge) merge ;;
  triton) triton_note ;;
  all) velocity; vllm_stage; batchcap; chunked; orders; tokens; merge; triton_note ;;
  *) echo "usage: $0 [velocity|vllm|batchcap|chunked|orders|tokens|merge|triton|all]"; exit 1 ;;
esac
