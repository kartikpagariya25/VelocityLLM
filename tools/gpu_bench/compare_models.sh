#!/bin/bash
# usage: compare_models.sh [SLA_ms] [N] [max_concurrency]
# env: MODELS="a b"  POLS="static dynamic smart"  MOCK=1 (no GPU)  REPO  MODELS_DIR  PORT  OUT  LOGDIR
#      WARMUP=30   untimed warm-up burst before measuring (0 = cold start, the old behaviour)
#      REPEATS=3   measured bursts per (model,policy) on the same running server
REPO=${REPO:-$HOME/VelocityLLM}; MODELS_DIR=${MODELS_DIR:-$REPO/models}; TOOLS=$(cd "$(dirname "$0")" && pwd)
PORT=${PORT:-8000}; SLA=${1:-8000}; N=${2:-100}; CONC=${3:-16}; WARMUP=${WARMUP:-0}; REPEATS=${REPEATS:-1}
OUT=${OUT:-$HOME/results_models.csv}; LOGDIR=${LOGDIR:-$HOME/velocity_logs}; mkdir -p "$LOGDIR"
cd "$REPO" || exit 1
[ -f .venv/bin/activate ] && source .venv/bin/activate
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '^/mnt/' | paste -sd:)
MODELS=${MODELS:-$(ls "$MODELS_DIR")}; POLS=${POLS:-static dynamic smart}
echo "model,policy,run,sla_ms,status,ok,rejected,errors,p50_s,p99_s,tok_s,goodput_pct,kv_est_tokens,kv_vllm_tokens,note" > "$OUT"
stop_server() { kill $PID 2>/dev/null; wait $PID 2>/dev/null; pkill -f EngineCore 2>/dev/null
  for i in $(seq 1 30); do curl -s localhost:$PORT/health >/dev/null 2>&1 || break; sleep 1; done; sleep 4; }
for M in $MODELS; do
  YAML="$LOGDIR/cfg_$M.yaml"
  python3 "$TOOLS/modelcfg.py" "$MODELS_DIR/$M" > "$YAML" 2>/dev/null || { echo "$M,-,0,$SLA,NO_CONFIG,,,,,,,,,,config.json unreadable" >> "$OUT"; continue; }
  echo "##### $M (warmup=$WARMUP repeats=$REPEATS)"; cat "$YAML"
  for POL in $POLS; do
    LOG="$LOGDIR/${M}_${POL}.log"; echo "=== $M | $POL | SLA ${SLA}ms | $N concurrent | conc $CONC ==="
    EXTRA=""; [ "$MOCK" = 1 ] && EXTRA="--mock"
    python3 -m scheduler_engine.server --policy $POL --config "$YAML" --model-path "$MODELS_DIR/$M" \
      --sla-ms $SLA --max-concurrency $CONC --port $PORT $EXTRA > "$LOG" 2>&1 &
    PID=$!; READY=0
    for i in $(seq 1 150); do
      kill -0 $PID 2>/dev/null || break
      curl -s localhost:$PORT/health 2>/dev/null | grep -q "\"policy\":\"$POL\"" && { READY=1; break; }
      sleep 2
    done
    if [ $READY != 1 ]; then
      WHY=$(grep -E "Error|error" "$LOG" | grep -v "^ " | tail -1 | tr ',"' '; ' | cut -c1-110)
      echo "  START FAILED: $WHY"; echo "$M,$POL,0,$SLA,START_FAILED,,,,,,,,,,$WHY" >> "$OUT"; stop_server; continue
    fi
    if [ "$MOCK" != 1 ] && { grep -q "Falling back to MockBackend" "$LOG" || ! grep -q "AsyncLLMEngine initialized successfully" "$LOG"; }; then
      echo "  NOT A REAL GPU RUN: fell back to MockBackend. Skipping."
      echo "$M,$POL,0,$SLA,FELL_BACK_TO_MOCK,,,,,,,,,,vLLM did not load - see $LOG" >> "$OUT"; stop_server; continue
    fi
    if [ "$WARMUP" -gt 0 ]; then python3 "$TOOLS/burst.py" $WARMUP $PORT >/dev/null 2>&1; sleep 3; fi
    for RUN in $(seq 1 $REPEATS); do
      R=$(python3 "$TOOLS/burst.py" $N $PORT | grep RESULT_JSON | sed 's/RESULT_JSON //')
      KVE=""; KVV=""
      if [ "$POL" = smart ]; then
        KVE=$(curl -s localhost:$PORT/smart/state | python3 -c "import sys,json; print(json.load(sys.stdin)['tokens']['kv']['capacity_tokens'])" 2>/dev/null)
        KVV=$(grep -o "GPU KV cache size: [0-9,]* tokens" "$LOG" | grep -o "[0-9,]*" | tr -d ',' | tail -1)
      fi
      python3 - "$M" "$POL" "$RUN" "$SLA" "$KVE" "$KVV" "$R" >> "$OUT" <<'PY'
import sys, json
m, p, run, sla, kve, kvv, r = sys.argv[1:8]
d = json.loads(r) if r else {}
why = "; ".join(f"{k}x{v}" for k, v in d.get("why", {}).items()).replace(",", " ")
print(",".join(str(x) for x in [m, p, run, sla, "OK" if d else "NO_RESULT", d.get("ok",""), d.get("rejected",""), d.get("errors",""),
      d.get("p50",""), d.get("p99",""), d.get("tok_s",""), d.get("goodput_pct",""), kve, kvv, why]))
PY
      tail -1 "$OUT"; [ $RUN -lt $REPEATS ] && sleep 5
    done
    stop_server
  done
done
echo; python3 "$TOOLS/summary.py" "$OUT"
