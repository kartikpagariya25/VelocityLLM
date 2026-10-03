#!/bin/bash
# usage: compare_mixed.sh [RATE_per_s] [DURATION_s] [max_concurrency]
# env: MODELS="a b"  POLS="static dynamic smart"  MOCK=1  REPEATS=3  WARMUP_S=8  SERVER_SLA=6000
#      REPO  MODELS_DIR  PORT  OUT  LOGDIR
REPO=${REPO:-$HOME/VelocityLLM}; MODELS_DIR=${MODELS_DIR:-$REPO/models}; TOOLS=$(cd "$(dirname "$0")" && pwd)
PORT=${PORT:-8000}; RATE=${1:-10}; DUR=${2:-15}; CONC=${3:-8}; REPEATS=${REPEATS:-3}; WARMUP_S=${WARMUP_S:-8}; SERVER_SLA=${SERVER_SLA:-6000}
OUT=${OUT:-$HOME/results_mixed.csv}; LOGDIR=${LOGDIR:-$HOME/velocity_logs}; mkdir -p "$LOGDIR"
cd "$REPO" || exit 1
[ -f .venv/bin/activate ] && source .venv/bin/activate
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '^/mnt/' | paste -sd:)
MODELS=${MODELS:-$(ls "$MODELS_DIR")}; POLS=${POLS:-static dynamic smart}
echo "model,policy,run,rate,status,json" > "$OUT"
stop_server() { kill $PID 2>/dev/null; wait $PID 2>/dev/null; pkill -f EngineCore 2>/dev/null
  for i in $(seq 1 30); do curl -s localhost:$PORT/health >/dev/null 2>&1 || break; sleep 1; done; sleep 4; }
for M in $MODELS; do
  YAML="$LOGDIR/cfg_$M.yaml"
  python3 "$TOOLS/modelcfg.py" "$MODELS_DIR/$M" > "$YAML" 2>/dev/null || { echo "$M,-,0,$RATE,NO_CONFIG," >> "$OUT"; continue; }
  echo "##### $M | mixed traffic ${RATE} req/s for ${DUR}s | concurrency cap $CONC | repeats $REPEATS"
  for POL in $POLS; do
    LOG="$LOGDIR/mixed_${M}_${POL}.log"; echo "=== $M | $POL ==="
    EXTRA=""; [ "$MOCK" = 1 ] && EXTRA="--mock"
    python3 -m scheduler_engine.server --policy $POL --config "$YAML" --model-path "$MODELS_DIR/$M" \
      --sla-ms $SERVER_SLA --max-concurrency $CONC --port $PORT $EXTRA > "$LOG" 2>&1 &
    PID=$!; READY=0
    for i in $(seq 1 150); do
      kill -0 $PID 2>/dev/null || break
      curl -s localhost:$PORT/health 2>/dev/null | grep -q "\"policy\":\"$POL\"" && { READY=1; break; }
      sleep 2
    done
    if [ $READY != 1 ]; then echo "  START FAILED"; echo "$M,$POL,0,$RATE,START_FAILED," >> "$OUT"; stop_server; continue; fi
    if [ "$MOCK" != 1 ] && { grep -q "Falling back to MockBackend" "$LOG" || ! grep -q "AsyncLLMEngine initialized successfully" "$LOG"; }; then
      echo "  NOT A REAL GPU RUN (fell back to MockBackend). Skipping."; echo "$M,$POL,0,$RATE,FELL_BACK_TO_MOCK," >> "$OUT"; stop_server; continue
    fi
    python3 "$TOOLS/mixed_load.py" $RATE $WARMUP_S 999 $PORT >/dev/null 2>&1; sleep 3        # untimed warm-up
    for RUN in $(seq 1 $REPEATS); do
      R=$(python3 "$TOOLS/mixed_load.py" $RATE $DUR $((100 + RUN)) $PORT | grep RESULT_JSON | sed 's/RESULT_JSON //')
      if [ -n "$R" ]; then echo "$M,$POL,$RUN,$RATE,OK,$(echo "$R" | tr ',' ';')" >> "$OUT"
        echo "$R" | python3 -c "import sys,json; d=json.load(sys.stdin); a=d['all']; i=d['interactive']; print(f'  run $RUN: goodput {a[\"goodput\"]}% | interactive {i[\"goodput\"]}% (p99 {i[\"p99\"]}s) | batch served {d[\"batch\"][\"served_pct\"]}% | rejected {a[\"rejected\"]}/{a[\"offered\"]} | {d[\"tok_s\"]} tok/s')"
      else echo "  run $RUN: NO RESULT"; echo "$M,$POL,$RUN,$RATE,NO_RESULT," >> "$OUT"; fi
      [ $RUN -lt $REPEATS ] && sleep 5
    done
    stop_server
  done
done
echo; python3 "$TOOLS/mixed_summary.py" "$OUT"
