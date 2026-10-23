#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
PORT="${PORT:-9000}"
SLA_MS="${SLA_MS:-8000}"
MODELS_DIR="${MODELS_DIR:-models}"
HF_MODEL="${HF_MODEL:-meta-llama/Llama-3.2-1B-Instruct}"
HF_DIR="${HF_DIR:-$MODELS_DIR/llama-3.2-1b}"

if [ -t 1 ]; then G=$'\033[92m'; R=$'\033[91m'; Y=$'\033[93m'; B=$'\033[1m'; D=$'\033[2m'; N=$'\033[0m'; else G=""; R=""; Y=""; B=""; D=""; N=""; fi
ok()   { printf "  %s✔%s %s\n" "$G" "$N" "$1"; }
bad()  { printf "  %s✘%s %s\n" "$R" "$N" "$1"; FAILED=1; }
warn() { printf "  %s!%s %s\n" "$Y" "$N" "$1"; }
head_() { printf "\n%s%s%s\n" "$B" "$1" "$N"; }

activate() {
  if [ ! -d .venv ]; then "$PY" -m venv .venv; fi
  . .venv/bin/activate
}

setup() {
  head_ "Setup"
  activate
  mkdir -p "$HOME/tmp_pip"; export TMPDIR="$HOME/tmp_pip"
  pip install -q --upgrade pip
  pip install -q -r requirements.txt "huggingface-hub>=1.5.0,<2.0" matplotlib pyyaml && ok "Python dependencies installed"
  if command -v npm > /dev/null; then
    (cd frontend && npm ci --silent && npm run build > /dev/null) && ok "Arena frontend built" || bad "Frontend build failed"
  else
    warn "npm not found; install Node.js 20+ to serve the Arena page"
  fi
}

model() {
  head_ "Model"
  activate
  if [ -f "$HF_DIR/config.json" ]; then ok "Model already present at $HF_DIR"; return; fi
  mkdir -p "$MODELS_DIR"
  hf auth whoami > /dev/null 2>&1 || { echo "  Hugging Face login needed:"; hf auth login; }
  hf download "$HF_MODEL" --local-dir "$HF_DIR" --exclude "original/*" && ok "Model downloaded to $HF_DIR" || bad "Model download failed"
}

check() {
  FAILED=0
  head_ "Preflight"
  activate
  if command -v nvidia-smi > /dev/null && nvidia-smi -L > /dev/null 2>&1; then
    ok "GPU: $(nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader | head -1)"
  else
    bad "nvidia-smi found no GPU; run this on the GPU machine"
  fi
  "$PY" -c "import vllm" > /dev/null 2>&1 && ok "vLLM importable" || bad "vLLM missing in .venv (run: ./run_lab.sh setup)"
  found=$(find "$MODELS_DIR" -maxdepth 2 -name config.json 2>/dev/null | wc -l)
  [ "$found" -gt 0 ] && ok "$found model folder(s) under $MODELS_DIR" || bad "No model under $MODELS_DIR (run: ./run_lab.sh model)"
  [ -f frontend/dist/index.html ] && ok "Arena frontend built" || warn "frontend/dist missing; API only (run: ./run_lab.sh setup)"
  if command -v lsof > /dev/null && lsof -i ":$PORT" > /dev/null 2>&1; then bad "Port $PORT already in use (fuser -k $PORT/tcp)"; else ok "Port $PORT free"; fi
  free_mb=$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | awk -F, '{print $1-$2}')
  [ -n "${free_mb:-}" ] && { [ "$free_mb" -ge 3000 ] && ok "${free_mb} MB GPU memory free" || warn "Only ${free_mb} MB GPU memory free; close other GPU programs"; }
  echo
  [ "$FAILED" -eq 0 ] && printf "%sReady.%s\n" "$G" "$N" || { printf "%sFix the items above first.%s\n" "$R" "$N"; return 1; }
}

tests() {
  head_ "Tests"
  activate
  python -m pytest tests -q
}

serve() {
  activate
  check || exit 1
  ip=$(hostname -I 2>/dev/null | awk '{print $1}')
  head_ "Arena"
  printf "  Local : %shttp://localhost:%s/#/lab%s\n" "$B" "$PORT" "$N"
  [ -n "${ip:-}" ] && printf "  LAN   : %shttp://%s:%s/#/lab%s\n" "$B" "$ip" "$PORT" "$N"
  printf "  %sChoose 'PC GPU (this machine)' and press Run comparison. Ctrl+C to stop.%s\n\n" "$D" "$N"
  exec python -m control_service --host 0.0.0.0 --port "$PORT" --models-dir "$MODELS_DIR" --cors-origin '*'
}

bench() {
  activate
  check || exit 1
  head_ "Static vs Dynamic benchmark (CLI)"
  mname=$(basename "$(dirname "$(find "$MODELS_DIR" -maxdepth 2 -name config.json | head -1)")")
  mpath=$(dirname "$(find "$MODELS_DIR" -maxdepth 2 -name config.json | head -1)")
  python load_generator/fair_benchmark.py --model "$mname=$mpath" --sla-ms "$SLA_MS" "$@"
}

mock() {
  head_ "Mock Arena (no GPU, pipeline check only)"
  activate
  ip=$(hostname -I 2>/dev/null | awk '{print $1}')
  printf "  Local : %shttp://localhost:%s/#/lab%s\n" "$B" "$PORT" "$N"
  [ -n "${ip:-}" ] && printf "  WSL/LAN: %shttp://%s:%s/#/lab%s\n\n" "$B" "$ip" "$PORT" "$N"
  exec python -m control_service --mock --host 0.0.0.0 --port "$PORT" --cors-origin '*'
}

all() { setup; model; tests; serve; }

case "${1:-serve}" in
  setup) setup ;;
  model) model ;;
  check) check ;;
  test)  tests ;;
  serve) serve ;;
  bench) shift; bench "$@" ;;
  mock)  mock ;;
  all)   all ;;
  *) echo "usage: $0 [setup|model|check|test|serve|bench|mock|all]"; exit 1 ;;
esac
