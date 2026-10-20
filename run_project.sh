#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

MODE="${1:-all}"
PY="${PYTHON:-python3}"

setup() {
  if [ ! -d .venv ]; then "$PY" -m venv .venv; fi
  . .venv/bin/activate
  pip install -q --upgrade pip
  pip install -q fastapi "uvicorn[standard]" pydantic pillow aiohttp huggingface_hub pytest pytest-asyncio httpx requests matplotlib
}

tests() { python -m pytest tests -q; }

sweep() {
  python tools/rate_sweep.py --policies static,static-batch,dynamic --mock --mock-contention 0.5 --rps 1,2,4,6,8,12,16 --out-dir results/sweep_mock
  python tools/rate_sweep.py --policies dynamic:fcfs,dynamic:sjf,dynamic:priority --mock --mock-contention 0.5 --rps 2,6,10 --repeats 3 --out-dir results/policy_order_mock
}

web() {
  command -v npm > /dev/null || { echo "npm not found: install Node.js 20 or newer"; exit 1; }
  (cd frontend && npm ci --silent && npm run build)
  (cd frontend && exec npx vite preview --host --port "${WEB_PORT:-4173}")
}

dev() {
  command -v npm > /dev/null || { echo "npm not found: install Node.js 20 or newer"; exit 1; }
  (cd frontend && npm ci --silent && exec npm run dev)
}

serve() {
  python -m scheduler_engine.server --policy "${POLICY:-dynamic}" --runner "${RUNNER:-vllm}" --model-path "${MODEL_PATH:?set MODEL_PATH}" --port "${PORT:-8000}"
}

case "$MODE" in
  web|dev) ;;
  *) setup ;;
esac
case "$MODE" in
  setup) ;;
  test) tests ;;
  sweep) sweep ;;
  serve) serve ;;
  web) web ;;
  dev) dev ;;
  all) tests; sweep ;;
  *) echo "usage: $0 [setup|test|sweep|serve|web|dev|all]"; exit 1 ;;
esac
