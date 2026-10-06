#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
(cd frontend && npm ci && npm run build)
rm -rf control_service/web build dist
cp -r frontend/dist control_service/web
python3 -m pip install --quiet build twine
python3 -m build
python3 -m twine check dist/*
