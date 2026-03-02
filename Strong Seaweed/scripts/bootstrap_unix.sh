#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "== BlueWeave + Strong Seaweed bootstrap (Unix) =="

ensure_from_example() {
  local target="$1"
  local example="$2"
  if [[ ! -f "$target" && -f "$example" ]]; then
    cp "$example" "$target"
    echo "Created $target from example."
  fi
}

pushd "$ROOT_DIR/blue-weave-aqua-main" >/dev/null
npm install
popd >/dev/null

pushd "$ROOT_DIR/blue-weave-aqua-main/server" >/dev/null
ensure_from_example ".env" ".env.example"
npm install
popd >/dev/null

pushd "$ROOT_DIR/blue-weave-aqua-main/server/agents_python" >/dev/null
if [[ ! -d ".venv" ]]; then
  python3 -m venv .venv
fi
.venv/bin/python -m pip install -r requirements.txt
popd >/dev/null

pushd "$ROOT_DIR/Strong-Seaweed-main" >/dev/null
ensure_from_example ".env" ".env.example"
if [[ ! -d ".venv_model_api" ]]; then
  python3 -m venv .venv_model_api
fi
.venv_model_api/bin/python -m pip install -r requirements-model-api.txt
popd >/dev/null

echo
echo "Bootstrap complete."
echo "Start model API, agent gateway, backend, and frontend per README.md."
