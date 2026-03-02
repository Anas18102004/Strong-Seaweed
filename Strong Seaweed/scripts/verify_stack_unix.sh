#!/usr/bin/env bash
set -euo pipefail

echo "== BlueWeave stack verification =="

check() {
  local name="$1"
  local url="$2"
  if curl -fsS --max-time 8 "$url" >/dev/null; then
    echo "[OK]   $name"
  else
    echo "[FAIL] $name"
  fi
}

check "Frontend" "http://127.0.0.1:8080"
check "Backend health" "http://127.0.0.1:4000/health"
check "Agent gateway health" "http://127.0.0.1:8101/health"
check "Model API health" "http://127.0.0.1:8000/health"

echo
echo "If some checks failed, start missing services using README instructions."
