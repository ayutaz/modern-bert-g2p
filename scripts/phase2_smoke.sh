#!/usr/bin/env bash
# Runs 3-pilot Phase 2 smoke tests (1-batch 1-step each).
# Assumes .venv/ is present at repo root and configs/{p_a,p_b,p_c}.yaml exist.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"

PYTHON="${PYTHON:-.venv/bin/python}"
SMOKE_ROOT="${SMOKE_ROOT:-/tmp/phase2-smoke}"
SEED="${SEED:-20260704}"

for pilot in p_a p_b p_c; do
    echo "=== smoke: ${pilot} ==="
    "${PYTHON}" -m modernbert_g2p train \
        --config "configs/${pilot}.yaml" \
        --seed "${SEED}" \
        --output-dir "${SMOKE_ROOT}/${pilot}" \
        --smoke
done

echo "All 3 pilots smoke-passed"
