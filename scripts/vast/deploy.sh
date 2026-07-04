#!/usr/bin/env bash
# Local kickoff: rsync src/, configs/, tests/, scripts/ to Vast, then pytest remotely.
# Corpus + model weights on Vast are NOT touched (they live under data/processed/ + /root/models/).
#
# Usage (from local Mac):
#   bash scripts/vast/deploy.sh
#
# Wall clock: ~1 min (30 s rsync + 30-60 s pytest tail).
set -euo pipefail

REPO="/Users/s19447/Desktop/modern-bert-g2p"
VAST_HOST="root@ssh2.vast.ai"
VAST_PORT=27868
REMOTE_ROOT="/root/modern-bert-g2p"

echo "=== [1/3] rsync code -> Vast (${VAST_HOST}:${REMOTE_ROOT}) ==="
rsync -avz \
    -e "ssh -p ${VAST_PORT}" \
    "${REPO}/src" \
    "${REPO}/configs" \
    "${REPO}/tests" \
    "${REPO}/scripts" \
    "${VAST_HOST}:${REMOTE_ROOT}/"

echo "=== [2/3] pytest on Vast (fast fail; --tb=short; tail 80) ==="
ssh -p "${VAST_PORT}" "${VAST_HOST}" bash -lc '
    set -euo pipefail
    cd /root/modern-bert-g2p
    source .venv/bin/activate
    pytest -x -q --tb=short 2>&1 | tail -80
'

echo "=== [3/3] summary ==="
ssh -p "${VAST_PORT}" "${VAST_HOST}" bash -lc '
    set -eu
    cd /root/modern-bert-g2p
    echo "  remote HEAD    : $(git rev-parse --short HEAD 2>/dev/null || echo n/a)"
    echo "  src files      : $(find src -name "*.py" | wc -l)"
    echo "  configs        : $(ls configs/*.yaml 2>/dev/null | wc -l)"
    echo "  test files     : $(find tests -name "test_*.py" | wc -l)"
'

echo "deploy: OK"
