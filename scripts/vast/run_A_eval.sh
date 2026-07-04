#!/usr/bin/env bash
# On-Vast: eval existing P-A 10K checkpoints (3 seeds) on JSUT Basic5000.
# Requires the P-A prediction_fn adapter (Task A) to be deployed via scripts/vast/deploy.sh first.
#
# Usage (from local Mac):
#   ssh -p 27868 root@ssh2.vast.ai 'bash /root/modern-bert-g2p/scripts/vast/run_A_eval.sh'
#
# Wall clock: ~15 min total (~5 min per seed on RTX 5090, batch_size=32).
# Cost: ~$0.09 @ $0.36/hr.
set -euo pipefail

REPO="/root/modern-bert-g2p"
cd "${REPO}"
source .venv/bin/activate

SEEDS=(20260704 20260705 20260706)
CONFIG="configs/p_a_gpu.yaml"
CKPT_STEP=10000
RESULTS_DIR="/root/reports/eval/p_a"
mkdir -p "${RESULTS_DIR}"

if [[ ! -f "${CONFIG}" ]]; then
    echo "FATAL: config not found: ${CONFIG}" >&2
    echo "  (Vast may host it at this path even if the local repo has p_a.yaml." >&2
    echo "   Verify with: ls configs/p_a*.yaml on the remote.)" >&2
    exit 2
fi

echo "=== P-A 10K JSUT eval ==="
echo "  config : ${CONFIG}"
echo "  seeds  : ${SEEDS[*]}"
echo "  out    : ${RESULTS_DIR}"

for seed in "${SEEDS[@]}"; do
    CKPT="reports/phase2/p_a/${seed}/checkpoint_step_${CKPT_STEP}.pt"
    OUT="${RESULTS_DIR}/jsut_${seed}.json"
    echo ""
    echo "--- eval P-A seed=${seed} ---"
    echo "  ckpt : ${CKPT}"
    echo "  out  : ${OUT}"
    if [[ ! -f "${CKPT}" ]]; then
        echo "MISSING checkpoint: ${CKPT} -- skipping" >&2
        continue
    fi
    python -m modernbert_g2p eval \
        --pilot P-A \
        --config "${CONFIG}" \
        --checkpoint "${CKPT}" \
        --dataset jsut \
        --output "${OUT}" \
        --batch-size 32
done

echo ""
echo "P-A 10K eval DONE. Results:"
ls -la "${RESULTS_DIR}"
