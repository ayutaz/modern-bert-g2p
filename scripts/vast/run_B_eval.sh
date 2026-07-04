#!/usr/bin/env bash
# On-Vast: eval 30K-step checkpoints from chain_B (6 total) on JSUT Basic5000 (PER micro).
# P-A uses the new prediction_fn adapter; P-C uses the existing implementation.
# Run AFTER scripts/vast/chain_B_training.sh has completed successfully.
#
# Usage (from local Mac):
#   ssh -p 27868 root@ssh2.vast.ai 'bash /root/modern-bert-g2p/scripts/vast/run_B_eval.sh'
#
# Wall clock: ~30 min total (~5 min per checkpoint x 6 + compare aggregation).
# Cost: ~$0.18 @ $0.36/hr.
set -euo pipefail

REPO="/root/modern-bert-g2p"
cd "${REPO}"
source .venv/bin/activate

SEEDS=(20260704 20260705 20260706)
CKPT_STEP=30000
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

RESULTS_ROOT="/root/reports/eval_v2"
mkdir -p "${RESULTS_ROOT}/p_a" "${RESULTS_ROOT}/p_c"

eval_one () {
    local pilot_dir="$1"    # p_a or p_c  (checkpoint / output directory name)
    local pilot_cli="$2"    # P-A or P-C  (CLI --pilot enum)
    local seed="$3"
    local config="$4"

    local ckpt="reports/phase2/${pilot_dir}_30k/${seed}/checkpoint_step_${CKPT_STEP}.pt"
    local out="${RESULTS_ROOT}/${pilot_dir}/jsut_${seed}.json"

    echo ""
    echo "--- eval ${pilot_cli} seed=${seed} ---"
    echo "  ckpt : ${ckpt}"
    echo "  out  : ${out}"
    if [[ ! -f "${ckpt}" ]]; then
        echo "MISSING checkpoint: ${ckpt} -- skipping" >&2
        return 0
    fi
    python -m modernbert_g2p eval \
        --pilot "${pilot_cli}" \
        --config "${config}" \
        --checkpoint "${ckpt}" \
        --dataset jsut \
        --output "${out}" \
        --batch-size 32
}

echo "=== chain_B post-eval (30K checkpoints) ==="
echo "  results root : ${RESULTS_ROOT}"
echo "  seeds        : ${SEEDS[*]}"
echo "  ckpt step    : ${CKPT_STEP}"

# eval P-A x 3
for seed in "${SEEDS[@]}"; do
    eval_one "p_a" "P-A" "${seed}" "configs/p_a_30k.yaml"
done

# eval P-C x 3
for seed in "${SEEDS[@]}"; do
    eval_one "p_c" "P-C" "${seed}" "configs/p_c_30k.yaml"
done

# Aggregate -> pilot table v2 (uses existing compare CLI, walks the two 30K dirs).
COMPARE_OUT="/root/reports/eval_v2/pilot_table_v2_${STAMP}.md"
echo ""
echo "=== aggregate -> pilot table v2 ==="
python -m modernbert_g2p compare \
    --checkpoint "reports/phase2/p_a_30k" \
    --checkpoint "reports/phase2/p_c_30k" \
    --output "${COMPARE_OUT}"

echo ""
echo "chain_B eval DONE."
echo "  per-run JSON  : ${RESULTS_ROOT}/{p_a,p_c}/jsut_<seed>.json"
echo "  aggregate MD  : ${COMPARE_OUT}"
ls -la "${RESULTS_ROOT}/p_a" "${RESULTS_ROOT}/p_c"
