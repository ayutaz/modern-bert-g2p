#!/usr/bin/env bash
# On-Vast: sequential 30K-step training chain -- P-A x 3 seed, then P-C x 3 seed (6 runs).
# Fail-fast on any run error. Emits [chain B] START / DONE markers to /root/logs/chain_b.log.
#
# Launch (from local Mac) inside screen so 24+ h session survives SSH drops:
#   ssh -p 27868 root@ssh2.vast.ai \
#       'cd /root/modern-bert-g2p && \
#        screen -dmS chainB bash scripts/vast/chain_B_training.sh'
#
# Monitor:
#   ssh -p 27868 root@ssh2.vast.ai 'tail -F /root/logs/chain_b.log'
# Reattach:
#   ssh -p 27868 root@ssh2.vast.ai 'screen -r chainB'
#
# Wall clock (RTX 5090 @ ~0.55 s/step, from Track 3 estimate):
#   30K steps * 0.55 s/step = ~4.58 h per run
#   6 runs sequential      = ~27.5 h total (range 26-30 h)
# Cost @ $0.36/hr = ~$9.90 (~$10-11)
set -euo pipefail

REPO="/root/modern-bert-g2p"
cd "${REPO}"
source .venv/bin/activate

SEEDS=(20260704 20260705 20260706)
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

MASTER_LOG="/root/logs/chain_b.log"
PER_RUN_LOG_DIR="/root/logs/chain_b_${STAMP}"
mkdir -p "$(dirname "${MASTER_LOG}")" "${PER_RUN_LOG_DIR}"

marker () {
    # Timestamped [chain B] marker -- always writes to master log AND stdout.
    local msg="$1"
    printf '[chain B] %s  %s\n' "$(date -u +%FT%TZ)" "${msg}" | tee -a "${MASTER_LOG}"
}

run_one () {
    local pilot="$1"
    local seed="$2"
    local config="$3"
    local out="reports/phase2/${pilot}_30k/${seed}"
    local run_log="${PER_RUN_LOG_DIR}/${pilot}_${seed}.log"

    mkdir -p "${out}"

    if [[ ! -f "${config}" ]]; then
        marker "FATAL config missing: ${config}"
        exit 2
    fi

    marker "START pilot=${pilot} seed=${seed} config=${config} out=${out}"

    python -m modernbert_g2p train \
        --config "${config}" \
        --seed "${seed}" \
        --output-dir "${out}" 2>&1 | tee "${run_log}"
    local rc=${PIPESTATUS[0]}

    if [[ ${rc} -ne 0 ]]; then
        marker "FAIL pilot=${pilot} seed=${seed} rc=${rc} log=${run_log}"
        exit "${rc}"
    fi
    marker "DONE pilot=${pilot} seed=${seed} log=${run_log}"
}

marker "CHAIN START stamp=${STAMP} host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

# P-A x 3 (bigger seq2seq model -- run first so failures surface early)
for seed in "${SEEDS[@]}"; do
    run_one "p_a" "${seed}" "configs/p_a_30k.yaml"
done

# P-C x 3
for seed in "${SEEDS[@]}"; do
    run_one "p_c" "${seed}" "configs/p_c_30k.yaml"
done

marker "CHAIN DONE stamp=${STAMP} per_run_logs=${PER_RUN_LOG_DIR}"
echo ""
echo "chain_B ALL DONE. Master log: ${MASTER_LOG}. Per-run logs: ${PER_RUN_LOG_DIR}"
