#!/usr/bin/env bash
# B-05: Phase 0 3-baseline 一括測定スクリプト
#
# 1 コマンドで下記 3 baseline を測定し、結果を tabular + JSON で出力する:
#   1) haqumei JSUT Basic5000  PER
#   2) pyopenjtalk JVS nonpara CER
#   3) haqumei ROHAN 4600      KER
#
# 未作成のスクリプトは自動 skip、失敗は fail-fast (exit 1)。
#
# Usage:
#   bash scripts/eval_baselines.sh                 # run all 3 (default)
#   bash scripts/eval_baselines.sh all             # run all 3
#   bash scripts/eval_baselines.sh haqumei_jsut    # run one target only
#   bash scripts/eval_baselines.sh pyopenjtalk_jvs
#   bash scripts/eval_baselines.sh haqumei_rohan
#
# 出力 (results/baselines.json は "all" 実行のときのみ上書き。
#       単一 target のときは results/baselines_<target>.json に書く。
#       これで matrix 並列実行が互いに上書きしない — F1 修正):
#   results/baselines.json                                # all target 用
#   results/baselines_<target>.json                       # 単一 target 用
#   results/history/baselines_<label>_<timestamp>.json    # 履歴
#   results/logs/<eval>_<timestamp>.log                    # 各 eval の生ログ
set -euo pipefail

TARGET="${1:-all}"
case "$TARGET" in
  all|haqumei_jsut|pyopenjtalk_jvs|haqumei_rohan) ;;
  *)
    echo "ERROR: unknown target '$TARGET'." >&2
    echo "Valid: all | haqumei_jsut | pyopenjtalk_jvs | haqumei_rohan" >&2
    exit 2
    ;;
esac

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

VENV="${VENV:-$REPO/.venv}"
DATA_DIR="${DATA_DIR:-$REPO/data}"
RESULTS_DIR="$REPO/results"
HISTORY_DIR="$RESULTS_DIR/history"
LOGS_DIR="$RESULTS_DIR/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

mkdir -p "$DATA_DIR" "$RESULTS_DIR" "$HISTORY_DIR" "$LOGS_DIR"

command -v uv >/dev/null 2>&1 || { echo "ERROR: uv not installed. See https://docs.astral.sh/uv/" >&2; exit 1; }

# ---- 1. venv (idempotent) ------------------------------------------------
if [ ! -x "$VENV/bin/python" ]; then
  echo "[setup] creating venv at $VENV (python 3.12)..." >&2
  uv venv --python 3.12 "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# ---- 2. deps -------------------------------------------------------------
echo "[setup] installing deps (haqumei==0.8.0 pyopenjtalk-plus pyyaml jiwer tqdm)..." >&2
# NB: pyopenjtalk-plus is the ~800K-entry-dict fork mandated by CLAUDE.md
# (§ "設計の核心思想 5"). Falls back to pyopenjtalk if the plus fork is not
# available on the current index — Phase 0 finish critique F3.
uv pip install --quiet haqumei==0.8.0 pyyaml jiwer tqdm
uv pip install --quiet pyopenjtalk-plus || uv pip install --quiet pyopenjtalk

# ---- 3. datasets (idempotent) --------------------------------------------
clone_if_missing() {
  local url="$1" dest="$2"
  if [ ! -d "$dest/.git" ]; then
    echo "[data] cloning $url -> $dest" >&2
    git clone --depth 1 "$url" "$dest"
  else
    echo "[data] present: $dest" >&2
  fi
}

clone_if_missing https://github.com/prj-beatrice/jsut-label.git         "$DATA_DIR/jsut-label"
clone_if_missing https://github.com/Hiroshiba/jvs_hiho.git              "$DATA_DIR/jvs_hiho"      || true
clone_if_missing https://github.com/mmorise/rohan4600.git               "$DATA_DIR/rohan4600"    || true

export JSUT_YAML="$DATA_DIR/jsut-label/text_kana/basic5000.yaml"
export JVS_DIR="$DATA_DIR/jvs_hiho"
export ROHAN_DIR="$DATA_DIR/rohan4600"

# ---- 4. eval runner ------------------------------------------------------
# run_eval <name> <script> <metric_key>  -> emits a JSON fragment on stdout
run_eval() {
  local name="$1" script="$2" metric="$3"
  local log="$LOGS_DIR/${name}_${TIMESTAMP}.log"
  if [ ! -f "$script" ]; then
    echo "[eval] SKIP $name ($(basename "$script") not present)" >&2
    printf '{"name":"%s","status":"skipped","reason":"script_missing","log":"%s"}' "$name" ""
    return 0
  fi
  echo "[eval] running $name -> $log" >&2
  if ! python "$script" >"$log" 2>&1; then
    echo "[eval] FAILED $name (last 30 lines):" >&2
    tail -30 "$log" >&2
    printf '{"name":"%s","status":"failed","log":"%s"}' "$name" "$log"
    return 1
  fi
  # metric_key like PER / CER / KER — grab the LAST occurrence
  local value
  value=$(grep -oE "${metric}=[0-9.]+%" "$log" | tail -1 | sed -E "s/${metric}=//;s/%//" || true)
  if [ -z "$value" ]; then
    echo "[eval] WARNING $name: could not parse ${metric} from log" >&2
    printf '{"name":"%s","status":"parsed_failed","metric":"%s","log":"%s"}' "$name" "$metric" "$log"
    return 0
  fi
  printf '{"name":"%s","status":"ok","metric":"%s","value":%s,"unit":"%%","log":"%s"}' \
    "$name" "$metric" "$value" "$log"
}

skipped_target() {
  local name="$1"
  printf '{"name":"%s","status":"filtered","reason":"target_arg=%s"}' "$name" "$TARGET"
}

# fail-fast: if a script exists but crashes, we abort. Missing scripts skip.
if [ "$TARGET" = "all" ] || [ "$TARGET" = "haqumei_jsut" ]; then
  HAQ_JSUT_JSON=$(run_eval haqumei_jsut  "$REPO/scripts/eval_haqumei_jsut.py"    PER)
else
  HAQ_JSUT_JSON=$(skipped_target haqumei_jsut)
fi
if [ "$TARGET" = "all" ] || [ "$TARGET" = "pyopenjtalk_jvs" ]; then
  PYOJT_JVS_JSON=$(run_eval pyopenjtalk_jvs "$REPO/scripts/eval_pyopenjtalk_jvs.py" CER)
else
  PYOJT_JVS_JSON=$(skipped_target pyopenjtalk_jvs)
fi
if [ "$TARGET" = "all" ] || [ "$TARGET" = "haqumei_rohan" ]; then
  HAQ_ROHAN_JSON=$(run_eval haqumei_rohan "$REPO/scripts/eval_haqumei_rohan.py"    KER)
else
  HAQ_ROHAN_JSON=$(skipped_target haqumei_rohan)
fi

# ---- 5. persist JSON -----------------------------------------------------
# For a single-target run we write to results/baselines_<target>.json so
# concurrent matrix jobs (test-integration.yml) do not clobber each other.
if [ "$TARGET" = "all" ]; then
  LATEST_JSON="$RESULTS_DIR/baselines.json"
  HISTORY_JSON="$HISTORY_DIR/baselines_all_${TIMESTAMP}.json"
else
  LATEST_JSON="$RESULTS_DIR/baselines_${TARGET}.json"
  HISTORY_JSON="$HISTORY_DIR/baselines_${TARGET}_${TIMESTAMP}.json"
fi

python - "$LATEST_JSON" "$HISTORY_JSON" "$TIMESTAMP" \
  "$HAQ_JSUT_JSON" "$PYOJT_JVS_JSON" "$HAQ_ROHAN_JSON" <<'PY'
import json, sys, datetime
latest, hist, ts, *entries = sys.argv[1:]
payload = {
    "timestamp": ts,
    "iso": datetime.datetime.now().isoformat(timespec="seconds"),
    "results": [json.loads(e) for e in entries],
}
for path in (latest, hist):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
print(f"[save] {latest}", file=sys.stderr)
print(f"[save] {hist}", file=sys.stderr)
PY

# ---- 6. tabular stdout ---------------------------------------------------
echo
printf '%-20s %-8s %-10s %-12s %s\n' "BASELINE" "METRIC" "VALUE" "STATUS" "LOG"
printf '%-20s %-8s %-10s %-12s %s\n' "--------------------" "--------" "----------" "------------" "----------------------------------------"
python - "$HAQ_JSUT_JSON" "$PYOJT_JVS_JSON" "$HAQ_ROHAN_JSON" <<'PY'
import json, sys, os
for raw in sys.argv[1:]:
    r = json.loads(raw)
    val = f"{r['value']:.4f}%" if r.get("status") == "ok" else "-"
    print(f"{r['name']:<20} {r.get('metric','-'):<8} {val:<10} {r['status']:<12} {os.path.basename(r.get('log','')) or '-'}")
PY

echo
echo "results -> $LATEST_JSON"
