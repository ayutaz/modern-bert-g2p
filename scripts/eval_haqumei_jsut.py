"""Reproduce haqumei's JSUT Basic5000 PER 1.17% measurement.

Verified 2026-07-03 with haqumei 0.8.0 on macOS aarch64:
  Ours:     PER 1.1657% (S=2107 D=540 I=825 N=297843)
  Official: PER 1.17%   (S=2117 D=527 I=831 N=297843)
  Diff: 0.0043 pt (AC-P0 requirement <= 0.1% PASSED)

Usage:
  # 1. Install haqumei in a Python 3.12 venv (Python 3.14 not supported yet)
  uv venv --python 3.12 .venv
  source .venv/bin/activate
  uv pip install -e ".[dev]"

  # 2. Clone JSUT-label
  git clone --depth 1 https://github.com/prj-beatrice/jsut-label.git

  # 3. Run
  JSUT_YAML=./jsut-label/text_kana/basic5000.yaml python scripts/eval_haqumei_jsut.py

Protocol (matches haqumei-eval/src/main.rs):
- Input text: yaml `text_level2` field (NOT text_level0; they differ in 153 sentences)
- HaqumeiOptions: use_unidic_yomi=True, normalize_iu=IuPronunciation.Yuu
- Reference: yaml `phone_level3` split by '-'
- Ignore `pau` in both hypothesis and reference
- Devoicing normalization: A/E/I/O/U -> a/e/i/o/u (haqumei-eval lowercases uppercase phonemes except N)
- Metric: token-level Levenshtein distance, PER = (S+D+I) / N_ref * 100

Uses the canonical PER implementation in :mod:`modernbert_g2p.metrics.per`
so any protocol change is reflected in both baseline reproduction and unit tests.
"""
import os
import sys
from pathlib import Path

# Ensure `src/` is importable when running from a source checkout without an
# editable install. Safe to keep even after `uv pip install -e .`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import yaml  # noqa: E402
from haqumei import Haqumei, IuPronunciation  # noqa: E402

from modernbert_g2p.metrics.per import compute_per  # noqa: E402

JSUT_YAML = Path(os.environ.get("JSUT_YAML", "jsut-label/text_kana/basic5000.yaml"))


def main():
    if not JSUT_YAML.exists():
        sys.exit(f"JSUT yaml not found at {JSUT_YAML}. Set JSUT_YAML env var or clone jsut-label first.")

    print(f"Loading {JSUT_YAML}...", file=sys.stderr)
    with open(JSUT_YAML) as f:
        data = yaml.safe_load(f)
    print(f"Loaded {len(data)} entries", file=sys.stderr)

    hq = Haqumei(use_unidic_yomi=True, normalize_iu=IuPronunciation.Yuu)

    ids = list(data.keys())
    texts = [data[i].get("text_level2") or data[i].get("text_level0") or "" for i in ids]
    refs = [data[i].get("phone_level3", "").split("-") for i in ids]

    print(f"Running g2p_batch on {len(texts)} sentences...", file=sys.stderr)
    hyps_all = hq.g2p_batch(texts)

    S_total = D_total = I_total = N_total = 0
    for hyp_all, ref in zip(hyps_all, refs, strict=True):
        # compute_per handles ignore={"pau"} and devoicing normalization; passing
        # raw sequences here keeps this script in lockstep with unit tests.
        result = compute_per(hyp_all, ref)
        S_total += int(result["s"])
        D_total += int(result["d"])
        I_total += int(result["i"])
        N_total += int(result["n"])

    per = (S_total + D_total + I_total) / max(N_total, 1) * 100
    print()
    print("=== haqumei JSUT Basic5000 PER (Python replication) ===")
    print(f"N={N_total}  S={S_total}  D={D_total}  I={I_total}  PER={per:.4f}%")
    print("Official (haqumei README): N=297843  S=2117  D=527  I=831  PER=1.17%")
    print(f"Diff: {abs(per - 1.17):.4f} pt")
    if abs(per - 1.17) <= 0.1:
        print(">>> AC-P0 PASSED (within +/- 0.1%)")
    else:
        print(">>> AC-P0 NOT MET")
    return 0 if abs(per - 1.17) <= 0.1 else 1


if __name__ == "__main__":
    sys.exit(main())
