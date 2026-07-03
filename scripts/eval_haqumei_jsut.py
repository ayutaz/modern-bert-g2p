"""Reproduce haqumei's JSUT Basic5000 PER 1.17% measurement.

Verified 2026-07-03 with haqumei 0.8.0 on macOS aarch64:
  Ours:     PER 1.1657% (S=2107 D=540 I=825 N=297843)
  Official: PER 1.17%   (S=2117 D=527 I=831 N=297843)
  Diff: 0.0043 pt (AC-P0 requirement <= 0.1% PASSED)

Usage:
  # 1. Install haqumei in a Python 3.12 venv (Python 3.14 not supported yet)
  uv venv --python 3.12 .venv
  source .venv/bin/activate
  uv pip install haqumei==0.8.0 pyyaml

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
"""
import os
import sys
from pathlib import Path

import yaml
from haqumei import Haqumei, IuPronunciation


JSUT_YAML = Path(os.environ.get("JSUT_YAML", "jsut-label/text_kana/basic5000.yaml"))
DEVOICED = {"A", "E", "I", "O", "U"}


def normalize_phoneme(p: str) -> str:
    return p.lower() if p in DEVOICED else p


def levenshtein(hyp, ref):
    n, m = len(ref), len(hyp)
    if n == 0:
        return 0, 0, m
    if m == 0:
        return 0, n, 0
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
        back[i][0] = "D"
    for j in range(m + 1):
        dp[0][j] = j
        back[0][j] = "I"
    back[0][0] = None
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                back[i][j] = "="
            else:
                sub = dp[i - 1][j - 1] + 1
                dele = dp[i - 1][j] + 1
                ins = dp[i][j - 1] + 1
                best = min(sub, dele, ins)
                dp[i][j] = best
                back[i][j] = "S" if best == sub else ("D" if best == dele else "I")
    S = D = I = 0
    i, j = n, m
    while i > 0 or j > 0:
        op = back[i][j]
        if op == "=":
            i -= 1; j -= 1
        elif op == "S":
            S += 1; i -= 1; j -= 1
        elif op == "D":
            D += 1; i -= 1
        elif op == "I":
            I += 1; j -= 1
        else:
            break
    return S, D, I


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
    refs = [
        [p for p in data[i].get("phone_level3", "").split("-") if p != "pau"]
        for i in ids
    ]

    print(f"Running g2p_batch on {len(texts)} sentences...", file=sys.stderr)
    hyps_all = hq.g2p_batch(texts)

    S_total = D_total = I_total = N_total = 0
    for hyp_all, ref in zip(hyps_all, refs):
        hyp = [normalize_phoneme(p) for p in hyp_all if p != "pau"]
        S, D, I = levenshtein(hyp, ref)
        S_total += S
        D_total += D
        I_total += I
        N_total += len(ref)

    per = (S_total + D_total + I_total) / max(N_total, 1) * 100
    print()
    print("=== haqumei JSUT Basic5000 PER (Python replication) ===")
    print(f"N={N_total}  S={S_total}  D={D_total}  I={I_total}  PER={per:.4f}%")
    print(f"Official (haqumei README): N=297843  S=2117  D=527  I=831  PER=1.17%")
    print(f"Diff: {abs(per - 1.17):.4f} pt")
    if abs(per - 1.17) <= 0.1:
        print(">>> AC-P0 PASSED (within +/- 0.1%)")
    else:
        print(">>> AC-P0 NOT MET")
    return 0 if abs(per - 1.17) <= 0.1 else 1


if __name__ == "__main__":
    sys.exit(main())
