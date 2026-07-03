#!/usr/bin/env python3
"""Reproduce haqumei's ROHAN 4600 KANA-level KER (nominal 1.64%).

Replicates ``haqumei-eval::evaluate_kana_dataset`` for the ROHAN4600 dataset:

- Data source: ``resources/Rohan4600_transcript_utf8.txt`` from the haqumei repo,
  same file that ``haqumei-eval/build.rs`` reads at build time.
- Line format: ``ID:text,kana``. Anything inside ``(...)`` in the text column
  is stripped along with the parentheses themselves (matches build.rs).
- Model: ``Haqumei(revert_long_vowels=True, revert_yotsugana=True)`` — the exact
  options used for the ROHAN kana evaluation in ``haqumei-eval/src/main.rs`` (all
  other options left at defaults, i.e. ``use_unidic_yomi=False``).
- Prediction: ``g2k_per_word_batch`` then concatenated per sentence, matching
  the Rust eval's ``g2k_per_word(text)`` + ``push_str`` loop.
- Metric: character-level Levenshtein distance (S+D+I) over the concatenated
  kana strings, summed over all sentences, divided by the total number of gold
  kana characters. Reported as ``100 * total_edits / total_chars`` (%).

Usage
-----
    /path/to/haqumei_repro/.venv/bin/python scripts/eval_haqumei_rohan.py \
        --transcript /path/to/haqumei_source/haqumei/resources/Rohan4600_transcript_utf8.txt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Levenshtein (character-level, unit costs) — mirrors compute_edit_ops in
# haqumei-eval/src/main.rs.
# ---------------------------------------------------------------------------
def levenshtein_sdi(expected: str, actual: str) -> Tuple[int, int, int]:
    m = len(expected)
    n = len(actual)
    if m == 0:
        return 0, 0, n
    if n == 0:
        return 0, m, 0

    # dp[i][j] = edit distance between expected[:i] and actual[:j]
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        curr[0] = i
        e_i = expected[i - 1]
        for j in range(1, n + 1):
            cost = 0 if e_i == actual[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,          # deletion
                curr[j - 1] + 1,      # insertion
                prev[j - 1] + cost,   # substitution / equal
            )
        prev, curr = curr, prev

    # We need S, D, I counts — reconstruct via a second pass with a full DP
    # matrix. Doing this on 4600 short strings is cheap enough.
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        e_i = expected[i - 1]
        row = dp[i]
        prow = dp[i - 1]
        for j in range(1, n + 1):
            cost = 0 if e_i == actual[j - 1] else 1
            row[j] = min(prow[j] + 1, row[j - 1] + 1, prow[j - 1] + cost)

    # Backtrack (same tie-breaking order as haqumei-eval::compute_edit_ops).
    s = d = ins = 0
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] and expected[i - 1] == actual[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            s += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            d += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return s, d, ins


# ---------------------------------------------------------------------------
# Data loading — mirrors the ROHAN parsing loop in haqumei-eval/build.rs.
# ---------------------------------------------------------------------------
def _strip_parens(text: str) -> str:
    out = []
    in_paren = False
    for ch in text:
        if ch == "(":
            in_paren = True
        elif ch == ")":
            in_paren = False
        elif not in_paren:
            out.append(ch)
    return "".join(out)


def load_rohan(path: Path) -> Tuple[List[str], List[str]]:
    texts: List[str] = []
    kanas: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line:
                continue
            # Split on the FIRST ':' — matches Rust's split_once(':').
            id_sep = line.find(":")
            if id_sep < 0:
                continue
            pair = line[id_sep + 1 :]
            # Split on the FIRST ',' — matches Rust's split_once(',').
            comma = pair.find(",")
            if comma < 0:
                continue
            text = pair[:comma]
            kana = pair[comma + 1 :]
            texts.append(_strip_parens(text))
            kanas.append(kana)
    return texts, kanas


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Reproduce haqumei ROHAN KER.")
    parser.add_argument(
        "--transcript",
        type=Path,
        default=Path(
            "/private/tmp/claude-1518468357/-Users-s19447-Desktop-modern-bert-g2p/"
            "45bf4e0a-a6e3-4bc2-b0d3-0d4204d41b28/scratchpad/haqumei_source/haqumei/"
            "resources/Rohan4600_transcript_utf8.txt"
        ),
        help="Path to Rohan4600_transcript_utf8.txt in the haqumei repo.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--target-ker",
        type=float,
        default=1.64,
        help="Nominal published KER (%) to compare against.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.10,
        help="Absolute tolerance in percentage points.",
    )
    args = parser.parse_args()

    from haqumei import Haqumei

    texts, gold_kanas = load_rohan(args.transcript)
    assert len(texts) == len(gold_kanas) == 4600, (
        f"Expected 4600 rows; got texts={len(texts)}, kanas={len(gold_kanas)}"
    )

    hq = Haqumei(revert_long_vowels=True, revert_yotsugana=True)

    t0 = time.time()
    preds: List[str] = []
    for start in range(0, len(texts), args.batch_size):
        chunk = texts[start : start + args.batch_size]
        per_word_batch = hq.g2k_per_word_batch(chunk)
        for parts in per_word_batch:
            preds.append("".join(parts))
    infer_secs = time.time() - t0

    total_s = total_d = total_i = 0
    total_chars = 0
    sent_errors = 0
    for gold, pred in zip(gold_kanas, preds):
        s, d, ins = levenshtein_sdi(gold, pred)
        total_s += s
        total_d += d
        total_i += ins
        total_chars += len(gold)
        if (s + d + ins) > 0:
            sent_errors += 1

    total_edits = total_s + total_d + total_i
    ker = 100.0 * total_edits / max(1, total_chars)
    exact_acc = 100.0 * (len(texts) - sent_errors) / len(texts)

    delta = ker - args.target_ker
    within = abs(delta) <= args.tolerance

    print("=" * 60)
    print("haqumei ROHAN 4600 KANA-level KER reproduction")
    print("=" * 60)
    print(f"Total sentences        : {len(texts)}")
    print(f"Sentences with errors  : {sent_errors}")
    print(f"Exact-match accuracy   : {exact_acc:.2f}%")
    print(
        f"Overall KANA KER       : {ker:.4f}%  "
        f"(S={total_s} D={total_d} I={total_i} N={total_chars})"
    )
    print(f"Nominal published KER  : {args.target_ker:.4f}%")
    print(f"Delta (measured-target): {delta:+.4f} pp")
    print(f"Within ±{args.tolerance:.2f} pp        : {'YES' if within else 'NO'}")
    print(f"Inference wall-time    : {infer_secs:.1f}s")
    print("=" * 60)

    return 0 if within else 1


if __name__ == "__main__":
    sys.exit(main())
