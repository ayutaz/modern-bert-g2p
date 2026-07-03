"""Shared token-level Levenshtein backtracker returning (S, D, I) counts.

Canonicalized from ``scripts/eval_haqumei_jsut.py`` (verified 2026-07-03 to
reproduce haqumei's JSUT Basic5000 PER 1.17% within 0.005 pt).

The routine returns raw substitution / deletion / insertion counts against the
reference sequence, matching the definition used by haqumei-eval's Rust code
and by the classic ASR PER/WER convention: ``PER = (S + D + I) / N_ref``.
"""

from __future__ import annotations

from collections.abc import Sequence


def levenshtein_sdi(hyp: Sequence[str], ref: Sequence[str]) -> tuple[int, int, int]:
    """Return ``(S, D, I)`` — substitutions, deletions, insertions.

    - ``D`` counts reference tokens missing from the hypothesis.
    - ``I`` counts hypothesis tokens absent from the reference.
    - ``S`` counts positions where both differ.

    Matches
    ``modernbert_g2p.metrics.per.compute_per`` semantics: an empty reference
    with a non-empty hypothesis returns ``(0, 0, len(hyp))``; an empty
    hypothesis against a non-empty reference returns ``(0, len(ref), 0)``.
    """
    n, m = len(ref), len(hyp)
    if n == 0:
        return 0, 0, m
    if m == 0:
        return 0, n, 0

    dp: list[list[int]] = [[0] * (m + 1) for _ in range(n + 1)]
    back: list[list[str | None]] = [[None] * (m + 1) for _ in range(n + 1)]
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

    s = d = i_count = 0
    i, j = n, m
    while i > 0 or j > 0:
        op = back[i][j]
        if op == "=":
            i -= 1
            j -= 1
        elif op == "S":
            s += 1
            i -= 1
            j -= 1
        elif op == "D":
            d += 1
            i -= 1
        elif op == "I":
            i_count += 1
            j -= 1
        else:
            break
    return s, d, i_count


def _summarize(s: int, d: int, i: int, n_ref: int) -> dict[str, float | int]:
    """Build the canonical metrics return dict from raw counts."""
    errors = s + d + i
    per = errors / n_ref * 100 if n_ref > 0 else 0.0
    # Accuracy is clamped at 0 because insertions can push errors above N_ref.
    accuracy = max(0.0, 100.0 - per)
    return {
        "n": n_ref,
        "s": s,
        "d": d,
        "i": i,
        "per": per,
        "accuracy": accuracy,
    }


__all__ = ["levenshtein_sdi", "_summarize"]
