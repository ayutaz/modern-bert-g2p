"""Phoneme Error Rate (PER) — canonical implementation.

Protocol notes (matches haqumei-eval Rust reference, verified 2026-07-03 to
reproduce JSUT Basic5000 PER 1.17% within 0.005 pt):

- Token unit is a JULIUS phoneme (e.g. ``a``, ``k``, ``ky``, ``N``).
- ``pau`` is discarded from both hypothesis and reference before scoring.
- Devoiced vowels ``A/E/I/O/U`` are lowercased to ``a/e/i/o/u`` when
  ``devoicing_normalize=True``; ``N`` (moraic nasal) is preserved.
- PER = (S + D + I) / N_ref * 100.

See :file:`docs/requirements.md` (FR-EVAL-*) for the wider evaluation contract.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence

from modernbert_g2p.metrics._edit_distance import _summarize, levenshtein_sdi

_DEVOICED = frozenset({"A", "E", "I", "O", "U"})

DEFAULT_IGNORE: frozenset[str] = frozenset({"pau"})


def _normalize_phoneme(p: str) -> str:
    return p.lower() if p in _DEVOICED else p


def compute_per(
    hypothesis: Sequence[str],
    reference: Sequence[str],
    ignore: Collection[str] = DEFAULT_IGNORE,
    devoicing_normalize: bool = True,
) -> dict[str, float | int]:
    """Compute Phoneme Error Rate for a single (hyp, ref) pair.

    Args:
        hypothesis: Predicted JULIUS phoneme sequence.
        reference: Ground-truth JULIUS phoneme sequence.
        ignore: Token set to strip from both sequences before scoring
            (defaults to ``{"pau"}``, matching the haqumei protocol).
        devoicing_normalize: If ``True``, map ``A/E/I/O/U`` → ``a/e/i/o/u``.
            ``N`` is intentionally preserved. Applied to both hyp and ref.

    Returns:
        Dict with keys ``n`` (reference length after filtering), ``s``, ``d``,
        ``i`` (raw edit-op counts), ``per`` (percentage), ``accuracy``
        (``100 - per`` clamped at 0). All values match the canonical schema
        used across :mod:`modernbert_g2p.metrics`.
    """
    ignore_set = frozenset(ignore)

    def _prepare(seq: Sequence[str]) -> list[str]:
        out: list[str] = []
        for tok in seq:
            if tok in ignore_set:
                continue
            out.append(_normalize_phoneme(tok) if devoicing_normalize else tok)
        return out

    hyp = _prepare(hypothesis)
    ref = _prepare(reference)
    s, d, i = levenshtein_sdi(hyp, ref)
    return _summarize(s, d, i, len(ref))


__all__ = ["compute_per", "DEFAULT_IGNORE"]
