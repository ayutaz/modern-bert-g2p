"""Kana Character Error Rate (CER) — canonical implementation.

Used against JVS-3000 (Koriyama Interspeech 2026 benchmark). The unit is a
single kana character; edit ops are computed with the same Levenshtein
backtracker as :func:`modernbert_g2p.metrics.compute_per`.

Character normalization (``char_normalize=True``, the default):

- Unicode NFKC normalization (folds full-width digits/letters, compat forms).
- Hiragana → Katakana (``\\u3041–\\u3096`` → ``\\u30A1–\\u30F6``) so mixed
  transcripts score consistently.
- Whitespace characters are stripped.

Callers who need to score against a strict kana-only reference should
pre-filter their strings; this function intentionally preserves punctuation
so it will surface as substitutions/insertions if a system emits them.
"""

from __future__ import annotations

import unicodedata

from modernbert_g2p.metrics._edit_distance import _summarize, levenshtein_sdi

_HIRAGANA_START = 0x3041
_HIRAGANA_END = 0x3096
_HIRA_TO_KATA_OFFSET = 0x30A1 - 0x3041


def _hira_to_kata(ch: str) -> str:
    cp = ord(ch)
    if _HIRAGANA_START <= cp <= _HIRAGANA_END:
        return chr(cp + _HIRA_TO_KATA_OFFSET)
    return ch


def _normalize_kana(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return "".join(_hira_to_kata(c) for c in text if not c.isspace())


def compute_cer(hyp: str, ref: str, char_normalize: bool = True) -> dict[str, float | int]:
    """Compute character-level CER for a single (hyp, ref) pair.

    Args:
        hyp: Predicted kana transcript.
        ref: Ground-truth kana transcript.
        char_normalize: Apply NFKC + hiragana→katakana + whitespace stripping
            before scoring. Default ``True``.

    Returns:
        Dict with keys ``n`` (reference length after normalization), ``s``,
        ``d``, ``i``, ``per`` (percentage — key kept as ``per`` for schema
        parity with :func:`compute_per` / :func:`compute_ker`), ``accuracy``.
    """
    if char_normalize:
        hyp_norm = _normalize_kana(hyp)
        ref_norm = _normalize_kana(ref)
    else:
        hyp_norm = hyp
        ref_norm = ref

    hyp_chars = list(hyp_norm)
    ref_chars = list(ref_norm)
    s, d, i = levenshtein_sdi(hyp_chars, ref_chars)
    return _summarize(s, d, i, len(ref_chars))


__all__ = ["compute_cer"]
