"""Katakana Kana Error Rate (KER) — canonical implementation.

Used against ROHAN 4600 with the haqumei-eval protocol. KER is character-level
Levenshtein over katakana, with hiragana folded to katakana and whitespace
stripped so the reference and hypothesis are on the same alphabet.

Unlike :func:`modernbert_g2p.metrics.compute_cer`, ``compute_ker`` also strips
JULIUS phrase-boundary markers (``|``, ``/``) if they appear, since ROHAN
annotations sometimes carry accent phrase delimiters that are not part of the
kana surface form we intend to score.
"""

from __future__ import annotations

import unicodedata

from modernbert_g2p.metrics._edit_distance import _summarize, levenshtein_sdi

_HIRAGANA_START = 0x3041
_HIRAGANA_END = 0x3096
_HIRA_TO_KATA_OFFSET = 0x30A1 - 0x3041
_STRIP_CHARS = frozenset({"|", "/", " ", "　"})


def _hira_to_kata(ch: str) -> str:
    cp = ord(ch)
    if _HIRAGANA_START <= cp <= _HIRAGANA_END:
        return chr(cp + _HIRA_TO_KATA_OFFSET)
    return ch


def _normalize_katakana(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    out: list[str] = []
    for c in text:
        if c in _STRIP_CHARS or c.isspace():
            continue
        out.append(_hira_to_kata(c))
    return "".join(out)


def compute_ker(hyp: str, ref: str) -> dict[str, float | int]:
    """Compute katakana KER for a single (hyp, ref) pair.

    Args:
        hyp: Predicted transcript (hiragana or katakana; folded to katakana).
        ref: Ground-truth transcript.

    Returns:
        Dict matching the schema of :func:`compute_per` / :func:`compute_cer`.
    """
    hyp_chars = list(_normalize_katakana(hyp))
    ref_chars = list(_normalize_katakana(ref))
    s, d, i = levenshtein_sdi(hyp_chars, ref_chars)
    return _summarize(s, d, i, len(ref_chars))


__all__ = ["compute_ker"]
