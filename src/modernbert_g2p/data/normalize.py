"""Bit-exact canonical normalization matching scripts/eval_haqumei_jsut.py.

Implements docs/design/phase1_data_pipeline.md §4 (Normalization spec).
"""

from __future__ import annotations

import functools
from collections.abc import Iterable
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

IGNORED_PHONEMES: frozenset[str] = frozenset({"pau"})

DEVOICING_MAP = MappingProxyType(
    {
        "A": "a",
        "E": "e",
        "I": "i",
        "O": "o",
        "U": "u",
    }
)


@dataclass(frozen=True, slots=True)
class NormalizedText:
    phonemes: tuple[str, ...]
    mora_accents: tuple[str, ...]
    accent_boundaries: tuple[int, ...]


@functools.lru_cache(maxsize=1)
def get_haqumei() -> Any:
    """Return a lazily-constructed Haqumei singleton matching scripts/eval_haqumei_jsut.py."""
    from haqumei import Haqumei, IuPronunciation

    return Haqumei(use_unidic_yomi=True, normalize_iu=IuPronunciation.Yuu)


def devoice_phonemes(phonemes: Iterable[str]) -> tuple[str, ...]:
    """Lowercase A/E/I/O/U to a/e/i/o/u; preserve N and all other tokens verbatim."""
    return tuple(DEVOICING_MAP.get(p, p) for p in phonemes)


def accent_nucleus_to_hl(nucleus: int, mora_count: int) -> tuple[str, ...]:
    """Convert (accent_nucleus, mora_count) to an H/L tuple of length mora_count.

    Semantics (classical Japanese accent, matches haqumei/OpenJTalk):
      - nucleus == 0 (heiban): mora 0 = L, moras 1.. = H (1-mora words: (L,)).
      - nucleus == 1 (atamadaka): mora 0 = H, moras 1.. = L.
      - nucleus >= 2 (nakadaka/odaka): mora 0 = L, moras 1..nucleus-1 = H, moras nucleus.. = L.
    """
    if mora_count <= 0:
        return ()
    if nucleus == 0:
        if mora_count == 1:
            return ("L",)
        return ("L", *("H",) * (mora_count - 1))
    first = "H" if nucleus == 1 else "L"
    body = ("H",) * max(0, nucleus - 1)
    tail = ("L",) * (mora_count - nucleus)
    return (first, *body, *tail)


def _read_attr(obj: object, name: str) -> Any:
    if isinstance(obj, dict):
        return obj[name]
    return getattr(obj, name)


def _try_read_attr(obj: object, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _extract_word_prosody(word: object) -> tuple[int, int, int]:
    nucleus = int(_read_attr(word, "accent_nucleus"))
    mora_count = int(_read_attr(word, "mora_count"))
    chain_flag = int(_try_read_attr(word, "chain_flag", -1))
    return nucleus, mora_count, chain_flag


def normalize_from_haqumei(text: str, *, hq: object | None = None) -> NormalizedText:
    """Run haqumei and return devoiced, pau-stripped phonemes plus per-mora H/L labels.

    Empty text short-circuits to an empty NormalizedText without touching haqumei.
    """
    if not text:
        return NormalizedText(phonemes=(), mora_accents=(), accent_boundaries=())

    hq = hq if hq is not None else get_haqumei()

    raw_phonemes: list[str] = hq.g2p_batch([text])[0]  # type: ignore[attr-defined]
    devoiced = devoice_phonemes(raw_phonemes)
    phonemes = tuple(p for p in devoiced if p not in IGNORED_PHONEMES)

    words = hq.g2p_mapping_prosody(text)  # type: ignore[attr-defined]

    mora_accents: list[str] = []
    accent_boundaries: list[int] = []

    for idx, word in enumerate(words):
        try:
            nucleus, mora_count, chain_flag = _extract_word_prosody(word)
        except (AttributeError, KeyError, ValueError, TypeError):
            continue

        if mora_count <= 0:
            continue

        if idx > 0 and chain_flag == 0:
            accent_boundaries.append(len(mora_accents))

        mora_accents.extend(accent_nucleus_to_hl(nucleus, mora_count))

    return NormalizedText(
        phonemes=phonemes,
        mora_accents=tuple(mora_accents),
        accent_boundaries=tuple(accent_boundaries),
    )
