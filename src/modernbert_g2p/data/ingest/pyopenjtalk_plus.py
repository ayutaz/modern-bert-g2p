"""S1 ingest: parse mecab-naist-jdic CSV rows into Row records.

Implements docs/design/phase1_data_pipeline.md §3 (S1 pyopenjtalk-plus).
"""

from __future__ import annotations

import csv
import warnings
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import ClassVar

from modernbert_g2p.data.ingest.base import (
    is_katakana,
    kana_to_julius_phonemes,
    kana_to_mora_count,
    normalize_katakana,
    register,
)
from modernbert_g2p.data.normalize import accent_nucleus_to_hl
from modernbert_g2p.data.schema import Row, make_id, validate_row
from modernbert_g2p.data.weighting import weight_for

NAIST_JDIC_COLS: int = 15

_COL_SURFACE = 0
_COL_POS1 = 4
_COL_POS2 = 5
_COL_PRON = 11
_COL_ACCENT = 13
_COL_CHAIN = 14


def _parse_accent(accent_field: str) -> tuple[int, int | None]:
    if not accent_field or accent_field == "*":
        return 0, None
    head = accent_field.split(",", 1)[0].strip()
    head = head.split(";", 1)[0].strip()
    if not head:
        return 0, None
    if "/" in head:
        left, _, right = head.partition("/")
        try:
            nucleus = int(left)
        except ValueError:
            return 0, None
        mora: int | None
        try:
            mora = int(right) if right else None
        except ValueError:
            mora = None
        return nucleus, mora
    try:
        return int(head), None
    except ValueError:
        return 0, None


def _is_all_katakana(kana: str) -> bool:
    return bool(kana) and all(is_katakana(c) for c in kana)


def _is_all_kanji(surface: str) -> bool:
    if not surface:
        return False
    return all("一" <= c <= "鿿" for c in surface)


def _has_ascii_letter(s: str) -> bool:
    return any(("a" <= c <= "z") or ("A" <= c <= "Z") for c in s)


def _has_non_ascii(s: str) -> bool:
    return any(ord(c) >= 128 for c in s)


def _has_digit(s: str) -> bool:
    return any(c.isdigit() for c in s)


def _detect_category(surface: str, pos1: str, pos2: str) -> str:
    if _has_ascii_letter(surface):
        if _has_non_ascii(surface):
            return "english_mixed"
        letters_only = "".join(c for c in surface if c.isalpha())
        if len(letters_only) >= 2 and letters_only.isupper():
            return "english_abbreviation"
        return "english_mixed"
    if _has_digit(surface) and _has_non_ascii(surface):
        return "numeric_unit"
    if _is_all_katakana(surface):
        return "loanword"
    if _is_all_kanji(surface) and pos1 == "名詞" and "固有名詞" in pos2:
        return "proper_noun_kanji"
    return "general"


def parse_naist_jdic_row(fields: Sequence[str]) -> Row | None:
    """Parse one 15-column mecab-naist-jdic row into a Row, or None if skippable."""
    if len(fields) != NAIST_JDIC_COLS:
        return None

    surface = fields[_COL_SURFACE]
    if not surface:
        return None

    pron_raw = fields[_COL_PRON]
    if not pron_raw:
        return None

    kana = normalize_katakana(pron_raw)
    if not _is_all_katakana(kana):
        return None

    try:
        phonemes = kana_to_julius_phonemes(kana)
    except ValueError:
        return None
    if not phonemes:
        return None

    nucleus, _mora_from_field = _parse_accent(fields[_COL_ACCENT])
    mora_count = kana_to_mora_count(kana)

    if mora_count <= 0:
        mora_accents: tuple[str, ...] = ()
    elif nucleus < 0 or nucleus > mora_count:
        mora_accents = ("L",) * mora_count
    else:
        mora_accents = accent_nucleus_to_hl(nucleus, mora_count)

    pos1 = fields[_COL_POS1]
    pos2 = fields[_COL_POS2]
    category = _detect_category(surface, pos1, pos2)

    row = Row(
        id=make_id("pyopenjtalk_plus", surface, phonemes),
        source="pyopenjtalk_plus",
        source_license="BSD3",
        text=surface,
        phonemes=phonemes,
        mora_accents=mora_accents,
        accent_boundaries=(),
        category=category,
        sample_weight=weight_for(category),
        extra={
            "pos1": pos1,
            "pos2": pos2,
            "accent_type": nucleus,
            "accent_raw": fields[_COL_ACCENT],
            "chain_flag": fields[_COL_CHAIN],
        },
    )
    try:
        validate_row(row)
    except ValueError:
        return None
    return row


@register("pyopenjtalk_plus")
class PyopenjtalkPlusSource:
    """S1 ingest source for mecab-naist-jdic CSV shipped with pyopenjtalk-plus."""

    source: ClassVar[str] = "pyopenjtalk_plus"
    source_license: ClassVar[str] = "BSD3"

    def entries(self, root: Path, *, limit: int | None = None) -> Iterator[Row]:
        paths = self._collect_paths(root)
        count = 0
        for path in paths:
            with path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                for line_idx, fields in enumerate(reader, start=1):
                    if not fields:
                        continue
                    row = parse_naist_jdic_row(fields)
                    if row is None:
                        warnings.warn(
                            f"skipping malformed row in {path.name}:{line_idx}",
                            stacklevel=2,
                        )
                        continue
                    yield row
                    count += 1
                    if limit is not None and count >= limit:
                        return

    @staticmethod
    def _collect_paths(root: Path) -> list[Path]:
        if root.is_file():
            return [root]
        if not root.exists():
            raise FileNotFoundError(str(root))
        return sorted(root.glob("*.csv"))


__all__ = [
    "NAIST_JDIC_COLS",
    "PyopenjtalkPlusSource",
    "parse_naist_jdic_row",
]
