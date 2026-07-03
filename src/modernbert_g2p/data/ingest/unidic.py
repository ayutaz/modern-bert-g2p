"""S2 ingest: parse UniDic-cwj lex.csv (33 columns for cwj-3.1.1) into Row records.

Implements docs/design/phase1_data_pipeline.md §3 (S2 UniDic-cwj 3.1.1). Column
indexes verified against unidic-cwj-3.1.1/lex_3_1.csv:
  0 surface, 4/5/6 POS1/2/3, 13 pron 発音形出現形, 16 goshu 語種,
  28 accent nucleus, 29 accent connection type.
"""

from __future__ import annotations

import csv
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

UNIDIC_COLS: int = 33

_COL_SURFACE = 0
_COL_PRON = 13
_COL_POS1 = 4
_COL_POS2 = 5
_COL_GOSHU = 16
_COL_ATYPE = 28


def _first_nucleus(atype_raw: str) -> int | None:
    head = atype_raw.split(",")[0].strip()
    if not head or head == "*":
        return None
    try:
        return int(head)
    except ValueError:
        return None


def _is_all_katakana(kana: str) -> bool:
    return bool(kana) and all(is_katakana(c) for c in kana)


def _category_for_goshu(goshu: str) -> str:
    return "loanword" if goshu == "外" else "general"


def parse_unidic_row(fields: Sequence[str]) -> Row | None:
    """Parse one 30-column UniDic lex.csv row into a Row, or None if the row is skippable."""
    if len(fields) != UNIDIC_COLS:
        return None

    surface = fields[_COL_SURFACE]
    pron = fields[_COL_PRON]
    atype_raw = fields[_COL_ATYPE]

    if not surface or not pron:
        return None
    if atype_raw == "*":
        return None

    nucleus = _first_nucleus(atype_raw)
    if nucleus is None:
        return None

    kana_norm = normalize_katakana(pron)
    if not _is_all_katakana(kana_norm):
        return None

    try:
        phonemes = kana_to_julius_phonemes(kana_norm)
    except ValueError:
        return None
    if not phonemes:
        return None

    mora_count = kana_to_mora_count(kana_norm)
    mora_accents = accent_nucleus_to_hl(nucleus, mora_count)

    goshu = fields[_COL_GOSHU]
    category = _category_for_goshu(goshu)

    row = Row(
        id=make_id("unidic", surface, phonemes),
        source="unidic",
        source_license="CC-BY-4.0",
        text=surface,
        phonemes=phonemes,
        mora_accents=mora_accents,
        accent_boundaries=(),
        category=category,
        sample_weight=weight_for(category),
        extra={
            "pos1": fields[_COL_POS1],
            "pos2": fields[_COL_POS2],
            "accent_type_raw": atype_raw,
        },
    )
    try:
        validate_row(row)
    except ValueError:
        return None
    return row


@register("unidic")
class UnidicSource:
    """S2 ingest source for UniDic-cwj 3.1.1 lex.csv."""

    source: ClassVar[str] = "unidic"
    source_license: ClassVar[str] = "CC-BY-4.0"

    def entries(self, root: Path, *, limit: int | None = None) -> Iterator[Row]:
        path = root if root.is_file() else root / "lex.csv"
        if not path.exists():
            raise FileNotFoundError(str(path))

        count = 0
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            for fields in reader:
                if not fields:
                    continue
                row = parse_unidic_row(fields)
                if row is None:
                    continue
                yield row
                count += 1
                if limit is not None and count >= limit:
                    return


__all__ = ["UNIDIC_COLS", "UnidicSource", "parse_unidic_row"]
