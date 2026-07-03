"""Category to sample_weight policy applied at Parquet build time.

Implements docs/design/phase1_data_pipeline.md §7 (Sample weighting).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from types import MappingProxyType

from modernbert_g2p.data.schema import Row

CATEGORY_WEIGHTS: Mapping[str, float] = MappingProxyType({
    "general": 1.0,
    "numeric_unit": 2.0,
    "proper_noun_kanji": 2.0,
    "proper_noun_katakana": 2.0,
    "counter": 2.0,
    "loanword": 2.0,
    "english_mixed": 2.0,
    "english_abbreviation": 2.0,
    "polyphone": 2.0,
})


def weight_for(category: str) -> float:
    return CATEGORY_WEIGHTS[category]


def apply_weight(row: Row) -> Row:
    return replace(row, sample_weight=weight_for(row.category))


__all__ = ["CATEGORY_WEIGHTS", "apply_weight", "weight_for"]
