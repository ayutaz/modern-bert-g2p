"""Unit tests for CATEGORY_WEIGHTS policy, weight_for, and apply_weight."""

from __future__ import annotations

import pytest

from modernbert_g2p.data.schema import KNOWN_CATEGORIES, Row
from modernbert_g2p.data.weighting import CATEGORY_WEIGHTS, apply_weight, weight_for


def _row(category: str) -> Row:
    return Row(
        id="pyopenjtalk_plus:000000000000",
        source="pyopenjtalk_plus",
        source_license="BSD3",
        text="今日",
        phonemes=("k", "y", "o", "o"),
        mora_accents=("H", "L"),
        accent_boundaries=(),
        category=category,
        sample_weight=1.0,
        extra={"pos": "名詞"},
    )


def test_weight_for_general_is_one():
    assert weight_for("general") == 1.0


def test_weight_for_numeric_unit_is_two():
    assert weight_for("numeric_unit") == 2.0


def test_weight_for_counter_is_two():
    assert weight_for("counter") == 2.0


def test_weight_for_loanword_is_two():
    assert weight_for("loanword") == 2.0


def test_weight_for_proper_noun_variants_are_two():
    assert weight_for("proper_noun_kanji") == 2.0
    assert weight_for("proper_noun_katakana") == 2.0


def test_weight_for_english_variants_are_two():
    assert weight_for("english_mixed") == 2.0
    assert weight_for("english_abbreviation") == 2.0


def test_weight_for_polyphone_is_two():
    assert weight_for("polyphone") == 2.0


def test_weight_for_unknown_category_raises_key_error():
    with pytest.raises(KeyError):
        weight_for("mystery")


def test_all_category_weights_positive():
    assert all(w > 0 for w in CATEGORY_WEIGHTS.values())


def test_category_weights_covers_every_known_category():
    assert set(CATEGORY_WEIGHTS.keys()) == set(KNOWN_CATEGORIES)


def test_category_weights_is_immutable():
    with pytest.raises(TypeError):
        CATEGORY_WEIGHTS["general"] = 99.0  # type: ignore[index]


def test_apply_weight_sets_sample_weight_from_category():
    row = _row("counter")
    result = apply_weight(row)
    assert result.sample_weight == 2.0


def test_apply_weight_preserves_other_fields():
    row = _row("english_abbreviation")
    result = apply_weight(row)
    assert result.id == row.id
    assert result.source == row.source
    assert result.source_license == row.source_license
    assert result.text == row.text
    assert result.phonemes == row.phonemes
    assert result.mora_accents == row.mora_accents
    assert result.accent_boundaries == row.accent_boundaries
    assert result.category == row.category
    assert result.extra == row.extra


def test_apply_weight_returns_new_frozen_row():
    row = _row("general")
    result = apply_weight(row)
    assert result is not row
    assert result.sample_weight == 1.0


def test_apply_weight_unknown_category_raises_key_error():
    row = Row(
        id="pyopenjtalk_plus:000000000000",
        source="pyopenjtalk_plus",
        source_license="BSD3",
        text="今日",
        phonemes=("k", "y", "o", "o"),
        mora_accents=("H", "L"),
        accent_boundaries=(),
        category="general",
    )
    bogus = Row(
        id=row.id,
        source=row.source,
        source_license=row.source_license,
        text=row.text,
        phonemes=row.phonemes,
        mora_accents=row.mora_accents,
        accent_boundaries=row.accent_boundaries,
        category="mystery",
    )
    with pytest.raises(KeyError):
        apply_weight(bogus)
