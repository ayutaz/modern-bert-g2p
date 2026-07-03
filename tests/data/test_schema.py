"""Unit tests for Row dataclass, constants, validate_row, and dict roundtrip."""

from __future__ import annotations

import dataclasses
import re

import pytest

from modernbert_g2p.data.schema import (
    KNOWN_CATEGORIES,
    KNOWN_LICENSES,
    KNOWN_SOURCES,
    SCHEMA_VERSION,
    Row,
    from_dict,
    make_id,
    to_dict,
    validate_row,
)


def _valid_row(**overrides: object) -> Row:
    defaults: dict[str, object] = {
        "id": "pyopenjtalk_plus:abc123def456",
        "source": "pyopenjtalk_plus",
        "source_license": "BSD3",
        "text": "今日",
        "phonemes": ("k", "y", "o", "o"),
        "mora_accents": ("H", "L"),
        "accent_boundaries": (),
        "category": "general",
        "sample_weight": 1.0,
        "extra": {},
    }
    defaults.update(overrides)
    return Row(**defaults)  # type: ignore[arg-type]


def test_schema_version_is_one_dot_zero():
    assert SCHEMA_VERSION == "1.0"


def test_known_constants_populated():
    assert "pyopenjtalk_plus" in KNOWN_SOURCES
    assert "aozora" in KNOWN_SOURCES
    assert "BSD3" in KNOWN_LICENSES
    assert "CC-BY-SA-4.0" in KNOWN_LICENSES
    assert "general" in KNOWN_CATEGORIES
    assert "polyphone" in KNOWN_CATEGORIES


def test_row_frozen():
    row = _valid_row()
    with pytest.raises(dataclasses.FrozenInstanceError):
        row.text = "違う"  # type: ignore[misc]


def test_row_defaults_general_and_weight_one():
    row = Row(
        id="pyopenjtalk_plus:000000000000",
        source="pyopenjtalk_plus",
        source_license="BSD3",
        text="今日",
        phonemes=("k", "y", "o", "o"),
        mora_accents=("H", "L"),
        accent_boundaries=(),
    )
    assert row.category == "general"
    assert row.sample_weight == 1.0
    assert row.extra == {}


def test_make_id_deterministic():
    a = make_id("pyopenjtalk_plus", "今日", ["k", "y", "o", "o"])
    b = make_id("pyopenjtalk_plus", "今日", ["k", "y", "o", "o"])
    c = make_id("pyopenjtalk_plus", "今日", ["k", "y", "o", "u"])
    assert a == b
    assert a != c


def test_make_id_format_matches_regex():
    ident = make_id("pyopenjtalk_plus", "今日", ["k", "y", "o", "o"])
    assert re.match(r"^[a-z_]+:[0-9a-f]{12}$", ident)


def test_validate_row_happy_path():
    row = _valid_row()
    validate_row(row)


def test_validate_row_rejects_unknown_source():
    row = _valid_row(source="foo")
    with pytest.raises(ValueError, match="source"):
        validate_row(row)


def test_validate_row_rejects_unknown_license():
    row = _valid_row(source_license="MIT")
    with pytest.raises(ValueError, match="source_license"):
        validate_row(row)


def test_validate_row_rejects_unknown_category():
    row = _valid_row(category="mystery")
    with pytest.raises(ValueError, match="category"):
        validate_row(row)


def test_validate_row_rejects_empty_text():
    row = _valid_row(text="")
    with pytest.raises(ValueError, match="text"):
        validate_row(row)


def test_validate_row_rejects_ignored_phoneme():
    row = _valid_row(phonemes=("k", "pau", "o"))
    with pytest.raises(ValueError, match="phonemes"):
        validate_row(row)


def test_validate_row_rejects_non_hl_accent_tag():
    row = _valid_row(mora_accents=("H", "X"))
    with pytest.raises(ValueError, match="mora_accents"):
        validate_row(row)


def test_validate_row_rejects_boundary_out_of_range():
    row = _valid_row(mora_accents=("H", "L"), accent_boundaries=(2,))
    with pytest.raises(ValueError, match="accent_boundaries"):
        validate_row(row)


def test_validate_row_rejects_negative_boundary():
    row = _valid_row(mora_accents=("H", "L", "H"), accent_boundaries=(-1,))
    with pytest.raises(ValueError, match="accent_boundaries"):
        validate_row(row)


def test_validate_row_rejects_non_monotonic_boundaries():
    row = _valid_row(mora_accents=("H", "L", "H", "L", "H"), accent_boundaries=(2, 1))
    with pytest.raises(ValueError, match="accent_boundaries"):
        validate_row(row)


def test_validate_row_accepts_monotonic_boundaries():
    row = _valid_row(mora_accents=("H", "L", "H", "L", "H"), accent_boundaries=(1, 3))
    validate_row(row)


def test_validate_row_rejects_zero_weight():
    row = _valid_row(sample_weight=0.0)
    with pytest.raises(ValueError, match="sample_weight"):
        validate_row(row)


def test_validate_row_rejects_negative_weight():
    row = _valid_row(sample_weight=-0.1)
    with pytest.raises(ValueError, match="sample_weight"):
        validate_row(row)


def test_roundtrip_dict():
    row = _valid_row(
        mora_accents=("H", "L", "H", "L"),
        accent_boundaries=(2,),
        category="counter",
        sample_weight=2.0,
        extra={"pos": "名詞", "accent_type": 1},
    )
    d = to_dict(row)
    assert isinstance(d["phonemes"], list)
    assert isinstance(d["accent_boundaries"], list)
    restored = from_dict(d)
    assert restored == row


def test_from_dict_rejects_unknown_keys():
    row = _valid_row()
    d = to_dict(row)
    d["mystery_field"] = 123
    with pytest.raises(ValueError, match="mystery_field"):
        from_dict(d)


def test_from_dict_fills_defaults_for_optional_fields():
    minimal = {
        "id": "pyopenjtalk_plus:000000000000",
        "source": "pyopenjtalk_plus",
        "source_license": "BSD3",
        "text": "今日",
        "phonemes": ["k", "y", "o", "o"],
        "mora_accents": ["H", "L"],
        "accent_boundaries": [],
    }
    row = from_dict(minimal)
    assert row.category == "general"
    assert row.sample_weight == 1.0
    assert row.extra == {}
