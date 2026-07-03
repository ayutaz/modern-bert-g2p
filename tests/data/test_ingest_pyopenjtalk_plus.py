"""Tests for Track 6 — S1 pyopenjtalk-plus mecab-naist-jdic ingestion."""

from __future__ import annotations

from pathlib import Path

import pytest

from modernbert_g2p.data.ingest import base as ingest_base
from modernbert_g2p.data.ingest.pyopenjtalk_plus import (
    NAIST_JDIC_COLS,
    PyopenjtalkPlusSource,
    parse_naist_jdic_row,
)
from modernbert_g2p.data.schema import validate_row

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "mecab_naist_jdic_sample.csv"


def _rows_by_text() -> dict[str, object]:
    return {r.text: r for r in PyopenjtalkPlusSource().entries(FIXTURE_PATH)}


def test_source_registered_in_registry() -> None:
    assert "pyopenjtalk_plus" in ingest_base.SOURCE_REGISTRY
    assert ingest_base.SOURCE_REGISTRY["pyopenjtalk_plus"] is PyopenjtalkPlusSource


def test_class_metadata() -> None:
    assert PyopenjtalkPlusSource.source == "pyopenjtalk_plus"
    assert PyopenjtalkPlusSource.source_license == "BSD3"


def test_entries_from_fixture_yields_15_rows() -> None:
    rows = list(PyopenjtalkPlusSource().entries(FIXTURE_PATH))
    assert len(rows) == 15


def test_every_row_passes_validate_and_carries_source_tags() -> None:
    for row in PyopenjtalkPlusSource().entries(FIXTURE_PATH):
        validate_row(row)
        assert row.source == "pyopenjtalk_plus"
        assert row.source_license == "BSD3"
        assert len(row.phonemes) > 0
        assert len(row.mora_accents) > 0
        assert row.accent_boundaries == ()


def test_heiban_row_mora_accents_L_H_H() -> None:
    rows = _rows_by_text()
    assert rows["桜"].mora_accents == ("L", "H", "H")  # type: ignore[union-attr]


def test_atamadaka_row_mora_accents_H_L() -> None:
    rows = _rows_by_text()
    assert rows["花"].mora_accents == ("H", "L")  # type: ignore[union-attr]


def test_odaka_row_mora_accents_L_H() -> None:
    rows = _rows_by_text()
    assert rows["犬"].mora_accents == ("L", "H")  # type: ignore[union-attr]


def test_nakadaka_row_mora_accents() -> None:
    rows = _rows_by_text()
    assert rows["iPhone"].mora_accents == ("L", "H", "H", "L")  # type: ignore[union-attr]


def test_loanword_category_all_katakana() -> None:
    rows = _rows_by_text()
    assert rows["コーヒー"].category == "loanword"  # type: ignore[union-attr]
    assert rows["パソコン"].category == "loanword"  # type: ignore[union-attr]


def test_english_abbreviation_category() -> None:
    rows = _rows_by_text()
    row = rows["PDF"]
    assert row.category == "english_abbreviation"  # type: ignore[union-attr]
    assert row.sample_weight == 2.0  # type: ignore[union-attr]


def test_english_mixed_category() -> None:
    rows = _rows_by_text()
    assert rows["iPhone"].category == "english_mixed"  # type: ignore[union-attr]


def test_numeric_unit_category() -> None:
    rows = _rows_by_text()
    assert rows["3日"].category == "numeric_unit"  # type: ignore[union-attr]


def test_proper_noun_kanji_category() -> None:
    rows = _rows_by_text()
    assert rows["山田"].category == "proper_noun_kanji"  # type: ignore[union-attr]
    assert rows["東京都"].category == "proper_noun_kanji"  # type: ignore[union-attr]


def test_general_category_for_common_noun() -> None:
    rows = _rows_by_text()
    assert rows["猫"].category == "general"  # type: ignore[union-attr]
    assert rows["猫"].sample_weight == 1.0  # type: ignore[union-attr]


def test_limit_caps_output_at_five() -> None:
    rows = list(PyopenjtalkPlusSource().entries(FIXTURE_PATH, limit=5))
    assert len(rows) == 5


def test_pdf_phoneme_and_mora_lengths() -> None:
    rows = _rows_by_text()
    row = rows["PDF"]
    assert len(row.mora_accents) == 6  # type: ignore[union-attr]
    assert row.mora_accents == ("L", "H", "H", "H", "L", "L")  # type: ignore[union-attr]


def test_parse_naist_jdic_row_happy() -> None:
    fields = [
        "花", "1301", "1301", "4820", "名詞", "一般",
        "*", "*", "*", "*", "花", "ハナ", "ハ'ナ", "1/2", "C1",
    ]
    row = parse_naist_jdic_row(fields)
    assert row is not None
    assert row.text == "花"
    assert row.mora_accents == ("H", "L")
    assert row.category == "general"
    assert row.extra["accent_type"] == 1
    assert row.extra["pos1"] == "名詞"
    assert row.extra["chain_flag"] == "C1"


def test_parse_skips_wrong_column_count() -> None:
    assert parse_naist_jdic_row(["花", "1", "2", "3"]) is None
    long_row = ["x"] * (NAIST_JDIC_COLS + 1)
    assert parse_naist_jdic_row(long_row) is None


def test_parse_skips_empty_surface() -> None:
    fields = ["", "1301", "1301", "4820", "名詞", "一般",
              "*", "*", "*", "*", "", "ハナ", "ハ'ナ", "1/2", "C1"]
    assert parse_naist_jdic_row(fields) is None


def test_parse_skips_empty_pron() -> None:
    fields = ["花", "1301", "1301", "4820", "名詞", "一般",
              "*", "*", "*", "*", "花", "", "", "1/2", "C1"]
    assert parse_naist_jdic_row(fields) is None


def test_parse_skips_non_katakana_pron() -> None:
    fields = ["花", "1301", "1301", "4820", "名詞", "一般",
              "*", "*", "*", "*", "花", "hana!!", "hana!!", "1/2", "C1"]
    assert parse_naist_jdic_row(fields) is None


def test_parse_multi_accent_takes_first() -> None:
    fields = ["花", "1301", "1301", "4820", "名詞", "一般",
              "*", "*", "*", "*", "花", "ハナ", "ハ'ナ", "2/2,1/2", "C1"]
    row = parse_naist_jdic_row(fields)
    assert row is not None
    assert row.extra["accent_type"] == 2
    assert row.mora_accents == ("L", "H")


def test_parse_accent_with_chain_suffix() -> None:
    fields = ["花", "1301", "1301", "4820", "名詞", "一般",
              "*", "*", "*", "*", "花", "ハナ", "ハ'ナ", "1;C1", "*"]
    row = parse_naist_jdic_row(fields)
    assert row is not None
    assert row.extra["accent_type"] == 1


def test_parse_accent_bare_int() -> None:
    fields = ["花", "1301", "1301", "4820", "名詞", "一般",
              "*", "*", "*", "*", "花", "ハナ", "ハ'ナ", "1", "C1"]
    row = parse_naist_jdic_row(fields)
    assert row is not None
    assert row.extra["accent_type"] == 1
    assert row.mora_accents == ("H", "L")


def test_parse_out_of_range_nucleus_falls_back_to_all_L() -> None:
    fields = ["花", "1301", "1301", "4820", "名詞", "一般",
              "*", "*", "*", "*", "花", "ハナ", "ハ'ナ", "9/2", "C1"]
    row = parse_naist_jdic_row(fields)
    assert row is not None
    assert row.mora_accents == ("L", "L")


def test_entries_skips_malformed_rows_in_file(tmp_path: Path) -> None:
    p = tmp_path / "mixed.csv"
    p.write_text(
        "花,1301,1301,4820,名詞,一般,*,*,*,*,花,ハナ,ハ'ナ,1/2,C1\n"
        "bogus,too,few,cols\n"
        "桜,1301,1301,4800,名詞,一般,*,*,*,*,桜,サクラ,サクラ,0/3,C1\n",
        encoding="utf-8",
    )
    with pytest.warns(UserWarning):
        rows = list(PyopenjtalkPlusSource().entries(p))
    assert len(rows) == 2
    assert {r.text for r in rows} == {"花", "桜"}


def test_entries_from_directory_iterates_all_csv(tmp_path: Path) -> None:
    (tmp_path / "a.csv").write_text(
        "花,1301,1301,4820,名詞,一般,*,*,*,*,花,ハナ,ハ'ナ,1/2,C1\n",
        encoding="utf-8",
    )
    (tmp_path / "b.csv").write_text(
        "桜,1301,1301,4800,名詞,一般,*,*,*,*,桜,サクラ,サクラ,0/3,C1\n",
        encoding="utf-8",
    )
    rows = list(PyopenjtalkPlusSource().entries(tmp_path))
    assert len(rows) == 2
    assert {r.text for r in rows} == {"花", "桜"}


def test_entries_missing_root_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        list(PyopenjtalkPlusSource().entries(tmp_path / "nonexistent"))
