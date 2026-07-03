from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from modernbert_g2p.data.ingest.base import SOURCE_REGISTRY, get_source
from modernbert_g2p.data.ingest.unidic import (
    UNIDIC_COLS,
    UnidicSource,
    parse_unidic_row,
)
from modernbert_g2p.data.schema import Row

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "unidic_lex_sample.csv"
_EXPECTED_VALID_ROWS = 7


def _hana_fields() -> list[str]:
    fields = ["*"] * UNIDIC_COLS
    fields[0] = "花"
    fields[9] = "ハナ"
    fields[10] = "名詞"
    fields[11] = "普通名詞"
    fields[16] = "和"
    fields[24] = "ハナ"
    fields[25] = "0"
    fields[26] = "C1"
    return fields


@pytest.fixture
def lex_root(tmp_path: Path) -> Path:
    dst = tmp_path / "lex.csv"
    shutil.copy(_FIXTURE_PATH, dst)
    return tmp_path


def test_unidic_source_registered() -> None:
    assert "unidic" in SOURCE_REGISTRY
    assert get_source("unidic") is UnidicSource


def test_unidic_source_license_and_name() -> None:
    assert UnidicSource.source == "unidic"
    assert UnidicSource.source_license == "CC-BY-4.0"


def test_parse_unidic_row_happy() -> None:
    row = parse_unidic_row(_hana_fields())
    assert row is not None
    assert row.source == "unidic"
    assert row.source_license == "CC-BY-4.0"
    assert row.text == "花"
    assert row.phonemes == ("h", "a", "n", "a")
    assert row.mora_accents == ("L", "H")
    assert row.accent_boundaries == ()
    assert row.extra["pos1"] == "名詞"
    assert row.extra["pos2"] == "普通名詞"
    assert row.extra["accent_type_raw"] == "0"


def test_parse_unidic_row_multi_accent_takes_first() -> None:
    fields = _hana_fields()
    fields[0] = "頭"
    fields[9] = "アタマ"
    fields[24] = "アタマ"
    fields[25] = "1,3"
    row = parse_unidic_row(fields)
    assert row is not None
    assert row.text == "頭"
    assert row.mora_accents == ("H", "L", "L")
    assert row.extra["accent_type_raw"] == "1,3"


def test_parse_skips_aType_star() -> None:
    fields = _hana_fields()
    fields[25] = "*"
    assert parse_unidic_row(fields) is None


def test_parse_skips_empty_surface() -> None:
    fields = _hana_fields()
    fields[0] = ""
    assert parse_unidic_row(fields) is None


def test_parse_skips_non_katakana_pron() -> None:
    fields = _hana_fields()
    fields[9] = "!!"
    assert parse_unidic_row(fields) is None


def test_parse_skips_wrong_column_count() -> None:
    fields = _hana_fields()[:-1]
    assert parse_unidic_row(fields) is None


def test_parse_loanword_from_goshu() -> None:
    fields = _hana_fields()
    fields[0] = "パン"
    fields[9] = "パン"
    fields[16] = "外"
    fields[24] = "パン"
    fields[25] = "1"
    row = parse_unidic_row(fields)
    assert row is not None
    assert row.category == "loanword"


def test_parse_general_when_goshu_not_gairai() -> None:
    row = parse_unidic_row(_hana_fields())
    assert row is not None
    assert row.category == "general"


def test_parse_general_row_sample_weight_is_one() -> None:
    row = parse_unidic_row(_hana_fields())
    assert row is not None
    assert row.category == "general"
    assert row.sample_weight == 1.0


def test_parse_loanword_row_sample_weight_is_two() -> None:
    fields = _hana_fields()
    fields[0] = "パン"
    fields[9] = "パン"
    fields[16] = "外"
    fields[24] = "パン"
    fields[25] = "1"
    row = parse_unidic_row(fields)
    assert row is not None
    assert row.category == "loanword"
    assert row.sample_weight == 2.0


def test_source_entries_reads_lex_csv(lex_root: Path) -> None:
    rows = list(UnidicSource().entries(lex_root))
    assert len(rows) == _EXPECTED_VALID_ROWS
    assert all(isinstance(r, Row) for r in rows)
    assert all(r.source == "unidic" for r in rows)
    assert all(r.source_license == "CC-BY-4.0" for r in rows)


def test_source_entries_categories_include_loanword(lex_root: Path) -> None:
    rows = list(UnidicSource().entries(lex_root))
    categories = {r.category for r in rows}
    assert "loanword" in categories
    assert "general" in categories


def test_source_entries_respects_limit(lex_root: Path) -> None:
    rows = list(UnidicSource().entries(lex_root, limit=3))
    assert len(rows) == 3


def test_source_entries_skips_malformed_rows(lex_root: Path, tmp_path: Path) -> None:
    dst = tmp_path / "lex_extra.csv"
    src_text = _FIXTURE_PATH.read_text(encoding="utf-8")
    dst_text = src_text + "malformed,too,few\n"
    dst.write_text(dst_text, encoding="utf-8")
    root = tmp_path / "root"
    root.mkdir()
    shutil.copy(dst, root / "lex.csv")
    rows = list(UnidicSource().entries(root))
    assert len(rows) == _EXPECTED_VALID_ROWS


def test_source_entries_missing_lex_raises(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        list(UnidicSource().entries(empty))


def test_source_entries_accepts_file_path_directly(tmp_path: Path) -> None:
    dst = tmp_path / "lex.csv"
    shutil.copy(_FIXTURE_PATH, dst)
    rows = list(UnidicSource().entries(dst))
    assert len(rows) == _EXPECTED_VALID_ROWS
