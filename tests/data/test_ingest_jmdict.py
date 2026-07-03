"""Tests for the JMDict (S3) ingest source."""

from __future__ import annotations

from pathlib import Path

import pytest

from modernbert_g2p.data.ingest import base as base_mod
from modernbert_g2p.data.ingest.jmdict import (
    JMDictSource,
    hira_to_kata,
    parse_jmdict_entry,
)

FIXTURE_XML = Path(__file__).parent / "fixtures" / "jmdict_sample.xml"


def test_hira_to_kata_basic():
    assert hira_to_kata("あいうえお") == "アイウエオ"
    assert hira_to_kata("きょう") == "キョウ"
    assert hira_to_kata("カタカナはそのまま") == "カタカナハソノママ"


def test_hira_to_kata_leaves_non_hiragana_untouched():
    assert hira_to_kata("今日はABC 123") == "今日はABC 123".replace("は", "ハ")


def test_source_registered():
    cls = base_mod.get_source("jmdict")
    assert cls is JMDictSource


def test_source_license_is_edrdg():
    assert JMDictSource.source_license == "EDRDG"
    assert JMDictSource.source == "jmdict"


def test_entries_yields_expected_count():
    rows = list(JMDictSource().entries(FIXTURE_XML))
    assert len(rows) == 12


def test_all_rows_carry_edrdg_license_and_jmdict_source():
    rows = list(JMDictSource().entries(FIXTURE_XML))
    assert {r.source for r in rows} == {"jmdict"}
    assert {r.source_license for r in rows} == {"EDRDG"}


def test_mora_accents_empty_for_jmdict():
    rows = list(JMDictSource().entries(FIXTURE_XML))
    assert all(r.mora_accents == () for r in rows)
    assert all(r.accent_boundaries == () for r in rows)


def test_sample_weight_is_general_for_jmdict():
    rows = list(JMDictSource().entries(FIXTURE_XML))
    assert all(r.category == "general" for r in rows)
    assert all(r.sample_weight == 1.0 for r in rows)


def test_parse_single_entry_two_keb_three_reb_yields_six_rows():
    entry_xml = """
    <entry>
      <ent_seq>2000001</ent_seq>
      <k_ele><keb>明日</keb></k_ele>
      <k_ele><keb>翌日</keb></k_ele>
      <r_ele><reb>あした</reb></r_ele>
      <r_ele><reb>あす</reb></r_ele>
      <r_ele><reb>みょうにち</reb></r_ele>
    </entry>
    """
    rows = list(parse_jmdict_entry(entry_xml))
    surfaces = {r.text for r in rows}
    assert len(rows) == 6
    assert surfaces == {"明日", "翌日"}


def test_parse_entry_with_re_restr_limits_pairs():
    entry_xml = """
    <entry>
      <ent_seq>2000002</ent_seq>
      <k_ele><keb>本</keb></k_ele>
      <k_ele><keb>書</keb></k_ele>
      <r_ele><reb>ほん</reb></r_ele>
      <r_ele>
        <reb>しょ</reb>
        <re_restr>書</re_restr>
      </r_ele>
    </entry>
    """
    rows = list(parse_jmdict_entry(entry_xml))
    pairs = {(r.text, r.extra["reb"]) for r in rows}
    assert len(rows) == 3
    assert ("本", "ホン") in pairs
    assert ("書", "ホン") in pairs
    assert ("書", "ショ") in pairs
    assert ("本", "ショ") not in pairs


def test_entry_without_keb_uses_reading_as_surface():
    entry_xml = """
    <entry>
      <ent_seq>2000003</ent_seq>
      <r_ele><reb>ありがとう</reb></r_ele>
    </entry>
    """
    rows = list(parse_jmdict_entry(entry_xml))
    assert len(rows) == 1
    assert rows[0].text == "ありがとう"
    assert rows[0].extra["reb"] == "アリガトウ"


def test_limit_truncates_iteration():
    rows_all = list(JMDictSource().entries(FIXTURE_XML))
    rows_lim = list(JMDictSource().entries(FIXTURE_XML, limit=3))
    assert len(rows_lim) == 3
    assert rows_lim == rows_all[:3]


def test_missing_xml_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(JMDictSource().entries(tmp_path))


def test_row_ids_are_deterministic_and_unique():
    rows = list(JMDictSource().entries(FIXTURE_XML))
    ids = [r.id for r in rows]
    assert len(set(ids)) == len(ids)
    rows_again = list(JMDictSource().entries(FIXTURE_XML))
    assert [r.id for r in rows_again] == ids


def test_extra_carries_ent_seq_and_reb():
    rows = list(JMDictSource().entries(FIXTURE_XML))
    neko = next(r for r in rows if r.text == "猫")
    assert neko.extra["ent_seq"] == "1000005"
    assert neko.extra["reb"] == "ネコ"
    assert neko.phonemes[0] == "n"
