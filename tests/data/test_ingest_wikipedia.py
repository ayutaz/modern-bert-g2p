"""Unit tests for the Wikipedia (S4) ingest source."""

from __future__ import annotations

from pathlib import Path

from modernbert_g2p.data.ingest.base import SOURCE_REGISTRY, get_source
from modernbert_g2p.data.ingest.wikipedia import (
    WikipediaSource,
    extract_ruby_pairs,
)

_FIXTURE = Path(__file__).parent / "fixtures" / "wikipedia_sample.html.json"


def test_wikipedia_source_registered() -> None:
    assert "wikipedia" in SOURCE_REGISTRY
    assert get_source("wikipedia") is WikipediaSource


def test_wikipedia_source_metadata() -> None:
    assert WikipediaSource.source == "wikipedia"
    assert WikipediaSource.source_license == "CC-BY-SA-4.0"


def test_entries_yields_non_empty_for_fixture() -> None:
    rows = list(WikipediaSource().entries(_FIXTURE))
    assert len(rows) > 0
    for row in rows:
        assert row.source == "wikipedia"
        assert row.source_license == "CC-BY-SA-4.0"


def test_first_row_surface_and_rt_extracted() -> None:
    rows = list(WikipediaSource().entries(_FIXTURE))
    first = rows[0]
    assert first.text == "桜"
    assert first.extra["rt_text"] == "さくら"
    assert first.phonemes == ("s", "a", "k", "u", "r", "a")


def test_katakana_rt_handled() -> None:
    rows = list(WikipediaSource().entries(_FIXTURE))
    match = next((r for r in rows if r.extra.get("rt_text") == "ワイファイ"), None)
    assert match is not None
    assert match.phonemes == ("w", "a", "i", "f", "a", "i")


def test_mixed_ascii_rb_accepted() -> None:
    rows = list(WikipediaSource().entries(_FIXTURE))
    match = next((r for r in rows if r.text == "Wi-Fi"), None)
    assert match is not None
    assert match.extra["rt_text"] == "ワイファイ"


def test_explicit_rb_form_extracted() -> None:
    rows = list(WikipediaSource().entries(_FIXTURE))
    match = next((r for r in rows if r.text == "東京"), None)
    assert match is not None
    assert match.extra["rt_text"] == "とうきょう"
    assert match.phonemes == ("t", "o", "u", "ky", "o", "u")


def test_empty_rt_skipped() -> None:
    rows = list(WikipediaSource().entries(_FIXTURE))
    surfaces = {r.text for r in rows}
    assert "空" not in surfaces
    for row in rows:
        assert row.extra["rt_text"]


def test_rp_parens_ignored_in_extractor() -> None:
    pairs = extract_ruby_pairs(
        "<ruby>桜<rp>（</rp><rt>さくら</rt><rp>）</rp></ruby>"
    )
    assert pairs == [("桜", "さくら")]


def test_extract_ruby_pairs_no_rb_form() -> None:
    pairs = extract_ruby_pairs("<ruby>桜<rt>さくら</rt></ruby>")
    assert pairs == [("桜", "さくら")]


def test_extract_ruby_pairs_explicit_rb_form() -> None:
    pairs = extract_ruby_pairs("<ruby><rb>東京</rb><rt>とうきょう</rt></ruby>")
    assert pairs == [("東京", "とうきょう")]


def test_extract_ruby_pairs_multiple() -> None:
    html = (
        "<p><ruby>桜<rt>さくら</rt></ruby>と"
        "<ruby><rb>桃</rb><rt>もも</rt></ruby></p>"
    )
    assert extract_ruby_pairs(html) == [("桜", "さくら"), ("桃", "もも")]


def test_limit_works() -> None:
    all_rows = list(WikipediaSource().entries(_FIXTURE))
    limited = list(WikipediaSource().entries(_FIXTURE, limit=2))
    assert len(limited) == 2
    assert all_rows[:2] == limited


def test_limit_zero_yields_nothing() -> None:
    rows = list(WikipediaSource().entries(_FIXTURE, limit=0))
    assert rows == []


def test_mora_accents_empty() -> None:
    rows = list(WikipediaSource().entries(_FIXTURE))
    assert len(rows) > 0
    for row in rows:
        assert row.mora_accents == ()
        assert row.accent_boundaries == ()


def test_sample_weight_matches_policy() -> None:
    from modernbert_g2p.data.weighting import weight_for

    rows = list(WikipediaSource().entries(_FIXTURE))
    assert len(rows) > 0
    for row in rows:
        assert row.category == "general"
        assert row.sample_weight == weight_for("general")


def test_fixture_yields_expected_total() -> None:
    rows = list(WikipediaSource().entries(_FIXTURE))
    surfaces = [r.text for r in rows]
    assert surfaces == ["桜", "東京", "Wi-Fi", "富士山", "京都", "夏"]


def test_row_id_format() -> None:
    rows = list(WikipediaSource().entries(_FIXTURE))
    for row in rows:
        assert row.id.startswith("wikipedia:")
        prefix, _, digest = row.id.partition(":")
        assert len(digest) == 12
        assert all(c in "0123456789abcdef" for c in digest)


def test_entries_reads_directory_root(tmp_path: Path) -> None:
    dest = tmp_path / "wiki.jsonl"
    dest.write_bytes(_FIXTURE.read_bytes())
    rows = list(WikipediaSource().entries(tmp_path))
    assert len(rows) == 6
