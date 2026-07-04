"""Tests for the Aozora Bunko XHTML S5 ingest source."""

from pathlib import Path

from modernbert_g2p.data.ingest import SOURCE_REGISTRY
from modernbert_g2p.data.ingest.aozora import (
    AozoraSource,
    detect_license,
    extract_ruby_pairs_aozora,
)
from modernbert_g2p.data.schema import KNOWN_LICENSES, validate_row

FIXTURE_DIR = Path(__file__).parent / "fixtures"
FIXTURE_FILE = FIXTURE_DIR / "aozora_sample.html"


def test_aozora_source_registered():
    assert "aozora" in SOURCE_REGISTRY
    assert SOURCE_REGISTRY["aozora"] is AozoraSource


def test_source_license_base_is_pdplusaoz():
    assert AozoraSource.source_license == "PDplusAoz"
    assert AozoraSource.source_license in KNOWN_LICENSES


def test_entries_yields_one_row_per_ruby_pair_for_fixture_file():
    rows = list(AozoraSource().entries(FIXTURE_FILE))
    assert len(rows) == 5
    surfaces = [r.text for r in rows]
    readings = [r.extra["rt_text"] for r in rows]
    assert surfaces == ["桜", "花", "咲", "春", "風"]
    assert readings == ["さくら", "はな", "さ", "はる", "かぜ"]
    for row in rows:
        assert row.source == "aozora"
        assert row.extra["file"] == "aozora_sample.html"
        validate_row(row)
    assert rows[0].phonemes == ("s", "a", "k", "u", "r", "a")
    assert rows[1].phonemes == ("h", "a", "n", "a")


def test_entries_sets_sample_weight_from_weighting_policy():
    from modernbert_g2p.data.weighting import weight_for

    rows = list(AozoraSource().entries(FIXTURE_FILE))
    assert len(rows) >= 1
    for r in rows:
        assert r.category == "general"
        assert r.sample_weight == weight_for("general")


def test_rp_parens_stripped_from_ruby_extraction():
    html = "<ruby><rb>桜</rb><rp>（</rp><rt>さくら</rt><rp>）</rp></ruby>"
    pairs = extract_ruby_pairs_aozora(html)
    assert pairs == [("桜", "さくら")]
    _, reading = pairs[0]
    assert "（" not in reading
    assert "）" not in reading


def test_extract_ruby_pairs_without_rb_tag():
    html = "<ruby>桜<rp>（</rp><rt>さくら</rt><rp>）</rp></ruby>"
    assert extract_ruby_pairs_aozora(html) == [("桜", "さくら")]


def test_detect_license_pd_from_fixture_header():
    html = FIXTURE_FILE.read_text(encoding="utf-8")
    assert detect_license(html) == "PD"


def test_detect_license_ccby_returns_pdplusaoz():
    html = (
        "<div>本作品はクリエイティブ・コモンズ 表示ライセンスの下で配布されます。</div>"
        "<hr /><ruby><rb>桜</rb><rt>さくら</rt></ruby>"
    )
    assert detect_license(html) == "PDplusAoz"


def test_detect_license_defaults_to_pdplusaoz_when_unknown():
    html = "<div>ライセンス情報なし</div><hr />body"
    assert detect_license(html) == "PDplusAoz"


def test_row_extra_file_license_matches_detected(tmp_path):
    rows = list(AozoraSource().entries(FIXTURE_FILE))
    assert rows[0].extra["file_license"] == "PD"
    assert rows[0].source_license == "PD"


def test_entries_skips_zero_ruby_document(tmp_path):
    empty = tmp_path / "empty.html"
    empty.write_text(
        "<html><body><p>ルビ無しの本文です。パブリック・ドメイン。</p><hr />"
        "<p>ここにも<span>ルビはありません</span></p></body></html>",
        encoding="utf-8",
    )
    rows = list(AozoraSource().entries(tmp_path))
    assert rows == []


def test_entries_respects_limit(tmp_path):
    source_html = FIXTURE_FILE.read_text(encoding="utf-8")
    for i in range(3):
        (tmp_path / f"work_{i}.html").write_text(source_html, encoding="utf-8")
    rows = list(AozoraSource().entries(tmp_path, limit=2))
    assert len(rows) == 2
    for row in rows:
        assert row.source == "aozora"


def test_entries_reads_shift_jis_encoded_file(tmp_path):
    sjis_html = (
        "<html><body><p>パブリック・ドメイン</p><hr />"
        "<ruby><rb>桜</rb><rp>（</rp><rt>さくら</rt><rp>）</rp></ruby></body></html>"
    )
    encoded = sjis_html.encode("shift_jis")
    path = tmp_path / "sjis.html"
    path.write_bytes(encoded)
    rows = list(AozoraSource().entries(path))
    assert len(rows) == 1
    assert rows[0].text == "桜"
    assert rows[0].source_license == "PD"


def test_entries_directory_iterates_multiple_files(tmp_path):
    (tmp_path / "a.html").write_text(
        "<div>パブリック・ドメイン</div><hr />"
        "<ruby><rb>山</rb><rp>（</rp><rt>やま</rt><rp>）</rp></ruby>",
        encoding="utf-8",
    )
    (tmp_path / "b.html").write_text(
        "<div>クリエイティブ・コモンズ 表示</div><hr />"
        "<ruby><rb>川</rb><rp>（</rp><rt>かわ</rt><rp>）</rp></ruby>",
        encoding="utf-8",
    )
    rows = sorted(AozoraSource().entries(tmp_path), key=lambda r: r.text)
    assert len(rows) == 2
    by_text = {r.text: r for r in rows}
    assert by_text["山"].source_license == "PD"
    assert by_text["川"].source_license == "PDplusAoz"
    assert by_text["山"].extra["file_license"] == "PD"
    assert by_text["川"].extra["file_license"] == "PDplusAoz"
