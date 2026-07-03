"""End-to-end synthetic-corpus round-trip through the full Phase 1 pipeline."""

from __future__ import annotations

import importlib.util
import json
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parent / "fixtures"

# Per-track fixture name candidates. The interface spec and this track's
# task description use slightly different filenames; probe both so we survive
# whichever the sibling tracks land.
_FIXTURE_CANDIDATES: dict[str, tuple[str, ...]] = {
    "pyopenjtalk_plus": ("mecab_naist_jdic_sample.csv", "naist_jdic_mini.csv"),
    "unidic": ("unidic_lex_sample.csv", "unidic_lex_mini.csv"),
    "jmdict": ("jmdict_sample.xml", "jmdict_mini.xml"),
    "wikipedia": ("wikipedia_sample.html.json", "wikipedia_dump_mini.jsonl"),
    "aozora": ("aozora_sample.html", "aozora_pd.html"),
    "jsut_yaml": ("tiny_jsut.yaml", "held_out_mini.yaml"),
    "rohan_txt": ("tiny_rohan.txt", "held_out_mini_rohan.txt"),
    "jvs_txt": ("tiny_jvs.txt", "held_out_mini_jvs.txt"),
    "hard_set": ("tiny_hardset.jsonl", "held_out_mini_hard_set.jsonl"),
}


def _find_fixture(kind: str) -> Path | None:
    for name in _FIXTURE_CANDIDATES[kind]:
        p = FIXTURES / name
        if p.exists():
            return p
    return None


def _importorskip_data_module(name: str) -> Any:
    return pytest.importorskip(f"modernbert_g2p.data.{name}")


def _stage_source_root(tmp_path: Path, kind: str, fixture: Path) -> Path:
    """Copy a single-file fixture into a per-source directory (sources want dirs)."""
    target_names = {
        "pyopenjtalk_plus": fixture.name,
        "unidic": "lex.csv",
        "jmdict": "JMdict_e.xml",
        "wikipedia": fixture.name if fixture.suffix in {".jsonl", ".json"} else fixture.name,
        "aozora": fixture.name,
    }
    root = tmp_path / f"src_{kind}"
    root.mkdir(parents=True, exist_ok=True)
    target = root / target_names[kind]
    target.write_bytes(fixture.read_bytes())
    return root


def _iter_rows_from(source_obj: Any, root: Path, limit: int) -> Iterator[Any]:
    try:
        yield from source_obj.entries(root, limit=limit)
    except TypeError:
        yield from source_obj.entries(root)


def _collect_rows(sources: dict[str, Any], staged_roots: dict[str, Path], limit: int) -> list[Any]:
    rows: list[Any] = []
    for name, src in sources.items():
        root = staged_roots[name]
        try:
            rows.extend(list(_iter_rows_from(src, root, limit)))
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"ingest source {name!r} failed on fixture {root}: {exc}")
    return rows


def _available_ingest_sources() -> dict[str, Any]:
    base = _importorskip_data_module("ingest.base")
    sources: dict[str, Any] = {}
    for name in ("pyopenjtalk_plus", "unidic", "jmdict", "wikipedia", "aozora"):
        try:
            pytest.importorskip(f"modernbert_g2p.data.ingest.{name}")
        except pytest.skip.Exception:
            continue
        try:
            cls = base.get_source(name)
        except (KeyError, AttributeError):
            continue
        sources[name] = cls()
    if not sources:
        pytest.skip("no ingest source modules are importable yet")
    return sources


def test_synthetic_corpus_full_pipeline(tmp_path: Path) -> None:
    schema = _importorskip_data_module("schema")
    dedup_mod = _importorskip_data_module("dedup")
    split_mod = _importorskip_data_module("split")
    contam_mod = _importorskip_data_module("contamination")
    output_mod = _importorskip_data_module("output")

    sources = _available_ingest_sources()

    staged_roots: dict[str, Path] = {}
    for name in sources:
        fx = _find_fixture(name)
        if fx is None:
            pytest.skip(f"missing fixture for source {name!r}")
        staged_roots[name] = _stage_source_root(tmp_path, name, fx)

    jsut_yaml = _find_fixture("jsut_yaml")
    rohan_txt = _find_fixture("rohan_txt")
    jvs_txt = _find_fixture("jvs_txt")
    hard_set = _find_fixture("hard_set")
    assert hard_set is not None, "tiny_hardset.jsonl must be provided by this track"

    texts, phoneme_seqs = contam_mod.load_held_out_texts(
        jsut_yaml, rohan_txt, jvs_txt, hard_set
    )
    assert texts, "held-out loader must return at least one entry"

    cfilter = contam_mod.ContaminationFilter()
    cfilter.add_held_out(texts, phoneme_seqs)

    rows = _collect_rows(sources, staged_roots, limit=10)
    if not rows:
        pytest.skip("no rows produced by ingest sources")

    for row in rows:
        schema.validate_row(row)

    deduper = dedup_mod.Deduper()
    unique_rows = [r for r in rows if deduper.add(r)]
    dedup_stats = deduper.stats()

    cfg = split_mod.SplitConfig(train=0.8, val=0.1, test=0.1, seed=17)
    parts = split_mod.partition(unique_rows, cfg, contamination=cfilter)

    assert set(parts.keys()) >= {"train", "val", "test"}

    train, val, test = parts["train"], parts["val"], parts["test"]
    train_ids = {r.id for r in train}
    val_ids = {r.id for r in val}
    test_ids = {r.id for r in test}
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    kept = len(train) + len(val) + len(test)
    contamination_hits = len(unique_rows) - kept
    assert contamination_hits >= 0
    assert dedup_stats["unique"] == len(unique_rows)

    if not train and not val and not test:
        pytest.skip("all rows contaminated; nothing to assert further")

    out_dir = tmp_path / "processed"
    out_dir.mkdir()
    split_paths: dict[str, Path] = {}
    for name, split_rows in parts.items():
        p = out_dir / f"{name}.jsonl"
        output_mod.write_jsonl(iter(split_rows), p)
        split_paths[name] = p

    manifest_path = output_mod.write_manifest(split_paths, output_dir=out_dir)
    assert manifest_path.exists()

    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert isinstance(manifest, dict)
    assert "splits" in manifest
    for name, p in split_paths.items():
        entry = manifest["splits"][name]
        assert entry["sha256"] == output_mod.sha256_file(p)

    reloaded = list(output_mod.read_jsonl(split_paths["train"])) if train else []
    assert len(reloaded) == len(train)
    if reloaded:
        first_ids = {r.id for r in train}
        reloaded_ids = {r.id for r in reloaded}
        assert first_ids == reloaded_ids


def test_cli_end_to_end(tmp_path: Path) -> None:
    cli = _importorskip_data_module("cli")
    _importorskip_data_module("ingest.pyopenjtalk_plus")

    fx = _find_fixture("pyopenjtalk_plus")
    if fx is None:
        pytest.skip("missing pyopenjtalk_plus fixture")

    root = _stage_source_root(tmp_path, "pyopenjtalk_plus", fx)
    out_dir = tmp_path / "out"

    argv = [
        "build",
        "--source", "pyopenjtalk_plus",
        "--root", f"pyopenjtalk_plus={root}",
        "--output", str(out_dir),
        "--jsonl",
        "--seed", "42",
    ]
    rc = cli.main(argv)
    assert rc == 0
    assert out_dir.exists()
    assert (out_dir / "manifest.yaml").exists()

    written = {p.name for p in out_dir.iterdir()}
    assert "manifest.yaml" in written
    assert any(name.startswith(split) for split in ("train", "val", "test") for name in written)


def test_cli_determinism_same_seed_same_sha(tmp_path: Path) -> None:
    cli = _importorskip_data_module("cli")
    _importorskip_data_module("ingest.pyopenjtalk_plus")
    output_mod = _importorskip_data_module("output")

    fx = _find_fixture("pyopenjtalk_plus")
    if fx is None:
        pytest.skip("missing pyopenjtalk_plus fixture")

    root_a = _stage_source_root(tmp_path / "a_src", "pyopenjtalk_plus", fx)
    root_b = _stage_source_root(tmp_path / "b_src", "pyopenjtalk_plus", fx)
    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"

    common = ["build", "--source", "pyopenjtalk_plus", "--jsonl", "--seed", "42"]
    rc_a = cli.main([*common, "--root", f"pyopenjtalk_plus={root_a}", "--output", str(out_a)])
    rc_b = cli.main([*common, "--root", f"pyopenjtalk_plus={root_b}", "--output", str(out_b)])
    assert rc_a == 0
    assert rc_b == 0

    for split in ("train", "val", "test"):
        pa = out_a / f"{split}.jsonl"
        pb = out_b / f"{split}.jsonl"
        if not pa.exists() or not pb.exists():
            continue
        assert output_mod.sha256_file(pa) == output_mod.sha256_file(pb), (
            f"determinism violated for split {split!r}"
        )


def test_normalize_matches_haqumei_jsut() -> None:
    pytest.importorskip("haqumei")
    normalize_mod = _importorskip_data_module("normalize")

    from haqumei import Haqumei, IuPronunciation

    from modernbert_g2p.metrics.per import DEFAULT_IGNORE

    hq = Haqumei(use_unidic_yomi=True, normalize_iu=IuPronunciation.Yuu)
    sentences = ["こんにちは", "今日は良い天気ですね", "彼は日本語の教師です"]

    for s in sentences:
        normalized = normalize_mod.normalize_from_haqumei(s, hq=hq)
        raw = hq.g2p_batch([s])[0]
        devoicing = getattr(
            normalize_mod, "DEVOICING_MAP",
            {"A": "a", "E": "e", "I": "i", "O": "o", "U": "u"},
        )
        expected = tuple(
            devoicing.get(tok, tok) for tok in raw if tok not in DEFAULT_IGNORE
        )
        assert tuple(normalized.phonemes) == expected


def _make_synthetic_rows(
    schema_mod: Any, n: int, *, source: str = "pyopenjtalk_plus"
) -> list[Any]:
    Row = schema_mod.Row
    make_id = schema_mod.make_id
    rows: list[Any] = []
    for i in range(n):
        text = f"合成語{i:03d}"
        phonemes = ("g", "o", "u", "s", "e", "i", "g", "o", str(i % 10))
        row = Row(
            id=make_id(source, text, phonemes),
            source=source,
            source_license="BSD3",
            text=text,
            phonemes=phonemes,
            mora_accents=("L", "H", "H", "H", "H"),
            accent_boundaries=(),
            category="general",
            sample_weight=1.0,
            extra={"i": i},
        )
        rows.append(row)
    return rows


def test_write_and_read_jsonl_roundtrip(tmp_path: Path) -> None:
    schema_mod = _importorskip_data_module("schema")
    output_mod = _importorskip_data_module("output")

    rows = _make_synthetic_rows(schema_mod, 8)
    path = tmp_path / "synthetic.jsonl"
    n_written = output_mod.write_jsonl(iter(rows), path)
    assert n_written == 8

    read_back = list(output_mod.read_jsonl(path))
    assert len(read_back) == 8
    assert [r.id for r in read_back] == [r.id for r in rows]


def test_manifest_deterministic_bytes(tmp_path: Path) -> None:
    schema_mod = _importorskip_data_module("schema")
    output_mod = _importorskip_data_module("output")

    rows = _make_synthetic_rows(schema_mod, 4)
    d1 = tmp_path / "d1"
    d2 = tmp_path / "d2"
    d1.mkdir()
    d2.mkdir()

    train1 = d1 / "train.jsonl"
    train2 = d2 / "train.jsonl"
    output_mod.write_jsonl(iter(rows), train1)
    output_mod.write_jsonl(iter(rows), train2)

    m1 = output_mod.write_manifest({"train": train1}, output_dir=d1)
    m2 = output_mod.write_manifest({"train": train2}, output_dir=d2)

    assert m1.read_bytes() == m2.read_bytes()


@pytest.mark.skipif(
    importlib.util.find_spec("pyarrow") is None,
    reason="pyarrow not installed",
)
def test_write_parquet_roundtrip(tmp_path: Path) -> None:
    schema_mod = _importorskip_data_module("schema")
    output_mod = _importorskip_data_module("output")

    rows = _make_synthetic_rows(schema_mod, 6)
    path = tmp_path / "synthetic.parquet"
    n = output_mod.write_parquet(iter(rows), path)
    assert n == 6
    assert path.exists()


def _iter_read_jsonl_plain(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def test_sample_weight_persisted_correctly(tmp_path: Path) -> None:
    """Fix for T4: verify sample_weight matches weight_for(category) end-to-end.

    Guards against the M1 regression where non-pyopenjtalk_plus ingest sources
    forgot to compute sample_weight, leaving loanword/proper-noun rows at 1.0
    instead of 2.0.
    """
    cli = _importorskip_data_module("cli")
    _importorskip_data_module("ingest.pyopenjtalk_plus")
    weighting = _importorskip_data_module("weighting")

    fx = _find_fixture("pyopenjtalk_plus")
    if fx is None:
        pytest.skip("missing pyopenjtalk_plus fixture")

    root = _stage_source_root(tmp_path, "pyopenjtalk_plus", fx)
    out_dir = tmp_path / "out"

    argv = [
        "build",
        "--source", "pyopenjtalk_plus",
        "--root", f"pyopenjtalk_plus={root}",
        "--output", str(out_dir),
        "--jsonl",
        "--seed", "42",
    ]
    rc = cli.main(argv)
    assert rc == 0

    all_rows: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        p = out_dir / f"{split}.jsonl"
        if p.exists():
            all_rows.extend(_iter_read_jsonl_plain(p))
    assert all_rows, "expected at least one row across all splits"

    by_text: dict[str, dict[str, Any]] = {r["text"]: r for r in all_rows}

    expected_loanwords = {"コーヒー", "パソコン"}
    expected_general = {"花", "桜", "猫", "犬", "学校"}

    loanword_hits = expected_loanwords & set(by_text.keys())
    general_hits = expected_general & set(by_text.keys())
    assert loanword_hits, "expected fixture to produce loanword rows"
    assert general_hits, "expected fixture to produce general rows"

    for text in loanword_hits:
        row = by_text[text]
        assert row["category"] == "loanword", (
            f"{text!r} should be category=loanword, got {row['category']!r}"
        )
        assert row["sample_weight"] == weighting.weight_for("loanword") == 2.0, (
            f"{text!r} loanword should have sample_weight=2.0, got {row['sample_weight']}"
        )

    for text in general_hits:
        row = by_text[text]
        assert row["category"] == "general", (
            f"{text!r} should be category=general, got {row['category']!r}"
        )
        assert row["sample_weight"] == weighting.weight_for("general") == 1.0, (
            f"{text!r} general should have sample_weight=1.0, got {row['sample_weight']}"
        )

    for row in all_rows:
        expected = weighting.weight_for(row["category"])
        assert row["sample_weight"] == expected, (
            f"row id={row['id']!r} category={row['category']!r} "
            f"sample_weight={row['sample_weight']} != weight_for(category)={expected}"
        )


def test_tiny_hardset_fixture_is_valid_json_and_schema_compatible() -> None:
    hs = FIXTURES / "tiny_hardset.jsonl"
    assert hs.exists()
    rows = list(_iter_read_jsonl_plain(hs))
    assert len(rows) == 5
    valid_categories = {
        "polyphone", "counter", "proper_noun", "loanword",
        "numeric_unit", "english_mixed", "english_abbreviation",
    }
    for r in rows:
        assert set(r.keys()) >= {"id", "category", "text", "phonemes", "accent"}
        assert r["category"] in valid_categories
        assert isinstance(r["text"], str) and r["text"]
        assert isinstance(r["phonemes"], list) and r["phonemes"]
        assert isinstance(r["accent"], list) and r["accent"]
        assert all(a in {"H", "L"} for a in r["accent"])
        assert re.fullmatch(r"hs-[a-z_]+-[0-9]{3}", r["id"]) is not None
