"""Tests for modernbert_g2p.data.output — Parquet + JSONL + manifest writers."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from modernbert_g2p.data.output import (
    read_jsonl,
    sha256_of_file,
    write_jsonl,
    write_manifest,
    write_parquet,
)
from modernbert_g2p.data.schema import Row


def _make_row(i: int, *, source: str = "pyopenjtalk_plus") -> Row:
    return Row(
        id=f"{source}:{i:012x}",
        source=source,
        source_license="BSD3",
        text=f"テスト{i}",
        phonemes=("t", "e", "s", "u", "t", "o"),
        mora_accents=("L", "H", "H"),
        accent_boundaries=(0,),
        category="general",
        sample_weight=1.0,
        extra={"idx": i, "note": f"サンプル{i}"},
    )


def _make_rows(n: int) -> list[Row]:
    return [_make_row(i) for i in range(n)]


def test_write_jsonl_roundtrip_preserves_rows(tmp_path: Path) -> None:
    rows = _make_rows(10)
    path = tmp_path / "sample.jsonl"

    written = write_jsonl(iter(rows), path)
    assert written == 10

    reloaded = list(read_jsonl(path))
    assert reloaded == rows


def test_write_jsonl_empty_iterable_yields_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    n = write_jsonl(iter([]), path)
    assert n == 0
    assert path.exists()
    assert path.read_bytes() == b""
    assert list(read_jsonl(path)) == []


def test_read_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    rows = _make_rows(3)
    path = tmp_path / "blanks.jsonl"
    write_jsonl(iter(rows), path)

    text = path.read_text(encoding="utf-8")
    with path.open("w", encoding="utf-8") as f:
        f.write("\n")
        f.write(text)
        f.write("\n\n")

    reloaded = list(read_jsonl(path))
    assert reloaded == rows


def test_read_jsonl_reports_line_number_on_error(tmp_path: Path) -> None:
    path = tmp_path / "broken.jsonl"
    good = _make_row(0)
    lines = [
        __import__("json").dumps(_row_dict(good), ensure_ascii=False, sort_keys=True),
        __import__("json").dumps(_row_dict(good), ensure_ascii=False, sort_keys=True),
        "not json",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="line 3"):
        list(read_jsonl(path))


def _row_dict(row: Row) -> dict[str, object]:
    from modernbert_g2p.data.schema import to_dict

    return to_dict(row)


def test_write_parquet_falls_back_to_jsonl_when_pyarrow_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("modernbert_g2p.data.output.HAS_PYARROW", False)
    rows = _make_rows(4)
    parquet_path = tmp_path / "shard.parquet"

    n = write_parquet(iter(rows), parquet_path)
    assert n == 4

    fallback = parquet_path.with_suffix(".jsonl")
    assert fallback.exists()
    assert not parquet_path.exists()
    reloaded = list(read_jsonl(fallback))
    assert reloaded == rows


def test_write_parquet_roundtrip_when_pyarrow_available(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    rows = _make_rows(6)
    path = tmp_path / "shard.parquet"
    n = write_parquet(iter(rows), path, batch_size=2)
    assert n == 6

    table = pq.read_table(str(path))
    assert table.num_rows == 6
    phonemes = table.column("phonemes").to_pylist()
    assert phonemes[0] == list(rows[0].phonemes)
    ids = table.column("id").to_pylist()
    assert ids == [r.id for r in rows]


def test_sha256_of_file_matches_hashlib(tmp_path: Path) -> None:
    data = b"the quick brown fox\n" * 500
    path = tmp_path / "blob.bin"
    path.write_bytes(data)

    expected = hashlib.sha256(data).hexdigest()
    assert sha256_of_file(path) == expected
    assert sha256_of_file(path) == sha256_of_file(path)


def test_write_manifest_records_counts_and_sha256(tmp_path: Path) -> None:
    output_dir = tmp_path / "processed"
    output_dir.mkdir()

    train_rows = _make_rows(8)
    val_rows = _make_rows(2)
    test_rows = _make_rows(2)

    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    test_path = output_dir / "test.jsonl"
    write_jsonl(iter(train_rows), train_path)
    write_jsonl(iter(val_rows), val_path)
    write_jsonl(iter(test_rows), test_path)

    manifest_path = write_manifest(
        {"train": train_path, "val": val_path, "test": test_path},
        output_dir=output_dir,
    )
    assert manifest_path == output_dir / "manifest.yaml"

    doc = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert doc["schema_version"] == "1.0"
    assert set(doc["splits"].keys()) == {"train", "val", "test"}
    assert doc["splits"]["train"]["rows"] == 8
    assert doc["splits"]["val"]["rows"] == 2
    assert doc["splits"]["test"]["rows"] == 2
    assert doc["splits"]["train"]["path"] == "train.jsonl"
    assert doc["splits"]["train"]["sha256"] == sha256_of_file(train_path)
    assert doc["splits"]["train"]["size_bytes"] == train_path.stat().st_size


def test_write_manifest_is_byte_deterministic(tmp_path: Path) -> None:
    output_dir = tmp_path / "processed"
    output_dir.mkdir()
    rows = _make_rows(5)
    a = output_dir / "a.jsonl"
    b = output_dir / "b.jsonl"
    write_jsonl(iter(rows), a)
    write_jsonl(iter(rows), b)

    m1 = write_manifest({"train": a, "val": b}, output_dir=output_dir)
    first = m1.read_bytes()
    m2 = write_manifest({"val": b, "train": a}, output_dir=output_dir)
    second = m2.read_bytes()
    assert first == second


def test_write_manifest_with_zero_splits(tmp_path: Path) -> None:
    output_dir = tmp_path / "processed"
    output_dir.mkdir()
    manifest_path = write_manifest({}, output_dir=output_dir)
    doc = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert doc == {"schema_version": "1.0", "splits": {}}
