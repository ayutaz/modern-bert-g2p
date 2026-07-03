"""Parquet + JSONL writers and manifest.yaml serialization.

Implements docs/design/phase1_data_pipeline.md §8 (output schema) and §9
(storage decision).

Extra note: the ``extra`` field of each ``Row`` is JSON-encoded as a string
column in the Parquet output (Parquet Map of heterogeneous values is awkward)
and JSON-decoded on read.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

import yaml

from modernbert_g2p.data.schema import SCHEMA_VERSION, Row, from_dict, to_dict

try:
    import pyarrow  # noqa: F401
    import pyarrow.parquet  # noqa: F401

    HAS_PYARROW: bool = True
except ImportError:
    HAS_PYARROW = False


def write_parquet(
    rows: Iterable[Row],
    path: Path,
    *,
    batch_size: int = 10000,
) -> int:
    """Write rows to Parquet, or fall back to JSONL when pyarrow is unavailable.

    When ``pyarrow`` is not installed the actual output is written to
    ``path.with_suffix(".jsonl")`` — the ``.parquet`` file is NOT created.
    Callers that pass the original ``path`` to downstream consumers (e.g.,
    ``write_manifest``, ``sha256_of_file``) MUST first check ``HAS_PYARROW``
    and swap in the ``.jsonl`` sibling. The CLI ``cmd_build`` gates on
    ``HAS_PYARROW`` and never routes a Parquet-named path through this
    fallback in production.
    """
    if not HAS_PYARROW:
        fallback = path.with_suffix(".jsonl")
        return write_jsonl(rows, fallback)

    import pyarrow as pa

    schema = pa.schema(
        [
            ("id", pa.string()),
            ("source", pa.string()),
            ("source_license", pa.string()),
            ("text", pa.string()),
            ("phonemes", pa.list_(pa.string())),
            ("mora_accents", pa.list_(pa.string())),
            ("accent_boundaries", pa.list_(pa.int32())),
            ("category", pa.string()),
            ("sample_weight", pa.float32()),
            ("extra", pa.string()),
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    batch: list[Row] = []
    writer: Any = None
    try:
        for row in rows:
            batch.append(row)
            if len(batch) >= batch_size:
                writer = _flush_parquet_batch(batch, schema, path, writer)
                total += len(batch)
                batch = []
        if batch:
            writer = _flush_parquet_batch(batch, schema, path, writer)
            total += len(batch)
    finally:
        if writer is not None:
            writer.close()
    return total


def _flush_parquet_batch(
    batch: list[Row],
    schema: Any,
    path: Path,
    writer: Any,
) -> Any:
    import pyarrow as pa
    import pyarrow.parquet as pq

    columns = {
        "id": [r.id for r in batch],
        "source": [r.source for r in batch],
        "source_license": [r.source_license for r in batch],
        "text": [r.text for r in batch],
        "phonemes": [list(r.phonemes) for r in batch],
        "mora_accents": [list(r.mora_accents) for r in batch],
        "accent_boundaries": [list(r.accent_boundaries) for r in batch],
        "category": [r.category for r in batch],
        "sample_weight": [float(r.sample_weight) for r in batch],
        "extra": [json.dumps(r.extra, ensure_ascii=False, sort_keys=True) for r in batch],
    }
    table = pa.Table.from_pydict(columns, schema=schema)
    if writer is None:
        writer = pq.ParquetWriter(str(path), schema)
    writer.write_table(table)
    return writer


def write_jsonl(rows: Iterable[Row], path: Path) -> int:
    """Write rows to newline-delimited JSON, one row per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(to_dict(row), ensure_ascii=False, sort_keys=True))
            f.write("\n")
            count += 1
    return count


def read_jsonl(path: Path) -> Iterator[Row]:
    """Stream rows from a JSONL file written by ``write_jsonl``."""
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as e:
                raise ValueError(f"malformed JSON at line {lineno}: {e}") from e
            yield from_dict(obj)


def sha256_of_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """Streaming SHA-256 hex digest of ``path``."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


sha256_file = sha256_of_file


def write_manifest(
    splits: Mapping[str, Path],
    *,
    output_dir: Path,
    schema_version: str = SCHEMA_VERSION,
) -> Path:
    """Write ``manifest.yaml`` describing every split's path, row count, and SHA-256."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.yaml"

    splits_info: dict[str, dict[str, object]] = {}
    for split_name, split_path in splits.items():
        p = Path(split_path)
        try:
            rel = str(p.relative_to(output_dir))
        except ValueError:
            rel = str(p)
        splits_info[split_name] = {
            "path": rel,
            "rows": _count_rows(p),
            "sha256": sha256_of_file(p),
            "size_bytes": p.stat().st_size,
        }

    doc: dict[str, object] = {
        "schema_version": schema_version,
        "splits": splits_info,
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, sort_keys=True, allow_unicode=True, default_flow_style=False)
    return manifest_path


def _count_rows(path: Path) -> int:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        if not HAS_PYARROW:
            raise RuntimeError("pyarrow required to count rows in a Parquet file")
        import pyarrow.parquet as pq

        return int(pq.ParquetFile(str(path)).metadata.num_rows)
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count
