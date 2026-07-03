"""argparse CLI for `python -m modernbert_g2p.data build`.

Implements docs/design/phase1_data_pipeline.md §3 (ingestion pipeline),
§5 (dedup), §6 (split & contamination), §7 (sample weighting), and §9
(storage/manifest).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modernbert_g2p.data.schema import Row

_SOURCE_CHOICES: tuple[str, ...] = (
    "pyopenjtalk_plus",
    "unidic",
    "jmdict",
    "wikipedia",
    "aozora",
)

_DEFAULT_RATIOS: tuple[float, float, float] = (0.85, 0.05, 0.10)


def parse_source_root_pairs(pairs: Sequence[str]) -> dict[str, Path]:
    """Parse repeatable ``NAME=PATH`` strings into a mapping."""
    out: dict[str, Path] = {}
    for spec in pairs:
        if "=" not in spec:
            raise ValueError(f"Expected NAME=PATH, got {spec!r}")
        name, _, raw = spec.partition("=")
        name = name.strip()
        raw = raw.strip()
        if not name or not raw:
            raise ValueError(f"Malformed NAME=PATH pair: {spec!r}")
        out[name] = Path(raw)
    return out


def make_parser() -> argparse.ArgumentParser:
    """Build the top-level argparse parser."""
    parser = argparse.ArgumentParser(
        prog="python -m modernbert_g2p.data",
        description="Phase 1 data pipeline CLI (see docs/design/phase1_data_pipeline.md)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    p_build = subparsers.add_parser(
        "build", help="Build normalized corpus splits from raw sources"
    )
    p_build.add_argument(
        "--source",
        action="append",
        choices=list(_SOURCE_CHOICES),
        metavar="NAME",
        help="Source to ingest (repeatable). One of: %(choices)s.",
    )
    p_build.add_argument(
        "--root",
        action="append",
        metavar="NAME=PATH",
        help="Per-source root directory in NAME=PATH form (repeatable).",
    )
    p_build.add_argument(
        "--output",
        type=Path,
        required=True,
        metavar="DIR",
        help="Output directory for train/val/test files and manifest.yaml.",
    )
    p_build.add_argument("--limit", type=int, default=None, help="Cap rows per source.")
    p_build.add_argument("--seed", type=int, default=42, help="Split seed (default: 42).")
    p_build.add_argument(
        "--ratios",
        nargs=3,
        type=float,
        default=list(_DEFAULT_RATIOS),
        metavar=("TRAIN", "VAL", "TEST"),
        help="Split ratios summing to 1.0 (default: %(default)s).",
    )
    p_build.add_argument("--jsut-yaml", type=Path, default=None)
    p_build.add_argument("--rohan-txt", type=Path, default=None)
    p_build.add_argument("--jvs-txt", type=Path, default=None)
    p_build.add_argument("--hard-set-jsonl", type=Path, default=None)
    fmt = p_build.add_mutually_exclusive_group()
    fmt.add_argument(
        "--parquet", dest="format", action="store_const", const="parquet",
        help="Force Parquet output.",
    )
    fmt.add_argument(
        "--jsonl", dest="format", action="store_const", const="jsonl",
        help="Force JSONL output (default fallback when pyarrow missing).",
    )
    p_build.set_defaults(format=None)

    p_info = subparsers.add_parser("info", help="Print registered ingest sources.")
    p_info.add_argument(
        "--sources", action="store_true", help="List registered sources (default action)."
    )

    p_verify = subparsers.add_parser(
        "verify-manifest",
        help="Recompute SHA-256 of each split and compare against manifest.yaml.",
    )
    p_verify.add_argument("directory", type=Path, help="Directory containing manifest.yaml.")

    return parser


def _known_sources_safe() -> list[str]:
    try:
        from modernbert_g2p.data.ingest import known_sources
    except ImportError:
        return list(_SOURCE_CHOICES)
    registered = list(known_sources())
    if not registered:
        return list(_SOURCE_CHOICES)
    return sorted(set(registered) | set(_SOURCE_CHOICES))


def _sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def cmd_build(args: argparse.Namespace) -> int:
    sources: list[str] = args.source or []
    if not sources:
        print("error: at least one --source is required", file=sys.stderr)
        return 2

    try:
        roots = parse_source_root_pairs(args.root or [])
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    missing = [s for s in sources if s not in roots]
    if missing:
        print(
            f"error: --root missing for source(s): {', '.join(missing)}",
            file=sys.stderr,
        )
        return 2

    train, val, test = args.ratios
    if abs((train + val + test) - 1.0) > 1e-6:
        print(
            f"error: --ratios must sum to 1.0, got {train + val + test}",
            file=sys.stderr,
        )
        return 2

    from modernbert_g2p.data.contamination import (
        ContaminationFilter,
        load_held_out_texts,
    )
    from modernbert_g2p.data.dedup import Deduper
    from modernbert_g2p.data.ingest import get_source
    from modernbert_g2p.data.output import (
        HAS_PYARROW,
        write_jsonl,
        write_manifest,
        write_parquet,
    )
    from modernbert_g2p.data.split import SplitConfig, partition
    from modernbert_g2p.data.weighting import apply_weight

    fmt = args.format
    if fmt is None:
        fmt = "parquet" if HAS_PYARROW else "jsonl"
    if fmt == "parquet" and not HAS_PYARROW:
        print(
            "error: pyarrow not installed; re-run with --jsonl or install pyarrow.",
            file=sys.stderr,
        )
        return 2

    held_texts, held_phonemes = load_held_out_texts(
        args.jsut_yaml, args.rohan_txt, args.jvs_txt, args.hard_set_jsonl
    )
    contamination = ContaminationFilter()
    contamination.add_held_out(held_texts, held_phonemes)

    cfg = SplitConfig(train=train, val=val, test=test, seed=args.seed)

    deduper = Deduper()
    unique_rows: list[Row] = []
    source_counts: dict[str, int] = {}
    for name in sources:
        cls = get_source(name)
        seen_in_source = 0
        for row in cls().entries(roots[name], limit=args.limit):
            seen_in_source += 1
            if deduper.add(row):
                unique_rows.append(apply_weight(row))
        source_counts[name] = seen_in_source

    splits = partition(unique_rows, cfg=cfg, contamination=contamination)

    args.output.mkdir(parents=True, exist_ok=True)
    ext = "parquet" if fmt == "parquet" else "jsonl"
    split_paths: dict[str, Path] = {}
    for split_name in ("train", "val", "test"):
        target = args.output / f"{split_name}.{ext}"
        rows = splits[split_name]
        if fmt == "parquet":
            write_parquet(rows, target)
        else:
            write_jsonl(rows, target)
        split_paths[split_name] = target

    manifest_path = write_manifest(split_paths, output_dir=args.output)

    _print_build_summary(
        args.output,
        source_counts,
        deduper,
        contamination,
        splits,
        split_paths,
        manifest_path,
    )
    return 0


def _print_build_summary(
    output_dir: Path,
    source_counts: dict[str, int],
    deduper: object,
    contamination: object,
    splits: dict[str, list[Row]],
    split_paths: dict[str, Path],
    manifest_path: Path,
) -> None:
    print(f"Wrote splits to {output_dir}")
    print("Per-source input rows:")
    for name, n in source_counts.items():
        print(f"  {name}: {n}")
    if hasattr(deduper, "stats"):
        print(f"Dedup stats: {deduper.stats()}")  # type: ignore[attr-defined]
    if hasattr(contamination, "stats"):
        print(f"Contamination stats: {contamination.stats()}")  # type: ignore[attr-defined]
    print("Split sizes:")
    for split_name in ("train", "val", "test"):
        p = split_paths[split_name]
        print(f"  {split_name}: {len(splits[split_name])} rows -> {p.name}")
    print(f"Manifest: {manifest_path}")


def cmd_info(args: argparse.Namespace) -> int:  # noqa: ARG001
    sources = _known_sources_safe()
    print("Registered ingest sources:")
    for name in sources:
        print(f"  {name}")
    return 0


def cmd_verify_manifest(args: argparse.Namespace) -> int:
    import yaml

    manifest_path = args.directory / "manifest.yaml"
    if not manifest_path.exists():
        print(f"error: manifest not found at {manifest_path}", file=sys.stderr)
        return 2
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f) or {}

    splits = manifest.get("splits") or {}
    if not isinstance(splits, dict) or not splits:
        print("error: manifest has no 'splits' section", file=sys.stderr)
        return 2

    all_ok = True
    for split_name, info in sorted(splits.items()):
        if not isinstance(info, dict):
            print(f"[FAIL] {split_name}: entry is not a mapping", file=sys.stderr)
            all_ok = False
            continue
        rel_path = info.get("path")
        expected = info.get("sha256")
        if rel_path is None or expected is None:
            print(
                f"[FAIL] {split_name}: missing 'path' or 'sha256'", file=sys.stderr
            )
            all_ok = False
            continue
        candidate = args.directory / str(rel_path)
        if not candidate.exists():
            print(f"[FAIL] {split_name}: file missing at {candidate}", file=sys.stderr)
            all_ok = False
            continue
        actual = _sha256_file(candidate)
        if actual == expected:
            print(f"[OK]   {split_name}: {rel_path}")
        else:
            print(f"[FAIL] {split_name}: {rel_path}")
            print(f"  expected: {expected}")
            print(f"  actual:   {actual}")
            all_ok = False
    return 0 if all_ok else 1


_COMMANDS = {
    "build": cmd_build,
    "info": cmd_info,
    "verify-manifest": cmd_verify_manifest,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    handler = _COMMANDS.get(args.command)
    if handler is None:
        parser.print_help(sys.stderr)
        return 2
    return handler(args)


__all__ = [
    "cmd_build",
    "cmd_info",
    "cmd_verify_manifest",
    "main",
    "make_parser",
    "parse_source_root_pairs",
]
