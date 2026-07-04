"""Parallel Wikipedia shard ingest → dump JSONL per shard, merged into one file.

Uses ProcessPoolExecutor to fan out 72 Wikipedia shards across CPU cores.
Each worker processes one shard and appends its rows to an intermediate JSONL
under ``--output``. The main process concatenates all worker outputs at the end.

Usage:
    python scripts/parallel_wiki_ingest.py \
        --shards-dir data/raw/wikipedia/extracted \
        --output data/interim/wiki_rows.jsonl \
        --workers 16
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _process_shard(shard_path_str: str, output_dir_str: str) -> tuple[str, int]:
    from modernbert_g2p.data.ingest.wikipedia import WikipediaSource
    from modernbert_g2p.data.schema import to_dict

    shard = Path(shard_path_str)
    output_dir = Path(output_dir_str)
    out_path = output_dir / f"{shard.stem}.jsonl"
    count = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for row in WikipediaSource().entries(shard):
            fh.write(json.dumps(to_dict(row), ensure_ascii=False))
            fh.write("\n")
            count += 1
    return shard.name, count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--pattern", default="*.ndjson")
    args = parser.parse_args(argv)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    interim_dir = args.output.parent / "_shards"
    interim_dir.mkdir(exist_ok=True)

    shards = sorted(args.shards_dir.rglob(args.pattern))
    if not shards:
        sys.exit(f"no shards matched {args.pattern} under {args.shards_dir}")
    print(f"[parallel] {len(shards)} shards, {args.workers} workers", file=sys.stderr)

    t0 = time.perf_counter()
    total = 0
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(_process_shard, str(s), str(interim_dir)): s for s in shards
        }
        for fut in as_completed(futures):
            name, cnt = fut.result()
            total += cnt
            done += 1
            elapsed = time.perf_counter() - t0
            print(
                f"[{done}/{len(shards)}] {name}: {cnt} rows "
                f"(cum {total}, elapsed {elapsed:.1f}s)",
                file=sys.stderr,
            )

    print(f"[merge] concatenating {len(shards)} shards -> {args.output}", file=sys.stderr)
    with args.output.open("w", encoding="utf-8") as out:
        for s in shards:
            piece = interim_dir / f"{s.stem}.jsonl"
            if piece.exists():
                with piece.open("r", encoding="utf-8") as p:
                    for line in p:
                        out.write(line)
    print(f"[done] total rows: {total} in {time.perf_counter() - t0:.1f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
