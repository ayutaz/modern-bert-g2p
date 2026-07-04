"""Final 5-source build using precomputed Wikipedia JSONL + fresh S1-S3/S5 ingest.

Skips re-ingesting Wikipedia (which is expensive) by loading rows from a
JSONL produced by ``scripts/parallel_wiki_ingest.py``. Everything else
(pyopenjtalk_plus / unidic / jmdict / aozora) is ingested fresh.

Downstream stages (dedup → contamination → split → Parquet + manifest) are
identical to the standard ``python -m modernbert_g2p.data build`` path.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wiki-jsonl", type=Path, required=True)
    parser.add_argument("--pyopenjtalk-plus", type=Path, required=True)
    parser.add_argument("--unidic", type=Path, required=True)
    parser.add_argument("--jmdict", type=Path, required=True)
    parser.add_argument("--aozora", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hard-set-jsonl", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ratios", nargs=3, type=float, default=(0.85, 0.05, 0.10), metavar=("TRAIN", "VAL", "TEST")
    )
    args = parser.parse_args(argv)

    from modernbert_g2p.data.contamination import ContaminationFilter, load_held_out_texts
    from modernbert_g2p.data.dedup import Deduper
    from modernbert_g2p.data.ingest.aozora import AozoraSource
    from modernbert_g2p.data.ingest.jmdict import JMDictSource
    from modernbert_g2p.data.ingest.pyopenjtalk_plus import PyopenjtalkPlusSource
    from modernbert_g2p.data.ingest.unidic import UnidicSource
    from modernbert_g2p.data.output import read_jsonl, write_manifest, write_parquet
    from modernbert_g2p.data.split import split_corpus

    args.output.mkdir(parents=True, exist_ok=True)

    print("[stage] fresh ingest 4 sources", file=sys.stderr)
    t0 = time.perf_counter()
    sources = {
        "pyopenjtalk_plus": PyopenjtalkPlusSource().entries(args.pyopenjtalk_plus),
        "unidic": UnidicSource().entries(args.unidic),
        "jmdict": JMDictSource().entries(args.jmdict),
        "aozora": AozoraSource().entries(args.aozora),
    }

    per_source_count: dict[str, int] = {}
    all_rows = []
    for name, gen in sources.items():
        n = 0
        for r in gen:
            all_rows.append(r)
            n += 1
        per_source_count[name] = n
        print(f"[ingest] {name}: {n} rows ({time.perf_counter() - t0:.1f}s)", file=sys.stderr)

    print(f"[stage] loading wiki jsonl {args.wiki_jsonl}", file=sys.stderr)
    wiki_n = 0
    for r in read_jsonl(args.wiki_jsonl):
        all_rows.append(r)
        wiki_n += 1
    per_source_count["wikipedia"] = wiki_n
    print(f"[ingest] wikipedia (precomputed): {wiki_n} rows ({time.perf_counter() - t0:.1f}s)", file=sys.stderr)

    total_input = sum(per_source_count.values())
    print(f"[stage] total input rows: {total_input}", file=sys.stderr)

    print("[stage] contamination filter setup", file=sys.stderr)
    contam = ContaminationFilter()
    held_texts, held_phonemes = load_held_out_texts(
        jsut_yaml=None,
        rohan_txt=None,
        jvs_txt=None,
        hard_set_jsonl=args.hard_set_jsonl,
    )
    contam.add_held_out(held_texts, held_phonemes)
    print(f"[contam] held_out={len(held_texts)}", file=sys.stderr)

    print("[stage] dedup + contamination + split", file=sys.stderr)
    deduper = Deduper()
    splits = split_corpus(
        all_rows,
        contamination_filter=contam,
        deduper=deduper,
        seed=args.seed,
        ratios=tuple(args.ratios),
    )
    for k, v in splits.items():
        print(f"[split] {k}: {len(v)} rows", file=sys.stderr)

    print("[stage] write parquet + manifest", file=sys.stderr)
    split_paths: dict[str, Path] = {}
    for name in ("train", "val", "test", "excluded_contaminated", "excluded_duplicate"):
        rows = splits.get(name, [])
        if not rows:
            continue
        p = args.output / f"{name}.parquet"
        n = write_parquet(rows, p)
        actual = p if p.exists() else p.with_suffix(".jsonl")
        print(f"[write] {name}: {n} rows -> {actual.name}", file=sys.stderr)
        split_paths[name] = actual
    manifest = write_manifest(split_paths, output_dir=args.output)
    print(f"[done] manifest: {manifest} elapsed {time.perf_counter() - t0:.1f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
