"""Deterministic hash-based train/val/test partitioner with contamination integration.

Implements docs/design/phase1_data_pipeline.md §6.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from modernbert_g2p.data.contamination import ContaminationFilter
    from modernbert_g2p.data.dedup import Deduper
    from modernbert_g2p.data.schema import Row


Split = Literal["train", "val", "test"]
SPLITS: tuple[Split, ...] = ("train", "val", "test")

_RATIO_TOL: float = 1e-6


@dataclass(frozen=True, slots=True)
class SplitConfig:
    train: float = 0.96
    val: float = 0.02
    test: float = 0.02
    seed: int = 42


_DEFAULT_SPLIT_CONFIG: SplitConfig = SplitConfig()


def _validate_ratios(ratios: Sequence[float]) -> None:
    if len(ratios) != 3:
        raise ValueError(f"ratios must be length 3, got {len(ratios)}")
    total = sum(ratios)
    if abs(total - 1.0) > _RATIO_TOL:
        raise ValueError(f"ratios must sum to 1.0 within {_RATIO_TOL}, got {total}")
    if any(r < 0 for r in ratios):
        raise ValueError(f"ratios must be non-negative, got {tuple(ratios)}")


def deterministic_split(
    row_id: str,
    seed: int = 42,
    ratios: Sequence[float] = (0.85, 0.05, 0.10),
) -> Split:
    _validate_ratios(ratios)
    digest = hashlib.blake2b(f"{seed}:{row_id}".encode(), digest_size=8).digest()
    bucket = int.from_bytes(digest, "big") / 2**64
    if bucket < ratios[0]:
        return "train"
    if bucket < ratios[0] + ratios[1]:
        return "val"
    return "test"


def assign_split(row: Row, cfg: SplitConfig = _DEFAULT_SPLIT_CONFIG) -> Split:
    return deterministic_split(
        row.id, seed=cfg.seed, ratios=(cfg.train, cfg.val, cfg.test)
    )


def split_corpus(
    rows: Iterable[Row],
    *,
    contamination_filter: ContaminationFilter | None = None,
    deduper: Deduper | None = None,
    seed: int = 42,
    ratios: Sequence[float] = (0.85, 0.05, 0.10),
) -> dict[str, list[Row]]:
    _validate_ratios(ratios)
    out: dict[str, list[Row]] = {
        "train": [],
        "val": [],
        "test": [],
        "excluded_contaminated": [],
        "excluded_duplicate": [],
    }
    for row in rows:
        if deduper is not None and not deduper.add(row):
            out["excluded_duplicate"].append(row)
            continue
        if contamination_filter is not None:
            hit, _layer = contamination_filter.is_contaminated(row.text, row.phonemes)
            if hit:
                out["excluded_contaminated"].append(row)
                continue
        out[deterministic_split(row.id, seed=seed, ratios=ratios)].append(row)
    return out


def partition(
    rows: Iterable[Row],
    cfg: SplitConfig = _DEFAULT_SPLIT_CONFIG,
    *,
    contamination: ContaminationFilter | None = None,
) -> dict[Split, list[Row]]:
    _validate_ratios((cfg.train, cfg.val, cfg.test))
    out: dict[Split, list[Row]] = {"train": [], "val": [], "test": []}
    for row in rows:
        if contamination is not None:
            hit, _layer = contamination.is_contaminated(row.text, row.phonemes)
            if hit:
                continue
        out[assign_split(row, cfg)].append(row)
    return out
