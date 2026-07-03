from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import pytest

from modernbert_g2p.data.dedup import Deduper
from modernbert_g2p.data.split import (
    SPLITS,
    SplitConfig,
    assign_split,
    deterministic_split,
    partition,
    split_corpus,
)

try:
    from modernbert_g2p.data.schema import Row as _SchemaRow
except ImportError:
    _SchemaRow = None


@dataclass(frozen=True, slots=True)
class _FallbackRow:
    id: str
    text: str
    phonemes: tuple[str, ...]
    source: str = "pyopenjtalk_plus"
    source_license: str = "BSD3"
    mora_accents: tuple[str, ...] = ()
    accent_boundaries: tuple[int, ...] = ()
    category: str = "general"
    sample_weight: float = 1.0
    extra: dict[str, object] = field(default_factory=dict)


def _row(row_id: str, text: str = "", phonemes: tuple[str, ...] = ()) -> object:
    if _SchemaRow is None:
        return _FallbackRow(id=row_id, text=text, phonemes=phonemes)
    return _SchemaRow(
        id=row_id,
        source="pyopenjtalk_plus",
        source_license="BSD3",
        text=text,
        phonemes=phonemes,
        mora_accents=(),
        accent_boundaries=(),
    )


class _AlwaysContaminated:
    def is_contaminated(self, text: str, phonemes: object) -> tuple[bool, str]:
        return (True, "L1")


class _NeverContaminated:
    def is_contaminated(self, text: str, phonemes: object) -> tuple[bool, str]:
        return (False, "")


class _SetContaminated:
    def __init__(self, bad_ids: set[str]) -> None:
        self._bad = bad_ids

    def is_contaminated(self, text: str, phonemes: object) -> tuple[bool, str]:
        return ((text in self._bad), "L1" if text in self._bad else "")


def test_splits_constant() -> None:
    assert SPLITS == ("train", "val", "test")


def test_deterministic_split_same_input_same_output() -> None:
    for i in range(200):
        row_id = f"pyopenjtalk_plus:abc{i:03d}"
        first = deterministic_split(row_id, seed=42)
        second = deterministic_split(row_id, seed=42)
        assert first == second
        assert first in SPLITS


def test_deterministic_split_seed_changes_assignment() -> None:
    row_ids = [f"id-{i}" for i in range(500)]
    a = [deterministic_split(r, seed=1) for r in row_ids]
    b = [deterministic_split(r, seed=2) for r in row_ids]
    assert a != b


def test_deterministic_split_distribution_matches_ratios() -> None:
    row_ids = [f"row-{i:05d}" for i in range(10_000)]
    ratios = (0.85, 0.05, 0.10)
    counts = Counter(deterministic_split(r, seed=42, ratios=ratios) for r in row_ids)
    n = len(row_ids)
    for split, expected in zip(SPLITS, ratios, strict=True):
        observed = counts[split] / n
        assert abs(observed - expected) < 0.02, f"{split}: {observed} vs {expected}"


def test_assign_split_uses_row_id_and_config_seed() -> None:
    row = _row("pyopenjtalk_plus:abc123")
    cfg = SplitConfig(train=0.96, val=0.02, test=0.02, seed=42)
    assert assign_split(row, cfg) == deterministic_split(
        row.id, seed=42, ratios=(0.96, 0.02, 0.02)
    )


def test_deterministic_split_rejects_bad_ratios() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        deterministic_split("id-x", ratios=(0.5, 0.2, 0.2))
    with pytest.raises(ValueError, match="non-negative"):
        deterministic_split("id-x", ratios=(1.2, 0.3, -0.5))
    with pytest.raises(ValueError, match="length 3"):
        deterministic_split("id-x", ratios=(0.5, 0.5))


def test_split_corpus_sum_equals_input_count() -> None:
    rows = [_row(f"id-{i}", text=f"text-{i}") for i in range(500)]
    result = split_corpus(rows, seed=7)
    total = sum(len(v) for v in result.values())
    assert total == len(rows)
    assert result["excluded_contaminated"] == []
    assert result["excluded_duplicate"] == []


def test_split_corpus_contamination_filter_removes_hits() -> None:
    rows = [_row(f"id-{i}", text=f"text-{i}") for i in range(50)]
    bad_texts = {"text-3", "text-17", "text-42"}
    cf = _SetContaminated(bad_texts)
    result = split_corpus(rows, contamination_filter=cf)
    assert len(result["excluded_contaminated"]) == 3
    assert {r.text for r in result["excluded_contaminated"]} == bad_texts
    assert sum(len(result[s]) for s in SPLITS) == len(rows) - 3


def test_split_corpus_deduper_removes_duplicates() -> None:
    rows = [
        _row("id-0", text="alpha", phonemes=("a",)),
        _row("id-1", text="beta", phonemes=("b",)),
        _row("id-2", text="alpha", phonemes=("c",)),
        _row("id-3", text="gamma", phonemes=("a",)),
        _row("id-4", text="delta", phonemes=("d",)),
    ]
    d = Deduper()
    result = split_corpus(rows, deduper=d)
    assert len(result["excluded_duplicate"]) == 2
    assert {r.id for r in result["excluded_duplicate"]} == {"id-2", "id-3"}
    kept = sum(len(result[s]) for s in SPLITS)
    assert kept == 3


def test_split_corpus_deduper_and_contamination_combined() -> None:
    rows = [
        _row("id-0", text="alpha", phonemes=("a",)),
        _row("id-1", text="beta", phonemes=("b",)),
        _row("id-2", text="alpha", phonemes=("c",)),
        _row("id-3", text="omega", phonemes=("z",)),
    ]
    d = Deduper()
    cf = _SetContaminated({"beta"})
    result = split_corpus(rows, deduper=d, contamination_filter=cf)
    assert len(result["excluded_duplicate"]) == 1
    assert len(result["excluded_contaminated"]) == 1
    assert sum(len(result[s]) for s in SPLITS) == 2


def test_split_corpus_ratio_validation() -> None:
    rows = [_row("x-1")]
    with pytest.raises(ValueError, match="sum to 1"):
        split_corpus(rows, ratios=(0.5, 0.2, 0.2))


def test_partition_returns_three_splits_only() -> None:
    rows = [_row(f"p-{i}", text=f"t-{i}") for i in range(100)]
    result = partition(rows, SplitConfig(seed=17))
    assert set(result.keys()) == set(SPLITS)
    assert sum(len(v) for v in result.values()) == len(rows)


def test_partition_contamination_drops_rows() -> None:
    rows = [_row(f"p-{i}", text=f"t-{i}") for i in range(30)]
    result = partition(rows, contamination=_AlwaysContaminated())
    assert result == {"train": [], "val": [], "test": []}


def test_partition_no_contamination_keeps_all() -> None:
    rows = [_row(f"p-{i}", text=f"t-{i}") for i in range(30)]
    result = partition(rows, contamination=_NeverContaminated())
    assert sum(len(v) for v in result.values()) == 30


def test_partition_rejects_nonunit_config() -> None:
    rows = [_row("z-1")]
    with pytest.raises(ValueError, match="sum to 1"):
        partition(rows, SplitConfig(train=0.5, val=0.2, test=0.2))


def test_partition_deterministic_across_seeds() -> None:
    rows = [_row(f"a-{i}", text=str(i)) for i in range(300)]
    r1 = partition(rows, SplitConfig(seed=5))
    r2 = partition(rows, SplitConfig(seed=5))
    assert {s: [r.id for r in r1[s]] for s in SPLITS} == {
        s: [r.id for r in r2[s]] for s in SPLITS
    }
