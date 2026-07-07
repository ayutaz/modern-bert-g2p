"""Row dataclass and canonical schema constants for the Phase 1 corpus.

Implements docs/design/phase1_data_pipeline.md §8 (Output schema).
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, replace

SCHEMA_VERSION: str = "1.0"

KNOWN_SOURCES: frozenset[str] = frozenset({
    "pyopenjtalk_plus",
    "unidic",
    "jmdict",
    "wikipedia",
    "aozora",
})

KNOWN_LICENSES: frozenset[str] = frozenset({
    "BSD3",
    "CC-BY-4.0",
    "EDRDG",
    "CC-BY-SA-4.0",
    "PD",
    "PDplusAoz",
})

KNOWN_CATEGORIES: frozenset[str] = frozenset({
    "general",
    "numeric_unit",
    "proper_noun",  # seed_v2 uses the umbrella tag; kanji/katakana split will land in Phase 1
    "proper_noun_kanji",
    "proper_noun_katakana",
    "counter",
    "loanword",
    "english_mixed",
    "english_abbreviation",
    "polyphone",
})

_IGNORED_PHONEMES: frozenset[str] = frozenset({"pau"})
_ACCENT_TAGS: frozenset[str] = frozenset({"H", "L"})


@dataclass(frozen=True, slots=True)
class Row:
    id: str
    source: str
    source_license: str
    text: str
    phonemes: tuple[str, ...]
    mora_accents: tuple[str, ...]
    accent_boundaries: tuple[int, ...]
    category: str = "general"
    sample_weight: float = 1.0
    extra: dict[str, object] = field(default_factory=dict)


def make_id(source: str, text: str, phonemes: Sequence[str]) -> str:
    payload = f"{text}|{'|'.join(phonemes)}".encode()
    digest = hashlib.md5(payload).hexdigest()[:12]
    return f"{source}:{digest}"


def validate_row(row: Row) -> None:
    if row.source not in KNOWN_SOURCES:
        raise ValueError(f"source: unknown value {row.source!r}, expected one of {sorted(KNOWN_SOURCES)}")
    if row.source_license not in KNOWN_LICENSES:
        raise ValueError(
            f"source_license: unknown value {row.source_license!r}, expected one of {sorted(KNOWN_LICENSES)}"
        )
    if row.category not in KNOWN_CATEGORIES:
        raise ValueError(f"category: unknown value {row.category!r}, expected one of {sorted(KNOWN_CATEGORIES)}")
    if not row.text:
        raise ValueError("text: must be non-empty")
    for p in row.phonemes:
        if p in _IGNORED_PHONEMES:
            raise ValueError(f"phonemes: contains ignored token {p!r}")
    for tag in row.mora_accents:
        if tag not in _ACCENT_TAGS:
            raise ValueError(f"mora_accents: unexpected tag {tag!r}, expected one of {sorted(_ACCENT_TAGS)}")
    mora_len = len(row.mora_accents)
    prev = -1
    for idx in row.accent_boundaries:
        if idx < 0 or idx >= mora_len:
            raise ValueError(
                f"accent_boundaries: index {idx} outside [0, {mora_len})"
            )
        if idx <= prev:
            raise ValueError(
                f"accent_boundaries: non-monotonic sequence at index {idx} (previous {prev})"
            )
        prev = idx
    if row.sample_weight <= 0:
        raise ValueError(f"sample_weight: must be > 0, got {row.sample_weight}")


def to_dict(row: Row) -> dict[str, object]:
    return {
        "id": row.id,
        "source": row.source,
        "source_license": row.source_license,
        "text": row.text,
        "phonemes": list(row.phonemes),
        "mora_accents": list(row.mora_accents),
        "accent_boundaries": list(row.accent_boundaries),
        "category": row.category,
        "sample_weight": row.sample_weight,
        "extra": dict(row.extra),
    }


_ROW_FIELD_NAMES: frozenset[str] = frozenset(f.name for f in fields(Row))


def from_dict(d: Mapping[str, object]) -> Row:
    unknown = set(d.keys()) - _ROW_FIELD_NAMES
    if unknown:
        raise ValueError(f"from_dict: unknown keys {sorted(unknown)}")
    missing = _ROW_FIELD_NAMES - set(d.keys()) - {"category", "sample_weight", "extra"}
    if missing:
        raise ValueError(f"from_dict: missing required keys {sorted(missing)}")
    return Row(
        id=str(d["id"]),
        source=str(d["source"]),
        source_license=str(d["source_license"]),
        text=str(d["text"]),
        phonemes=tuple(d["phonemes"]),  # type: ignore[arg-type]
        mora_accents=tuple(d["mora_accents"]),  # type: ignore[arg-type]
        accent_boundaries=tuple(d["accent_boundaries"]),  # type: ignore[arg-type]
        category=str(d.get("category", "general")),
        sample_weight=float(d.get("sample_weight", 1.0)),  # type: ignore[arg-type]
        extra=dict(d.get("extra", {})),  # type: ignore[arg-type]
    )


def replace_row(row: Row, **changes: object) -> Row:
    return replace(row, **changes)


__all__ = [
    "KNOWN_CATEGORIES",
    "KNOWN_LICENSES",
    "KNOWN_SOURCES",
    "Row",
    "SCHEMA_VERSION",
    "from_dict",
    "make_id",
    "replace_row",
    "to_dict",
    "validate_row",
]
