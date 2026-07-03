"""Text-hash + phoneme-hash deduplication.

Implements docs/design/phase1_data_pipeline.md §5.1.
"""

from __future__ import annotations

import hashlib
import unicodedata
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modernbert_g2p.data.schema import Row


KeyExtractor = str | Callable[["Row"], str]

_DIGEST_SIZE: int = 16


def _normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def text_hash(text: str) -> str:
    """Deterministic blake2b-128 hex of NFKC-normalized, whitespace-collapsed text."""
    return hashlib.blake2b(
        _normalize_text(text).encode("utf-8"), digest_size=_DIGEST_SIZE
    ).hexdigest()


def phoneme_hash(phonemes: Iterable[str]) -> str:
    """Deterministic blake2b-128 hex of a pipe-joined phoneme sequence (order-sensitive)."""
    return hashlib.blake2b(
        "|".join(phonemes).encode("utf-8"), digest_size=_DIGEST_SIZE
    ).hexdigest()


class Deduper:
    """Row deduplicator keyed by any combination of text-hash, phoneme-hash, or callable."""

    def __init__(self, keys: Sequence[KeyExtractor] = ("text", "phonemes")) -> None:
        if not keys:
            raise ValueError("Deduper requires at least one key")
        for k in keys:
            if not (callable(k) or k in {"text", "phonemes"}):
                raise ValueError(f"Unknown key spec: {k!r}")
        self.keys: tuple[KeyExtractor, ...] = tuple(keys)
        self._seen: list[set[str]] = [set() for _ in self.keys]
        self._unique: int = 0
        self._dup_text: int = 0
        self._dup_phoneme: int = 0
        self._dup_other: int = 0

    def _token(self, key: KeyExtractor, row: Row) -> str:
        if callable(key):
            return "fn:" + key(row)
        if key == "text":
            return "t:" + text_hash(row.text)
        if key == "phonemes":
            return "p:" + phoneme_hash(row.phonemes)
        raise ValueError(f"Unknown key: {key!r}")

    def _bump_dup(self, key: KeyExtractor) -> None:
        if key == "text":
            self._dup_text += 1
        elif key == "phonemes":
            self._dup_phoneme += 1
        else:
            self._dup_other += 1

    def is_dup(self, row: Row) -> bool:
        for key, seen in zip(self.keys, self._seen, strict=True):
            if self._token(key, row) in seen:
                return True
        return False

    def add(self, row: Row) -> bool:
        tokens = [self._token(k, row) for k in self.keys]
        for key, seen, token in zip(self.keys, self._seen, tokens, strict=True):
            if token in seen:
                self._bump_dup(key)
                return False
        for seen, token in zip(self._seen, tokens, strict=True):
            seen.add(token)
        self._unique += 1
        return True

    def add_all(self, rows: Iterable[Row]) -> Iterator[Row]:
        for row in rows:
            if self.add(row):
                yield row

    def seen_count(self) -> int:
        return self._unique

    def dup_count(self) -> int:
        return self._dup_text + self._dup_phoneme + self._dup_other

    @property
    def size(self) -> int:
        return self._unique

    def stats(self) -> dict[str, int]:
        return {
            "unique": self._unique,
            "dup_text": self._dup_text,
            "dup_phoneme": self._dup_phoneme,
            "dup_other": self._dup_other,
        }
