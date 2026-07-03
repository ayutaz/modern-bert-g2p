from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from modernbert_g2p.data.dedup import Deduper, phoneme_hash, text_hash

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


def _row(row_id: str, text: str, phonemes: tuple[str, ...]) -> object:
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


def test_text_hash_deterministic() -> None:
    h1 = text_hash("こんにちは、世界")
    h2 = text_hash("こんにちは、世界")
    assert h1 == h2
    assert len(h1) == 32
    assert all(c in "0123456789abcdef" for c in h1)


def test_text_hash_normalizes_whitespace_and_nfkc() -> None:
    assert text_hash("abc") == text_hash("abc\n")
    assert text_hash("abc") == text_hash("  abc  ")
    assert text_hash("abc") == text_hash("a b c".replace(" ", ""))
    assert text_hash("ABC") == text_hash("ＡＢＣ")


def test_phoneme_hash_order_sensitive() -> None:
    h1 = phoneme_hash(["k", "o", "N"])
    h2 = phoneme_hash(["k", "N", "o"])
    assert h1 != h2


def test_phoneme_hash_deterministic() -> None:
    seq = ("k", "o", "N", "n", "i", "ch", "i", "w", "a")
    assert phoneme_hash(seq) == phoneme_hash(list(seq))
    assert phoneme_hash(iter(seq)) == phoneme_hash(seq)


def test_deduper_add_new_returns_true() -> None:
    d = Deduper()
    row = _row("a:1", "abc", ("a", "b"))
    assert d.add(row) is True
    assert d.add(row) is False
    assert d.seen_count() == 1
    assert d.dup_count() == 1


def test_deduper_stats_split_between_text_and_phoneme() -> None:
    d = Deduper()
    d.add(_row("a:1", "hello world", ("h", "e", "l")))
    assert d.add(_row("a:2", "hello world", ("x", "y", "z"))) is False
    assert d.add(_row("a:3", "different text entirely here", ("h", "e", "l"))) is False
    stats = d.stats()
    assert stats == {"unique": 1, "dup_text": 1, "dup_phoneme": 1, "dup_other": 0}
    assert d.seen_count() == 1
    assert d.dup_count() == 2


def test_deduper_add_all_preserves_order_and_filters() -> None:
    rows = [
        _row("i:0", "sentence a", ("a",)),
        _row("i:1", "sentence b", ("b",)),
        _row("i:2", "sentence a", ("c",)),
        _row("i:3", "sentence c", ("a",)),
        _row("i:4", "sentence d", ("d",)),
    ]
    d = Deduper()
    kept = list(d.add_all(rows))
    assert [r.id for r in kept] == ["i:0", "i:1", "i:4"]
    assert d.seen_count() == 3
    assert d.dup_count() == 2


def test_deduper_keys_phonemes_only_allows_same_text_different_phonemes() -> None:
    d = Deduper(keys=("phonemes",))
    r1 = _row("p:1", "同じ文", ("a", "b"))
    r2 = _row("p:2", "同じ文", ("c", "d"))
    r3 = _row("p:3", "全く違う", ("a", "b"))
    assert d.add(r1) is True
    assert d.add(r2) is True
    assert d.add(r3) is False


def test_deduper_keys_text_only_drops_same_text_different_phonemes() -> None:
    d = Deduper(keys=("text",))
    r1 = _row("t:1", "同じ文", ("a", "b"))
    r2 = _row("t:2", "同じ文", ("c", "d"))
    assert d.add(r1) is True
    assert d.add(r2) is False
    assert d.stats()["dup_text"] == 1


def test_deduper_callable_key() -> None:
    d = Deduper(keys=(lambda r: r.text[:3],))
    r1 = _row("c:1", "abcdef", ("x",))
    r2 = _row("c:2", "abcxyz", ("y",))
    r3 = _row("c:3", "defghi", ("z",))
    assert d.add(r1) is True
    assert d.add(r2) is False
    assert d.add(r3) is True


def test_deduper_callable_key_bumps_dup_other() -> None:
    d = Deduper(keys=(lambda r: r.text[:3],))
    d.add(_row("c:1", "abcdef", ("x",)))
    assert d.add(_row("c:2", "abcxyz", ("y",))) is False
    stats = d.stats()
    assert stats == {"unique": 1, "dup_text": 0, "dup_phoneme": 0, "dup_other": 1}
    assert d.dup_count() == 1


def test_deduper_empty_keys_raises() -> None:
    with pytest.raises(ValueError, match="at least one key"):
        Deduper(keys=())


def test_deduper_unknown_key_raises() -> None:
    with pytest.raises(ValueError, match="Unknown key"):
        Deduper(keys=("foobar",))


def test_deduper_is_dup_does_not_mutate_state() -> None:
    d = Deduper()
    r1 = _row("m:1", "text one", ("a",))
    r2 = _row("m:2", "text two", ("b",))
    d.add(r1)
    assert d.is_dup(r1) is True
    assert d.is_dup(r2) is False
    assert d.seen_count() == 1
    assert d.add(r2) is True
    assert d.seen_count() == 2


def test_deduper_size_property_matches_seen_count() -> None:
    d = Deduper()
    for i in range(7):
        d.add(_row(f"s:{i}", f"unique text {i}", (str(i),)))
    assert d.size == 7
    assert d.size == d.seen_count()
