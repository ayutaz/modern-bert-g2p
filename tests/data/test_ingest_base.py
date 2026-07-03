from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import ClassVar

import pytest

from modernbert_g2p.data.ingest import base as base_mod
from modernbert_g2p.data.ingest.base import (
    SOURCE_REGISTRY,
    get_source,
    is_katakana,
    kana_to_julius_phonemes,
    kana_to_mora_count,
    known_sources,
    normalize_katakana,
    register,
)


@pytest.fixture
def isolated_registry(monkeypatch: pytest.MonkeyPatch) -> dict[str, type]:
    fresh: dict[str, type] = {}
    monkeypatch.setattr(base_mod, "SOURCE_REGISTRY", fresh)
    return fresh


def test_register_decorator_adds_to_registry(isolated_registry: dict[str, type]) -> None:
    @register("fake")
    class Dummy:
        source: ClassVar[str] = "fake"
        source_license: ClassVar[str] = "MIT"

        def entries(self, root: Path, *, limit: int | None = None) -> Iterator:
            if False:
                yield None

    assert isolated_registry["fake"] is Dummy
    assert get_source("fake") is Dummy


def test_register_duplicate_raises(isolated_registry: dict[str, type]) -> None:
    @register("dupe")
    class A:
        source: ClassVar[str] = "dupe"
        source_license: ClassVar[str] = "MIT"

        def entries(self, root: Path, *, limit: int | None = None) -> Iterator:
            if False:
                yield None

    with pytest.raises(ValueError, match="already registered"):

        @register("dupe")
        class B:
            source: ClassVar[str] = "dupe"
            source_license: ClassVar[str] = "MIT"

            def entries(self, root: Path, *, limit: int | None = None) -> Iterator:
                if False:
                    yield None


def test_get_source_unknown_raises_with_helpful_message() -> None:
    with pytest.raises(KeyError) as excinfo:
        get_source("does_not_exist")
    msg = str(excinfo.value)
    assert "does_not_exist" in msg
    assert "Known sources" in msg


def test_known_sources_is_sorted() -> None:
    result = known_sources()
    assert result == sorted(result)


def test_source_registry_populated_after_import() -> None:
    for name in ("pyopenjtalk_plus", "unidic", "jmdict", "wikipedia", "aozora"):
        if name in SOURCE_REGISTRY:
            cls = get_source(name)
            assert getattr(cls, "source", None) == name


def test_normalize_katakana_hiragana_to_katakana() -> None:
    assert normalize_katakana("こんにちは") == "コンニチハ"
    assert normalize_katakana("キャベツ") == "キャベツ"


def test_normalize_katakana_strips_whitespace_and_nfkc() -> None:
    assert normalize_katakana("  カナ  ") == "カナ"
    assert normalize_katakana("ｶﾅ") == "カナ"


def test_is_katakana() -> None:
    assert is_katakana("ア")
    assert is_katakana("ー")
    assert not is_katakana("あ")
    assert not is_katakana("a")
    assert not is_katakana("")


def test_kana_to_julius_konnichiwa() -> None:
    assert kana_to_julius_phonemes("コンニチハ") == ("k", "o", "N", "n", "i", "ch", "i", "h", "a")


def test_kana_to_julius_kyabetsu() -> None:
    assert kana_to_julius_phonemes("キャベツ") == ("ky", "a", "b", "e", "ts", "u")


def test_kana_to_julius_long_vowels_toukyou() -> None:
    assert kana_to_julius_phonemes("トウキョウ") == ("t", "o", "u", "ky", "o", "u")


def test_kana_to_julius_choonpu_expands_to_prev_vowel() -> None:
    assert kana_to_julius_phonemes("コーヒー") == ("k", "o", "o", "h", "i", "i")


def test_kana_to_julius_sokuon_sukkiri() -> None:
    assert kana_to_julius_phonemes("スッキリ") == ("s", "u", "q", "k", "i", "r", "i")


def test_kana_to_julius_accepts_hiragana_input() -> None:
    assert kana_to_julius_phonemes("こんにちは") == ("k", "o", "N", "n", "i", "ch", "i", "h", "a")


def test_kana_to_julius_empty_string() -> None:
    assert kana_to_julius_phonemes("") == ()


def test_kana_to_julius_unknown_raises_when_no_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(base_mod, "_try_pyopenjtalk_fallback", lambda _kana: None)
    with pytest.raises(ValueError, match="Cannot phonemize kana"):
        kana_to_julius_phonemes("X")


def test_kana_to_mora_count_konnichiwa() -> None:
    assert kana_to_mora_count("コンニチハ") == 5


def test_kana_to_mora_count_kyabetsu() -> None:
    assert kana_to_mora_count("キャベツ") == 3


def test_kana_to_mora_count_sokuon() -> None:
    assert kana_to_mora_count("スッキリ") == 4


def test_kana_to_mora_count_choonpu_counts() -> None:
    assert kana_to_mora_count("コーヒー") == 4


def test_kana_to_mora_count_kyakkya() -> None:
    assert kana_to_mora_count("キャッキャ") == 3


def test_kana_to_mora_count_nippon() -> None:
    assert kana_to_mora_count("ニッポン") == 4


def test_kana_to_mora_count_empty() -> None:
    assert kana_to_mora_count("") == 0
