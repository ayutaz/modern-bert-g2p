"""Unit tests for modernbert_g2p.data.normalize."""

from __future__ import annotations

import pytest

from modernbert_g2p.data.normalize import (
    DEVOICING_MAP,
    IGNORED_PHONEMES,
    NormalizedText,
    accent_nucleus_to_hl,
    devoice_phonemes,
    get_haqumei,
    normalize_from_haqumei,
)


def test_ignored_phonemes_is_pau_only() -> None:
    assert IGNORED_PHONEMES == frozenset({"pau"})


def test_devoice_phonemes_lowers_uppercase_vowels() -> None:
    result = devoice_phonemes(("k", "A", "s", "I", "t", "O", "E", "U"))
    assert result == ("k", "a", "s", "i", "t", "o", "e", "u")


def test_devoice_phonemes_preserves_moraic_nasal_and_others() -> None:
    result = devoice_phonemes(("h", "o", "N", "ky", "o", "u", "q", "pau"))
    assert result == ("h", "o", "N", "ky", "o", "u", "q", "pau")


def test_devoice_phonemes_only_maps_declared_uppercase_vowels() -> None:
    # Only A/E/I/O/U are downcased; other uppercase tokens are preserved.
    assert set(DEVOICING_MAP.keys()) == {"A", "E", "I", "O", "U"}
    assert devoice_phonemes(("X", "Y", "Z")) == ("X", "Y", "Z")


def test_accent_nucleus_heiban_multimora() -> None:
    assert accent_nucleus_to_hl(0, 4) == ("L", "H", "H", "H")


def test_accent_nucleus_heiban_single_mora() -> None:
    assert accent_nucleus_to_hl(0, 1) == ("L",)


def test_accent_nucleus_atamadaka() -> None:
    assert accent_nucleus_to_hl(1, 3) == ("H", "L", "L")


def test_accent_nucleus_nakadaka() -> None:
    assert accent_nucleus_to_hl(3, 5) == ("L", "H", "H", "L", "L")


def test_accent_nucleus_odaka() -> None:
    # nucleus == mora_count → last mora carries the drop.
    assert accent_nucleus_to_hl(4, 4) == ("L", "H", "H", "H")


def test_accent_nucleus_zero_moras() -> None:
    assert accent_nucleus_to_hl(0, 0) == ()


def test_normalize_from_haqumei_empty_text_short_circuits() -> None:
    # Empty input must not trigger haqumei construction.
    result = normalize_from_haqumei("")
    assert result == NormalizedText(phonemes=(), mora_accents=(), accent_boundaries=())


def test_normalized_text_is_frozen() -> None:
    nt = NormalizedText(phonemes=("k",), mora_accents=("L",), accent_boundaries=())
    import dataclasses

    with pytest.raises(dataclasses.FrozenInstanceError):
        nt.phonemes = ("q",)  # type: ignore[misc]


def test_get_haqumei_returns_singleton() -> None:
    pytest.importorskip("haqumei")
    a = get_haqumei()
    b = get_haqumei()
    assert a is b


@pytest.mark.parametrize(
    "text",
    [
        "こんにちは",
        "今日は良い天気ですね",
        "彼は日本語の教師です",
    ],
)
def test_normalize_from_haqumei_phoneme_stream_matches_eval_pipeline(text: str) -> None:
    pytest.importorskip("haqumei")
    from haqumei import Haqumei, IuPronunciation

    from modernbert_g2p.data.normalize import devoice_phonemes as devoice
    from modernbert_g2p.metrics.per import DEFAULT_IGNORE

    reference_hq = Haqumei(use_unidic_yomi=True, normalize_iu=IuPronunciation.Yuu)
    raw = reference_hq.g2p_batch([text])[0]
    expected = tuple(p for p in devoice(raw) if p not in DEFAULT_IGNORE)

    result = normalize_from_haqumei(text)
    assert result.phonemes == expected


def test_normalize_from_haqumei_mora_accents_are_hl_labels() -> None:
    pytest.importorskip("haqumei")
    result = normalize_from_haqumei("こんにちは")
    assert len(result.mora_accents) > 0
    assert all(label in {"H", "L"} for label in result.mora_accents)


def test_normalize_from_haqumei_boundaries_within_mora_range() -> None:
    pytest.importorskip("haqumei")
    result = normalize_from_haqumei("今日は良い天気ですね")
    assert all(0 <= b < len(result.mora_accents) for b in result.accent_boundaries)
    # Boundaries must be monotonically strictly increasing.
    assert list(result.accent_boundaries) == sorted(set(result.accent_boundaries))
