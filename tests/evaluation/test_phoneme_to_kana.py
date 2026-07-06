"""Unit tests for phoneme_to_kana converter."""

from __future__ import annotations

import pytest

from modernbert_g2p.evaluation.phoneme_to_kana import phonemes_to_kana


@pytest.mark.parametrize(
    "phonemes, expected",
    [
        # Simple hiragana / katakana pairs
        (("k", "o", "n", "n", "i", "ch", "i", "w", "a"), "コンニチワ"),
        (("s", "a", "k", "u", "r", "a"), "サクラ"),
        (("h", "a", "n", "a"), "ハナ"),
        # Long vowels via repeated vowels
        (("k", "a", "a"), "カー"),
        (("t", "o", "o", "k", "y", "o", "o"), "トーキョー"),
        (("o", "b", "a", "a", "s", "a", "N"), "オバーサン"),
        # Sokuon (q)
        (("k", "i", "q", "t", "e"), "キッテ"),
        (("m", "a", "q", "ch", "a"), "マッチャ"),
        # Palatalized (yō'on)
        (("ky", "o", "o"), "キョー"),
        (("j", "a", "n"), "ジャン"),  # trailing bare n treated as moraic ン
        (("ry", "u", "u"), "リュー"),
        # Bare vowels
        (("a", "i", "u", "e", "o"), "アイウエオ"),
        # sh, ch, ts, f, v
        (("sh", "i", "t", "a"), "シタ"),
        (("ch", "i", "z", "u"), "チズ"),
        (("ts", "u", "k", "u", "e"), "ツクエ"),
        (("f", "a", "N"), "ファン"),
        (("v", "a", "i", "o", "r", "i", "N"), "ヴァイオリン"),
        # pau / unknown tokens dropped
        (("k", "a", "pau", "s", "a"), "カサ"),
        (("k", "a", "<unk>", "s", "a"), "カサ"),
        # Empty / all silence
        ((), ""),
        (("pau",), ""),
    ],
)
def test_phonemes_to_kana(phonemes: tuple[str, ...], expected: str) -> None:
    assert phonemes_to_kana(phonemes) == expected


def test_trailing_consonant_dropped() -> None:
    """A trailing consonant with no vowel is silently dropped."""
    assert phonemes_to_kana(("k", "a", "s")) == "カ"


def test_repeated_vowel_after_different_vowel_emits_new_kana() -> None:
    """Different vowel → new kana, not ー."""
    assert phonemes_to_kana(("a", "i")) == "アイ"
    assert phonemes_to_kana(("i", "a")) == "イア"


def test_moraic_n_after_consonant_flushes_pending() -> None:
    """A dangling consonant preceding N: consonant dropped, N emitted alone."""
    assert phonemes_to_kana(("k", "N")) == "ン"


def test_bare_n_before_consonant_is_moraic() -> None:
    """Bare "n" followed by another consonant is treated as moraic ン (JULIUS often emits ``n n`` for ``ん + n_...``)."""
    assert phonemes_to_kana(("k", "o", "n", "n", "i", "ch", "i")) == "コンニチ"


def test_c_plus_y_digraph_composition() -> None:
    """Separate ``k`` + ``y`` tokens should compose to a kyō'on digraph."""
    assert phonemes_to_kana(("k", "y", "o")) == "キョ"
    assert phonemes_to_kana(("r", "y", "u")) == "リュ"
