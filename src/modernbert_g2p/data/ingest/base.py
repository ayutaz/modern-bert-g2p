"""IngestSource Protocol and shared kana/mora helpers.

Implements docs/design/phase1_data_pipeline.md §3.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Protocol

if TYPE_CHECKING:
    from modernbert_g2p.data.schema import Row


class IngestSource(Protocol):
    source: ClassVar[str]
    source_license: ClassVar[str]

    def entries(self, root: Path, *, limit: int | None = None) -> Iterator[Row]: ...


SOURCE_REGISTRY: dict[str, type[IngestSource]] = {}


def register(name: str) -> Callable[[type[IngestSource]], type[IngestSource]]:
    """Class decorator registering an IngestSource under `name`."""

    def deco(cls: type[IngestSource]) -> type[IngestSource]:
        if name in SOURCE_REGISTRY:
            raise ValueError(
                f"Source '{name}' already registered to {SOURCE_REGISTRY[name].__name__}"
            )
        SOURCE_REGISTRY[name] = cls
        return cls

    return deco


def get_source(name: str) -> type[IngestSource]:
    if name not in SOURCE_REGISTRY:
        raise KeyError(
            f"Unknown source '{name}'. Known sources: {sorted(SOURCE_REGISTRY)}"
        )
    return SOURCE_REGISTRY[name]


def known_sources() -> list[str]:
    return sorted(SOURCE_REGISTRY)


_HIRAGANA_START = 0x3041
_HIRAGANA_END = 0x3096
_KATAKANA_OFFSET = 0x60

_SMALL_VOWELS: frozenset[str] = frozenset("ァィゥェォヮ")
_SMALL_YA_YU_YO: frozenset[str] = frozenset("ャュョ")
_SUTEGANA: frozenset[str] = _SMALL_VOWELS | _SMALL_YA_YU_YO

_VOWEL_OF: dict[str, str] = {
    "ア": "a", "イ": "i", "ウ": "u", "エ": "e", "オ": "o",
    "ァ": "a", "ィ": "i", "ゥ": "u", "ェ": "e", "ォ": "o",
    "ヴ": "u",
}


_KANA_TO_JULIUS: dict[str, tuple[str, ...]] = {
    "ア": ("a",), "イ": ("i",), "ウ": ("u",), "エ": ("e",), "オ": ("o",),
    "ァ": ("a",), "ィ": ("i",), "ゥ": ("u",), "ェ": ("e",), "ォ": ("o",),
    "カ": ("k", "a"), "キ": ("k", "i"), "ク": ("k", "u"), "ケ": ("k", "e"), "コ": ("k", "o"),
    "ガ": ("g", "a"), "ギ": ("g", "i"), "グ": ("g", "u"), "ゲ": ("g", "e"), "ゴ": ("g", "o"),
    "サ": ("s", "a"), "シ": ("sh", "i"), "ス": ("s", "u"), "セ": ("s", "e"), "ソ": ("s", "o"),
    "ザ": ("z", "a"), "ジ": ("j", "i"), "ズ": ("z", "u"), "ゼ": ("z", "e"), "ゾ": ("z", "o"),
    "タ": ("t", "a"), "チ": ("ch", "i"), "ツ": ("ts", "u"), "テ": ("t", "e"), "ト": ("t", "o"),
    "ダ": ("d", "a"), "ヂ": ("j", "i"), "ヅ": ("z", "u"), "デ": ("d", "e"), "ド": ("d", "o"),
    "ナ": ("n", "a"), "ニ": ("n", "i"), "ヌ": ("n", "u"), "ネ": ("n", "e"), "ノ": ("n", "o"),
    "ハ": ("h", "a"), "ヒ": ("h", "i"), "フ": ("f", "u"), "ヘ": ("h", "e"), "ホ": ("h", "o"),
    "バ": ("b", "a"), "ビ": ("b", "i"), "ブ": ("b", "u"), "ベ": ("b", "e"), "ボ": ("b", "o"),
    "パ": ("p", "a"), "ピ": ("p", "i"), "プ": ("p", "u"), "ペ": ("p", "e"), "ポ": ("p", "o"),
    "マ": ("m", "a"), "ミ": ("m", "i"), "ム": ("m", "u"), "メ": ("m", "e"), "モ": ("m", "o"),
    "ヤ": ("y", "a"), "ユ": ("y", "u"), "ヨ": ("y", "o"),
    "ラ": ("r", "a"), "リ": ("r", "i"), "ル": ("r", "u"), "レ": ("r", "e"), "ロ": ("r", "o"),
    "ワ": ("w", "a"), "ヰ": ("i",), "ヱ": ("e",), "ヲ": ("o",),
    "ン": ("N",),
    "ヴ": ("v", "u"),
}


_KANA_DIGRAPHS: dict[str, tuple[str, ...]] = {
    "キャ": ("ky", "a"), "キュ": ("ky", "u"), "キョ": ("ky", "o"), "キェ": ("ky", "e"),
    "ギャ": ("gy", "a"), "ギュ": ("gy", "u"), "ギョ": ("gy", "o"), "ギェ": ("gy", "e"),
    "シャ": ("sh", "a"), "シュ": ("sh", "u"), "ショ": ("sh", "o"), "シェ": ("sh", "e"),
    "ジャ": ("j", "a"), "ジュ": ("j", "u"), "ジョ": ("j", "o"), "ジェ": ("j", "e"),
    "チャ": ("ch", "a"), "チュ": ("ch", "u"), "チョ": ("ch", "o"), "チェ": ("ch", "e"),
    "ヂャ": ("j", "a"), "ヂュ": ("j", "u"), "ヂョ": ("j", "o"),
    "ニャ": ("ny", "a"), "ニュ": ("ny", "u"), "ニョ": ("ny", "o"),
    "ヒャ": ("hy", "a"), "ヒュ": ("hy", "u"), "ヒョ": ("hy", "o"),
    "ビャ": ("by", "a"), "ビュ": ("by", "u"), "ビョ": ("by", "o"),
    "ピャ": ("py", "a"), "ピュ": ("py", "u"), "ピョ": ("py", "o"),
    "ミャ": ("my", "a"), "ミュ": ("my", "u"), "ミョ": ("my", "o"),
    "リャ": ("ry", "a"), "リュ": ("ry", "u"), "リョ": ("ry", "o"),
    "ファ": ("f", "a"), "フィ": ("f", "i"), "フェ": ("f", "e"), "フォ": ("f", "o"),
    "フュ": ("fy", "u"),
    "ウィ": ("w", "i"), "ウェ": ("w", "e"), "ウォ": ("w", "o"),
    "ティ": ("t", "i"), "トゥ": ("t", "u"), "テュ": ("ty", "u"),
    "ディ": ("d", "i"), "ドゥ": ("d", "u"), "デュ": ("dy", "u"),
    "ヴァ": ("v", "a"), "ヴィ": ("v", "i"), "ヴェ": ("v", "e"), "ヴォ": ("v", "o"),
    "ヴュ": ("vy", "u"),
    "ツァ": ("ts", "a"), "ツィ": ("ts", "i"), "ツェ": ("ts", "e"), "ツォ": ("ts", "o"),
    "イェ": ("y", "e"),
    "クァ": ("k", "w", "a"), "クィ": ("k", "w", "i"), "クェ": ("k", "w", "e"),
    "クォ": ("k", "w", "o"),
    "グァ": ("g", "w", "a"),
}


def is_katakana(char: str) -> bool:
    if not char:
        return False
    code = ord(char[0])
    return 0x30A0 <= code <= 0x30FF or code == 0x30FC


def _hiragana_to_katakana(text: str) -> str:
    out: list[str] = []
    for ch in text:
        code = ord(ch)
        if _HIRAGANA_START <= code <= _HIRAGANA_END:
            out.append(chr(code + _KATAKANA_OFFSET))
        else:
            out.append(ch)
    return "".join(out)


def normalize_katakana(kana: str) -> str:
    return _hiragana_to_katakana(unicodedata.normalize("NFKC", kana)).strip()


def _try_pyopenjtalk_fallback(kana: str) -> tuple[str, ...] | None:
    try:
        import pyopenjtalk
    except ImportError:
        return None
    try:
        phonemes = pyopenjtalk.g2p(kana, kana=False).split()
    except Exception:
        return None
    return tuple(phonemes) if phonemes else None


def kana_to_julius_phonemes(kana: str) -> tuple[str, ...]:
    norm = normalize_katakana(kana)
    if not norm:
        return ()

    out: list[str] = []
    last_vowel: str | None = None
    i = 0
    n = len(norm)
    fallback_needed = False

    while i < n:
        ch = norm[i]

        if ch == "ー":
            if last_vowel is None:
                fallback_needed = True
                break
            out.append(last_vowel)
            i += 1
            continue

        if ch == "ッ":
            out.append("q")
            i += 1
            continue

        if i + 1 < n:
            digraph = norm[i : i + 2]
            if digraph in _KANA_DIGRAPHS:
                phones = _KANA_DIGRAPHS[digraph]
                out.extend(phones)
                last_vowel = phones[-1]
                i += 2
                continue

        if ch in _KANA_TO_JULIUS:
            phones = _KANA_TO_JULIUS[ch]
            out.extend(phones)
            tail = phones[-1]
            if tail in {"a", "i", "u", "e", "o"}:
                last_vowel = tail
            i += 1
            continue

        fallback_needed = True
        break

    if fallback_needed:
        fallback = _try_pyopenjtalk_fallback(norm)
        if fallback is None:
            raise ValueError(f"Cannot phonemize kana: {kana!r}")
        return fallback

    return tuple(out)


def kana_to_mora_count(kana: str) -> int:
    norm = normalize_katakana(kana)
    if not norm:
        return 0

    count = 0
    i = 0
    n = len(norm)
    while i < n:
        ch = norm[i]
        if ch in _SUTEGANA:
            i += 1
            continue
        if i + 1 < n and norm[i + 1] in _SUTEGANA:
            count += 1
            i += 2
            continue
        count += 1
        i += 1
    return count


__all__ = [
    "IngestSource",
    "SOURCE_REGISTRY",
    "get_source",
    "is_katakana",
    "kana_to_julius_phonemes",
    "kana_to_mora_count",
    "known_sources",
    "normalize_katakana",
    "register",
]
