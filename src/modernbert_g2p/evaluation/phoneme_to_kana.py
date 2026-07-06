"""JULIUS phoneme sequence → katakana string converter.

Used to bridge P-A / P-C model output (phoneme tuples) with JVS kana CER and
ROHAN KER metrics that expect kana character strings. Not a general-purpose
IPA/JULIUS converter — targeted at the phoneme inventory used in this project.

Design principles:
- Deterministic single-pass state machine on the phoneme stream.
- Missing / unknown phonemes are silently dropped rather than raising, since
  the caller feeds noisy model output; alignment errors are absorbed by the
  edit-distance metric downstream.
- Long-vowel duplicates (`aa`, `oo`, ...) are emitted as `ー` (chōonpu).
- `pau` is treated as a silent marker (dropped).
"""

from __future__ import annotations

from collections.abc import Iterable

_CV_TABLE: dict[tuple[str, str], str] = {
    # Row: a i u e o
    ("k", "a"): "カ", ("k", "i"): "キ", ("k", "u"): "ク", ("k", "e"): "ケ", ("k", "o"): "コ",
    ("g", "a"): "ガ", ("g", "i"): "ギ", ("g", "u"): "グ", ("g", "e"): "ゲ", ("g", "o"): "ゴ",
    ("s", "a"): "サ", ("s", "u"): "ス", ("s", "e"): "セ", ("s", "o"): "ソ",
    ("z", "a"): "ザ", ("z", "u"): "ズ", ("z", "e"): "ゼ", ("z", "o"): "ゾ",
    ("t", "a"): "タ", ("t", "e"): "テ", ("t", "o"): "ト",
    ("d", "a"): "ダ", ("d", "e"): "デ", ("d", "o"): "ド",
    ("n", "a"): "ナ", ("n", "i"): "ニ", ("n", "u"): "ヌ", ("n", "e"): "ネ", ("n", "o"): "ノ",
    ("h", "a"): "ハ", ("h", "i"): "ヒ", ("h", "e"): "ヘ", ("h", "o"): "ホ",
    ("b", "a"): "バ", ("b", "i"): "ビ", ("b", "u"): "ブ", ("b", "e"): "ベ", ("b", "o"): "ボ",
    ("p", "a"): "パ", ("p", "i"): "ピ", ("p", "u"): "プ", ("p", "e"): "ペ", ("p", "o"): "ポ",
    ("m", "a"): "マ", ("m", "i"): "ミ", ("m", "u"): "ム", ("m", "e"): "メ", ("m", "o"): "モ",
    ("y", "a"): "ヤ", ("y", "u"): "ユ", ("y", "o"): "ヨ", ("y", "e"): "イェ",
    ("r", "a"): "ラ", ("r", "i"): "リ", ("r", "u"): "ル", ("r", "e"): "レ", ("r", "o"): "ロ",
    ("w", "a"): "ワ", ("w", "i"): "ウィ", ("w", "e"): "ウェ", ("w", "o"): "ヲ",
    # Digraphs (sh/ch/ts/j/f + vowel) — mostly single-mora
    ("sh", "a"): "シャ", ("sh", "i"): "シ", ("sh", "u"): "シュ", ("sh", "e"): "シェ", ("sh", "o"): "ショ",
    ("ch", "a"): "チャ", ("ch", "i"): "チ", ("ch", "u"): "チュ", ("ch", "e"): "チェ", ("ch", "o"): "チョ",
    ("ts", "a"): "ツァ", ("ts", "i"): "ツィ", ("ts", "u"): "ツ", ("ts", "e"): "ツェ", ("ts", "o"): "ツォ",
    ("j", "a"): "ジャ", ("j", "i"): "ジ", ("j", "u"): "ジュ", ("j", "e"): "ジェ", ("j", "o"): "ジョ",
    ("f", "a"): "ファ", ("f", "i"): "フィ", ("f", "u"): "フ", ("f", "e"): "フェ", ("f", "o"): "フォ",
    ("v", "a"): "ヴァ", ("v", "i"): "ヴィ", ("v", "u"): "ヴ", ("v", "e"): "ヴェ", ("v", "o"): "ヴォ",
    # Palatalized (ky/gy/ny/hy/by/py/my/ry) + vowel
    ("ky", "a"): "キャ", ("ky", "u"): "キュ", ("ky", "o"): "キョ",
    ("gy", "a"): "ギャ", ("gy", "u"): "ギュ", ("gy", "o"): "ギョ",
    ("ny", "a"): "ニャ", ("ny", "u"): "ニュ", ("ny", "o"): "ニョ",
    ("hy", "a"): "ヒャ", ("hy", "u"): "ヒュ", ("hy", "o"): "ヒョ",
    ("by", "a"): "ビャ", ("by", "u"): "ビュ", ("by", "o"): "ビョ",
    ("py", "a"): "ピャ", ("py", "u"): "ピュ", ("py", "o"): "ピョ",
    ("my", "a"): "ミャ", ("my", "u"): "ミュ", ("my", "o"): "ミョ",
    ("ry", "a"): "リャ", ("ry", "u"): "リュ", ("ry", "o"): "リョ",
    # ti / di / tu / du (foreign / rare)
    ("t", "i"): "ティ", ("t", "u"): "トゥ",
    ("d", "i"): "ディ", ("d", "u"): "ドゥ",
    ("s", "i"): "スィ",
    ("z", "i"): "ズィ",
    # h/b/p + i via yōon (hy/by/py handled above; hi/bi/pi are canonical)
    ("h", "u"): "フ",  # some transcriptions use h+u; canonically fu
}

_VOWELS = frozenset({"a", "i", "u", "e", "o"})
_CONSONANTS_SINGLE = frozenset(
    {"k", "g", "s", "z", "t", "d", "n", "h", "b", "p", "m", "y", "r", "w", "f", "v", "j"}
)
_CONSONANTS_DIGRAPH = frozenset(
    {"sh", "ch", "ts", "ky", "gy", "sy", "zy", "ny", "hy", "by", "py", "my", "ry"}
)
_VOWEL_TO_LONG = {"a": "ー", "i": "ー", "u": "ー", "e": "ー", "o": "ー"}


def phonemes_to_kana(phonemes: Iterable[str]) -> str:
    """Convert a JULIUS phoneme sequence into a katakana string.

    Rules:
    - CV or CV-yō pairs → single kana (dictionary lookup).
    - Bare vowel (no preceding consonant) → single kana (ア/イ/ウ/エ/オ).
    - Repeat vowel (aa/ii/uu/ee/oo) → ー (chōonpu, long-vowel mark).
    - ``N`` → ン (moraic n).
    - ``q`` → ッ (sokuon, geminate).
    - ``pau`` and unknown tokens are dropped silently.
    - Trailing consonant with no following vowel is dropped.
    """
    out: list[str] = []
    pending_c: str | None = None
    last_vowel: str | None = None
    for p in phonemes:
        if not p:
            continue
        if p == "pau":
            pending_c = None
            last_vowel = None
            continue
        if p == "N":
            if pending_c is not None:
                pending_c = None
            out.append("ン")
            last_vowel = None
            continue
        if p == "q":
            if pending_c is not None:
                pending_c = None
            out.append("ッ")
            last_vowel = None
            continue
        if p in _VOWELS:
            if pending_c is not None:
                kana = _CV_TABLE.get((pending_c, p))
                if kana is not None:
                    out.append(kana)
                else:
                    # unknown CV — try consonant alone dropped, emit vowel
                    out.append({"a": "ア", "i": "イ", "u": "ウ", "e": "エ", "o": "オ"}[p])
                pending_c = None
                last_vowel = p
            elif p == last_vowel:
                out.append(_VOWEL_TO_LONG[p])
            else:
                out.append({"a": "ア", "i": "イ", "u": "ウ", "e": "エ", "o": "オ"}[p])
                last_vowel = p
            continue
        if p == "n" and pending_c is None:
            # Bare "n" not followed by anything yet: could be moraic (before another consonant)
            # We tentatively buffer it, but if the next token is a consonant we emit ン and reset.
            pending_c = "n"
            last_vowel = None
            continue
        if p == "y" and pending_c in {"k", "g", "s", "z", "n", "h", "b", "p", "m", "r"}:
            # Consonant + y digraph (JULIUS may emit as two tokens): combine.
            pending_c = pending_c + "y"
            last_vowel = None
            continue
        if p in _CONSONANTS_DIGRAPH or p in _CONSONANTS_SINGLE:
            # Flush any stale pending_c: if it was "n", treat as moraic N.
            if pending_c == "n":
                out.append("ン")
            pending_c = p
            last_vowel = None
            continue
        # unknown phoneme: silently drop
    if pending_c == "n":
        # Trailing "n" — treat as moraic N.
        out.append("ン")
    return "".join(out)


__all__ = ["phonemes_to_kana"]
