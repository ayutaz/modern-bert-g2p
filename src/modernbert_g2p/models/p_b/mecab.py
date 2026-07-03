"""MeCab pretokenizer wrapper for pilot P-B.

Wraps ``fugashi`` (UniDic-3.1.1 + pyopenjtalk-plus 追加語彙) and returns morpheme
boundaries as ``MeCabToken`` records. All fugashi/unidic imports are lazy so
importing this module never requires the C++ MeCab runtime.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MeCabToken:
    """One morpheme returned by :class:`MeCabPretokenizer`."""

    surface: str
    start: int
    end: int
    pos: str


class MeCabPretokenizer:
    """MeCab pretokenizer with a lazily-loaded fugashi backend."""

    def __init__(self, dict_path: str | None = None, user_dict: str | None = None) -> None:
        self._dict_path = dict_path
        self._user_dict = user_dict
        self._tagger: object | None = None

    def _ensure_tagger(self) -> object:
        if self._tagger is None:
            try:
                import fugashi
            except ImportError as exc:
                raise RuntimeError(
                    "fugashi is required for MeCabPretokenizer. Install with "
                    "`pip install fugashi[unidic]`."
                ) from exc
            args: list[str] = []
            if self._dict_path is not None:
                args.extend(["-d", self._dict_path])
            if self._user_dict is not None:
                args.extend(["-u", self._user_dict])
            self._tagger = fugashi.Tagger(" ".join(args)) if args else fugashi.Tagger()
        return self._tagger

    def pretokenize(self, text: str) -> list[MeCabToken]:
        """Segment ``text`` into morphemes with byte-safe (char-index) spans."""
        if not text:
            return []
        tagger = self._ensure_tagger()
        tokens: list[MeCabToken] = []
        cursor = 0
        for word in tagger(text):  # type: ignore[operator]
            surface = word.surface
            if not surface:
                continue
            found = text.find(surface, cursor)
            if found < 0:
                found = cursor
            start = found
            end = start + len(surface)
            pos = getattr(getattr(word, "feature", None), "pos1", None) or ""
            tokens.append(MeCabToken(surface=surface, start=start, end=end, pos=pos))
            cursor = end
        return tokens


def build_mock_pretokenizer(mapping: dict[str, list[MeCabToken]]) -> MeCabPretokenizer:
    """Return a MeCabPretokenizer that dispatches from a fixed mapping (tests only)."""
    pre = MeCabPretokenizer()

    def _mock_pretokenize(text: str) -> list[MeCabToken]:
        return list(mapping.get(text, []))

    pre.pretokenize = _mock_pretokenize  # type: ignore[method-assign]
    return pre
