"""S4 ingest: per-ruby annotation extraction from Wikipedia HTML dump JSONL.

Implements docs/design/phase1_data_pipeline.md §3 (S4 Wikipedia source).

MVP scope: one Row per ``<ruby>`` annotation, using the ``<rb>`` text as the
surface and the ``<rt>`` text as the katakana/hiragana reading which is
converted to JULIUS phonemes. Sentence-level grouping with mora accent
labels via forced alignment is deferred to Phase 3, so ``mora_accents`` is
emitted as an empty tuple — Wikipedia ruby carries no accent information
and fabricating all-L labels would poison the BAS/APBP training head.

Parser: ``html.parser.HTMLParser`` from the standard library. lxml is not a
dependency for CI-lightness; the parser gracefully handles the two ruby
shapes we see in the wild (``<ruby>rb<rt>rt</rt></ruby>`` and
``<ruby><rb>rb</rb><rt>rt</rt></ruby>``) and skips ``<rp>`` parens.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from html.parser import HTMLParser
from pathlib import Path
from typing import ClassVar

from modernbert_g2p.data.ingest.base import (
    kana_to_julius_phonemes,
    register,
)
from modernbert_g2p.data.schema import Row, make_id
from modernbert_g2p.data.weighting import weight_for


class _RubyParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.pairs: list[tuple[str, str]] = []
        self._in_ruby: bool = False
        self._in_rt: bool = False
        self._in_rp: bool = False
        self._rb_parts: list[str] = []
        self._rt_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "ruby":
            self._in_ruby = True
            self._in_rt = False
            self._in_rp = False
            self._rb_parts = []
            self._rt_parts = []
        elif self._in_ruby and tag == "rt":
            self._in_rt = True
        elif self._in_ruby and tag == "rp":
            self._in_rp = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "ruby" and self._in_ruby:
            rb = "".join(self._rb_parts).strip()
            rt = "".join(self._rt_parts).strip()
            if rb and rt:
                self.pairs.append((rb, rt))
            self._in_ruby = False
            self._in_rt = False
            self._in_rp = False
        elif self._in_ruby and tag == "rt":
            self._in_rt = False
        elif self._in_ruby and tag == "rp":
            self._in_rp = False

    def handle_data(self, data: str) -> None:
        if not self._in_ruby or self._in_rp:
            return
        if self._in_rt:
            self._rt_parts.append(data)
        else:
            self._rb_parts.append(data)


def extract_ruby_pairs(html: str) -> list[tuple[str, str]]:
    """Return ``(surface, reading)`` pairs from HTML, ignoring ``<rp>`` parens."""
    parser = _RubyParser()
    parser.feed(html)
    parser.close()
    return parser.pairs


def _iter_jsonl_paths(root: Path) -> Iterator[Path]:
    if root.is_file():
        yield root
        return
    yield from sorted(root.rglob("*.json*"))


def _iter_articles(root: Path) -> Iterator[tuple[str, str]]:
    for path in _iter_jsonl_paths(root):
        with path.open(encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                body = doc.get("article_body") or {}
                html = body.get("html") if isinstance(body, dict) else None
                if not html:
                    continue
                title = str(doc.get("name") or doc.get("title") or "")
                yield title, html


def _make_row(rb: str, rt: str, title: str) -> Row | None:
    try:
        phonemes = kana_to_julius_phonemes(rt)
    except ValueError:
        return None
    if not phonemes:
        return None
    category = "general"
    row_id = make_id("wikipedia", rb, phonemes)
    return Row(
        id=row_id,
        source="wikipedia",
        source_license="CC-BY-SA-4.0",
        text=rb,
        phonemes=phonemes,
        mora_accents=(),
        accent_boundaries=(),
        category=category,
        sample_weight=weight_for(category),
        extra={"article_title": title, "rt_text": rt},
    )


@register("wikipedia")
class WikipediaSource:
    """S4 Wikipedia enterprise HTML dump ingest — per-ruby row emission."""

    source: ClassVar[str] = "wikipedia"
    source_license: ClassVar[str] = "CC-BY-SA-4.0"

    def entries(self, root: Path, *, limit: int | None = None) -> Iterator[Row]:
        yielded = 0
        for title, html in _iter_articles(root):
            for rb, rt in extract_ruby_pairs(html):
                if limit is not None and yielded >= limit:
                    return
                row = _make_row(rb, rt, title)
                if row is None:
                    continue
                yield row
                yielded += 1


__all__ = ["WikipediaSource", "extract_ruby_pairs"]
