"""S5 ingest: Aozora XHTML parser with per-work PD vs Aozora-CC-BY branching.

Implements docs/design/phase1_data_pipeline.md §3 (S5 青空文庫).
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from typing import ClassVar, Literal

from modernbert_g2p.data.ingest.base import kana_to_julius_phonemes, register
from modernbert_g2p.data.schema import Row, make_id
from modernbert_g2p.data.weighting import weight_for

_HR_SPLIT = re.compile(r"<hr\s*/?>", re.IGNORECASE)
_RUBY_PATTERN = re.compile(r"<ruby[^>]*>(.*?)</ruby>", re.DOTALL | re.IGNORECASE)
_RP_PATTERN = re.compile(r"<rp[^>]*>.*?</rp>", re.DOTALL | re.IGNORECASE)
_RT_PATTERN = re.compile(r"<rt[^>]*>(.*?)</rt>", re.DOTALL | re.IGNORECASE)
_RB_PATTERN = re.compile(r"<rb[^>]*>(.*?)</rb>", re.DOTALL | re.IGNORECASE)
_TAG_PATTERN = re.compile(r"<[^>]+>")

_PD_MARKERS: tuple[str, ...] = (
    "パブリック・ドメイン",
    "パブリックドメイン",
    "著作権の保護期間は満了",
)
_CCBY_MARKERS: tuple[str, ...] = (
    "クリエイティブ・コモンズ",
    "クリエイティブコモンズ",
    "Creative Commons",
)


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("shift_jis", errors="replace")


def detect_license(html: str) -> Literal["PD", "PDplusAoz"]:
    parts = _HR_SPLIT.split(html, maxsplit=1)
    header = parts[0] if len(parts) > 1 else html
    if any(marker in header for marker in _PD_MARKERS):
        return "PD"
    if any(marker in header for marker in _CCBY_MARKERS):
        return "PDplusAoz"
    return "PDplusAoz"


def extract_ruby_pairs_aozora(html: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for m in _RUBY_PATTERN.finditer(html):
        inner = m.group(1)
        rt_m = _RT_PATTERN.search(inner)
        if not rt_m:
            continue
        reading = rt_m.group(1).strip()
        rb_m = _RB_PATTERN.search(inner)
        if rb_m:
            surface = rb_m.group(1).strip()
        else:
            without_rp = _RP_PATTERN.sub("", inner)
            without_rt = _RT_PATTERN.sub("", without_rp)
            surface = _TAG_PATTERN.sub("", without_rt).strip()
        if surface and reading:
            pairs.append((surface, reading))
    return pairs


@register("aozora")
class AozoraSource:
    source: ClassVar[str] = "aozora"
    source_license: ClassVar[str] = "PDplusAoz"

    def entries(self, root: Path, *, limit: int | None = None) -> Iterator[Row]:
        if root.is_file():
            files: list[Path] = [root]
        else:
            found: list[Path] = []
            for pattern in ("*.html", "*.xhtml"):
                found.extend(root.rglob(pattern))
            files = sorted(found)
        emitted = 0
        for f in files:
            if limit is not None and emitted >= limit:
                return
            html = _read_text(f)
            license_tag = detect_license(html)
            parts = _HR_SPLIT.split(html, maxsplit=1)
            body = parts[1] if len(parts) > 1 else html
            pairs = extract_ruby_pairs_aozora(body)
            if not pairs:
                continue
            for surface, reading in pairs:
                if limit is not None and emitted >= limit:
                    return
                try:
                    phonemes = kana_to_julius_phonemes(reading)
                except ValueError:
                    continue
                if not phonemes:
                    continue
                yield Row(
                    id=make_id("aozora", surface, phonemes),
                    source="aozora",
                    source_license=license_tag,
                    text=surface,
                    phonemes=phonemes,
                    mora_accents=(),
                    accent_boundaries=(),
                    sample_weight=weight_for("general"),
                    extra={
                        "file": f.name,
                        "file_license": license_tag,
                        "rt_text": reading,
                    },
                )
                emitted += 1


__all__ = [
    "AozoraSource",
    "detect_license",
    "extract_ruby_pairs_aozora",
]
