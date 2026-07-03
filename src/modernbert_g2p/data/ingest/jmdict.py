"""S3 ingest: parse JMDict XML entries (k_ele x r_ele) into Row records.

Implements docs/design/phase1_data_pipeline.md §4 (S3 JMDict extraction).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import ClassVar
from xml.etree.ElementTree import Element, fromstring, iterparse

from modernbert_g2p.data.ingest.base import (
    kana_to_julius_phonemes,
    normalize_katakana,
    register,
)
from modernbert_g2p.data.schema import Row, make_id, validate_row
from modernbert_g2p.data.weighting import weight_for

_HIRA_START = 0x3041
_HIRA_END = 0x3096
_HIRA_TO_KATA_OFFSET = 0x60


def hira_to_kata(text: str) -> str:
    """Convert hiragana codepoints (U+3041-U+3096) to katakana; leave other chars untouched."""
    return "".join(
        chr(ord(c) + _HIRA_TO_KATA_OFFSET) if _HIRA_START <= ord(c) <= _HIRA_END else c
        for c in text
    )


def parse_jmdict_entry(entry_xml: str) -> Iterator[Row]:
    """Parse a single ``<entry>...</entry>`` XML string into zero or more Rows."""
    yield from _iter_rows_from_entry(fromstring(entry_xml))


@register("jmdict")
class JMDictSource:
    """S3 ingest source for the EDRDG JMDict XML dump."""

    source: ClassVar[str] = "jmdict"
    source_license: ClassVar[str] = "EDRDG"

    def entries(self, root: Path, *, limit: int | None = None) -> Iterator[Row]:
        path = root if root.is_file() else root / "JMdict_e.xml"
        if not path.exists():
            raise FileNotFoundError(str(path))

        count = 0
        for _event, elem in iterparse(str(path), events=("end",)):
            if elem.tag != "entry":
                continue
            for row in _iter_rows_from_entry(elem):
                yield row
                count += 1
                if limit is not None and count >= limit:
                    elem.clear()
                    return
            elem.clear()


def _iter_rows_from_entry(entry: Element) -> Iterator[Row]:
    ent_seq_el = entry.find("ent_seq")
    ent_seq = (ent_seq_el.text or "").strip() if ent_seq_el is not None else ""

    kebs: list[str] = []
    for k in entry.findall("k_ele"):
        keb_el = k.find("keb")
        if keb_el is not None and keb_el.text:
            kebs.append(keb_el.text)

    r_entries: list[tuple[str, list[str] | None]] = []
    for r in entry.findall("r_ele"):
        reb_el = r.find("reb")
        if reb_el is None or not reb_el.text:
            continue
        restrs = [x.text for x in r.findall("re_restr") if x.text]
        r_entries.append((reb_el.text, restrs or None))

    if not kebs:
        for reb, _ in r_entries:
            row = _build_row(reb, reb, ent_seq)
            if row is not None:
                yield row
        return

    for reb, allowed in r_entries:
        for keb in kebs:
            if allowed is not None and keb not in allowed:
                continue
            row = _build_row(keb, reb, ent_seq)
            if row is not None:
                yield row


def _build_row(surface: str, reading: str, ent_seq: str) -> Row | None:
    kana = normalize_katakana(hira_to_kata(reading))
    if not kana or not surface:
        return None
    try:
        phonemes = kana_to_julius_phonemes(kana)
    except ValueError:
        return None
    if not phonemes:
        return None
    row_id = make_id("jmdict", surface, phonemes)
    row = Row(
        id=row_id,
        source="jmdict",
        source_license="EDRDG",
        text=surface,
        phonemes=phonemes,
        mora_accents=(),
        accent_boundaries=(),
        sample_weight=weight_for("general"),
        extra={"ent_seq": ent_seq, "reb": kana},
    )
    try:
        validate_row(row)
    except ValueError:
        return None
    return row
