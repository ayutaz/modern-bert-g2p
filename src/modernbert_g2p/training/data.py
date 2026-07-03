"""Datasets and pilot collators for the three Phase 2 tokenizer pilots.

Implements docs/design/phase2_tokenizer_pilots.md Track T6.

The collators are duck-typed against the tokenizer objects specified in
Track T2. Two encode-side interfaces are supported for compatibility:

- ``encode(list[str]) -> {"input_ids", "attention_mask", "extras": {...}}``
  (batched; matches the fake-tokenizer style used by unit tests).
- ``encode_input(text: str) -> {"input_ids", "attention_mask", ...}``
  (single-string; matches the real Track T2 wrappers around HF tokenizers).

Extras (``morph_positions``, ``morph_lens``, ``dict_hit_mask``,
``char_positions``, etc.) are read from ``enc["extras"][key]`` first, then
from ``enc[key]`` at the top level as a fallback.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol

from modernbert_g2p.data.schema import Row, make_id
from modernbert_g2p.models.canonical import build_default_vocab


class _PATokenizerLike(Protocol):
    pad_token_id: int

    def encode_input(self, text: str) -> dict[str, Any]: ...
    def encode_target(self, canonical: Any) -> Any: ...


class _PBTokenizerLike(Protocol):
    pad_token_id: int
    morph_id: int

    def encode_input(self, text: str) -> dict[str, Any]: ...


class _PCTokenizerLike(Protocol):
    pad_token_id: int
    max_slot: int

    def encode_input(self, text: str) -> dict[str, Any]: ...


def make_dummy_row(
    *,
    text: str = "こんにちは",
    phonemes: Sequence[str] = ("k", "o", "N", "n", "i", "ch", "i", "h", "a"),
    mora_accents: Sequence[str] = ("L", "H", "H", "H", "L"),
    accent_boundaries: Sequence[int] = (),
    category: str = "general",
    sample_weight: float = 1.0,
    source: str = "pyopenjtalk_plus",
    source_license: str = "BSD3",
    id_: str | None = None,
) -> Row:
    """Build a valid :class:`Row` for unit tests.

    Defaults produce a Phase-1-schema-compliant row for ``こんにちは``.
    """
    phon_tuple = tuple(phonemes)
    return Row(
        id=id_ if id_ is not None else make_id(source, text, phon_tuple),
        source=source,
        source_license=source_license,
        text=text,
        phonemes=phon_tuple,
        mora_accents=tuple(mora_accents),
        accent_boundaries=tuple(accent_boundaries),
        category=category,
        sample_weight=float(sample_weight),
    )


class G2PDataset:
    """Random-access dataset over Phase 1 Parquet / JSONL output or in-memory rows.

    Parameters
    ----------
    source
        Either a ``Path`` to a ``.jsonl`` or ``.parquet`` file, or an
        in-memory ``list[Row]``. Path suffix determines the reader.
    cache
        Retained for API forward-compatibility with a future streaming
        mode. Currently ignored — rows are always materialised.
    """

    __slots__ = ("_rows",)

    def __init__(self, source: Path | str | list[Row], *, cache: bool = True) -> None:
        del cache
        if isinstance(source, list):
            rows: list[Row] = []
            for r in source:
                if not isinstance(r, Row):
                    raise TypeError(f"G2PDataset: expected Row, got {type(r).__name__}")
                rows.append(r)
            self._rows = rows
            return

        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"G2PDataset source not found: {path}")
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            from modernbert_g2p.data.output import read_jsonl

            self._rows = list(read_jsonl(path))
        elif suffix == ".parquet":
            self._rows = _read_parquet_rows(path)
        else:
            raise ValueError(
                f"G2PDataset: unknown suffix {suffix!r}; expected .jsonl or .parquet"
            )

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> Row:
        return self._rows[idx]

    def __iter__(self) -> Iterator[Row]:
        return iter(self._rows)


def _read_parquet_rows(path: Path) -> list[Row]:
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise RuntimeError(
            "pyarrow is required to read Parquet files; install with `pip install pyarrow`"
        ) from e

    table = pq.read_table(str(path))
    columns = table.to_pydict()
    n = len(columns["id"])
    rows: list[Row] = []
    for i in range(n):
        extra_raw = columns["extra"][i]
        extra = json.loads(extra_raw) if extra_raw else {}
        rows.append(
            Row(
                id=columns["id"][i],
                source=columns["source"][i],
                source_license=columns["source_license"][i],
                text=columns["text"][i],
                phonemes=tuple(columns["phonemes"][i]),
                mora_accents=tuple(columns["mora_accents"][i]),
                accent_boundaries=tuple(columns["accent_boundaries"][i]),
                category=columns["category"][i],
                sample_weight=float(columns["sample_weight"][i]),
                extra=extra,
            )
        )
    return rows


def _canonical_from_row(row: Row) -> Any:
    """Build a ``CanonicalForm``-shaped object from a Row.

    If :mod:`modernbert_g2p.models.canonical` has landed, the real dataclass
    is used; otherwise a lightweight :class:`types.SimpleNamespace` with the
    same attribute names is returned. This keeps the collators usable while
    Track T1 (canonical vocab) is being written in parallel.
    """
    try:
        from modernbert_g2p.models.canonical import CanonicalForm  # type: ignore[import-not-found]

        return CanonicalForm(
            phonemes=tuple(row.phonemes),
            mora_accents=tuple(row.mora_accents),
            accent_boundaries=tuple(row.accent_boundaries),
        )
    except ImportError:
        return SimpleNamespace(
            phonemes=tuple(row.phonemes),
            mora_accents=tuple(row.mora_accents),
            accent_boundaries=tuple(row.accent_boundaries),
        )


def _round_up(value: int, multiple: int | None) -> int:
    if multiple is None or multiple <= 1 or value == 0:
        return value
    return ((value + multiple - 1) // multiple) * multiple


def _pad_1d(
    seqs: list[list[int]],
    pad_value: int,
    pad_to_multiple_of: int | None,
) -> list[list[int]]:
    if not seqs:
        return []
    max_len = max(len(s) for s in seqs)
    max_len = _round_up(max_len, pad_to_multiple_of)
    return [list(s) + [pad_value] * (max_len - len(s)) for s in seqs]


def _pad_2d(
    seqs: list[list[list[int]]],
    pad_value: int,
    inner: int,
    pad_to_multiple_of: int | None,
) -> list[list[list[int]]]:
    if not seqs:
        return []
    max_len = max(len(s) for s in seqs)
    max_len = _round_up(max_len, pad_to_multiple_of)
    pad_slot = [pad_value] * inner
    return [[list(x) for x in s] + [list(pad_slot) for _ in range(max_len - len(s))] for s in seqs]


def _check_non_empty(batch: list[Row], name: str) -> None:
    if not batch:
        raise ValueError(f"{name}: received empty batch")


def _encode_batch(
    tokenizer: Any,
    texts: list[str],
    extras_keys: Sequence[str] = (),
) -> tuple[list[list[int]], list[list[int]], dict[str, list[Any]]]:
    """Uniformly encode a batch of texts against either tokenizer interface.

    Prefers ``tokenizer.encode(list[str])`` (fake-style, batched) and falls
    back to per-text ``tokenizer.encode_input(text)`` (real Track T2 style).
    Extras are hoisted from ``enc["extras"]`` when present and from the
    top-level of the tokenizer output as a fallback, so morph/char position
    payloads land in the same place regardless of tokenizer style.
    """
    if hasattr(tokenizer, "encode") and callable(tokenizer.encode):
        enc = tokenizer.encode(texts)
        input_ids = [list(x) for x in enc["input_ids"]]
        attention_mask = [list(x) for x in enc["attention_mask"]]
        extras: dict[str, list[Any]] = {k: list(v) for k, v in enc.get("extras", {}).items()}
        for key in extras_keys:
            if key not in extras and key in enc:
                extras[key] = list(enc[key])
        return input_ids, attention_mask, extras

    input_ids = []
    attention_mask = []
    collected: dict[str, list[Any]] = {}
    for text in texts:
        e = tokenizer.encode_input(text)
        input_ids.append(list(e["input_ids"]))
        attention_mask.append(list(e["attention_mask"]))
        for key in extras_keys:
            if key in e:
                collected.setdefault(key, []).append(list(e[key]))
    return input_ids, attention_mask, collected


def _prepare_pa_target(
    tokenizer: Any,
    canonical: Any,
) -> tuple[list[int], list[int]]:
    """Return ``(decoder_input_ids, labels)`` from a P-A tokenizer output.

    Handles two return shapes:

    - ``list[int]`` (fake-style, ``[bos, ..., eos]``): the collator shifts
      internally to produce ``ids[:-1]`` / ``ids[1:]``.
    - ``dict{"decoder_input_ids", "labels"}`` (Track T2 shape): both keys
      are already same-length shift-aligned; the collator consumes them
      directly.

    Raises ``ValueError`` if a list-shape return has length < 2 (no room to
    shift out a paired decoder input / label sequence).
    """
    result = tokenizer.encode_target(canonical)
    if isinstance(result, dict):
        decoder_input_ids = list(result["decoder_input_ids"])
        labels = list(result["labels"])
        if len(decoder_input_ids) != len(labels):
            raise ValueError(
                "PACollator: encode_target dict must have decoder_input_ids and "
                f"labels of equal length (got {len(decoder_input_ids)} vs {len(labels)})"
            )
        return decoder_input_ids, labels
    ids = list(result)
    if len(ids) < 2:
        raise ValueError(
            "PACollator: encode_target must return a sequence of length >= 2 "
            "(expected <bos>/<eos> wrapped output)"
        )
    return ids[:-1], ids[1:]


@dataclass(frozen=True, slots=True)
class _MoraSpan:
    """Mora index range assigned to a single morpheme."""
    start: int
    end: int


def _split_mora_by_morph(
    row: Row,
    morph_lens_in_moras: Sequence[int],
) -> list[_MoraSpan]:
    total = len(row.mora_accents)
    spans: list[_MoraSpan] = []
    cursor = 0
    for m in morph_lens_in_moras:
        end = min(cursor + int(m), total)
        spans.append(_MoraSpan(cursor, end))
        cursor = end
    return spans


class PACollator:
    """Collate a list of :class:`Row` into a P-A seq2seq training batch.

    ``tokenizer`` MUST expose:
    - ``encode(texts: list[str]) -> {"input_ids": list[list[int]], "attention_mask": list[list[int]], ...}``
    - ``encode_target(canonical) -> list[int]`` where the returned sequence
      is assumed to have ``<bos>`` prepended and ``<eos>`` appended.
    - ``pad_token_id: int``

    Returned dict keys:
    - ``input_ids``, ``attention_mask`` (encoder side, padded)
    - ``decoder_input_ids``, ``decoder_attention_mask`` (right-shifted target)
    - ``labels`` (target[1:], pad positions replaced by ``label_pad_id``)
    - ``sample_weights``
    - ``ids`` (raw list[str], not a tensor)
    """

    def __init__(
        self,
        tokenizer: _PATokenizerLike,
        *,
        pad_to_multiple_of: int | None = 8,
        label_pad_id: int = -100,
    ) -> None:
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.label_pad_id = label_pad_id

    def __call__(self, batch: list[Row]) -> dict[str, Any]:
        _check_non_empty(batch, "PACollator")
        import torch

        texts = [r.text for r in batch]
        input_ids, attention_mask, _extras = _encode_batch(self.tokenizer, texts)

        input_pad = int(self.tokenizer.pad_token_id)
        input_ids_padded = _pad_1d(input_ids, input_pad, self.pad_to_multiple_of)
        attention_mask_padded = _pad_1d(attention_mask, 0, self.pad_to_multiple_of)

        decoder_input_ids: list[list[int]] = []
        labels_raw: list[list[int]] = []
        for row in batch:
            canonical = _canonical_from_row(row)
            dec_ids, lab_ids = _prepare_pa_target(self.tokenizer, canonical)
            decoder_input_ids.append(dec_ids)
            labels_raw.append(lab_ids)

        target_pad = int(self.tokenizer.pad_token_id)
        decoder_input_ids_padded = _pad_1d(
            decoder_input_ids, target_pad, self.pad_to_multiple_of
        )
        decoder_max_len = len(decoder_input_ids_padded[0])
        decoder_attention_mask_padded = [
            [1] * len(s) + [0] * (decoder_max_len - len(s)) for s in decoder_input_ids
        ]
        labels_padded = _pad_1d(labels_raw, self.label_pad_id, self.pad_to_multiple_of)

        sample_weights = [float(r.sample_weight) for r in batch]
        ids = [r.id for r in batch]

        return {
            "input_ids": torch.tensor(input_ids_padded, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask_padded, dtype=torch.long),
            "decoder_input_ids": torch.tensor(decoder_input_ids_padded, dtype=torch.long),
            "decoder_attention_mask": torch.tensor(
                decoder_attention_mask_padded, dtype=torch.long
            ),
            "labels": torch.tensor(labels_padded, dtype=torch.long),
            "sample_weights": torch.tensor(sample_weights, dtype=torch.float32),
            "ids": ids,
        }


class PBCollator:
    """Collate a list of :class:`Row` into a P-B MeCab-pretokenized training batch.

    ``tokenizer`` MUST expose:
    - ``encode(texts: list[str]) -> {"input_ids", "attention_mask", "extras": {"morph_positions", "morph_lens", "dict_hit_mask", "morph_lens_in_moras" (optional), "apbp" (optional list of BIO strings)}}``
    - ``pad_token_id: int``
    - ``morph_id: int``

    Optional extras (``morph_lens_in_moras``, ``apbp``) are consumed if the
    tokenizer supplies them; otherwise the collator falls back to a uniform
    split of the ``Row``'s mora sequence across morphemes and inserts ``B``
    at the mora index preceding each ``Row.accent_boundaries`` value.

    Returned dict keys:
    - ``input_ids``, ``attention_mask``
    - ``morph_positions``, ``morph_lens``  (both ``(B, M_max)`` int64)
    - ``morph_labels_phon``  ``(B, M_max, max_slot)`` int64, ``pad_id`` for slot pad
    - ``morph_labels_hl``    ``(B, M_max, max_slot)`` int64
    - ``morph_labels_apbp``  ``(B, M_max)`` int64 with ``0=O, 1=B, 2=I``
    - ``dict_hit_mask``       ``(B, M_max)`` bool
    - ``sample_weights``, ``ids``
    """

    _BIO_INDEX = {"O": 0, "B": 1, "I": 2}

    def __init__(
        self,
        tokenizer: _PBTokenizerLike,
        *,
        pad_to_multiple_of: int | None = 8,
        max_slot: int = 8,
        phoneme_pad_id: int = -100,
        hl_pad_id: int = -100,
        apbp_pad_id: int = -100,
    ) -> None:
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.max_slot = max_slot
        self.phoneme_pad_id = phoneme_pad_id
        self.hl_pad_id = hl_pad_id
        self.apbp_pad_id = apbp_pad_id

    def __call__(self, batch: list[Row]) -> dict[str, Any]:
        _check_non_empty(batch, "PBCollator")
        import torch

        texts = [r.text for r in batch]
        input_ids, attention_mask, extras = _encode_batch(
            self.tokenizer,
            texts,
            extras_keys=(
                "morph_positions",
                "morph_lens",
                "dict_hit_mask",
                "morph_lens_in_moras",
                "apbp",
            ),
        )

        morph_positions_lists: list[list[int]] = [
            list(x) for x in extras["morph_positions"]
        ]
        morph_lens_lists: list[list[int]] = [list(x) for x in extras["morph_lens"]]
        dict_hit_lists: list[list[bool]] = [
            list(x) for x in extras.get(
                "dict_hit_mask",
                [[False] * len(m) for m in morph_positions_lists],
            )
        ]

        morph_lens_in_moras_all: list[list[int]] = extras.get(
            "morph_lens_in_moras",
            [self._infer_mora_lens(row, len(mp))
             for row, mp in zip(batch, morph_positions_lists, strict=True)],
        )
        apbp_all: list[list[str]] = extras.get(
            "apbp",
            [self._infer_apbp(row, mora_lens)
             for row, mora_lens in zip(batch, morph_lens_in_moras_all, strict=True)],
        )

        m_max_raw = max(len(mp) for mp in morph_positions_lists)
        m_max = _round_up(m_max_raw, self.pad_to_multiple_of)

        input_pad = int(self.tokenizer.pad_token_id)
        input_ids_padded = _pad_1d(input_ids, input_pad, self.pad_to_multiple_of)
        attention_mask_padded = _pad_1d(attention_mask, 0, self.pad_to_multiple_of)

        morph_positions_padded = _pad_1d(morph_positions_lists, 0, self.pad_to_multiple_of)
        morph_lens_padded = _pad_1d(morph_lens_lists, 0, self.pad_to_multiple_of)
        dict_hit_padded = _pad_1d(
            [[int(x) for x in row] for row in dict_hit_lists], 0, self.pad_to_multiple_of
        )

        vocab = build_default_vocab()

        phon_labels_per_row: list[list[list[int]]] = []
        hl_labels_per_row: list[list[list[int]]] = []
        apbp_per_row: list[list[int]] = []
        for row, mora_lens, apbp_tags in zip(
            batch, morph_lens_in_moras_all, apbp_all, strict=True
        ):
            spans = _split_mora_by_morph(row, mora_lens)
            phon_per_morph: list[list[str]] = self._segment_phonemes(row, spans)
            hl_per_morph: list[list[str]] = self._segment_accents(row, spans)
            phon_labels_per_row.append(
                [self._encode_phoneme_slot(p, vocab) for p in phon_per_morph]
            )
            hl_labels_per_row.append(
                [self._encode_hl_slot(h, vocab) for h in hl_per_morph]
            )
            apbp_per_row.append([self._BIO_INDEX.get(t, 0) for t in apbp_tags])

        phon_labels_padded = _pad_2d(
            phon_labels_per_row, self.phoneme_pad_id, self.max_slot, self.pad_to_multiple_of
        )
        hl_labels_padded = _pad_2d(
            hl_labels_per_row, self.hl_pad_id, self.max_slot, self.pad_to_multiple_of
        )
        apbp_padded = _pad_1d(apbp_per_row, self.apbp_pad_id, self.pad_to_multiple_of)

        sample_weights = [float(r.sample_weight) for r in batch]
        ids = [r.id for r in batch]

        assert len(phon_labels_padded[0]) == m_max
        assert len(hl_labels_padded[0]) == m_max

        return {
            "input_ids": torch.tensor(input_ids_padded, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask_padded, dtype=torch.long),
            "morph_positions": torch.tensor(morph_positions_padded, dtype=torch.long),
            "morph_lens": torch.tensor(morph_lens_padded, dtype=torch.long),
            "morph_labels_phon": torch.tensor(phon_labels_padded, dtype=torch.long),
            "morph_labels_hl": torch.tensor(hl_labels_padded, dtype=torch.long),
            "morph_labels_apbp": torch.tensor(apbp_padded, dtype=torch.long),
            "dict_hit_mask": torch.tensor(dict_hit_padded, dtype=torch.bool),
            "sample_weights": torch.tensor(sample_weights, dtype=torch.float32),
            "ids": ids,
        }

    @staticmethod
    def _infer_mora_lens(row: Row, num_morphs: int) -> list[int]:
        total = len(row.mora_accents)
        if num_morphs <= 0:
            return []
        base, rem = divmod(total, num_morphs)
        lens = [base + (1 if i < rem else 0) for i in range(num_morphs)]
        return lens

    @staticmethod
    def _infer_apbp(row: Row, mora_lens: Sequence[int]) -> list[str]:
        tags = ["O"] * len(mora_lens)
        boundaries = {int(x) for x in row.accent_boundaries}
        cursor = 0
        for i, m in enumerate(mora_lens):
            cursor += int(m)
            if cursor in boundaries:
                tags[i] = "B"
        return tags

    def _segment_phonemes(self, row: Row, spans: Sequence[_MoraSpan]) -> list[list[str]]:
        phonemes = list(row.phonemes)
        mora_accents = row.mora_accents
        if not mora_accents:
            return [[] for _ in spans]
        phoneme_ranges = _phoneme_ranges_for_moras(phonemes, len(mora_accents))
        out: list[list[str]] = []
        for span in spans:
            if span.end <= span.start:
                out.append([])
                continue
            lo = phoneme_ranges[span.start][0] if span.start < len(phoneme_ranges) else len(phonemes)
            hi_idx = min(span.end - 1, len(phoneme_ranges) - 1)
            hi = phoneme_ranges[hi_idx][1] if hi_idx >= 0 else lo
            slot = phonemes[lo:hi]
            if len(slot) > self.max_slot:
                slot = slot[: self.max_slot]
            out.append(slot)
        return out

    def _segment_accents(self, row: Row, spans: Sequence[_MoraSpan]) -> list[list[str]]:
        out: list[list[str]] = []
        for span in spans:
            slot = list(row.mora_accents[span.start : span.end])
            if len(slot) > self.max_slot:
                slot = slot[: self.max_slot]
            out.append(slot)
        return out

    def _encode_phoneme_slot(self, phonemes: Sequence[str], vocab: Any) -> list[int]:
        ids = [vocab.id_of(p) for p in phonemes]
        if len(ids) > self.max_slot:
            ids = ids[: self.max_slot]
        ids = ids + [self.phoneme_pad_id] * (self.max_slot - len(ids))
        return ids

    def _encode_hl_slot(self, tags: Sequence[str], vocab: Any) -> list[int]:
        ids = [vocab.id_of(t) for t in tags]
        if len(ids) > self.max_slot:
            ids = ids[: self.max_slot]
        ids = ids + [self.hl_pad_id] * (self.max_slot - len(ids))
        return ids


class PCCollator:
    """Collate a list of :class:`Row` into a P-C char-level BERT training batch.

    ``tokenizer`` MUST expose:
    - ``encode(texts: list[str]) -> {"input_ids", "attention_mask", "extras": {"char_positions"}}``
      where ``char_positions[b][i]`` maps raw char ``i`` to its token index.
    - ``pad_token_id: int``
    - ``max_slot: int``

    Returned dict keys:
    - ``input_ids``, ``attention_mask`` (encoder side)
    - ``phoneme_labels``  ``(B, T_max, max_slot)`` int64
    - ``hl_labels``       ``(B, T_max, max_slot)`` int64
    - ``apbp_labels``     ``(B, T_max)`` int64 with ``0=O, 1=B, 2=I``
    - ``sample_weights``, ``ids``
    """

    _BIO_INDEX = {"O": 0, "B": 1, "I": 2}

    def __init__(
        self,
        tokenizer: _PCTokenizerLike,
        *,
        pad_to_multiple_of: int | None = 8,
        phoneme_pad_id: int = -100,
        hl_pad_id: int = -100,
        apbp_pad_id: int = -100,
    ) -> None:
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.max_slot = int(getattr(tokenizer, "max_slot", 8))
        self.phoneme_pad_id = phoneme_pad_id
        self.hl_pad_id = hl_pad_id
        self.apbp_pad_id = apbp_pad_id

    def _pad_slot(self, values: list[int], pad_value: int) -> list[int]:
        trimmed = list(values)[: self.max_slot]
        return trimmed + [pad_value] * (self.max_slot - len(trimmed))

    def __call__(self, batch: list[Row]) -> dict[str, Any]:
        _check_non_empty(batch, "PCCollator")
        import torch

        texts = [r.text for r in batch]
        input_ids, attention_mask, extras = _encode_batch(
            self.tokenizer,
            texts,
            extras_keys=("char_positions",),
        )
        char_positions_all: list[list[int]] = [
            list(x) for x in extras.get(
                "char_positions",
                [list(range(len(t))) for t in texts],
            )
        ]

        input_pad = int(self.tokenizer.pad_token_id)
        input_ids_padded = _pad_1d(input_ids, input_pad, self.pad_to_multiple_of)
        attention_mask_padded = _pad_1d(attention_mask, 0, self.pad_to_multiple_of)
        t_max = len(input_ids_padded[0])

        vocab = build_default_vocab()

        phon_labels_batch: list[list[list[int]]] = []
        hl_labels_batch: list[list[list[int]]] = []
        apbp_labels_batch: list[list[int]] = []

        for row, tok_len, char_positions in zip(
            batch,
            [len(x) for x in input_ids],
            char_positions_all,
            strict=True,
        ):
            phon_labels_row = [
                [self.phoneme_pad_id] * self.max_slot for _ in range(tok_len)
            ]
            hl_labels_row = [
                [self.hl_pad_id] * self.max_slot for _ in range(tok_len)
            ]
            apbp_labels_row = [self.apbp_pad_id] * tok_len

            char_phon_slots, char_hl_slots, char_bio = _row_to_char_slots(
                row, len(row.text), self.max_slot
            )
            for char_idx, tok_idx in enumerate(char_positions):
                if tok_idx >= tok_len or char_idx >= len(char_phon_slots):
                    continue
                phon_labels_row[tok_idx] = self._pad_slot(
                    [
                        vocab.id_of(p) if p != "<pad>" else self.phoneme_pad_id
                        for p in char_phon_slots[char_idx]
                    ],
                    self.phoneme_pad_id,
                )
                hl_labels_row[tok_idx] = self._pad_slot(
                    [
                        vocab.id_of(h) if h != "<pad>" else self.hl_pad_id
                        for h in char_hl_slots[char_idx]
                    ],
                    self.hl_pad_id,
                )
                apbp_labels_row[tok_idx] = self._BIO_INDEX.get(char_bio[char_idx], 0)

            phon_labels_batch.append(phon_labels_row)
            hl_labels_batch.append(hl_labels_row)
            apbp_labels_batch.append(apbp_labels_row)

        phon_labels_padded = _pad_2d_to_len(
            phon_labels_batch, self.phoneme_pad_id, self.max_slot, t_max
        )
        hl_labels_padded = _pad_2d_to_len(
            hl_labels_batch, self.hl_pad_id, self.max_slot, t_max
        )
        apbp_labels_padded = _pad_1d_to_len(apbp_labels_batch, self.apbp_pad_id, t_max)

        sample_weights = [float(r.sample_weight) for r in batch]
        ids = [r.id for r in batch]

        return {
            "input_ids": torch.tensor(input_ids_padded, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask_padded, dtype=torch.long),
            "phoneme_labels": torch.tensor(phon_labels_padded, dtype=torch.long),
            "hl_labels": torch.tensor(hl_labels_padded, dtype=torch.long),
            "apbp_labels": torch.tensor(apbp_labels_padded, dtype=torch.long),
            "sample_weights": torch.tensor(sample_weights, dtype=torch.float32),
            "ids": ids,
        }


def _pad_1d_to_len(seqs: list[list[int]], pad_value: int, target_len: int) -> list[list[int]]:
    return [list(s) + [pad_value] * (target_len - len(s)) for s in seqs]


def _pad_2d_to_len(
    seqs: list[list[list[int]]], pad_value: int, inner: int, target_len: int
) -> list[list[list[int]]]:
    pad_slot = [pad_value] * inner
    return [[list(x) for x in s] + [list(pad_slot) for _ in range(target_len - len(s))] for s in seqs]


def _phoneme_ranges_for_moras(
    phonemes: Sequence[str], n_moras: int
) -> list[tuple[int, int]]:
    """Partition ``phonemes`` into ranges belonging to each mora (best-effort).

    A mora is a vowel, ``N``, or ``q``; each such phoneme closes the current
    mora (consonants before it belong to that mora). The output length is
    exactly ``n_moras`` — if the phoneme sequence has fewer / more moras
    than expected, the ranges are truncated / extended with empty spans so
    the length invariant holds.
    """
    moraic = {"a", "i", "u", "e", "o", "N", "q"}
    ranges: list[tuple[int, int]] = []
    cursor = 0
    start = 0
    while cursor < len(phonemes) and len(ranges) < n_moras:
        if phonemes[cursor] in moraic:
            ranges.append((start, cursor + 1))
            start = cursor + 1
        cursor += 1
    while len(ranges) < n_moras:
        ranges.append((start, start))
    return ranges


def _row_to_char_slots(
    row: Row, n_chars: int, max_slot: int
) -> tuple[list[list[str]], list[list[str]], list[str]]:
    """Compute (phoneme_slots, hl_slots, bio) per raw character.

    This is a best-effort projection of the row's mora sequence onto its
    characters. When the character↔mora alignment is ambiguous (kanji chords
    that map to multiple morae), the trailing mora block is attached to the
    last character in the ambiguous stretch. Real training data uses
    per-source aligners in Phase 1; the collator's role here is to produce a
    well-shaped tensor, not perfect labels.
    """
    phon_slots: list[list[str]] = [[] for _ in range(n_chars)]
    hl_slots: list[list[str]] = [[] for _ in range(n_chars)]
    bio: list[str] = ["O"] * n_chars

    if n_chars == 0 or not row.mora_accents:
        return phon_slots, hl_slots, bio

    n_moras = len(row.mora_accents)
    phoneme_ranges = _phoneme_ranges_for_moras(row.phonemes, n_moras)

    base, rem = divmod(n_moras, n_chars)
    per_char_mora_counts: list[int] = []
    for i in range(n_chars):
        per_char_mora_counts.append(base + (1 if i < rem else 0))

    boundaries = {int(x) for x in row.accent_boundaries}
    mora_cursor = 0
    for char_idx, mora_count in enumerate(per_char_mora_counts):
        phon_bucket: list[str] = []
        hl_bucket: list[str] = []
        for _ in range(mora_count):
            if mora_cursor >= n_moras:
                break
            lo, hi = phoneme_ranges[mora_cursor]
            phon_bucket.extend(row.phonemes[lo:hi])
            hl_bucket.append(row.mora_accents[mora_cursor])
            mora_cursor += 1
            if mora_cursor in boundaries and char_idx + 1 < n_chars:
                bio[char_idx + 1] = "B"
        phon_slots[char_idx] = list(phon_bucket)[:max_slot]
        hl_slots[char_idx] = list(hl_bucket)[:max_slot]

    return phon_slots, hl_slots, bio


__all__ = [
    "G2PDataset",
    "PACollator",
    "PBCollator",
    "PCCollator",
    "make_dummy_row",
]
