"""Canonical vocab + per-pilot output-to-canonical conversion.

All three Phase 2 pilots (P-A seq2seq / P-B MeCab+[MORPH] / P-C char-BERT) emit
model-specific output shapes but must be projected onto a single canonical
representation before PER/CER/KER scoring so that comparisons are fair.

The canonical form follows ``docs/design/phase2_tokenizer_pilots.md §3.4``:

- JULIUS phoneme sequence (``a``, ``k``, ``ky``, ``N``, ...).
- Per-mora accent labels ``H`` / ``L`` in emission order.
- Accent-phrase-boundary marker positions (mora indices where ``/`` is inserted).

This module has no runtime dependency on ``torch`` / ``transformers`` — it is
pure Python so tests can import it without paying the ML stack import cost.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

JULIUS_PHONEMES: tuple[str, ...] = (
    "a", "i", "u", "e", "o",
    "N",
    "q", "cl",
    "k", "s", "sh", "t", "ch", "ts", "n", "h", "f", "m", "r", "y", "w",
    "g", "z", "j", "d", "b", "p", "v",
    "ky", "gy", "sy", "zy", "ny", "hy", "my", "ry", "by", "py",
    "dy", "ty", "fy", "kw", "gw",
)

SPECIAL_TOKENS: tuple[str, ...] = ("<pad>", "<bos>", "<eos>", "<unk>", "/", "H", "L")

PROSODY_TOKENS: tuple[str, ...] = ("H", "L", "/")

PAUSE_TOKEN: str = "pau"

DROPPED_PHONEMES: frozenset[str] = frozenset({PAUSE_TOKEN})


@dataclass(frozen=True, slots=True)
class CanonicalForm:
    """Immutable canonical G2P output shared across all three Phase 2 pilots."""

    phonemes: tuple[str, ...]
    mora_accents: tuple[str, ...]
    accent_boundaries: tuple[int, ...]


class Vocab:
    """Bijective token/id vocab for the canonical phoneme+prosody space.

    Layout (id ascending):

    - ``[0..3]``: 4 reserved specials ``<pad>``, ``<bos>``, ``<eos>``, ``<unk>``.
    - ``[4..3+N]``: JULIUS phonemes in :data:`JULIUS_PHONEMES` order.
    - ``[4+N..6+N]``: prosody tokens ``H``, ``L``, ``/``.
    """

    PAD_TOKEN = "<pad>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"
    BOUNDARY_TOKEN = "/"
    HIGH_TOKEN = "H"
    LOW_TOKEN = "L"

    def __init__(self, phoneme_tokens: Sequence[str] = JULIUS_PHONEMES) -> None:
        ordered = [
            self.PAD_TOKEN,
            self.BOS_TOKEN,
            self.EOS_TOKEN,
            self.UNK_TOKEN,
            *phoneme_tokens,
            self.HIGH_TOKEN,
            self.LOW_TOKEN,
            self.BOUNDARY_TOKEN,
        ]
        token_to_id: dict[str, int] = {}
        id_to_token: list[str] = []
        for tok in ordered:
            if tok in token_to_id:
                continue
            token_to_id[tok] = len(id_to_token)
            id_to_token.append(tok)
        self._token_to_id: dict[str, int] = token_to_id
        self._id_to_token: tuple[str, ...] = tuple(id_to_token)

    def id_of(self, token: str) -> int:
        return self._token_to_id.get(token, self._token_to_id[self.UNK_TOKEN])

    def token_of(self, idx: int) -> str:
        if 0 <= idx < len(self._id_to_token):
            return self._id_to_token[idx]
        return self.UNK_TOKEN

    @property
    def size(self) -> int:
        return len(self._id_to_token)

    @property
    def pad_id(self) -> int:
        return self._token_to_id[self.PAD_TOKEN]

    @property
    def bos_id(self) -> int:
        return self._token_to_id[self.BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self._token_to_id[self.EOS_TOKEN]

    @property
    def unk_id(self) -> int:
        return self._token_to_id[self.UNK_TOKEN]

    @property
    def boundary_id(self) -> int:
        return self._token_to_id[self.BOUNDARY_TOKEN]

    @property
    def high_id(self) -> int:
        return self._token_to_id[self.HIGH_TOKEN]

    @property
    def low_id(self) -> int:
        return self._token_to_id[self.LOW_TOKEN]


def build_default_vocab() -> Vocab:
    """Return the default :class:`Vocab` populated with :data:`JULIUS_PHONEMES`."""
    return Vocab()


_STRUCTURAL_SPECIALS: frozenset[str] = frozenset({"<pad>", "<bos>", "<eos>"})


def _is_droppable(token: str, unk_token: str) -> bool:
    return token in _STRUCTURAL_SPECIALS or token == unk_token or token in DROPPED_PHONEMES


def p_a_to_canonical(token_ids: Sequence[int], vocab: Vocab) -> CanonicalForm:
    """Convert a P-A decoder id sequence into a :class:`CanonicalForm`.

    Algorithm (see design doc §3.4, row P-A):
      1. Skip structural specials ``<pad>``, ``<bos>``, ``<eos>``, ``<unk>`` and
         the pause marker ``pau`` (silent segment; must not surface in the
         canonical phoneme sequence used for PER scoring).
      2. Buffer phoneme tokens until an ``H`` or ``L`` token is seen; that
         accent tag consumes the buffered phones as a single mora and appends
         the tag to ``mora_accents``.
      3. ``/`` pushes ``len(mora_accents)`` into ``accent_boundaries``.
      4. Any trailing buffered phones (no accent tag before end) are appended
         to ``phonemes`` without an accent contribution.
    """
    phonemes: list[str] = []
    mora_accents: list[str] = []
    accent_boundaries: list[int] = []
    pending: list[str] = []
    for tid in token_ids:
        s = vocab.token_of(tid)
        if _is_droppable(s, vocab.UNK_TOKEN):
            continue
        if s == vocab.HIGH_TOKEN or s == vocab.LOW_TOKEN:
            mora_accents.append(s)
            phonemes.extend(pending)
            pending = []
        elif s == vocab.BOUNDARY_TOKEN:
            accent_boundaries.append(len(mora_accents))
        else:
            pending.append(s)
    phonemes.extend(pending)
    return CanonicalForm(
        phonemes=tuple(phonemes),
        mora_accents=tuple(mora_accents),
        accent_boundaries=tuple(accent_boundaries),
    )


def p_b_to_canonical(
    per_morph_phon_ids: Sequence[Sequence[int]],
    per_morph_hl: Sequence[Sequence[str]],
    apbp_bio: Sequence[str],
    vocab: Vocab,
) -> CanonicalForm:
    """Convert P-B per-morpheme head output into a :class:`CanonicalForm`.

    Boundary rule: ``apbp_bio[i] == "B"`` at morph ``i`` (with ``i > 0``)
    inserts a ``/`` at the mora index reached just before morph ``i``.
    Phoneme slots equal to ``vocab.PAD_TOKEN``, structural specials, ``<unk>``,
    and the pause marker ``pau`` are dropped from the phoneme stream.
    """
    phonemes: list[str] = []
    mora_accents: list[str] = []
    accent_boundaries: list[int] = []
    for morph_idx, (phon_slots, hl_slots, apbp_tag) in enumerate(
        zip(per_morph_phon_ids, per_morph_hl, apbp_bio, strict=True)
    ):
        if apbp_tag == "B" and morph_idx > 0:
            accent_boundaries.append(len(mora_accents))
        for ph_id in phon_slots:
            s = vocab.token_of(ph_id)
            if s == vocab.PAD_TOKEN or _is_droppable(s, vocab.UNK_TOKEN):
                continue
            phonemes.append(s)
        for hl in hl_slots:
            if hl == vocab.PAD_TOKEN or hl == "":
                continue
            mora_accents.append(hl)
    return CanonicalForm(
        phonemes=tuple(phonemes),
        mora_accents=tuple(mora_accents),
        accent_boundaries=tuple(accent_boundaries),
    )


def p_c_to_canonical(
    per_char_phon_ids: Sequence[Sequence[int]],
    per_char_hl: Sequence[Sequence[str]],
    apbp_bio: Sequence[str],
    vocab: Vocab,
) -> CanonicalForm:
    """Convert P-C per-character fixed-slot output into a :class:`CanonicalForm`.

    Same aggregation as :func:`p_b_to_canonical` but the outer axis is
    per-character rather than per-morpheme. Structural specials, ``<unk>``,
    and the pause marker ``pau`` are dropped from the phoneme stream.
    """
    phonemes: list[str] = []
    mora_accents: list[str] = []
    accent_boundaries: list[int] = []
    for char_idx, (phon_slots, hl_slots, apbp_tag) in enumerate(
        zip(per_char_phon_ids, per_char_hl, apbp_bio, strict=True)
    ):
        if apbp_tag == "B" and char_idx > 0:
            accent_boundaries.append(len(mora_accents))
        for ph_id in phon_slots:
            s = vocab.token_of(ph_id)
            if s == vocab.PAD_TOKEN or _is_droppable(s, vocab.UNK_TOKEN):
                continue
            phonemes.append(s)
        for hl in hl_slots:
            if hl == vocab.PAD_TOKEN or hl == "":
                continue
            mora_accents.append(hl)
    return CanonicalForm(
        phonemes=tuple(phonemes),
        mora_accents=tuple(mora_accents),
        accent_boundaries=tuple(accent_boundaries),
    )
