"""Shared protocol / typed dicts for the three pilot tokenizers.

All three tokenizers expose the following methods:

- :meth:`BaseTokenizer.encode_input`: text → ``input_ids`` / ``attention_mask``
  plus any pilot-specific side channels (e.g. ``char_positions`` for P-C).
- :meth:`BaseTokenizer.encode`: ``list[str]`` → batched dict shaped
  ``{"input_ids", "attention_mask", "extras": {...}}`` for the T6 collators.
- :meth:`BaseTokenizer.encode_target`: :class:`CanonicalForm` → per-pilot
  target-side tensors keyed by well-known names.
- :meth:`BaseTokenizer.decode_output`: id sequence → readable string, for
  debug logging only.

The Protocol is duck-typed so callers can pass any object that structurally
satisfies the interface — useful for unit tests that inject fake tokenizers.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol, TypedDict, runtime_checkable

if TYPE_CHECKING:
    from modernbert_g2p.models.canonical import CanonicalForm


BIO_LABEL_TO_ID: dict[str, int] = {"O": 0, "B": 1, "I": 2}
"""Fixed integer mapping used by P-C for APBP BIO labels."""


ID_TO_BIO_LABEL: dict[int, str] = {v: k for k, v in BIO_LABEL_TO_ID.items()}
"""Inverse of :data:`BIO_LABEL_TO_ID`, for decode-side helpers."""


class EncodedTarget(TypedDict, total=False):
    """Union of possible keys returned by ``encode_target``.

    Only a subset is populated per pilot; consumers should inspect the keys.
    """

    decoder_input_ids: list[int]
    labels: list[int]
    per_morph_phon_ids: list[list[int]]
    per_morph_hl_ids: list[list[int]]
    apbp_bio_ids: list[int]
    per_char_phon_ids: list[list[int]]
    per_char_hl_ids: list[list[int]]


@runtime_checkable
class BaseTokenizer(Protocol):
    """Structural protocol satisfied by every pilot tokenizer."""

    name: str
    pad_token_id: int
    max_input_length: int

    def encode_input(self, text: str) -> dict[str, list[int]]:
        """Return at least ``input_ids`` and ``attention_mask`` for one text."""

    def encode(self, texts: list[str]) -> dict[str, Any]:
        """Batched encode for the T6 collator layer.

        Returns a dict with at least ``input_ids`` / ``attention_mask``
        (``list[list[int]]``) plus a per-pilot ``extras`` sub-dict that carries
        any auxiliary tensors (e.g. ``char_positions`` for P-C).
        """

    def encode_target(
        self, canonical: CanonicalForm, *args: object, **kwargs: object
    ) -> Any:
        """Return pilot-specific target-side id tensors.

        The return type varies per pilot: P-A yields a raw ``list[int]``
        (``<bos>`` prepended, ``<eos>`` appended) for the seq2seq decoder;
        P-C yields an :class:`EncodedTarget` dict.
        """

    def decode_output(self, ids: Sequence[int]) -> str:
        """Human-readable rendering of an id sequence for debug logs."""
