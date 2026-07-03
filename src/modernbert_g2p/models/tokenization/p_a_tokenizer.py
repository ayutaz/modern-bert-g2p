"""P-A seq2seq tokenizer wrapper.

Wraps a Hugging Face ``AutoTokenizer`` (default: ``sbintuitions/modernbert-ja-130m``)
for the encoder-side input, and produces an interleaved
``phoneme + H|L + /`` id stream for the decoder-side target.

Only stdlib and the canonical vocab are imported at module load. The
Hugging Face tokenizer is fetched lazily inside :meth:`PATokenizer.get_hf_tokenizer`
so unit tests that do not exercise the encoder path do not pay the
``transformers`` import cost.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from modernbert_g2p.models.canonical import CanonicalForm, Vocab, build_default_vocab

MORAIC_PHONEMES: frozenset[str] = frozenset({"a", "i", "u", "e", "o", "N", "q", "cl"})
"""Phonemes that count as a mora nucleus for interleaved H/L emission."""


def split_moras(phonemes: Sequence[str]) -> list[list[str]]:
    """Split a phoneme sequence into per-mora chunks by moraic nucleus.

    Each chunk contains the (optional) onset consonants followed by exactly
    one moraic phoneme. Trailing non-moraic phonemes are attached to the last
    mora chunk when present.
    """
    moras: list[list[str]] = []
    current: list[str] = []
    for ph in phonemes:
        current.append(ph)
        if ph in MORAIC_PHONEMES:
            moras.append(current)
            current = []
    if current:
        if moras:
            moras[-1].extend(current)
        else:
            moras.append(current)
    return moras


class PATokenizer:
    """SentencePiece wrapper + phoneme-side interleaver for pilot P-A."""

    name: str = "p_a"

    def __init__(
        self,
        tokenizer_name: str = "sbintuitions/modernbert-ja-130m",
        *,
        phoneme_vocab: Vocab | None = None,
        max_input_length: int = 512,
        max_target_length: int = 256,
        tokenizer_kwargs: dict[str, Any] | None = None,
        hf_tokenizer: Any = None,
    ) -> None:
        self.tokenizer_name: str = tokenizer_name
        self.max_input_length: int = max_input_length
        self.max_target_length: int = max_target_length
        self.tokenizer_kwargs: dict[str, Any] = dict(tokenizer_kwargs or {})
        self.vocab: Vocab = phoneme_vocab if phoneme_vocab is not None else build_default_vocab()
        self._hf_tokenizer: Any = hf_tokenizer

    @property
    def pad_token_id(self) -> int:
        return self.vocab.pad_id

    def get_hf_tokenizer(self) -> Any:
        """Return a cached ``AutoTokenizer`` instance (lazy)."""
        if self._hf_tokenizer is None:
            from transformers import AutoTokenizer

            self._hf_tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name, **self.tokenizer_kwargs
            )
        return self._hf_tokenizer

    def encode_input(self, text: str) -> dict[str, list[int]]:
        """Encode ``text`` with the HF tokenizer, truncating to ``max_input_length``."""
        tok = self.get_hf_tokenizer()
        encoded = tok(
            text,
            truncation=True,
            max_length=self.max_input_length,
            padding=False,
            return_attention_mask=True,
        )
        return {
            "input_ids": list(encoded["input_ids"]),
            "attention_mask": list(encoded["attention_mask"]),
        }

    def encode(self, texts: list[str]) -> dict[str, list[list[int]]]:
        """Batched wrapper around :meth:`encode_input` for the T6 collator layer."""
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        for text in texts:
            out = self.encode_input(text)
            input_ids.append(out["input_ids"])
            attention_mask.append(out["attention_mask"])
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def encode_target(self, canonical: CanonicalForm) -> list[int]:
        """Interleave phoneme / H|L / boundary and return a single id sequence.

        The returned list has ``<bos>`` prepended and ``<eos>`` appended so the
        T6 :class:`PACollator` can slice off ``[:-1]`` for ``decoder_input_ids``
        and ``[1:]`` for ``labels``.

        Emission order per mora ``i``:
          1. If ``i`` in ``accent_boundaries`` and ``i > 0``: emit ``/``.
          2. Emit every constituent phoneme (consonants + moraic nucleus).
          3. If ``mora_accents[i]`` is ``H`` or ``L``: emit it.

        Truncation: if the assembled sequence would exceed
        ``max_target_length``, it is clipped to ``max_target_length - 1``
        tokens and ``<eos>`` is appended so the final token is always
        ``<eos>``.
        """
        vocab = self.vocab
        stream_ids: list[int] = [vocab.bos_id]
        moras = split_moras(canonical.phonemes)
        boundary_set = set(canonical.accent_boundaries)
        accents = canonical.mora_accents
        for i, mora in enumerate(moras):
            if i in boundary_set and i > 0:
                stream_ids.append(vocab.boundary_id)
            for ph in mora:
                stream_ids.append(vocab.id_of(ph))
            if i < len(accents):
                accent = accents[i]
                if accent == vocab.HIGH_TOKEN:
                    stream_ids.append(vocab.high_id)
                elif accent == vocab.LOW_TOKEN:
                    stream_ids.append(vocab.low_id)
        target_len = self.max_target_length
        if len(stream_ids) > target_len - 1:
            stream_ids = stream_ids[: target_len - 1]
        stream_ids.append(vocab.eos_id)
        return stream_ids

    def decode_output(self, ids: Sequence[int]) -> str:
        """Render a target id sequence back to space-separated tokens."""
        vocab = self.vocab
        tokens = [vocab.token_of(int(i)) for i in ids]
        return " ".join(tokens)
