"""P-C char-level BERT tokenizer wrapper.

Design (see ``docs/design/phase2_tokenizer_pilots.md §3.3``):

Uses the ``tohoku-nlp/bert-base-japanese-char-v2`` tokenizer, which yields
one WordPiece token per character for the vast majority of Japanese
characters (rare CJK chars may fall back to ``##``-continuation and are
counted here as part of the previous character's slot).

Target-side encoding produces per-character fixed-slot phoneme + H/L ids
and a per-character APBP BIO label. Caller supplies ``char_alignment``:
for each character, the phoneme indices from ``canonical.phonemes`` that
belong to that character.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

from modernbert_g2p.models.canonical import CanonicalForm, Vocab, build_default_vocab
from modernbert_g2p.models.tokenization.base import BIO_LABEL_TO_ID, EncodedTarget
from modernbert_g2p.models.tokenization.p_a_tokenizer import MORAIC_PHONEMES


class PCTokenizer:
    """Char-level BertJapaneseTokenizer wrapper with per-char slot targets."""

    name: str = "p_c"

    DEFAULT_TOKENIZER_KWARGS: dict[str, Any] = {"word_tokenizer_type": "basic"}
    """Bypasses the fugashi requirement for char-v2 by using basic word split.

    Char-v2's default configuration invokes MeCab as its word tokenizer even
    though the subword step is single-character, which uselessly forces the
    fugashi dependency. For pilot P-C we only need per-char ids so the basic
    word tokenizer produces the same char-level output.
    """

    def __init__(
        self,
        tokenizer_name: str = "tohoku-nlp/bert-base-japanese-char-v2",
        *,
        phoneme_vocab: Vocab | None = None,
        max_input_length: int = 512,
        max_slot: int = 8,
        tokenizer_kwargs: dict[str, Any] | None = None,
        hf_tokenizer: Any = None,
    ) -> None:
        self.tokenizer_name: str = tokenizer_name
        self.max_input_length: int = max_input_length
        self.max_slot: int = max_slot
        merged: dict[str, Any] = dict(self.DEFAULT_TOKENIZER_KWARGS)
        if tokenizer_kwargs:
            merged.update(tokenizer_kwargs)
        self.tokenizer_kwargs: dict[str, Any] = merged
        self.vocab: Vocab = phoneme_vocab if phoneme_vocab is not None else build_default_vocab()
        self._hf_tokenizer: Any = hf_tokenizer

    @property
    def pad_token_id(self) -> int:
        return self.vocab.pad_id

    def get_hf_tokenizer(self) -> Any:
        """Return a cached HF tokenizer (lazy load)."""
        if self._hf_tokenizer is None:
            from transformers import AutoTokenizer

            self._hf_tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name, **self.tokenizer_kwargs
            )
        return self._hf_tokenizer

    def encode_input(self, text: str) -> dict[str, list[int]]:
        """Encode ``text`` char-by-char with the BERT-Japanese char tokenizer."""
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

    def encode(self, texts: list[str]) -> dict[str, Any]:
        """Batched encode for the T6 :class:`PCCollator`.

        Returns:
            dict with keys ``input_ids`` / ``attention_mask`` (both
            ``list[list[int]]``) plus an ``extras`` sub-dict carrying
            ``char_positions`` (``list[list[int]]``): for each raw character
            in the input text, the index of the corresponding token inside
            ``input_ids``. Assumes the char-v2 tokenizer emits exactly one
            token per character (verified by
            :meth:`TestPCTokenizer.test_encode_input_real_char_v2_tokenizer`).
            Positions account for a leading ``[CLS]`` and a trailing
            ``[SEP]`` when the HF tokenizer emits them.
        """
        hf_tok = self.get_hf_tokenizer()
        has_cls = getattr(hf_tok, "cls_token_id", None) is not None
        has_sep = getattr(hf_tok, "sep_token_id", None) is not None
        cls_offset = 1 if has_cls else 0
        sep_offset = 1 if has_sep else 0

        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        char_positions: list[list[int]] = []
        for text in texts:
            enc = self.encode_input(text)
            ids = enc["input_ids"]
            input_ids.append(ids)
            attention_mask.append(enc["attention_mask"])
            seq_len = len(ids)
            max_char = max(0, seq_len - cls_offset - sep_offset)
            n_chars = min(len(text), max_char)
            char_positions.append([cls_offset + i for i in range(n_chars)])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "extras": {"char_positions": char_positions},
        }

    def encode_target(
        self,
        canonical: CanonicalForm,
        char_alignment: Sequence[Sequence[int]],
    ) -> EncodedTarget:
        """Emit per-character phoneme + H/L fixed-slot ids + BIO APBP labels.

        Args:
            canonical: reference canonical form.
            char_alignment: ``char_alignment[c]`` is the list of phoneme
                indices (into ``canonical.phonemes``) that belong to
                character ``c``. Length equals the number of characters
                whose target labels we are producing (i.e. excluding
                ``[CLS]`` / ``[SEP]``).

        Shape:
            ``per_char_phon_ids``: ``(len(char_alignment), max_slot)``
            ``per_char_hl_ids``:   ``(len(char_alignment), max_slot)``
            ``apbp_bio_ids``:      ``(len(char_alignment),)``
        """
        vocab = self.vocab
        phonemes = canonical.phonemes
        boundary_set = set(canonical.accent_boundaries)

        phon_to_mora: list[int] = []
        m_idx = 0
        for ph in phonemes:
            phon_to_mora.append(m_idx)
            if ph in MORAIC_PHONEMES:
                m_idx += 1

        per_char_phon_ids: list[list[int]] = []
        per_char_hl_ids: list[list[int]] = []
        apbp_bio_ids: list[int] = []
        max_slot = self.max_slot

        for c_idx, phon_indices in enumerate(char_alignment):
            indices = list(phon_indices)
            if len(indices) > max_slot:
                warnings.warn(
                    f"PCTokenizer: char {c_idx} has {len(indices)} phonemes "
                    f"exceeding max_slot={max_slot}; truncating",
                    stacklevel=2,
                )
                indices = indices[:max_slot]

            phon_ids: list[int] = [vocab.id_of(phonemes[pi]) for pi in indices]
            phon_ids.extend([vocab.pad_id] * (max_slot - len(phon_ids)))

            char_mora_indices: list[int] = []
            hl_ids: list[int] = []
            for pi in indices:
                if pi < len(phonemes) and phonemes[pi] in MORAIC_PHONEMES:
                    mora_idx = phon_to_mora[pi]
                    char_mora_indices.append(mora_idx)
                    if mora_idx < len(canonical.mora_accents):
                        tag = canonical.mora_accents[mora_idx]
                        if tag == vocab.HIGH_TOKEN:
                            hl_ids.append(vocab.high_id)
                        elif tag == vocab.LOW_TOKEN:
                            hl_ids.append(vocab.low_id)
                        else:
                            hl_ids.append(vocab.pad_id)
                    else:
                        hl_ids.append(vocab.pad_id)
            hl_ids = hl_ids[:max_slot]
            hl_ids.extend([vocab.pad_id] * (max_slot - len(hl_ids)))

            per_char_phon_ids.append(phon_ids)
            per_char_hl_ids.append(hl_ids)

            is_boundary = any(m in boundary_set and m > 0 for m in char_mora_indices)
            if is_boundary:
                apbp_bio_ids.append(BIO_LABEL_TO_ID["B"])
            elif c_idx == 0:
                apbp_bio_ids.append(BIO_LABEL_TO_ID["O"])
            else:
                apbp_bio_ids.append(BIO_LABEL_TO_ID["I"])

        return {
            "per_char_phon_ids": per_char_phon_ids,
            "per_char_hl_ids": per_char_hl_ids,
            "apbp_bio_ids": apbp_bio_ids,
        }

    def decode_output(self, ids: Sequence[int]) -> str:
        """Render a phoneme-id sequence back to space-separated tokens."""
        vocab = self.vocab
        return " ".join(vocab.token_of(int(i)) for i in ids)
