"""P-B tokenizer: MeCab pretokenize + ``[MORPH]`` boundary + ModernBERT SP.

Design (see ``docs/design/phase2_tokenizer_pilots.md §3.2``):

1. Split the raw text into morphemes with :class:`MeCabPretokenizer`
   (UniDic-3.1.1 + pyopenjtalk-plus additional vocabulary).
2. Insert a special token ``[MORPH]`` between adjacent morphemes so the
   ModernBERT encoder can attend to explicit boundaries.
3. Track the index of each morpheme's first subword token inside the
   final ``input_ids`` sequence to enable per-morpheme mean-pooling in T4.

Target-side encoding produces per-morpheme phoneme + H/L fixed-slot ids
and a per-morpheme APBP BIO label. Caller supplies ``morph_alignment``
(mora → morph index) since the mapping is upstream of the tokenizer.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from modernbert_g2p.models.canonical import CanonicalForm, Vocab, build_default_vocab
from modernbert_g2p.models.tokenization.base import BIO_LABEL_TO_ID, EncodedTarget
from modernbert_g2p.models.tokenization.p_a_tokenizer import split_moras


class PBTokenizer:
    """MeCab-pretokenizing wrapper that inserts ``[MORPH]`` between morphemes."""

    name: str = "p_b"
    MORPH_TOKEN: str = "[MORPH]"

    def __init__(
        self,
        tokenizer_name: str = "sbintuitions/modernbert-ja-130m",
        *,
        morph_token: str = "[MORPH]",
        mecab_dict: str | None = None,
        phoneme_vocab: Vocab | None = None,
        max_input_length: int = 512,
        max_mora_per_morph: int = 8,
        tokenizer_kwargs: dict[str, Any] | None = None,
        hf_tokenizer: Any = None,
        pretokenizer: Any = None,
    ) -> None:
        self.tokenizer_name: str = tokenizer_name
        self.morph_token: str = morph_token
        self.mecab_dict: str | None = mecab_dict
        self.max_input_length: int = max_input_length
        self.max_mora_per_morph: int = max_mora_per_morph
        self.tokenizer_kwargs: dict[str, Any] = dict(tokenizer_kwargs or {})
        self.vocab: Vocab = phoneme_vocab if phoneme_vocab is not None else build_default_vocab()
        self._hf_tokenizer: Any = hf_tokenizer
        self._pretokenizer: Any = pretokenizer
        self._morph_id_cache: int | None = None

    @property
    def pad_token_id(self) -> int:
        return self.vocab.pad_id

    @property
    def morph_id(self) -> int:
        """HF tokenizer id assigned to ``[MORPH]`` (cached after first call)."""
        if self._morph_id_cache is None:
            tok = self.get_hf_tokenizer()
            self._morph_id_cache = int(tok.convert_tokens_to_ids(self.morph_token))
        return self._morph_id_cache

    def get_pretokenizer(self) -> Any:
        """Return a cached :class:`MeCabPretokenizer` (lazy fugashi load)."""
        if self._pretokenizer is None:
            from modernbert_g2p.models.p_b.mecab import MeCabPretokenizer

            self._pretokenizer = MeCabPretokenizer(dict_path=self.mecab_dict)
        return self._pretokenizer

    def get_hf_tokenizer(self) -> Any:
        """Return a cached HF tokenizer with ``[MORPH]`` registered."""
        if self._hf_tokenizer is None:
            from transformers import AutoTokenizer

            self._hf_tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name, **self.tokenizer_kwargs
            )
        if self.morph_token not in self._hf_tokenizer.get_vocab():
            self._hf_tokenizer.add_special_tokens(
                {"additional_special_tokens": [self.morph_token]}
            )
            self._morph_id_cache = None
        return self._hf_tokenizer

    def encode(self, texts: list[str]) -> dict[str, Any]:
        """Batched encode for the T6 :class:`PBCollator`.

        Returns:
            dict with keys ``input_ids`` / ``attention_mask`` (both
            ``list[list[int]]``) plus an ``extras`` sub-dict carrying
            ``morph_positions`` and ``morph_lens`` (both ``list[list[int]]``).
            ``dict_hit_mask`` is not populated here — Phase 3 will thread
            per-morpheme dictionary lookup through the pretokenizer.
        """
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        morph_positions: list[list[int]] = []
        morph_lens: list[list[int]] = []
        for text in texts:
            enc = self.encode_input(text)
            input_ids.append(enc["input_ids"])
            attention_mask.append(enc["attention_mask"])
            morph_positions.append(enc["morph_positions"])
            morph_lens.append(enc["morph_lens"])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "extras": {
                "morph_positions": morph_positions,
                "morph_lens": morph_lens,
            },
        }

    def encode_input(self, text: str) -> dict[str, list[int]]:
        """Encode ``text`` with MeCab pretokenize + ``[MORPH]`` insertion.

        Returns keys:

        - ``input_ids``: full sequence including ``[CLS]``, subwords with
          ``[MORPH]`` between adjacent morphemes, and ``[SEP]``.
        - ``attention_mask``: 1s of the same length.
        - ``morph_positions``: index inside ``input_ids`` of each morpheme's
          first subword token (length = number of morphemes emitted).
        - ``morph_lens``: number of subword tokens per morpheme (excluding
          the trailing ``[MORPH]``).
        """
        pretokenizer = self.get_pretokenizer()
        morphs = pretokenizer.pretokenize(text)
        hf_tok = self.get_hf_tokenizer()

        cls_id = getattr(hf_tok, "cls_token_id", None)
        sep_id = getattr(hf_tok, "sep_token_id", None)
        morph_id = self.morph_id

        input_ids: list[int] = []
        morph_positions: list[int] = []
        morph_lens: list[int] = []

        if cls_id is not None:
            input_ids.append(int(cls_id))

        for idx, tok in enumerate(morphs):
            surface = tok.surface
            enc = hf_tok(
                surface,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_input_length,
                return_attention_mask=False,
            )
            sub_ids = [int(x) for x in enc["input_ids"]]
            if not sub_ids:
                continue
            morph_positions.append(len(input_ids))
            input_ids.extend(sub_ids)
            morph_lens.append(len(sub_ids))
            if idx < len(morphs) - 1:
                input_ids.append(morph_id)
            if len(input_ids) >= self.max_input_length - 1:
                break

        if sep_id is not None:
            input_ids.append(int(sep_id))
        if len(input_ids) > self.max_input_length:
            input_ids = input_ids[: self.max_input_length]
        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "morph_positions": morph_positions,
            "morph_lens": morph_lens,
        }

    def encode_target(
        self,
        canonical: CanonicalForm,
        morph_alignment: Sequence[int],
    ) -> EncodedTarget:
        """Emit per-morpheme phoneme + H/L fixed-slot ids + APBP BIO ids.

        Args:
            canonical: reference canonical form (phonemes / mora_accents / boundaries).
            morph_alignment: per-mora morph index, length equals the number of
                moras in ``canonical`` (i.e. number of moraic phonemes).

        Slots per morph are padded to ``max_mora_per_morph``. Morphs exceeding
        that limit are truncated (upstream pipeline is expected to split them).
        """
        vocab = self.vocab
        moras = split_moras(canonical.phonemes)
        if len(morph_alignment) != len(moras):
            raise ValueError(
                f"morph_alignment length {len(morph_alignment)} must equal "
                f"mora count {len(moras)}"
            )
        n_morphs = (max(morph_alignment) + 1) if morph_alignment else 0

        per_morph_phon: list[list[str]] = [[] for _ in range(n_morphs)]
        per_morph_hl_str: list[list[str]] = [[] for _ in range(n_morphs)]
        for mora_idx, (mora, m_idx) in enumerate(zip(moras, morph_alignment, strict=True)):
            per_morph_phon[m_idx].extend(mora)
            if mora_idx < len(canonical.mora_accents):
                per_morph_hl_str[m_idx].append(canonical.mora_accents[mora_idx])
            else:
                per_morph_hl_str[m_idx].append("")

        boundary_set = set(canonical.accent_boundaries)
        apbp_bio_tags: list[str] = []
        cum_moras = 0
        for m_idx in range(n_morphs):
            n_moras_here = sum(1 for x in morph_alignment if x == m_idx)
            if m_idx == 0:
                apbp_bio_tags.append("O")
            elif cum_moras in boundary_set:
                apbp_bio_tags.append("B")
            else:
                apbp_bio_tags.append("I")
            cum_moras += n_moras_here

        max_slot = self.max_mora_per_morph
        per_morph_phon_ids: list[list[int]] = []
        per_morph_hl_ids: list[list[int]] = []
        for phon, hl in zip(per_morph_phon, per_morph_hl_str, strict=True):
            phon_ids = [vocab.id_of(p) for p in phon[:max_slot]]
            phon_ids.extend([vocab.pad_id] * (max_slot - len(phon_ids)))
            hl_ids: list[int] = []
            for tag in hl[:max_slot]:
                if tag == vocab.HIGH_TOKEN:
                    hl_ids.append(vocab.high_id)
                elif tag == vocab.LOW_TOKEN:
                    hl_ids.append(vocab.low_id)
                else:
                    hl_ids.append(vocab.pad_id)
            hl_ids.extend([vocab.pad_id] * (max_slot - len(hl_ids)))
            per_morph_phon_ids.append(phon_ids)
            per_morph_hl_ids.append(hl_ids)

        apbp_bio_ids = [BIO_LABEL_TO_ID[t] for t in apbp_bio_tags]
        return {
            "per_morph_phon_ids": per_morph_phon_ids,
            "per_morph_hl_ids": per_morph_hl_ids,
            "apbp_bio_ids": apbp_bio_ids,
        }

    def decode_output(self, ids: Sequence[int]) -> str:
        """Render a phoneme-id sequence back to space-separated tokens."""
        vocab = self.vocab
        return " ".join(vocab.token_of(int(i)) for i in ids)
