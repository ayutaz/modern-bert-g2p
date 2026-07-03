"""P-C char-level BERT model + optional linear-chain CRF for C2 APBP head.

The neural classes here (``LinearChainCRF``, ``PCCharBERT``) inherit from
``torch.nn.Module`` which is unavailable without the training extra. To honour
the project-wide "no top-level torch import" rule they are constructed inside
``_build_classes`` on first access and re-exposed through the module-level
``__getattr__`` hook.

Reference: docs/design/phase2_tokenizer_pilots.md §3.3 and §3.4 (P-C row).
"""

from __future__ import annotations

from typing import Any

from modernbert_g2p.models.p_c.config import PCConfig

__all__ = ["PCConfig", "LinearChainCRF", "PCCharBERT", "build_p_c"]  # noqa: F822


_CLASS_CACHE: dict[str, type] | None = None


def _build_classes() -> dict[str, type]:
    import torch
    from torch import nn
    from torch.nn import functional as F  # noqa: N812

    class LinearChainCRF(nn.Module):
        """Minimal linear-chain CRF used by the P-C C2 APBP head.

        Provides log-likelihood scoring for training and Viterbi decoding
        for inference. All tensor operations use ``logsumexp`` for numerical
        stability. The ``mask`` argument follows the HuggingFace convention
        (``1`` for real positions, ``0`` for padding).
        """

        def __init__(self, num_tags: int) -> None:
            super().__init__()
            if num_tags <= 0:
                raise ValueError(f"num_tags must be positive, got {num_tags}")
            self.num_tags = num_tags
            self.transitions = nn.Parameter(torch.randn(num_tags, num_tags) * 0.01)
            self.start_transitions = nn.Parameter(torch.randn(num_tags) * 0.01)
            self.end_transitions = nn.Parameter(torch.randn(num_tags) * 0.01)

        def forward(
            self,
            emissions: torch.Tensor,
            tags: torch.Tensor,
            mask: torch.Tensor,
        ) -> torch.Tensor:
            self._validate(emissions, tags, mask)
            mask_f = mask.to(dtype=emissions.dtype)
            numerator = self._score(emissions, tags, mask_f)
            denominator = self._log_partition(emissions, mask_f)
            return (denominator - numerator).mean()

        def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> list[list[int]]:
            if emissions.dim() != 3 or emissions.size(-1) != self.num_tags:
                raise ValueError(
                    f"emissions must be (B, T, {self.num_tags}), got {tuple(emissions.shape)}"
                )
            if mask.shape != emissions.shape[:2]:
                raise ValueError(
                    f"mask shape {tuple(mask.shape)} incompatible with "
                    f"emissions {tuple(emissions.shape[:2])}"
                )
            with torch.no_grad():
                return self._viterbi(emissions, mask.to(dtype=emissions.dtype))

        def _validate(
            self,
            emissions: torch.Tensor,
            tags: torch.Tensor,
            mask: torch.Tensor,
        ) -> None:
            if emissions.dim() != 3 or emissions.size(-1) != self.num_tags:
                raise ValueError(
                    f"emissions must be (B, T, {self.num_tags}), got {tuple(emissions.shape)}"
                )
            if tags.shape != emissions.shape[:2]:
                raise ValueError(
                    f"tags shape {tuple(tags.shape)} incompatible with "
                    f"emissions {tuple(emissions.shape[:2])}"
                )
            if mask.shape != emissions.shape[:2]:
                raise ValueError(
                    f"mask shape {tuple(mask.shape)} incompatible with "
                    f"emissions {tuple(emissions.shape[:2])}"
                )

        def _score(
            self,
            emissions: torch.Tensor,
            tags: torch.Tensor,
            mask: torch.Tensor,
        ) -> torch.Tensor:
            batch, seq_len, _ = emissions.shape
            long_tags = tags.long()
            score = self.start_transitions[long_tags[:, 0]] + emissions[:, 0].gather(
                1, long_tags[:, 0:1]
            ).squeeze(1)
            for step in range(1, seq_len):
                emit = emissions[:, step].gather(1, long_tags[:, step : step + 1]).squeeze(1)
                trans = self.transitions[long_tags[:, step - 1], long_tags[:, step]]
                score = score + mask[:, step] * (emit + trans)
            seq_lens = mask.sum(dim=1).long().clamp(min=1)
            last_idx = (seq_lens - 1).unsqueeze(1)
            last_tags = long_tags.gather(1, last_idx).squeeze(1)
            return score + self.end_transitions[last_tags]

        def _log_partition(
            self,
            emissions: torch.Tensor,
            mask: torch.Tensor,
        ) -> torch.Tensor:
            _, seq_len, _ = emissions.shape
            alpha = self.start_transitions.unsqueeze(0) + emissions[:, 0]
            for step in range(1, seq_len):
                broadcast = (
                    alpha.unsqueeze(2)
                    + self.transitions.unsqueeze(0)
                    + emissions[:, step].unsqueeze(1)
                )
                new_alpha = torch.logsumexp(broadcast, dim=1)
                mask_step = mask[:, step].unsqueeze(1)
                alpha = mask_step * new_alpha + (1.0 - mask_step) * alpha
            alpha = alpha + self.end_transitions.unsqueeze(0)
            return torch.logsumexp(alpha, dim=1)

        def _viterbi(
            self,
            emissions: torch.Tensor,
            mask: torch.Tensor,
        ) -> list[list[int]]:
            batch, seq_len, _ = emissions.shape
            score = self.start_transitions.unsqueeze(0) + emissions[:, 0]
            history: list[torch.Tensor] = []
            for step in range(1, seq_len):
                broadcast = score.unsqueeze(2) + self.transitions.unsqueeze(0)
                best_score, best_prev = broadcast.max(dim=1)
                new_score = best_score + emissions[:, step]
                mask_step = mask[:, step].unsqueeze(1)
                score = mask_step * new_score + (1.0 - mask_step) * score
                history.append(best_prev)
            final_score = score + self.end_transitions.unsqueeze(0)
            seq_lens = mask.sum(dim=1).long()
            result: list[list[int]] = []
            for b in range(batch):
                length = int(seq_lens[b].item())
                if length <= 0:
                    result.append([])
                    continue
                last_tag = int(final_score[b].argmax().item())
                tags_rev = [last_tag]
                for step_back in range(length - 1, 0, -1):
                    prev = int(history[step_back - 1][b, tags_rev[-1]].item())
                    tags_rev.append(prev)
                tags_rev.reverse()
                result.append(tags_rev)
            return result

    class PCCharBERT(nn.Module):
        """Char-level BERT with per-char per-slot phoneme + H/L heads and APBP head.

        - ``phon_head``: ``Linear(H, S * V)`` reshaped to ``(B, L, S, V)``
        - ``hl_head``: ``Linear(H, S * K)`` reshaped to ``(B, L, S, K)``
          where ``K = config.hl_vocab_size`` (default 2).
        - ``apbp_head``: ``Linear(H, 3)`` for BIO tags
        - ``crf``: :class:`LinearChainCRF` with ``num_tags=3`` when
          ``head_variant == 'C2'``, else ``None``.

        Loss (when labels are provided): phoneme CE (label smoothing) +
        optional H/L CE + ``apbp_alpha`` * APBP loss (CRF NLL for C2,
        cross-entropy for C1). Padding positions in per-slot labels carry
        ``config.label_pad_id`` (default ``-100``, torch's default
        ``ignore_index``) and are excluded from the loss.
        """

        def __init__(self, config: PCConfig, encoder: nn.Module | None = None) -> None:
            super().__init__()
            self.config = config
            self.encoder = encoder if encoder is not None else self._build_encoder(config)
            hidden = config.encoder_hidden
            slots = config.max_slot
            vocab = config.phoneme_vocab_size
            hl_vocab = config.hl_vocab_size
            self.phon_head = nn.Linear(hidden, slots * vocab)
            self.hl_head = nn.Linear(hidden, slots * hl_vocab)
            self.apbp_head = nn.Linear(hidden, 3)
            if config.head_variant == "C2":
                self.crf: LinearChainCRF | None = LinearChainCRF(3)
            else:
                self.crf = None

        @staticmethod
        def _build_encoder(config: PCConfig) -> nn.Module:
            if config.encoder_name == "tiny":
                raise ValueError(
                    "PCConfig.tiny() requires an explicit encoder argument. "
                    "Pass a mock encoder that returns .last_hidden_state."
                )
            from transformers import AutoModel

            return AutoModel.from_pretrained(config.encoder_name)

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            phoneme_labels: torch.Tensor | None = None,
            hl_labels: torch.Tensor | None = None,
            apbp_labels: torch.Tensor | None = None,
        ) -> dict[str, torch.Tensor]:
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            hidden = _extract_last_hidden(outputs)
            batch, seq_len, hidden_dim = hidden.shape
            if hidden_dim != self.config.encoder_hidden:
                raise ValueError(
                    f"encoder hidden {hidden_dim} does not match "
                    f"config.encoder_hidden={self.config.encoder_hidden}"
                )
            slots = self.config.max_slot
            vocab = self.config.phoneme_vocab_size
            hl_vocab = self.config.hl_vocab_size

            phon_logits = self.phon_head(hidden).reshape(batch, seq_len, slots, vocab)
            hl_logits = self.hl_head(hidden).reshape(batch, seq_len, slots, hl_vocab)
            apbp_logits = self.apbp_head(hidden)

            result: dict[str, torch.Tensor] = {
                "phon_logits": phon_logits,
                "hl_logits": hl_logits,
                "apbp_logits": apbp_logits,
            }
            if phoneme_labels is not None:
                result["loss"] = self._compute_loss(
                    phon_logits=phon_logits,
                    hl_logits=hl_logits,
                    apbp_logits=apbp_logits,
                    phoneme_labels=phoneme_labels,
                    hl_labels=hl_labels,
                    apbp_labels=apbp_labels,
                    attention_mask=attention_mask,
                )
            return result

        def _compute_loss(
            self,
            *,
            phon_logits: torch.Tensor,
            hl_logits: torch.Tensor,
            apbp_logits: torch.Tensor,
            phoneme_labels: torch.Tensor,
            hl_labels: torch.Tensor | None,
            apbp_labels: torch.Tensor | None,
            attention_mask: torch.Tensor,
        ) -> torch.Tensor:
            eps = self.config.label_smoothing
            alpha = self.config.apbp_alpha
            vocab = self.config.phoneme_vocab_size
            hl_vocab = self.config.hl_vocab_size
            pad_id = self.config.label_pad_id

            phon_loss = F.cross_entropy(
                phon_logits.reshape(-1, vocab),
                phoneme_labels.reshape(-1).long(),
                ignore_index=pad_id,
                label_smoothing=eps,
            )
            loss = phon_loss

            if hl_labels is not None:
                hl_loss = F.cross_entropy(
                    hl_logits.reshape(-1, hl_vocab),
                    hl_labels.reshape(-1).long(),
                    ignore_index=pad_id,
                    label_smoothing=eps,
                )
                loss = loss + hl_loss

            if apbp_labels is not None:
                if self.crf is not None:
                    mask = attention_mask.to(dtype=apbp_logits.dtype)
                    safe_tags = apbp_labels.long().clamp(min=0)
                    crf_nll = self.crf(apbp_logits, safe_tags, mask)
                    loss = loss + alpha * crf_nll
                else:
                    apbp_loss = F.cross_entropy(
                        apbp_logits.reshape(-1, 3),
                        apbp_labels.reshape(-1).long(),
                        ignore_index=pad_id,
                        label_smoothing=eps,
                    )
                    loss = loss + alpha * apbp_loss
            return loss

        def count_params(self) -> int:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _extract_last_hidden(outputs: Any) -> torch.Tensor:
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None and isinstance(outputs, dict):
            hidden = outputs.get("last_hidden_state")
        if hidden is None and isinstance(outputs, (tuple, list)) and outputs:
            hidden = outputs[0]
        if hidden is None:
            raise TypeError(
                "encoder output must expose 'last_hidden_state' or be a tuple whose "
                "first element is a (B, L, H) tensor"
            )
        return hidden

    return {"LinearChainCRF": LinearChainCRF, "PCCharBERT": PCCharBERT}


def _get_classes() -> dict[str, type]:
    global _CLASS_CACHE
    if _CLASS_CACHE is None:
        _CLASS_CACHE = _build_classes()
    return _CLASS_CACHE


def __getattr__(name: str) -> object:
    if name in {"LinearChainCRF", "PCCharBERT"}:
        return _get_classes()[name]
    raise AttributeError(f"module 'modernbert_g2p.models.p_c.model' has no attribute {name!r}")


def build_p_c(config: PCConfig, encoder: Any | None = None) -> Any:
    """Instantiate a :class:`PCCharBERT` from ``config``.

    Passing ``encoder`` bypasses the HuggingFace hub call and is required for
    :meth:`PCConfig.tiny` in unit tests.
    """
    classes = _get_classes()
    return classes["PCCharBERT"](config, encoder=encoder)
