"""ModernBERT + morpheme-pool heads for pilot P-B (MeCab + [MORPH] boundary).

MVP simplification (P-B, Phase 2)
=================================

For the Phase 2 MVP, ``morph_positions`` (B, M) is an index tensor pointing at
the FIRST subword of each morpheme and ``morph_lens`` (B, M) counts the number
of subwords in that morpheme. The pooled representation of morpheme ``i`` is
the concatenation of ``hidden[morph_positions[i]]`` (single-subword gather)
with ``hidden[morph_positions[i] + morph_lens[i]]`` (the ``[MORPH]`` sentinel
token that immediately follows). Phase 3 will replace the first gather with a
mean pool over ``[morph_positions[i], morph_positions[i] + morph_lens[i])``
subwords, using the tokenizer's attention mask; the ``(B, M)`` semantics of
``morph_positions`` and ``morph_lens`` are preserved.

TODO(Phase 3): swap the first gather for a masked mean-pool over the full
morpheme span and align with :class:`PBTokenizer` extras from track T2.

Build-time invariant
--------------------
:func:`build_p_b` REQUIRES the runtime ``[MORPH]`` token id (resolved by the
P-B tokenizer after ``resize_token_embeddings``) to be threaded in. The
``PBConfig.morph_token_id`` default is a safe placeholder (``0``); production
call sites (trainer factory in track T7, ``build_smoke_pipeline`` etc.) must
pass ``morph_token_id=tokenizer.morph_id`` so the ``[MORPH]`` embedding row
is Xavier-initialised at the correct index.

This module imports ``torch`` at module load; the package :mod:`p_b.__init__`
loads it lazily via ``__getattr__`` so config/mecab imports stay torch-free.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as func

from modernbert_g2p.models.p_b.config import PBConfig


class _TinyEncoder(nn.Module):
    """Self-contained tiny encoder used when ``config.encoder_name == "tiny"``.

    Emits a HuggingFace-style ``last_hidden_state`` attribute so downstream code
    consumes it identically to a real ModernBERT encoder.
    """

    def __init__(self, hidden: int, vocab_size: int, max_pos: int = 512) -> None:
        super().__init__()
        self.hidden_size = hidden
        self.embed = nn.Embedding(vocab_size, hidden)
        self.pos_embed = nn.Embedding(max_pos, hidden)
        n_heads = max(1, min(4, hidden // 8))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=n_heads,
            dim_feedforward=hidden * 2,
            batch_first=True,
            norm_first=True,
            dropout=0.0,
        )
        self.layer = nn.TransformerEncoder(encoder_layer, num_layers=1, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden)

    def resize_token_embeddings(self, new_size: int) -> None:
        old = self.embed
        if new_size == old.num_embeddings:
            return
        new_embed = nn.Embedding(new_size, old.embedding_dim)
        with torch.no_grad():
            keep = min(new_size, old.num_embeddings)
            new_embed.weight[:keep] = old.weight[:keep]
        self.embed = new_embed

    def get_input_embeddings(self) -> nn.Module:
        return self.embed

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        hidden = self.embed(input_ids) + self.pos_embed(positions)
        key_padding_mask: torch.Tensor | None = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0
        hidden = self.layer(hidden, src_key_padding_mask=key_padding_mask)
        hidden = self.norm(hidden)
        return SimpleNamespace(last_hidden_state=hidden)


class _TinyLSTMDecoder(nn.Module):
    """B2 head: 2-layer LSTM that emits ``max_step`` phoneme logits per morpheme."""

    def __init__(
        self,
        input_dim: int,
        hidden: int,
        phoneme_vocab_size: int,
        max_step: int,
    ) -> None:
        super().__init__()
        self.max_step = max_step
        self.hidden = hidden
        self.vocab_size = phoneme_vocab_size
        self.num_layers = 2
        self.h0_proj = nn.Linear(input_dim, hidden * self.num_layers)
        self.c0_proj = nn.Linear(input_dim, hidden * self.num_layers)
        self.phoneme_embed = nn.Embedding(phoneme_vocab_size + 1, hidden)
        self.lstm = nn.LSTM(hidden, hidden, num_layers=self.num_layers, batch_first=True)
        self.out_proj = nn.Linear(hidden, phoneme_vocab_size)

    def _initial_state(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n = context.size(0)
        h0 = self.h0_proj(context).view(n, self.num_layers, self.hidden).transpose(0, 1)
        c0 = self.c0_proj(context).view(n, self.num_layers, self.hidden).transpose(0, 1)
        return torch.tanh(h0).contiguous(), torch.tanh(c0).contiguous()

    def forward(
        self,
        context: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        n = context.size(0)
        state = self._initial_state(context)
        if labels is not None:
            shifted = torch.full_like(labels, self.vocab_size)
            shifted[:, 1:] = labels[:, :-1].clamp(min=0, max=self.vocab_size - 1)
            embeds = self.phoneme_embed(shifted)
            out, _ = self.lstm(embeds, state)
            return self.out_proj(out)

        step_input = torch.full(
            (n, 1), self.vocab_size, dtype=torch.long, device=context.device
        )
        logits_list: list[torch.Tensor] = []
        for _ in range(self.max_step):
            embed = self.phoneme_embed(step_input)
            out, state = self.lstm(embed, state)
            step_logits = self.out_proj(out)
            logits_list.append(step_logits)
            step_input = step_logits.argmax(dim=-1)
        return torch.cat(logits_list, dim=1)


class PBMorphBERT(nn.Module):
    """ModernBERT encoder + morpheme-pool heads for pilot P-B.

    Two head variants:

    - ``B1`` flat: ``Linear(2H, max_mora * V)`` reshaped to ``(B, M, max_mora, V)``.
    - ``B2`` LSTM: 2-layer LSTM decoder that emits ``max_mora`` steps.

    Both variants share the H/L linear head and the APBP BIO 3-class head.

    See module docstring for the MVP simplification of morpheme pooling.
    """

    def __init__(self, config: PBConfig, encoder: nn.Module | None = None) -> None:
        super().__init__()
        self.config = config
        self.encoder = encoder if encoder is not None else self._build_encoder(config)

        pooled_dim = config.encoder_hidden * 2
        self.phon_head: nn.Module
        if config.head_variant == "B1":
            self.phon_head = nn.Linear(
                pooled_dim, config.max_mora_per_morph * config.phoneme_vocab_size
            )
        elif config.head_variant == "B2":
            self.phon_head = _TinyLSTMDecoder(
                input_dim=pooled_dim,
                hidden=config.lstm_hidden,
                phoneme_vocab_size=config.phoneme_vocab_size,
                max_step=config.max_mora_per_morph,
            )
        else:
            raise ValueError(f"Unknown head_variant: {config.head_variant}")

        self.hl_head = nn.Linear(pooled_dim, config.max_mora_per_morph * 2)
        self.apbp_head = nn.Linear(pooled_dim, 3)

    @staticmethod
    def _build_encoder(config: PBConfig) -> nn.Module:
        if config.encoder_name == "tiny":
            return _TinyEncoder(
                hidden=config.encoder_hidden,
                vocab_size=config.encoder_vocab_size,
            )
        from transformers import AutoModel

        return AutoModel.from_pretrained(config.encoder_name)

    def _pool_morphemes(
        self,
        hidden_states: torch.Tensor,
        morph_positions: torch.Tensor,
        morph_lens: torch.Tensor,
    ) -> torch.Tensor:
        _, seq_len, dim = hidden_states.shape
        clamped_pos = morph_positions.clamp(min=0, max=seq_len - 1)
        next_positions = (morph_positions + morph_lens.clamp(min=1)).clamp(
            min=0, max=seq_len - 1
        )
        idx_pos = clamped_pos.unsqueeze(-1).expand(-1, -1, dim)
        idx_next = next_positions.unsqueeze(-1).expand(-1, -1, dim)
        pooled = torch.gather(hidden_states, dim=1, index=idx_pos)
        next_pooled = torch.gather(hidden_states, dim=1, index=idx_next)
        return torch.cat([pooled, next_pooled], dim=-1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        morph_positions: torch.Tensor,
        morph_lens: torch.Tensor,
        morph_labels_phon: torch.Tensor | None = None,
        morph_labels_hl: torch.Tensor | None = None,
        morph_labels_apbp: torch.Tensor | None = None,
        dict_hit_mask: torch.Tensor | None = None,
        sample_weights: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        encoder_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = encoder_out.last_hidden_state
        pooled = self._pool_morphemes(hidden_states, morph_positions, morph_lens)

        b, m, _ = pooled.shape
        cfg = self.config
        max_mora = cfg.max_mora_per_morph
        phoneme_vocab = cfg.phoneme_vocab_size

        if cfg.head_variant == "B1":
            phon_logits = self.phon_head(pooled).view(b, m, max_mora, phoneme_vocab)
        else:
            context = pooled.view(b * m, -1)
            labels_flat = (
                morph_labels_phon.view(b * m, max_mora)
                if morph_labels_phon is not None
                else None
            )
            logits_flat = self.phon_head(context, labels_flat)
            phon_logits = logits_flat.view(b, m, max_mora, phoneme_vocab)

        hl_logits = self.hl_head(pooled).view(b, m, max_mora, 2)
        apbp_logits = self.apbp_head(pooled)

        loss: torch.Tensor | None = None
        has_any_label = (
            morph_labels_phon is not None
            or morph_labels_hl is not None
            or morph_labels_apbp is not None
        )
        if has_any_label:
            device = phon_logits.device
            loss = torch.zeros((), device=device, dtype=phon_logits.dtype)

            morph_weight = torch.ones((b, m), device=device, dtype=phon_logits.dtype)
            if dict_hit_mask is not None:
                morph_weight = torch.where(
                    dict_hit_mask.bool(),
                    torch.full_like(morph_weight, cfg.dict_hit_weight),
                    morph_weight,
                )
            if sample_weights is not None:
                morph_weight = morph_weight * sample_weights.to(device=device, dtype=morph_weight.dtype).view(b, 1)

            if morph_labels_phon is not None:
                loss = loss + self._per_morph_ce(
                    phon_logits.reshape(-1, phoneme_vocab),
                    morph_labels_phon,
                    shape=(b, m, max_mora),
                    weight=morph_weight,
                    label_smoothing=cfg.label_smoothing,
                    alpha=1.0,
                )

            if morph_labels_hl is not None:
                loss = loss + self._per_morph_ce(
                    hl_logits.reshape(-1, 2),
                    morph_labels_hl,
                    shape=(b, m, max_mora),
                    weight=morph_weight,
                    label_smoothing=cfg.label_smoothing,
                    alpha=cfg.apbp_alpha,
                )

            if morph_labels_apbp is not None:
                loss = loss + self._per_morph_flat_ce(
                    apbp_logits.reshape(-1, 3),
                    morph_labels_apbp,
                    weight=morph_weight,
                    alpha=cfg.apbp_alpha,
                )

        return {
            "loss": loss,
            "phon_logits": phon_logits,
            "hl_logits": hl_logits,
            "apbp_logits": apbp_logits,
        }

    @staticmethod
    def _per_morph_ce(
        flat_logits: torch.Tensor,
        labels: torch.Tensor,
        *,
        shape: tuple[int, int, int],
        weight: torch.Tensor,
        label_smoothing: float,
        alpha: float,
    ) -> torch.Tensor:
        per_slot = func.cross_entropy(
            flat_logits,
            labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
            label_smoothing=label_smoothing,
        )
        b, m, max_mora = shape
        per_slot = per_slot.view(b, m, max_mora)
        valid = (labels != -100).to(per_slot.dtype)
        slot_count = valid.sum(dim=-1).clamp(min=1.0)
        per_morph = (per_slot * valid).sum(dim=-1) / slot_count
        active = (valid.sum(dim=-1) > 0).to(per_slot.dtype)
        weighted = per_morph * weight * active
        denom = active.sum().clamp(min=1.0)
        return alpha * weighted.sum() / denom

    @staticmethod
    def _per_morph_flat_ce(
        flat_logits: torch.Tensor,
        labels: torch.Tensor,
        *,
        weight: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        per = func.cross_entropy(
            flat_logits,
            labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        )
        b, m = labels.shape
        per = per.view(b, m)
        valid = (labels != -100).to(per.dtype)
        weighted = per * weight * valid
        denom = valid.sum().clamp(min=1.0)
        return alpha * weighted.sum() / denom

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_p_b(config: PBConfig, morph_token_id: int | None = None) -> PBMorphBERT:
    """Factory. Constructs a :class:`PBMorphBERT` and Xavier-inits the ``[MORPH]`` row.

    ``morph_token_id`` overrides ``config.morph_token_id`` when given (via
    :func:`dataclasses.replace`). New embedding rows are Xavier-initialised so
    the ``[MORPH]`` special token starts with a non-degenerate representation.

    Callers (e.g. the trainer factory built in track T7) MUST resolve the
    runtime ``[MORPH]`` id from the P-B tokenizer and thread it in via
    ``build_p_b(cfg, morph_token_id=tokenizer.morph_id)`` — the ``PBConfig``
    default is a safe placeholder, not a production value, and any negative
    id is rejected here to keep the resize/init path from silently no-op-ing.
    """
    if morph_token_id is not None:
        config = replace(config, morph_token_id=morph_token_id)
    if config.morph_token_id < 0:
        raise ValueError(
            "PBConfig.morph_token_id must be a non-negative token id; "
            "pass the tokenizer's [MORPH] id via "
            "build_p_b(cfg, morph_token_id=tokenizer.morph_id). "
            f"Got morph_token_id={config.morph_token_id}."
        )
    model = PBMorphBERT(config)

    target_size = max(config.morph_token_id + 1, config.encoder_vocab_size)
    encoder = model.encoder
    embed_size = _current_embed_size(encoder)
    if embed_size is not None and target_size > embed_size and hasattr(
        encoder, "resize_token_embeddings"
    ):
        encoder.resize_token_embeddings(target_size)

    embed = _current_embed(encoder)
    if embed is not None and 0 <= config.morph_token_id < embed.num_embeddings:
        with torch.no_grad():
            row = embed.weight[config.morph_token_id : config.morph_token_id + 1]
            nn.init.xavier_uniform_(row)
    return model


def _current_embed(encoder: nn.Module) -> nn.Embedding | None:
    embed = getattr(encoder, "embed", None)
    if isinstance(embed, nn.Embedding):
        return embed
    get = getattr(encoder, "get_input_embeddings", None)
    if callable(get):
        maybe = get()
        if isinstance(maybe, nn.Embedding):
            return maybe
    return None


def _current_embed_size(encoder: nn.Module) -> int | None:
    embed = _current_embed(encoder)
    if embed is None:
        return None
    return embed.num_embeddings
