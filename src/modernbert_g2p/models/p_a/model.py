"""PASeq2Seq model definition (ModernBERT encoder + Transformer decoder).

All ``torch`` / ``transformers`` imports are deferred: the ``nn.Module`` subclasses
are constructed inside :func:`_define_pa_classes` and exposed through module-level
``__getattr__``, so importing this module does not require the training stack.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from modernbert_g2p.models.p_a.config import PAConfig

if TYPE_CHECKING:
    import torch


_class_cache: dict[str, type] = {}


def _define_pa_classes() -> tuple[type, type]:
    import math
    from types import SimpleNamespace

    import torch
    from torch import nn

    class SinusoidalPositionalEncoding(nn.Module):
        """Fixed sinusoidal PE — safe for target lengths exceeding training max."""

        def __init__(self, dim: int, max_len: int = 8192) -> None:
            super().__init__()
            pe = torch.zeros(max_len, dim)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, dim, 2, dtype=torch.float)
                * (-math.log(10000.0) / dim)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.pe[:, : x.size(1)].to(x.dtype)

    class TinyEncoder(nn.Module):
        """Random-init Transformer encoder used only for CPU tests.

        Provides an HF-compatible ``forward`` returning a namespace with
        ``last_hidden_state`` so :class:`PASeq2Seq` can treat it identically to
        a real ModernBERT encoder loaded via ``AutoModel.from_pretrained``.
        """

        def __init__(
            self,
            hidden: int = 32,
            layers: int = 1,
            heads: int = 2,
            ffn: int = 64,
            vocab_size: int = 1000,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.hidden_size = hidden
            self.embed = nn.Embedding(vocab_size, hidden)
            self.pos = SinusoidalPositionalEncoding(hidden)
            layer = nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=heads,
                dim_feedforward=ffn,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.enc = nn.TransformerEncoder(
                layer, num_layers=layers, enable_nested_tensor=False
            )

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            **_: object,
        ) -> Any:
            x = self.pos(self.embed(input_ids))
            key_padding_mask = None
            if attention_mask is not None:
                key_padding_mask = ~attention_mask.bool()
            h = self.enc(x, src_key_padding_mask=key_padding_mask)
            return SimpleNamespace(last_hidden_state=h)

    class PASeq2Seq(nn.Module):
        """ModernBERT encoder + from-scratch Transformer decoder for pilot P-A.

        Follows ``docs/design/phase2_tokenizer_pilots.md`` §3.1: encoder frozen or
        fine-tuned by the caller's optimizer, autoregressive decoder emits
        interleaved phoneme / H / L / '/' tokens over a ~68-token vocab.
        """

        def __init__(
            self,
            config: PAConfig,
            encoder: nn.Module | None = None,
        ) -> None:
            super().__init__()
            self.config = config
            self.encoder = encoder if encoder is not None else self._build_encoder(config)
            enc_hidden = self._encoder_hidden(self.encoder)
            self.enc_proj = (
                nn.Linear(enc_hidden, config.decoder_hidden)
                if enc_hidden != config.decoder_hidden
                else nn.Identity()
            )
            self.tgt_embed = nn.Embedding(
                config.phoneme_vocab_size,
                config.decoder_hidden,
                padding_idx=config.pad_id,
            )
            self.tgt_pos = SinusoidalPositionalEncoding(
                config.decoder_hidden,
                max_len=max(config.max_target_length, 512),
            )
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=config.decoder_hidden,
                nhead=config.decoder_heads,
                dim_feedforward=config.decoder_ffn,
                dropout=config.decoder_dropout,
                batch_first=True,
                norm_first=True,
            )
            self.decoder = nn.TransformerDecoder(
                decoder_layer, num_layers=config.decoder_layers
            )
            self.output_head = nn.Linear(config.decoder_hidden, config.phoneme_vocab_size)
            self.output_head.weight = self.tgt_embed.weight

        @staticmethod
        def _build_encoder(config: PAConfig) -> nn.Module:
            if config.encoder_name == "tiny":
                return TinyEncoder(
                    hidden=config.tiny_encoder_hidden,
                    layers=config.tiny_encoder_layers,
                    heads=config.tiny_encoder_heads,
                    ffn=config.tiny_encoder_ffn,
                    vocab_size=config.tiny_encoder_vocab_size,
                )
            from transformers import AutoModel

            return AutoModel.from_pretrained(config.encoder_name)

        @staticmethod
        def _encoder_hidden(encoder: nn.Module) -> int:
            hidden = getattr(encoder, "hidden_size", None)
            if hidden is not None:
                return int(hidden)
            cfg = getattr(encoder, "config", None)
            if cfg is not None and hasattr(cfg, "hidden_size"):
                return int(cfg.hidden_size)
            raise AttributeError("Cannot determine encoder hidden size")

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            decoder_input_ids: torch.Tensor,
            decoder_attention_mask: torch.Tensor | None = None,
            labels: torch.Tensor | None = None,
        ) -> dict[str, torch.Tensor]:
            enc_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            memory = self.enc_proj(enc_out.last_hidden_state)
            memory_key_padding_mask = ~attention_mask.bool()

            tgt = self.tgt_pos(self.tgt_embed(decoder_input_ids))
            tgt_len = decoder_input_ids.size(1)
            causal_mask = torch.triu(
                torch.ones(
                    tgt_len,
                    tgt_len,
                    dtype=torch.bool,
                    device=decoder_input_ids.device,
                ),
                diagonal=1,
            )
            tgt_key_padding_mask = None
            if decoder_attention_mask is not None:
                tgt_key_padding_mask = ~decoder_attention_mask.bool()
            dec_out = self.decoder(
                tgt=tgt,
                memory=memory,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
            logits = self.output_head(dec_out)
            result: dict[str, torch.Tensor] = {"logits": logits}
            if labels is not None:
                result["loss"] = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    ignore_index=-100,
                    label_smoothing=self.config.label_smoothing,
                )
            return result

        def count_params(self) -> int:
            return sum(p.numel() for p in self.parameters())

        def generate(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            *,
            max_len: int | None = None,
            beam: int | None = None,
            length_penalty: float | None = None,
            coverage_penalty: float | None = None,
        ) -> torch.Tensor:
            """Autoregressive decode. Supports greedy (``beam=1``) and beam search.

            When ``beam > 1`` runs beam search with length normalization
            ``(5+t)^lp / 6^lp`` (Wu et al. 2016 GNMT) and optional coverage
            penalty on attention memory positions. Returns int64 ids of shape
            ``(B, L)`` without the leading ``<bos>``, right-padded with
            ``pad_id`` after each row's ``<eos>``.
            """
            resolved_max_len = max_len if max_len is not None else self.config.max_decode_len
            resolved_beam = beam if beam is not None else self.config.beam_size
            resolved_lp = (
                length_penalty if length_penalty is not None else self.config.length_penalty
            )
            resolved_cp = (
                coverage_penalty
                if coverage_penalty is not None
                else self.config.coverage_penalty
            )
            was_training = self.training
            self.eval()
            try:
                with torch.inference_mode():
                    if resolved_beam <= 1:
                        return self._greedy_decode(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_len=resolved_max_len,
                        )
                    return self._beam_decode(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_len=resolved_max_len,
                        beam=resolved_beam,
                        length_penalty=resolved_lp,
                        coverage_penalty=resolved_cp,
                    )
            finally:
                if was_training:
                    self.train()

        def _greedy_decode(
            self,
            *,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            max_len: int,
        ) -> torch.Tensor:
            enc_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            memory = self.enc_proj(enc_out.last_hidden_state)
            memory_key_padding_mask = ~attention_mask.bool()
            batch = input_ids.size(0)
            device = input_ids.device
            ys = torch.full(
                (batch, 1), self.config.bos_id, dtype=torch.long, device=device
            )
            finished = torch.zeros(batch, dtype=torch.bool, device=device)
            for _ in range(max_len):
                tgt = self.tgt_pos(self.tgt_embed(ys))
                tgt_len = ys.size(1)
                causal_mask = torch.triu(
                    torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=device),
                    diagonal=1,
                )
                dec_out = self.decoder(
                    tgt=tgt,
                    memory=memory,
                    tgt_mask=causal_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                )
                next_logits = self.output_head(dec_out[:, -1, :])
                next_token = next_logits.argmax(dim=-1)
                next_token = torch.where(
                    finished,
                    torch.full_like(next_token, self.config.pad_id),
                    next_token,
                )
                ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)
                finished = finished | (next_token == self.config.eos_id)
                if bool(finished.all()):
                    break
            return ys[:, 1:]

        def _beam_decode(
            self,
            *,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            max_len: int,
            beam: int,
            length_penalty: float,
            coverage_penalty: float,
        ) -> torch.Tensor:
            enc_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            memory = self.enc_proj(enc_out.last_hidden_state)
            batch = input_ids.size(0)
            src_len = memory.size(1)
            device = input_ids.device
            pad_id = self.config.pad_id
            eos_id = self.config.eos_id
            bos_id = self.config.bos_id
            vocab_size = self.config.phoneme_vocab_size

            expand_shape = (batch, beam, src_len, memory.size(-1))
            memory_b = memory.unsqueeze(1).expand(expand_shape).reshape(
                batch * beam, src_len, memory.size(-1)
            )
            attn_b = attention_mask.unsqueeze(1).expand(batch, beam, src_len).reshape(
                batch * beam, src_len
            )
            memory_key_padding_mask = ~attn_b.bool()

            ys = torch.full(
                (batch * beam, 1), bos_id, dtype=torch.long, device=device
            )
            scores = torch.zeros(batch, beam, device=device)
            scores[:, 1:] = float("-inf")
            scores = scores.reshape(batch * beam)
            finished = torch.zeros(batch * beam, dtype=torch.bool, device=device)
            coverage = (
                torch.zeros(batch * beam, src_len, device=device)
                if coverage_penalty > 0.0
                else None
            )

            for step in range(max_len):
                tgt = self.tgt_pos(self.tgt_embed(ys))
                tgt_len = ys.size(1)
                causal_mask = torch.triu(
                    torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=device),
                    diagonal=1,
                )
                dec_out = self.decoder(
                    tgt=tgt,
                    memory=memory_b,
                    tgt_mask=causal_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                )
                next_logits = self.output_head(dec_out[:, -1, :])
                log_probs = torch.log_softmax(next_logits, dim=-1)

                finished_mask = finished.unsqueeze(-1)
                pad_logp = torch.full_like(log_probs, float("-inf"))
                pad_logp[:, pad_id] = 0.0
                log_probs = torch.where(finished_mask, pad_logp, log_probs)

                cand_scores = scores.unsqueeze(-1) + log_probs
                cand_scores = cand_scores.view(batch, beam * vocab_size)

                cur_lens = torch.full(
                    (batch, beam), step + 1, dtype=torch.float, device=device
                )
                lp_denom = ((5.0 + cur_lens) / 6.0).pow(length_penalty)
                norm_score = cand_scores.view(batch, beam, vocab_size) / lp_denom.unsqueeze(-1)
                norm_score = norm_score.view(batch, beam * vocab_size)

                top_scores, top_idx = norm_score.topk(beam, dim=-1)
                beam_idx = top_idx // vocab_size
                token_idx = top_idx % vocab_size

                flat_beam = (
                    torch.arange(batch, device=device).unsqueeze(-1) * beam + beam_idx
                )
                flat_beam = flat_beam.reshape(-1)
                token_flat = token_idx.reshape(-1)

                raw_new_scores = cand_scores.gather(1, top_idx).reshape(-1)
                scores = raw_new_scores

                ys = ys[flat_beam]
                ys = torch.cat([ys, token_flat.unsqueeze(1)], dim=1)
                finished = finished[flat_beam] | (token_flat == eos_id)

                if coverage is not None:
                    coverage = coverage[flat_beam]

                if bool(finished.all()):
                    break

            ys = ys.view(batch, beam, -1)
            scores = scores.view(batch, beam)
            cur_lens = torch.full(
                (batch, beam), ys.size(-1) - 1, dtype=torch.float, device=device
            )
            lp_denom = ((5.0 + cur_lens) / 6.0).pow(length_penalty)
            final_scores = scores / lp_denom
            best = final_scores.argmax(dim=-1)
            batch_idx = torch.arange(batch, device=device)
            best_ys = ys[batch_idx, best]
            return best_ys[:, 1:]

    return PASeq2Seq, TinyEncoder


def _resolve(name: str) -> type:
    if not _class_cache:
        pa_cls, tiny_cls = _define_pa_classes()
        _class_cache["PASeq2Seq"] = pa_cls
        _class_cache["TinyEncoder"] = tiny_cls
    if name not in _class_cache:
        raise AttributeError(name)
    return _class_cache[name]


def __getattr__(name: str) -> object:
    if name in ("PASeq2Seq", "TinyEncoder"):
        return _resolve(name)
    raise AttributeError(name)


def build_p_a(
    config: PAConfig | None = None,
    encoder: Any | None = None,
) -> Any:
    """Instantiate a :class:`PASeq2Seq` with a default or supplied config."""
    cfg = config if config is not None else PAConfig()
    cls = _resolve("PASeq2Seq")
    return cls(cfg, encoder=encoder)


__all__ = ("build_p_a",)
