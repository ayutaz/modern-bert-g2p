"""Configuration dataclass for pilot P-B (MeCab + [MORPH] + ModernBERT)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PBConfig:
    """Hyperparameters for the P-B pilot.

    Design ref: ``docs/design/phase2_tokenizer_pilots.md`` §3.2.

    ``morph_token_id`` defaults to ``0`` (a safe placeholder) so that omitting
    an explicit override never triggers the -100/CE-ignore path silently in
    :func:`modernbert_g2p.models.p_b.model.build_p_b`. Callers MUST override
    it with the runtime ``[MORPH]`` id resolved from the tokenizer via
    ``build_p_b(cfg, morph_token_id=tokenizer.morph_id)``.
    """

    encoder_name: str = "sbintuitions/modernbert-ja-130m"
    head_variant: str = "B1"
    max_mora_per_morph: int = 8
    phoneme_vocab_size: int = 68
    apbp_alpha: float = 0.2
    label_smoothing: float = 0.05
    dict_hit_weight: float = 0.3
    morph_token: str = "[MORPH]"
    morph_token_id: int = 0
    encoder_hidden: int = 512
    encoder_vocab_size: int = 102400
    lstm_hidden: int = 256

    @classmethod
    def tiny(cls, head_variant: str = "B1") -> PBConfig:
        """Return a CPU-friendly config for unit tests."""
        return cls(
            encoder_name="tiny",
            head_variant=head_variant,
            encoder_hidden=32,
            phoneme_vocab_size=16,
            max_mora_per_morph=4,
            encoder_vocab_size=128,
            lstm_hidden=16,
            morph_token_id=15,
        )
