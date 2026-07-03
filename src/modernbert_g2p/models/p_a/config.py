"""Configuration dataclass for the P-A seq2seq pilot (ModernBERT + Transformer decoder)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PAConfig:
    """Hyperparameters for the P-A seq2seq pilot.

    Defaults match ``docs/design/phase2_tokenizer_pilots.md`` §3.1: ModernBERT-ja-130m
    encoder plus a from-scratch 6-layer Transformer decoder emitting a 68-token
    phoneme + prosody vocab (JULIUS phonemes + H/L + boundary + specials).
    """

    encoder_name: str = "sbintuitions/modernbert-ja-130m"
    decoder_layers: int = 6
    decoder_hidden: int = 512
    decoder_heads: int = 8
    decoder_ffn: int = 2048
    decoder_dropout: float = 0.1
    phoneme_vocab_size: int = 68
    label_smoothing: float = 0.1
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    max_target_length: int = 256
    max_decode_len: int = 128
    beam_size: int = 4
    length_penalty: float = 1.0
    coverage_penalty: float = 0.0
    tiny_encoder_vocab_size: int = 1000
    tiny_encoder_hidden: int = 32
    tiny_encoder_layers: int = 1
    tiny_encoder_heads: int = 2
    tiny_encoder_ffn: int = 64

    @classmethod
    def tiny(cls) -> PAConfig:
        """Return a minimal random-init config for CPU unit tests (no HF hub call)."""
        return cls(
            encoder_name="tiny",
            decoder_layers=1,
            decoder_hidden=32,
            decoder_heads=2,
            decoder_ffn=64,
            phoneme_vocab_size=16,
        )
