"""Configuration dataclass for the P-C char-level BERT pilot."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PCConfig:
    """Hyperparameters for :class:`PCCharBERT`.

    Defaults match docs/design/phase2_tokenizer_pilots.md §3.3.
    Use :meth:`tiny` in tests to obtain a CPU-friendly configuration paired
    with a caller-provided dummy encoder (avoids a HuggingFace hub download).
    """

    encoder_name: str = "tohoku-nlp/bert-base-japanese-char-v2"
    head_variant: str = "C1"
    max_slot: int = 8
    phoneme_vocab_size: int = 68
    hl_vocab_size: int = 2
    apbp_alpha: float = 0.2
    label_smoothing: float = 0.05
    encoder_hidden: int = 768
    label_pad_id: int = -100

    def __post_init__(self) -> None:
        if self.head_variant not in {"C1", "C2"}:
            raise ValueError(
                f"head_variant must be 'C1' or 'C2', got {self.head_variant!r}"
            )
        if self.max_slot <= 0:
            raise ValueError(f"max_slot must be positive, got {self.max_slot}")
        if self.phoneme_vocab_size <= 0:
            raise ValueError(
                f"phoneme_vocab_size must be positive, got {self.phoneme_vocab_size}"
            )
        if self.hl_vocab_size < 2:
            raise ValueError(
                f"hl_vocab_size must be >= 2, got {self.hl_vocab_size}"
            )
        if self.encoder_hidden <= 0:
            raise ValueError(f"encoder_hidden must be positive, got {self.encoder_hidden}")

    @classmethod
    def tiny(cls, head_variant: str = "C1") -> PCConfig:
        """Return a small config suitable for CPU unit tests with a mock encoder."""
        return cls(
            encoder_name="tiny",
            head_variant=head_variant,
            encoder_hidden=32,
            phoneme_vocab_size=16,
            max_slot=4,
        )
