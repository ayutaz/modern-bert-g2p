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
    pad_class_weight: float = 1.0
    """Multiplicative weight applied to the phoneme *pad class* (index 0) in the
    phoneme-head CE loss.

    Purpose: fight the P-C insertion pathology where the per-slot phoneme
    head (``Linear(H, S*V)`` → reshape ``(B, L, S, V)``) receives no gradient
    on empty slots because the collator marks them with ``-100`` (see design
    doc ``P-C Insertion Penalty 設計分析`` §2). Coupled with a collator that
    labels *empty in-token slots* with class ``0`` (the phoneme pad symbol)
    while continuing to mark out-of-sequence padding with ``label_pad_id``,
    raising ``pad_class_weight`` above ``1.0`` increases the loss penalty
    when the model fails to emit ``<pad>`` on empty slots — pushing it to
    suppress spurious phonemes and reducing insertion-driven PER.

    Directionality (per review-1 §B1): ``F.cross_entropy(weight=w)``
    multiplies the loss *on samples whose target is class c* by ``w[c]``.
    So ``pad_class_weight > 1.0`` amplifies the pressure to predict pad
    (fewer insertions); ``pad_class_weight < 1.0`` relaxes it (more
    insertions). Default ``1.0`` is a no-op — no weight tensor is built and
    behaviour is byte-identical to the pre-knob code path.

    Range: must be ``>= 0``. Suggested sweep once the collator changes land:
    ``{2.0, 3.0, 5.0}``.
    """

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
        if self.pad_class_weight < 0.0:
            raise ValueError(
                f"pad_class_weight must be >= 0, got {self.pad_class_weight}"
            )

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
