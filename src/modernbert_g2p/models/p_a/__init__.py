"""P-A seq2seq pilot: ModernBERT encoder + from-scratch Transformer decoder."""

from __future__ import annotations

from modernbert_g2p.models.p_a.config import PAConfig

__all__ = ["PAConfig", "PASeq2Seq", "TinyEncoder", "build_p_a"]


def __getattr__(name: str) -> object:
    if name in ("PASeq2Seq", "TinyEncoder", "build_p_a"):
        from modernbert_g2p.models.p_a import model as _model

        if name == "build_p_a":
            return _model.build_p_a
        return getattr(_model, name)
    raise AttributeError(f"module 'modernbert_g2p.models.p_a' has no attribute {name!r}")
