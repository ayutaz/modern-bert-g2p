"""Pilot P-B: MeCab pretokenize + ``[MORPH]`` boundary + ModernBERT.

Design reference: ``docs/design/phase2_tokenizer_pilots.md`` §3.2.

``PBMorphBERT`` and :func:`build_p_b` are loaded lazily on attribute access so
that importing :data:`PBConfig` (or the MeCab pretokenizer) never drags in the
PyTorch runtime.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from modernbert_g2p.models.p_b.config import PBConfig
from modernbert_g2p.models.p_b.mecab import MeCabPretokenizer, MeCabToken, build_mock_pretokenizer

if TYPE_CHECKING:
    from modernbert_g2p.models.p_b.model import PBMorphBERT, build_p_b

__all__ = [
    "MeCabPretokenizer",
    "MeCabToken",
    "PBConfig",
    "PBMorphBERT",
    "build_mock_pretokenizer",
    "build_p_b",
]


def __getattr__(name: str) -> Any:
    if name in {"PBMorphBERT", "build_p_b"}:
        from modernbert_g2p.models.p_b import model as _model

        return getattr(_model, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
