"""Model components for ModernBERT-based Japanese G2P (Phase 2).

Public API surface:

- Canonical vocab / conversion utilities:
  :class:`Vocab`, :class:`CanonicalForm`, :data:`JULIUS_PHONEMES`,
  :data:`SPECIAL_TOKENS`, :data:`PROSODY_TOKENS`, :data:`PAUSE_TOKEN`,
  :data:`DROPPED_PHONEMES`, :func:`build_default_vocab`,
  :func:`p_a_to_canonical`, :func:`p_c_to_canonical`.
- Pilot P-A (seq2seq): :class:`PAConfig`, :class:`PASeq2Seq`,
  :func:`build_p_a` — lazy re-exported so importing this package does not
  drag in ``torch`` / ``transformers`` unless the pilot classes are touched.
- Pilot P-C (char BERT): :class:`PCConfig`, :class:`PCCharBERT`,
  :func:`build_p_c` — same lazy convention.

P-B (MeCab + [MORPH]) was dropped in the v2.0 pure-NN pivot (2026-07-04):
MeCab is a rule-based morphological analyzer, so having it in the inference
path violated the pure-NN constraint the project adopted.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from modernbert_g2p.models.canonical import (
    DROPPED_PHONEMES,
    JULIUS_PHONEMES,
    PAUSE_TOKEN,
    PROSODY_TOKENS,
    SPECIAL_TOKENS,
    CanonicalForm,
    Vocab,
    build_default_vocab,
    p_a_to_canonical,
    p_c_to_canonical,
)
from modernbert_g2p.models.p_a import PAConfig
from modernbert_g2p.models.p_c import PCConfig

if TYPE_CHECKING:
    from modernbert_g2p.models.p_a import PASeq2Seq, build_p_a
    from modernbert_g2p.models.p_c import PCCharBERT, build_p_c

__all__ = [
    "DROPPED_PHONEMES",
    "JULIUS_PHONEMES",
    "PAUSE_TOKEN",
    "PROSODY_TOKENS",
    "SPECIAL_TOKENS",
    "CanonicalForm",
    "PAConfig",
    "PASeq2Seq",
    "PCCharBERT",
    "PCConfig",
    "Vocab",
    "build_default_vocab",
    "build_p_a",
    "build_p_c",
    "p_a_to_canonical",
    "p_c_to_canonical",
]


def __getattr__(name: str) -> Any:
    if name in {"PASeq2Seq", "build_p_a"}:
        from modernbert_g2p.models import p_a as _p_a

        return getattr(_p_a, name)
    if name in {"PCCharBERT", "build_p_c"}:
        from modernbert_g2p.models import p_c as _p_c

        return getattr(_p_c, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
