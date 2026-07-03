"""P-C pilot — char-level BERT (tohoku-nlp/bert-base-japanese-char-v2).

Reference: docs/design/phase2_tokenizer_pilots.md §3.3.
License note: default encoder tohoku-nlp/bert-base-japanese-char-v2 is
CC-BY-SA-4.0; weight-redistribution downstream must consider Share-Alike.
"""

from __future__ import annotations

from modernbert_g2p.models.p_c.config import PCConfig
from modernbert_g2p.models.p_c.model import build_p_c

__all__ = ["PCConfig", "build_p_c"]


def __getattr__(name: str) -> object:
    if name in {"LinearChainCRF", "PCCharBERT"}:
        from modernbert_g2p.models.p_c import model as _m

        return getattr(_m, name)
    raise AttributeError(f"module 'modernbert_g2p.models.p_c' has no attribute {name!r}")
