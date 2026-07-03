"""Per-pilot tokenizer wrappers for Phase 2 (P-A / P-B / P-C).

Each wrapper produces the pilot-specific ``input_ids`` / ``attention_mask``
plus target-side labels that align with the canonical vocab defined in
:mod:`modernbert_g2p.models.canonical`. Heavy runtime dependencies
(``transformers``, ``fugashi``) are imported lazily inside methods so that
``import modernbert_g2p.models.tokenization`` remains a cheap operation
that can run in a bare unit-test process.
"""

from __future__ import annotations

from typing import Any

from modernbert_g2p.models.tokenization.base import BaseTokenizer, EncodedTarget
from modernbert_g2p.models.tokenization.p_a_tokenizer import PATokenizer
from modernbert_g2p.models.tokenization.p_b_tokenizer import PBTokenizer
from modernbert_g2p.models.tokenization.p_c_tokenizer import PCTokenizer

_PILOT_ALIASES: dict[str, str] = {
    "p_a": "p_a", "pa": "p_a", "p-a": "p_a",
    "p_b": "p_b", "pb": "p_b", "p-b": "p_b",
    "p_c": "p_c", "pc": "p_c", "p-c": "p_c",
}


def get_tokenizer(pilot: str, **kwargs: Any) -> BaseTokenizer:
    """Return the tokenizer instance appropriate for ``pilot``.

    Args:
        pilot: One of ``"p_a"`` / ``"p_b"`` / ``"p_c"`` (case-insensitive;
            dashes and missing underscores are also accepted).
        **kwargs: Forwarded to the underlying tokenizer's constructor.

    Raises:
        ValueError: If ``pilot`` does not match any known pilot label.
    """
    key = pilot.lower().strip()
    normalized = _PILOT_ALIASES.get(key)
    if normalized == "p_a":
        return PATokenizer(**kwargs)
    if normalized == "p_b":
        return PBTokenizer(**kwargs)
    if normalized == "p_c":
        return PCTokenizer(**kwargs)
    raise ValueError(
        f"get_tokenizer: unknown pilot {pilot!r}; expected one of 'p_a', 'p_b', 'p_c'"
    )


__all__ = [
    "BaseTokenizer",
    "EncodedTarget",
    "PATokenizer",
    "PBTokenizer",
    "PCTokenizer",
    "get_tokenizer",
]
