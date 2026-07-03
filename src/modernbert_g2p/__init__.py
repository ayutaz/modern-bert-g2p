"""ModernBERT-based Japanese Grapheme-to-Phoneme (G2P) toolkit.

Public API surface:

- :mod:`modernbert_g2p.metrics` — canonical PER / kana CER / KER implementations.
- :mod:`modernbert_g2p.data` — Phase 1 data pipeline (Row schema, source
  ingestors, contamination filters, Parquet + manifest writer).
- :mod:`modernbert_g2p.models` — Phase 2 pilot models
  (P-A seq2seq, P-B MeCab+[MORPH], P-C char BERT).
- :mod:`modernbert_g2p.training` — Trainer / collators / loss / optimizer.
- :mod:`modernbert_g2p.evaluation` — bootstrap CI + dataset-specific scorers.
- :func:`load_config` — YAML → :class:`Phase2Config` loader.
- :func:`main` — ``python -m modernbert_g2p`` CLI entry point.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "0.1.0-alpha.0"

if TYPE_CHECKING:
    from modernbert_g2p.cli import main
    from modernbert_g2p.config import load_config

__all__ = ["__version__", "load_config", "main"]


def __getattr__(name: str) -> Any:
    if name == "load_config":
        from modernbert_g2p.config import load_config

        return load_config
    if name == "main":
        from modernbert_g2p.cli import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
