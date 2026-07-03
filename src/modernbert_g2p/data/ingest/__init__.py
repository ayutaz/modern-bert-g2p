"""Ingest source registry — see base.py for the Protocol.

Implements docs/design/phase1_data_pipeline.md §3.
"""

from __future__ import annotations

import importlib
import warnings

from modernbert_g2p.data.ingest.base import (
    SOURCE_REGISTRY,
    IngestSource,
    get_source,
    known_sources,
    register,
)

_SIBLING_MODULES: tuple[str, ...] = (
    "pyopenjtalk_plus",
    "unidic",
    "jmdict",
    "wikipedia",
    "aozora",
)


def _load_sibling_sources() -> None:
    for name in _SIBLING_MODULES:
        try:
            importlib.import_module(f"modernbert_g2p.data.ingest.{name}")
        except ImportError as exc:
            warnings.warn(
                f"Ingest source module '{name}' not importable ({exc}); "
                "it will be unavailable until implemented.",
                stacklevel=2,
            )


_load_sibling_sources()


__all__ = [
    "IngestSource",
    "SOURCE_REGISTRY",
    "get_source",
    "known_sources",
    "register",
]
