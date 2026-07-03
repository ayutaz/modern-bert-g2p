"""Phase 1 data pipeline — ingestion, normalization, dedup, contamination, split.

Implements docs/design/phase1_data_pipeline.md.

Any new submodule added here MUST cite the section of that design doc it
implements, and MUST NOT diverge from the canonical normalization order
defined there (§4) without updating the design doc first.
"""

from __future__ import annotations

from modernbert_g2p.data.contamination import (
    BloomFilter,
    ContaminationFilter,
    load_held_out_texts,
)
from modernbert_g2p.data.dedup import Deduper, phoneme_hash, text_hash
from modernbert_g2p.data.ingest import (
    SOURCE_REGISTRY,
    IngestSource,
    get_source,
    known_sources,
    register,
)
from modernbert_g2p.data.normalize import (
    NormalizedText,
    devoice_phonemes,
    normalize_from_haqumei,
)
from modernbert_g2p.data.output import (
    HAS_PYARROW,
    read_jsonl,
    write_jsonl,
    write_manifest,
    write_parquet,
)
from modernbert_g2p.data.schema import (
    SCHEMA_VERSION,
    Row,
    from_dict,
    to_dict,
    validate_row,
)
from modernbert_g2p.data.split import SPLITS, deterministic_split, split_corpus
from modernbert_g2p.data.weighting import (
    CATEGORY_WEIGHTS,
    apply_weight,
    weight_for,
)

compute_weight = weight_for
annotate_row_weight = apply_weight


__all__ = [
    "BloomFilter",
    "CATEGORY_WEIGHTS",
    "ContaminationFilter",
    "Deduper",
    "HAS_PYARROW",
    "IngestSource",
    "NormalizedText",
    "Row",
    "SCHEMA_VERSION",
    "SOURCE_REGISTRY",
    "SPLITS",
    "annotate_row_weight",
    "apply_weight",
    "compute_weight",
    "deterministic_split",
    "devoice_phonemes",
    "from_dict",
    "get_source",
    "known_sources",
    "load_held_out_texts",
    "normalize_from_haqumei",
    "phoneme_hash",
    "read_jsonl",
    "register",
    "split_corpus",
    "text_hash",
    "to_dict",
    "validate_row",
    "weight_for",
    "write_jsonl",
    "write_manifest",
    "write_parquet",
]
