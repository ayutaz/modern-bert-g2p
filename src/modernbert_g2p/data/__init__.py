"""Data pipeline module (Phase 1 target — not yet implemented).

This module will host ingestion, normalization, deduplication, contamination
check, and Parquet shard writers for the 5 primary data sources
(pyopenjtalk-plus dict, UniDic-cwj 3.1.1, JMDict, Wikipedia JA HTML dump,
Aozora Bunko).

The authoritative design is:
    docs/design/phase1_data_pipeline.md

Any new submodule added here MUST cite the section of that design doc it
implements, and MUST NOT diverge from the canonical normalization order
defined there (§4) without updating the design doc first.
"""

__all__: list[str] = []
