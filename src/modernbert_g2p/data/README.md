# `modernbert_g2p.data` — Phase 1 data pipeline module

This module hosts the ingestion, normalization, deduplication, contamination
check, and Parquet shard writers that produce the fine-tuning corpus for
Phase 2 onwards. It is currently a namespace-only placeholder; all
implementation is deferred to Phase 1 execution.

## Authoritative design

See `docs/design/phase1_data_pipeline.md` — this is the single source of
truth. Do not add ingestion or normalization logic here without keeping the
design doc in sync.

## Scope

| Concern | Design doc section |
|---|---|
| 5 primary sources + license tags | §1, §2 |
| Per-source ingestion | §3 |
| Canonical normalization (JULIUS + H/L + boundaries) | §4 |
| Dedup (exact + MinHash near-dup) | §5.1 |
| Quality filter | §5.2 |
| Train/val/dev-test/held-out split | §6.1 |
| Contamination prevention (4-layer defense) | §6.2 |
| Sample weighting (Phase 2 empirical) | §7 |
| Output Parquet schema | §8 |
| Storage layout | §9 |

## Related code

- `src/modernbert_g2p/metrics/` — canonical PER / kana CER / KER used for
  evaluation and for regression tests on the pipeline output.
- `scripts/eval_haqumei_jsut.py` — reference implementation of the
  normalization order; Phase 0.5 will extract its normalization helpers into
  `src/modernbert_g2p/normalize.py` so this module can import them directly
  (design doc §4, F4 fix).
- `data/hard_set_seed/schema.json` — parent JSON schema; the Phase 1 output
  schema extends it via `$ref` (design doc §8).

## Non-goals

- No LLM-based data augmentation in Phase 1 (deferred to Phase 5 or later,
  subject to license review).
- No speech-side processing (this project's scope is text-only G2P).
- No `sample_weight` column baked into Parquet — categories are stored,
  weights are computed at train time (design doc §7).

## Status

- [ ] `normalize.py` (Phase 0.5 prep, imported by this module)
- [ ] `ingest/` submodule (one file per source)
- [ ] `dedup.py`
- [ ] `contamination.py` (4-layer defense)
- [ ] `schema.py` (Parquet schema + JSON schema extension)
- [ ] `writer.py` (Parquet shard writer)
- [ ] Integration test producing 100 rows per source

Track progress in the Phase 1 execution plan (to be written under
`docs/phase1_execution_plan.md` via `superpowers:writing-plans`).
