# Hard-set Seed Curation

This directory holds the **seed** (21 fully-labeled examples) for the Phase 1
hard-set — the 7 × 200 = **1,400-sentence** human-curated evaluation set
mandated by `CLAUDE.md` §"Hard-set". The seed:

1. fixes the annotation schema (`schema.json`),
2. demonstrates the labeling conventions on 3 canonical examples per category
   (`samples.jsonl`), and
3. documents the semi-automatic curation SOP (this file) so the Phase 1 batch
   can be produced under a fixed budget with reproducible inter-annotator
   agreement targets.

Nothing here is training data. All 1,400 sentences are **eval-only** and are
excluded from every training / dev split by ID.

## Files

| File | Purpose |
|---|---|
| `schema.json` | JSON Schema Draft 2020-12 for one gold label. Enforced in CI. |
| `samples.jsonl` | 21 seed examples (7 categories × 3), all passing `schema.json`. |
| `README.md` | This file — curation SOP, agreement targets, cost budget. |

## Categories (7)

Fixed by `CLAUDE.md`. `category` enum in `schema.json` is the source of truth.

| category | Example failure mode |
|---|---|
| `polyphone` | 行った=いった / おこなった |
| `counter` | 3匹=さんびき (rendaku), 10冊=じゅっさつ (sokuon) |
| `proper_noun` | 山田 (LHH), 大阪 (HLLL) accent unpredictable from surface |
| `loanword` | コンピュータ accent 3, スマートフォン accent 2 |
| `numeric_unit` | 10km=じゅっきろ, 2025年=にせんにじゅうごねん |
| `english_mixed` | iPhone→アイフォン, Wi-Fi→ワイファイ |
| `english_abbreviation` | NASA=ナサ (word), HTML=エイチティーエムエル (spell) |

## Labeling conventions (must-read before annotating)

Follow `CLAUDE.md` §"実装時の技術的注意事項":

- **Phoneme set**: extended JULIUS. Vowels `a i u e o`; consonants
  `k g s z j t d ch ts n h b p m y r w f v`; specials `N` (moraic n),
  `q` (geminate / sokuon), `sil`, `pau`. Foreign-syllable extensions permitted:
  `f a` (ふぁ), `f o` (フォ), `d i` (でぃ), `t i` (ティ).
- **Long vowels**: canonical form is **vowel repetition**, not `:` or `ー`.
  せんせい → `s e N s e e`, けーしょん → `k e e sh o N`.
- **Mora tier ≠ phoneme tier**. A mora is CV, CyV, N, or q. `accent` length
  MUST equal mora count, not phoneme count. Enforced by the CI validator
  (see "Validation" below).
- **Accent tier**: only `H` / `L`. Compatible with pyopenjtalk's HL string.
  Accent nucleus = last H before an L → tail; 平板 = `L H H … H`; 頭高 =
  `H L L … L`.
- **Accent phrase boundaries**: 0-indexed mora positions where a **new**
  accent phrase begins. Position 0 is implicit and MUST NOT be included.
  Empty array = single accent phrase.
- **Dialect**: Tokyo Yamanote NHK accent dictionary (2016) is the reference
  when two Tokyo-standard readings compete. Record dialect assumption in
  `notes` if it affects the label.

## Curation SOP — semi-automatic C-plan

Phase 1 target: **1,400** sentences (7 × 200) labeled to the same standard as
the 21 seeds here, with **Cohen's κ ≥ 0.70** between two independent human
annotators on both the phoneme string and the H/L tier.

### Pipeline

```
                sentence pool
                     │
                     ▼
         (1) LLM pre-annotation                 ← Claude Sonnet / haqumei
                     │  proposes phonemes+accent
                     ▼
         (2) 3-way diff against
             haqumei + pyopenjtalk-plus         ← automatic
                     │  agreement, disagreement, ambiguity flags
                     ▼
         (3) Annotator A human review           ← accept / correct
                     │
                     ▼
         (4) Annotator B blind re-label         ← 20% subsample only
             on 20% subsample
                     │
                     ▼
         (5) Adjudicator resolves               ← senior linguist / PL
             every A/B disagreement
                     │
                     ▼
            gold JSONL, schema-valid,
              Cohen's κ ≥ 0.70
```

### (1) LLM pre-annotation

- Model: Claude Sonnet 4.5 via API (or local haqumei-v0.8.0 as fallback).
- Prompt template lives at `../../scripts/curation/prompts/hard_set_v1.txt`
  (to be added in the Phase 1 kickoff PR).
- Output MUST already be schema-valid JSON. Reject and retry on schema
  violation; do not silently pass through malformed rows.

### (2) 3-way diff format

Every sentence enters the human queue with a diff record:

```
id: hs-polyphone-042
text: 会議を行った
                      phonemes                              accent (HL)
llm       :  k a i g i w o o k o n a q t a               LHHH LHHHH
haqumei   :  k a i g i o o k o n a q t a                 LHHH LHHHH
openjtalk :  k a i g i w o o k o n a q t a               LHHL LHHHH   ← disagree
flags     : ["accent-disagreement"]
notes     : haqumei drops を surface w. Verify particle.
```

Rules for the diff:

- One column per source. Missing tokens rendered as `_`.
- Disagreement columns highlighted (color in TSV → `\033[31m…\033[0m`, or
  `!!` prefix in plain text).
- `flags` is a fixed vocabulary:
  `phoneme-disagreement`, `accent-disagreement`, `mora-count-disagreement`,
  `unknown-word`, `long-vowel-form-mismatch`, `dictionary-miss`.
- Annotators MUST resolve every disagreement or explicitly write the
  rationale in `notes` (e.g. "haqumei wrong: を particle is not silent
  before こ").

### (3) Annotator A review — one pass

- Fix errors inline in a working JSONL. Do not delete flags without
  addressing them.
- If the LLM pre-annotation is already correct, mark `agree` in a shadow
  status column (kept out of the schema).
- Throughput target: 25 sentences / hour for polyphone & counter,
  35 sentences / hour for the other 5 categories → ~50 hours per annotator
  for the full 1,400.

### (4) Annotator B blind re-label — 20% subsample

- 280 sentences (20% of 1,400), randomly sampled with a category-stratified
  seed. Same schema, same conventions, **no** access to A's labels.
- Purpose: measure inter-annotator agreement, not to double-label the whole
  set. This is the standard blind-double protocol used in JVS / JSUT
  labeling audits.

### (5) Adjudication and agreement metrics

- Adjudicator (senior linguist or project lead) resolves every A/B
  disagreement on the 280-sentence subsample.
- Compute Cohen's κ **before** adjudication on:
  - **phoneme-level κ**: token-aligned; token = phoneme symbol,
    alignment by Levenshtein-min-edit;
  - **mora-level κ on H/L**: token = single mora label ∈ {H, L}.
- **Target**: κ ≥ 0.70 on both tiers. If κ falls below 0.70 on a category,
  re-train annotators on that category, revise the diff heuristics, and
  re-label the affected subset.

## Validation (must pass in CI)

The following invariants are enforced by
`../../scripts/curation/validate_hard_set.py` (added in the Phase 1
kickoff PR). Any PR touching `samples.jsonl` or the Phase 1 hard-set JSONL
runs this check.

1. Row parses as JSON.
2. Row validates against `schema.json` (Draft 2020-12,
   `jsonschema.Draft202012Validator`).
3. `id` is unique across the entire hard-set corpus.
4. `len(accent) == mora_count(phonemes)` — mora count derived by the
   grouping rule `[Cy]V | V | N | q → 1 mora`.
5. Every value in `accent_phrase_boundaries` is `> 0` and `<= len(accent)`.
6. All phoneme symbols ∈ JULIUS-extended set (see §Labeling conventions).

The seed rows in `samples.jsonl` are the fixture used by
`test_validate_hard_set.py`. Do not edit them without updating the tests.

## Cost budget

Per-sentence cost, based on the C-plan pipeline:

| Item | Unit | Sentences | Unit cost | Subtotal |
|---|---|---|---|---|
| LLM pre-annotation (Claude Sonnet input+output, ~600 tokens/sentence) | JPY | 1,400 | ~0.45 | ~630 |
| Annotator A human review, 30/hour avg, ¥2,000/hour | JPY | 1,400 | 2000/30 ≈ 67 | ~94,000 |
| Annotator B 20% blind pass, 30/hour, ¥2,000/hour | JPY | 280 | 67 | ~18,700 |
| Adjudicator resolution, 15 min/disagreement × est. 60 disagreements | JPY | 60 | 3000 · 0.25 = 750 | ~45,000 |
| Storage / CI compute | JPY | — | — | ~500 |
| **Total (human-in-the-loop)** | JPY | | | **~158,830** |

Human labor dominates; that is the honest number. The **LLM half-automation
cost alone**, if we run only the pre-annotation + 3-way diff + schema
validation without paying human annotators (i.e. the machine-only scaffold
this seed enables), is:

| Item | JPY |
|---|---|
| LLM pre-annotation, 1,400 rows | ~630 |
| 3-way diff generation (offline scripts, no cloud) | ~0 |
| Schema validation CI runs (Actions minutes) | ~200 |
| **Machine-only scaffold total** | **~830** |

Both budgets are well under the **¥5,000** machine-scaffold ceiling
requested. The full human-labeled budget of ~¥159K is on the Phase 1
budget line, not the Phase 0 seed line — the ¥5K figure specifically
covers the machine scaffold this seed unlocks.

## Handoff to Phase 1

When Phase 1 starts, do the following in order:

1. Freeze `schema.json` at v1.0 and tag `hard-set-schema-v1.0`.
2. Copy `samples.jsonl` into `data/hard_set/gold.jsonl` as the first 21
   rows, keeping IDs stable.
3. Draft `scripts/curation/prompts/hard_set_v1.txt` using the notes in this
   file as the annotation manual.
4. Recruit two annotators (native Tokyo-Yamanote speakers with prior
   pyopenjtalk annotation experience preferred) and one adjudicator.
5. Run the C-plan pipeline in category-by-category order:
   polyphone → counter → proper_noun → loanword → numeric_unit →
   english_mixed → english_abbreviation.
6. After each category of 200, publish a κ report. Do not proceed to the
   next category if κ < 0.70 on the just-finished one.

## Provenance & disagreement with haqumei

Several seed rows deliberately disagree with haqumei v0.8.0 baseline output
(see `notes` field on individual rows). This is intentional: haqumei is a
Tier-2 target we must **beat**, not mimic. Where the seed's gold label
differs from haqumei, the disagreement is grounded in the NHK accent
dictionary or in pyopenjtalk-plus dictionary entries and is documented
inline. The Phase 1 hard-set must apply the same rule — the LLM-proposed
label is a starting point, not authority.
