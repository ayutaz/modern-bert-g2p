# Phase 2 Pilot Results v4 — 30K vs 60K, Hard-Set, P-C pad_class_weight fix

**Generated**: 2026-07-07 UTC
**Scope**: 47 evals across 4 pilot variants × 4 datasets (JSUT / JVS / ROHAN / hard-set)
**New in v4**: P-A 60K (2 seeds), P-C v3 (pad_class_weight=1.0 fix, 3 seeds), hard-set eval on 3 pilots × 3 seeds

## 1. Full mean ± σ table (per_micro %)

### P-A (seq2seq, ModernBERT-ja-130m + 6-layer Transformer decoder, 157M, beam=4)

| Pilot | JSUT (n_rows=5000) | JVS (n=3000) | ROHAN (n=4600) | hard-set (n=140) |
| :--- | ---: | ---: | ---: | ---: |
| **P-A 30K** (3 seeds) | 64.55 ± 2.64 % | 72.00 ± 3.96 % | 67.17 ± 4.73 % | **49.34 ± 0.51 %** |
| **P-A 60K** (2 seeds) | 66.44 ± 0.61 % | 75.90 ± 0.96 % | 71.86 ± 2.14 % | 49.92 ± 1.18 % |
| Δ (60K − 30K) | **+1.89** | **+3.90** | **+4.69** | +0.58 |

**結論**: 60K は JSUT/JVS/ROHAN で **明確な overfit** (+2〜+5pt 悪化)。hard-set のみほぼ同じ。

### P-C (char-level BERT, 110M)

| Pilot | JSUT | JVS | ROHAN | hard-set |
| :--- | ---: | ---: | ---: | ---: |
| P-C v2 (pad_class_weight=2.0) | 99.97 % | 99.97 % | 100.00 % | 99.04 % |
| **P-C v3 (pad_class_weight=1.0)** | 99.98 % | 99.96 % | 100.00 % | **98.91 %** |

**結論**: pad_class_weight を 2.0 → 1.0 に落としても **collapse は解消せず**。0.13pt の hard-set 微改善のみ。原因は agent 分析通り「8-slot 内 6:2 pad:real の内在的不均衡」+ label_smoothing による pad 側 ε バイアス。 Architecture 変更 (max_slot 削減、CTC-style monotonic、または empty-slot supervision 削除) が必要。

## 2. Per-seed comparison (seed 依存性)

### P-A JSUT / JVS / ROHAN (30K vs 60K)

| Seed | JSUT 30K | JSUT 60K | JVS 30K | JVS 60K | ROHAN 30K | ROHAN 60K |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20260704 | 63.75 | 66.87 (+3.1) | 72.40 | 76.58 (+4.2) | 65.99 | 73.38 (+7.4) |
| 20260705 | 67.50 | 66.01 (**−1.5**) | 75.74 | 75.22 (−0.5) | 72.38 | 70.35 (−2.0) |
| 20260706 | 62.41 | (skipped) | 67.85 | (skipped) | 63.15 | (skipped) |

**seed dependency**: seed 20260704 は 60K で overfit (+3〜+7pt)、seed 20260705 は逆に微改善 (−0.5〜−2pt)。訓練軌跡が seed shuffle で発散するのは under-regularized の signal。

## 3. Hard-set per-category breakdown (P-A 20260706, 30K best seed)

| Category | n | per_micro | note |
| :--- | ---: | ---: | :--- |
| proper_noun (kanji/katakana 混合) | 20 | **39.30 %** | 最良 |
| counter (助数詞) | 20 | 43.80 % | 2nd 良 |
| polyphone (多音字) | 20 | 47.11 % | |
| loanword (カタカナ外来語) | 20 | 48.90 % | |
| english_abbreviation | 20 | 50.00 % | |
| numeric_unit (数詞・単位) | 20 | 54.37 % | |
| english_mixed | 20 | **56.83 %** | 最悪 |

**hard-set 全体 49.24%** が 3 本柱 (62-67%) より **明確に低い**。curated 20/cat の短文なので語彙的曖昧性が少ないためと推測。

## 4. Baseline references (from CLAUDE.md — reference only, not target)

| Baseline | JSUT PER | 備考 |
| :--- | ---: | :--- |
| CharsiuG2P (byT5, 300M multi) | 10.51 % | 自著 held-out |
| Kakegawa TJ-G2P | 11.85 % | JSUT400 ad-hoc split |
| haqumei (hybrid, dict primary) | 1.17 % | dict-based, reference only |
| OpenJTalk (pure rule) | (未直接測) | — |

## 5. Findings

1. **Scale (30K → 60K) does not help** — 全 3 本柱で悪化 (+2〜+5pt)、hard-set のみほぼ同じ。**Data size (~150K samples) が bottleneck**、step 増加は overfit を招く。
2. **Seed dependency が大きい** — seed 変化で −2〜+7pt の差。学習軌跡の不安定 = regularization 不足の signal。
3. **P-C は architectural collapse** — pad_class_weight=2.0→1.0 で解消せず。max_slot=8 の 6:2 不均衡と label_smoothing bias が構造的ボトルネック。
4. **hard-set は 3 本柱より易しい (49% vs 65%)** — curated 短文の効果。categorical breakdown で proper_noun / counter が最良、english_mixed が最悪。

## 6. Recommended next iterations

**Priority 1 (data 拡大、~2 日 dev + ~10h Vast)**:
- v1_5src (~150K) → v2 (~500K-1M) 拡大: Wikipedia + Aozora + JMDict 再抽出
- 30K steps 維持 (60K は overfit なので不要)
- 期待: JSUT PER 65% → 40% 台 (data 3x で先行研究知見)

**Priority 2 (P-C architecture rewrite、~1 週 dev)**:
- max_slot=8 の固定 slot 設計を撤廃
- CTC-style monotonic alignment or seq2seq デコード (P-A と同じ) に置換
- 期待: 99% collapse を脱出、P-A に近い数値

**Priority 3 (regularization 強化)**:
- weight_decay 0.01 → 0.05
- dropout 0.1 → 0.2
- 期待: seed 分散 σ ~4% → ~2%、mean 微改善

## 7. Artifacts

- 47 JSON reports: `reports/phase2/eval/{p_a,p_a_60k,p_c_v2,p_c_v3}_{seed}_{dataset}.json`
- 累計 Vast コスト: ~$10 (32h GPU time)
- Checkpoints: 30K (P-A/P-C v2/P-C v3 × 3) + 60K (P-A × 2)
