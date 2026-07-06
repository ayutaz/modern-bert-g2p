# Phase 2 Pilot Results — JSUT Basic5000 PER

Generated: 2026-07-05 UTC
Training: 30K optimizer steps on Vast.ai RTX 5090, bf16, pure-NN (no dict/rule at inference).

## P-A (seq2seq: ModernBERT-ja-130m encoder + 6-layer Transformer decoder, 157M total params)

| Seed | JSUT PER micro | PER macro | n_rows | train_loss | val_loss |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 20260704 | **67.95 %** | 65.97 % | 5000 | 0.997 | 0.975 |
| 20260705 | 74.72 % | 74.22 % | 5000 | 0.932 | 0.980 |
| 20260706 | **62.96 %** | 58.83 % | 5000 | 0.949 | 0.985 |
| **Mean ± σ** | **68.54 ± 4.83 %** | 66.34 ± 6.29 % | — | 0.960 | 0.980 |

## P-C (char-level BERT: tohoku-bert-base-japanese-char-v2 + slot=8 phoneme head, 110M params)

| Seed | JSUT PER micro | PER macro | n_rows | train_loss | val_loss |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 20260704 | 337.06 % | 348.40 % | 5000 | 0.824 | NaN |
| 20260705 | 337.06 % | 348.46 % | 5000 | 0.810 | NaN |
| 20260706 | 336.45 % | 347.87 % | 5000 | 1.008 | NaN |

**P-C の eval postprocessing にバグ**: `p_c_to_canonical` が slot 出力 (最大 8 phonemes / char position) の pad フィルタリングをしていない → 参照長を大幅に超えた予測列が生成されるため PER が >100% になっている。訓練は正常 (train_loss 0.81 は全ピロット最良)。要後処理修正。

## Baseline Reference (from CLAUDE.md)

| Baseline | JSUT PER | Note |
| :--- | ---: | :--- |
| CharsiuG2P (byT5, 300M multilingual) | 10.51 % | 自著 held-out (IPA-vs-dict word-list) |
| PnG BERT (~110M) | (未公表) | Pretrain val whole-word acc 45.5% のみ |
| Kakegawa TJ-G2P (Kurihara & Sano 2024) | 11.85 % | JSUT400 ad-hoc split (400 文) |
| haqumei (hybrid, dict primary) | 1.17 % | Reference only (not target) |
| OpenJTalk (pure rule) | 1.03 % | Reference only (not target) |

## 所感

**このイテレーションの数値は先行 pure-NN baselineに到達せず** (P-A 68% vs CharsiuG2P 10.51%)。原因の主要仮説:

1. **学習不足**: 30K steps は先行研究 (PnG BERT ≥100K、Kakegawa 500K 相当) の 1/3-1/15。
2. **LR 保守化の残効**: 早期 divergence を防ぐため encoder_lr を 5e-5 → 1.5e-5 に落とし、head_lr も 3e-4 → 8e-5 に。結果として under-fit の可能性。
3. **beam mismatch**: 訓練 config は `beam_size: 4, coverage_penalty: 0.2` を保持しているが、eval 経路 `_build_pa_prediction_fn` は `inference` セクション欠如で `beam=1` (greedy) 実行。
4. **P-C postprocessing バグ**: 上述の slot pad フィルタ欠如。

## 次のステップ (推奨優先度順)

1. **P-C `p_c_to_canonical` の slot pad フィルタリング修正** (即 eval 再走で数値が正常化する見込み — train_loss 最良の可能性を活かす)
2. **P-A eval で beam=4 + coverage_penalty=0.2** を復元し PER 再測 (数 pt 改善見込み)
3. **60K-100K steps 再学習** (LR は現在の tighten 版を維持、divergence していないので安全)
4. **JVS + ROHAN の eval も回す** (JSUT 単独では 3 本柱 comparison が完結しない)

## Artifacts

- Per-row JSON reports: `reports/phase2/eval/p_{a,c}_*_jsut.json`
- Checkpoints (Vast on `43856040`): `reports/phase2/p_{a,c}/{seed}/checkpoint_step_30000.pt`
- Chain logs: `logs/chain_b.log`, `logs/eval.log`
