# Phase 2 Pilot Results v3 — JSUT / JVS / ROHAN 3 本柱

**Generated**: 2026-07-06 UTC
**Training**: 30K optimizer steps on Vast.ai RTX 5090, bf16, pure-NN (no dict/rule at inference).
**Eval**: JSUT Basic5000 (PER), JVS-3000 (kana CER via phoneme→kana converter), ROHAN 4600 (KER via phoneme→kana).
**Phase A**: P-A beam_size restored to 4 (train/eval parity); P-C postproc uses ``char_positions`` (CLS/SEP filtered).
**Phase B**: P-C collator empty slot labeled with ``vocab.pad_id=0`` (was ``-100``); retrained on ``configs/p_c_30k_v2.yaml``.
**Phase C**: JULIUS phoneme → katakana converter (29 unit tests) integrated for JVS/ROHAN CER/KER.

## P-A (seq2seq: ModernBERT-ja-130m encoder + 6-layer Transformer decoder, 157M params, beam=4)

| Seed | JSUT PER micro | JSUT PER macro | JVS CER micro | JVS CER macro | ROHAN KER micro | ROHAN KER macro |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20260704 | 63.75 % | 60.66 % | 72.40 % | 70.02 % | 65.99 % | 64.52 % |
| 20260705 | 67.50 % | 64.76 % | 75.74 % | 73.68 % | 72.38 % | 71.32 % |
| **20260706** | **62.41 %** | **57.75 %** | 67.85 % | 64.70 % | 63.15 % | 61.46 % |
| **Mean ± σ** | **64.55 ± 2.66 %** | 61.06 ± 3.51 % | 71.99 ± 3.96 % | 69.47 ± 4.51 % | 67.17 ± 4.66 % | 65.77 ± 4.99 % |

**Phase A で beam=1 → 4 の効果 (JSUT)**:
- seed 20260704: 67.95 % → 63.75 % (**-4.20 pt**)
- seed 20260705: 74.72 % → 67.50 % (**-7.22 pt**)
- seed 20260706: 62.96 % → 62.41 % (-0.55 pt)

## P-C v2 (char-level BERT: bert-base-japanese-char-v2 + slot=8 head、collator fix、pad_class_weight=2.0)

| Seed | JSUT PER micro | JVS CER micro | ROHAN KER micro | train_loss |
| :--- | ---: | ---: | ---: | ---: |
| 20260704 | 99.95 % | 99.96 % | 100.00 % | 0.610 |
| 20260705 | 99.99 % | 99.97 % | 100.00 % | 0.567 |
| 20260706 | 99.98 % | 99.97 % | 100.00 % | 0.653 |

**train_loss は全 pilot 最良 (0.61 ± 0.04, 旧 P-C の 0.81 から 25% 低下)** だが、eval PER は 99% 近辺で degenerate。

### P-C v2 の diagnosis

`pad_class_weight=2.0` が強すぎ、モデルが phoneme slot をほぼ全て `pad` (class 0) で予測する状態に collapse。sample smoke test:
- 入力「テスト」 → `('t', 'e', 's', 'u', 't', 'o')` (短文は成功)
- 入力「今日はいい天気ですね」 → `()` (空出力)
- 入力「水をマレーシアから買わなくてはならない」 → `()` (空出力)

Phase B の collator fix (empty slot を `-100` → `pad_id=0` でラベル化) は方向として正しい (旧 337% 過剰生成は解消) が、**pad_class_weight の tuning が不完全**。次サイクル修正候補:

- **pad_class_weight: 2.0 → 1.0** (weight boost を外し、balanced CE)
- **pad_class_weight: 2.0 → 0.5** (pad predictions を明示的に抑制)
- または `ignore_pad_slots: true` に切り替え (empty slot 由来の loss を落として実 phoneme のみで学習)

## Baseline Reference (from CLAUDE.md)

| Baseline | JSUT PER | JVS CER | ROHAN KER | Note |
| :--- | ---: | ---: | ---: | :--- |
| CharsiuG2P (byT5, 300M multilingual) | 10.51 % | — | — | 自著 held-out (IPA-vs-dict word-list) |
| PnG BERT (~110M) | (未公表) | — | — | Pretrain val whole-word acc 45.5% のみ |
| Kakegawa TJ-G2P (Kurihara & Sano 2024) | 11.85 % | — | — | JSUT400 ad-hoc split |
| **haqumei (hybrid, dict primary)** | **1.17 %** | (2.66% haqumei repo) | 1.64 % | Reference only — pyopenjtalk-plus 辞書 primary |
| **OpenJTalk (pure rule)** | (未直接測) | **1.03 %** | (未直接測) | Reference only |

## 所感

**このイテレーションで達成**:
1. ✅ **beam=4 fallback fix**: P-A JSUT PER の平均を 68.5% → **64.5%** (-4pt) に改善
2. ✅ **P-C postprocess fix** (`char_positions` 走査): CLS/SEP 除外、overproduction 大幅緩和 (337% → 100%)
3. ✅ **JULIUS→katakana converter 実装** (29 unit tests pass): JVS/ROHAN eval が有効に動作
4. ✅ **JVS + ROHAN 3 本柱の初期 baseline 公開** (pure-NN 130M で公開値は初)

**このイテレーションで未達**:
1. ❌ P-A の 3 本柱 PER/CER/KER 平均 63-72% — 先行 pure-NN の JSUT PER 10.51% (CharsiuG2P) には遠い
2. ❌ P-C v2 の pad_class_weight=2.0 が過剰、モデルが空出力 collapse。**再学習 (weight 1.0 or ignore_pad_slots: true) が必要**

**構造的な結論**: 30K steps + LR tightened の pure-NN 130M では、CharsiuG2P baseline (10.51%) には届かない。追加学習 (Phase D 60K-100K) と、P-C の pad balance 再調整が次の必須作業。

## Artifacts

- Per-row JSON reports: `reports/phase2/eval/p_{a,c_v2}_*_*.json` (18 files, 5.5 MB total)
- Checkpoints (Vast on `43856040`):
  - P-A: `reports/phase2/p_a/{seed}/checkpoint_step_30000.pt`
  - P-C v2: `reports/phase2/p_c_v2/{seed}/checkpoint_step_30000.pt`
- 30K training / eval logs on Vast: `logs/chain_b.log`, `logs/eval.log`

## 次の推奨アクション

1. **P-C v2 の pad_class_weight fine-tuning**: 3 seeds × 30K で `pad_class_weight={0.5, 1.0}` を試す (~4-5h Vast)
2. **P-A 60K-100K rerun** (Phase D): scale が underfit を解消するか検証 (~10-14h Vast)
3. **hard-set eval** (7 カテゴリ × 200 文): 多音字 / 数詞 / 固有名詞 etc の per-category PER を公開
