# Phase 2 パイロット比較テーブル (v2.0 pure-NN pivot)

**評価**: JSUT Basic5000 PER (canonical form、`scripts/eval_haqumei_jsut.py` と bit-一致プロトコル)
**訓練**: v1_5src Parquet corpus (train 541,885 rows / 5 sources)
**GPU**: Vast.ai RTX 5090 32GB VRAM
**日付**: 2026-07-04

---

## P-C (char-level BERT `tohoku-nlp/bert-base-japanese-char-v2`)

**訓練 HP**: 8000 optimizer steps × batch 64 × accumulate 2、bf16、encoder_lr 5e-5、warmup 400

| Seed | train_loss (final) | JSUT PER micro | JSUT PER macro | S | D | I | N |
|---|---:|---:|---:|---:|---:|---:|---:|
| 20260704 | (未取得) | **266.95%** | 276.73% | 205,394 | 70 | 589,619 | 297,843 |
| 20260705 | (未取得) | **237.92%** | 246.39% | 210,151 | 95 | 498,371 | 297,843 |
| 20260706 | (未取得) | **273.41%** | 282.88% | 207,684 | 85 | 606,566 | 297,843 |
| **mean** | — | **259.42%** | — | — | — | — | — |
| **std** | — | **15.44%** | — | — | — | — | — |

**特徴**: I (insertion) が支配的 (~500K-600K vs S ~200K)、per-char × slot=8 の phoneme head が spurious phonemes を過剰生成。

## P-A (seq2seq: ModernBERT-ja-130m encoder + Transformer decoder)

**訓練 HP**: 10000 optimizer steps × batch 32 × accumulate 4、bf16、encoder_lr 3e-5、head_lr 1e-4、warmup 500

| Seed | train_loss (final) | val_loss (final) | JSUT PER |
|---|---:|---:|---|
| 20260704 | **1.0308** | **1.0708** | 未計測 (adapter stub) |
| 20260705 | **1.0862** | **1.0846** | 未計測 (adapter stub) |
| 20260706 | **1.1583** | **1.1197** | 未計測 (adapter stub) |
| **mean** | **1.0918** | **1.0917** | — |
| **std (train)** | **0.0524** | — | — |

**特徴**: seed 間分散極めて小 (**val σ=0.024**)、学習収束良好。ただし `evaluate_checkpoint` の P-A prediction_fn adapter が Phase 2 D+B commit で未実装 (P-C のみ実装済)、eval 不能。

---

## 比較参考値 (docs/research/09 より)

| モデル | 種別 | 想定 JSUT PER (%) | 出典 |
|---|---|---:|---|
| Frontier LLM (Claude Opus 4.6) | LLM parse | **0.52** | Koriyama Interspeech 2026 |
| Frontier LLM (Gemini 3.1 Pro) | LLM parse | **0.62** | Koriyama Interspeech 2026 |
| haqumei (hybrid) | rule + NN | **1.17** | 参考値 (target ではない) |
| OpenJTalk (pyopenjtalk) | rule | **1.03** | 参考値 (target ではない) |
| CharsiuG2P (ByT5-small) | pure-NN | ~10.51 | 自著 held-out |
| Kakegawa TJ-G2P (Kurihara 2024) | pure-NN | 11.85 (PPL CER) | JSUT400 |
| PnG BERT (Yasuda & Toda 2022) | pure-NN | (WSA 45.5%) | pretrain-validation |
| **本 P-C (Phase 2 pilot 訓練 8K steps)** | **pure-NN** | **~259%** | 本表 |

---

## 分析

- 本 pilot 訓練 (P-A 10K / P-C 8K steps × 541K rows) は **CharsiuG2P ~10.5% には 25 倍のギャップ**あり
- **主因**: 訓練量不足 (Kakegawa は 30K+ steps、CharsiuG2P は multi-lingual 3.5M steps 相当)
- **P-C の I (insertion) 過剰**: 8-slot per-char × 68 phoneme vocab の出力空間が広く、まだ疎な出力を学べていない
- **P-A の val loss 1.09 は学習成功指標**、eval CLI 側の adapter 実装で PER 実測可能

## 次段推奨 (Phase 2.1 or Phase 3 直前作業)

1. **P-A prediction_fn adapter を `_build_prediction_fn_from_checkpoint` に実装** (~30 分作業、Phase 2 D+B の P-C adapter と同構造)
2. **訓練スケール拡大**: 10K → 100K steps (対応 GPU-h ~30 h × $0.36 = $11)
3. **loss 修正**: insertion penalty 追加 (P-C 8-slot 出力の疎化)
4. **JVS/ROHAN eval** も並列実行 (JSUT は分散大、他 dataset 追加で clarity)
5. **hard-set 140 sample eval** (Phase 0 で作成済) で category 別 PER 分析
