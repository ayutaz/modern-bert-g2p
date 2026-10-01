# modernbert-g2p

ModernBERT-Ja をベースに、推論時に辞書 lookup を一切呼ばない **pure-NN** の日本語 G2P (Grapheme-to-Phoneme) を試した研究プロジェクトです。

方針としては pyopenjtalk / OpenJTalk のような辞書 + ルール、あるいは haqumei のような hybrid ではなく、テキストを直接 neural network に入力して音素列を得る構成を採ります。学習時に辞書由来の教師信号を使うことは許容しますが、推論パスに MeCab / UniDic / 辞書 lookup を含めません。

## 現在のステータス

Phase 0 (baseline 再現) と Phase 2 (2 アーキ並列 pilot 学習・評価) までを実施し、その結果を本 README に記録しています。Phase 3 (マルチタスク学習) 以降はコード未着手です。

- Phase 0: baseline 再現 (haqumei / pyopenjtalk / OpenJTalk) — 完了
- Phase 1: 5-source データパイプライン構築 (150K sentence-level pairs) — 完了
- Phase 2: seq2seq / char-BERT 2 パイロット学習 + JSUT / JVS / ROHAN / hard-set 評価 — 完了
- Phase 3-6: 未着手

## 結果サマリ (Phase 2)

pure-NN 制約下で 2 種類のアーキテクチャを並列学習し、JSUT Basic5000 / JVS-3000 / ROHAN 4600 / hard-set (140 文) の 4 dataset で評価しました。

### seq2seq モデル (ModernBERT-ja-130m encoder + 6-layer Transformer decoder, 157M params)

| Pilot | JSUT PER | JVS CER | ROHAN KER | hard-set PER |
| :--- | ---: | ---: | ---: | ---: |
| 30K (3 seeds mean ± σ) | 64.55 ± 2.64 % | 72.00 ± 3.96 % | 67.17 ± 4.73 % | 49.34 ± 0.51 % |
| 60K (2 seeds mean ± σ) | 66.44 ± 0.61 % | 75.90 ± 0.96 % | 71.86 ± 2.14 % | 49.92 ± 1.18 % |
| Δ (60K − 30K) | +1.89 | +3.90 | +4.69 | +0.58 |
| 30K best seed (20260706) | 62.41 % | 67.85 % | 63.15 % | 49.24 % |

### char-BERT モデル (tohoku-nlp/bert-base-japanese-char-v2 + 8-slot phoneme head, 110M params)

| Pilot | JSUT | JVS | ROHAN | hard-set |
| :--- | ---: | ---: | ---: | ---: |
| v2 (pad_class_weight=2.0) | 99.97 % | 99.97 % | 100.00 % | 99.04 % |
| v3 (pad_class_weight=1.0) | 99.98 % | 99.96 % | 100.00 % | 98.91 % |

char-BERT は eval で 99% 以上に collapse。pad_class_weight を弱めても復旧しませんでした。

### 先行 pure-NN 研究との比較 (プロトコル不揃いのため参考値扱い)

| モデル | パラメータ | JSUT PER | プロトコル |
| :--- | ---: | ---: | :--- |
| CharsiuG2P byT5 (多言語) | 300M | 10.51 % | 自著 held-out (単語単位) |
| CharsiuG2P monolingual JA-only | 300M | 66.89 % | 自著 held-out |
| Kakegawa TJ-G2P | ~110M | 11.85 % | JSUT400 (400 文 subset) |
| 本 seq2seq 30K best | 157M | 62.41 % | JSUT Basic5000 (5000 文) |
| 本 seq2seq 30K mean | 157M | 64.55 % | JSUT Basic5000 |
| haqumei (hybrid, 参考) | — | 1.17 % | JSUT Basic5000 |

CharsiuG2P monolingual (66.89%) は 4.5pt 上回りましたが、多言語版 (10.51%) には 52pt 及ばず、先行 pure-NN 4 モデル全てを 3 本柱で越える当初目標は達成できていません。

### hard-set カテゴリ別 breakdown (seq2seq 30K best seed)

| Category | n | per_micro |
| :--- | ---: | ---: |
| proper_noun (kanji/katakana 混合) | 20 | 39.30 % |
| counter (助数詞) | 20 | 43.80 % |
| polyphone (多音字) | 20 | 47.11 % |
| loanword (カタカナ外来語) | 20 | 48.90 % |
| english_abbreviation | 20 | 50.00 % |
| numeric_unit (数詞・単位) | 20 | 54.37 % |
| english_mixed | 20 | 56.83 % |

英単語混在が最も苦手 (56.83%)、固有名詞は curated 短文の中では相対的に得意 (39.30%)。

## わかったこと

1. **30K → 60K で全 dataset が悪化**。data ~150K が bottleneck で、step scaling では改善しません。JSUT +1.89 / JVS +3.90 / ROHAN +4.69pt の悪化と、val_loss 0.97 → 1.03 の上昇が overfit signal と一致します。
2. **char-BERT の fixed-slot 設計は architectural に collapse**。max_slot=8 に対して日本語 1 文字が使う音素は typically 2 (子音 + 母音) 程度で、残り 6 slot が pad で埋まる 6:2 = 3:1 の class imbalance が構造的ボトルネックになります。pad_class_weight による重み調整だけでは解消せず、CTC-style monotonic alignment 等への置換が必要と判断しました。
3. **Seed 分散 σ ~3-5pt** と大きく、weight_decay / dropout の regularization 不足を示唆しています。
4. **hard-set は 3 本柱より易しい (49% vs 65%)**。curated 20 文/cat の短文なので語彙的曖昧性が少ないためと推測されます。

## アーキテクチャ (2 種の並列比較)

ModernBERT-ja のトークナイザーは SentencePiece 系の subword で、SB Intuitions 自身が [HF カード](https://huggingface.co/sbintuitions/modernbert-ja-130m)で「token classification タスクでは性能が悪い」と明記しています。そのため各文字にラベルを付ける naive token classification は最初から避け、次の 2 通りを並列に走らせて比較しました。

- seq2seq: ModernBERT-ja-130m を encoder に、6-layer Transformer decoder に音素列を自由生成させる。beam=4 の beam search + coverage penalty 0.2 で推論
- char-BERT: そもそも subword を捨てて、文字レベル BERT (東北大 bert-base-japanese-char-v2) を encoder にし、各文字位置で 8-slot の phoneme head で分類

base LM を統一した A/B ではなく「ModernBERT で行く」vs 「char-level BERT に切り替える」の trade-off 比較になっています。

## データパイプライン (Phase 1)

pyopenjtalk-plus 辞書 + UniDic + Wikipedia + Aozora + JMDict の 5 source から (surface, phoneme) の pair を抽出し、統合コーパスを構築しました。

- 統合サイズ: train 541,885 rows (v1_5src)
- Phase 2 学習で使う sentence-level pair: 約 150K

Wikipedia は並列 ingest infra、Aozora は per-ruby emission (per-file → per-ruby で 300 倍増加)、UniDic は 33 列の実 CSV に合わせた parser で処理しています。

## Quick start

### 1. ベースライン再現 (Phase 0 AC-P0)

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ".[dev,baselines]"   # haqumei / pyopenjtalk-plus は baselines extra

# JSUT-label を取得
git clone --depth 1 https://github.com/prj-beatrice/jsut-label.git

# haqumei の JSUT Basic5000 PER (公式 1.17%) を再現
JSUT_YAML=./jsut-label/text_kana/basic5000.yaml \
  python scripts/eval_haqumei_jsut.py
```

期待出力: `PER 1.1657%` (公式 1.17% との差 0.005 pt、AC-P0 要件 ≤ 0.1% を満たす)。

### 2. Phase 2 pilot 学習 (GPU 環境)

```bash
uv pip install -e ".[training,pipeline]"   # Parquet 読み込みに pyarrow (pipeline extra) が必要

# seq2seq (30K steps)
python -m modernbert_g2p train --config configs/p_a_30k.yaml --seed 20260706

# char-BERT (30K steps)
python -m modernbert_g2p train --config configs/p_c_30k_v2.yaml --seed 20260706
```

### 3. 評価

```bash
python -m modernbert_g2p eval --pilot P-A --config configs/p_a_30k.yaml \
  --checkpoint reports/phase2/p_a_30k/20260706/checkpoint_step_30000.pt \
  --dataset jsut --output reports/phase2/eval/p_a_20260706_jsut.json
```

### 4. テスト

```bash
pytest -q
```

## 学習設定 (Phase 2)

- Steps: 30K (baseline) / 60K (scale ablation)
- LR schedule: linear warmup 1500 → total 30000, grad_clip 0.5
- 途中発散対策で encoder_lr は 5.0e-5 → 1.5e-5、head_lr は 3.0e-4 → 8.0e-5 に絞りました
- Seeds: 20260704 / 20260705 / 20260706 の 3 種
- Precision: bf16
- GPU: RTX 5090 32GB (Blackwell sm_120, PyTorch 2.11.0+cu128)

## 評価データセット

- JSUT Basic5000 (5000 文): PER (canonical form、`scripts/eval_haqumei_jsut.py` と bit-一致プロトコル)
- JVS-3000 (3000 文): kana CER (Koriyama Interspeech 2026 benchmark)
- ROHAN 4600 (4600 文): kana KER
- hard-set seed_v2 (140 文、7 カテゴリ × 20): PER (per-category)

JVS / ROHAN 用に、モデルの JULIUS phoneme 出力を katakana に変換する converter (`src/modernbert_g2p/evaluation/phoneme_to_kana.py`) を実装しています。CV / digraph / palatalized 音素、moraic N (ン)、geminate q (ッ)、長音 (ー) をカバーする state machine で、unit test 26 ケース全て pass しています。

## ディレクトリ構造

```
modern-bert-g2p/
├── src/modernbert_g2p/          # ソースコード
│   ├── metrics/                 # PER / CER / KER canonical 実装
│   ├── models/                  # seq2seq / char-BERT アーキテクチャ
│   ├── training/                # 学習ループ + collator
│   ├── evaluation/              # phoneme_to_kana converter
│   └── data/                    # 5-source データパイプライン
├── tests/                       # pytest スイート (529 tests)
├── scripts/                     # baseline 測定・データ生成の CLI
├── configs/                     # Phase 2 pilot YAML (p_a_30k / p_c_30k_v2 等)
├── docs/
│   ├── requirements.md          # 要求定義書 (FR/NFR/CR/AC)
│   ├── research/                # 9 本の技術ドキュメント
│   └── design/                  # Phase 別実装設計
├── reports/phase2/eval/         # Phase 2 全 eval JSON (50 files)
├── data/                        # (gitignored) raw / processed
├── pyproject.toml
└── README.md
```

## 依存関係

`pyproject.toml` でランタイムと optional extras に分離しています。

- ランタイム (`dependencies`): `pyyaml`, `numpy` — metrics / evaluation に必要な最小構成
- baseline (`optional-dependencies.baselines`): `haqumei==0.8.0`, `pyopenjtalk-plus`, `jiwer>=3`, `tqdm` — `scripts/eval_*.py` の baseline 測定で使用
- 学習 (`optional-dependencies.training`): `torch>=2.1`, `transformers>=4.48`, `wandb`, `omegaconf`, `fugashi[unidic]` — Phase 2 以降で使用 (`uv.lock` の解決版は torch 2.12.1 / transformers 5.13.0)
- Phase 2 前処理 (`optional-dependencies.phase2`): `fugashi>=1.3`, `unidic-lite>=1.0`
- データパイプライン (`optional-dependencies.pipeline`): `pyarrow>=15`, `lxml>=5` — Parquet 入出力
- 開発 (`optional-dependencies.dev`): `pytest>=8`, `ruff`, `mypy`, `ipython`

Python サポート: `>=3.10,<3.13` (haqumei 0.8.0 は 3.13/3.14 未対応)。

## 設計原則 (pure-NN 制約)

1. 推論パスに pyopenjtalk / MeCab / 辞書 lookup を含めない。学習時に辞書由来の教師信号を使うことは許容
2. char-level BERT の pre-tokenize は「内部トークナイザー」として許容 (推論に外部辞書を呼ばない)
3. NHK Kurihara Interspeech 2024 の TJ-G2P + BAS architecture (T5 seq2seq + char-BERT) を multi-task 1 head として encoder に統合する構想 (Phase 3 で実装予定)
4. Hida ICASSP 2022 のマルチタスク (G2P + 多音字 + APBP + ANPP + BAS) を multi-head で実装する構想 (Phase 3)
5. データ量 5-10 倍拡大 + マルチタスク supervision + JSUT/JVS/ROHAN 3 本柱統一評価を pretrain・fine-tune 両方で活用

詳細は [`CLAUDE.md`](CLAUDE.md) と [`docs/requirements.md`](docs/requirements.md) を参照。

## Refuted (反証済み) の主張

- 「pure-NN が pure-text 入力で hybrid を越えた事例が公開ベンチに存在する」— 2026-07 時点で存在せず (Ohnaka INTERSPEECH 2025 arxiv 2506.04527 は speech+text 入力で PER 0.93% のためスコープ外)
- 「NHK Kurihara 2024 が Japanese G2P を pure neural では unsolvable と framing」— 該当論文にこの記述なし
- 「PnG BERT の training labels が Kuromoji+Neologd pseudo-labels」— 論文本文は "morphological analysis" とのみ記述、断定不能

詳細は [`CLAUDE.md`](CLAUDE.md#refuted-反証済み-の主張--参照禁止) 参照。

## 参照ドキュメント

- [`CLAUDE.md`](CLAUDE.md) — プロジェクト方針、pure-NN 制約、v2.0 pivot 経緯
- [`docs/requirements.md`](docs/requirements.md) — 要求定義書 (FR/NFR/CR/AC 付き)
- [`docs/research/01_overview.md`](docs/research/01_overview.md) — サマリーと開発戦略
- [`docs/research/02_existing_systems.md`](docs/research/02_existing_systems.md) — 既存 G2P サーベイ
- [`docs/research/06_implementation_roadmap.md`](docs/research/06_implementation_roadmap.md) — v2.0 実装ロードマップ
- [`docs/research/07_nn_only_benchmarks.md`](docs/research/07_nn_only_benchmarks.md) — pure-NN 先行研究の徹底調査
- [`docs/research/09_pure_nn_g2p_benchmarks.md`](docs/research/09_pure_nn_g2p_benchmarks.md) — 越えるべき pure-NN 4 モデルの整理
- [`reports/phase2/eval/pilot_table.md`](reports/phase2/eval/pilot_table.md) — Phase 2 の full 数値表 (v4)

## 参考文献 (pure-NN 先行研究)

- CharsiuG2P: [Zhu et al., Interspeech 2022 (arxiv:2204.03067)](https://arxiv.org/abs/2204.03067)
- PnG BERT: [Yasuda & Toda 2022 (arxiv:2212.08321)](https://arxiv.org/abs/2212.08321)
- TJ-G2P: [Kurihara & Sano 2024](https://arxiv.org/abs/2403.02714)
- CC-G2PnP: [Shirahata & Yamamoto 2026 (arxiv:2602.17157)](https://arxiv.org/abs/2602.17157)
- ModernBERT-Ja: [sbintuitions/modernbert-ja-130m](https://huggingface.co/sbintuitions/modernbert-ja-130m)
- BERT-base Japanese char v2: [tohoku-nlp/bert-base-japanese-char-v2](https://huggingface.co/tohoku-nlp/bert-base-japanese-char-v2)

## License

Apache License 2.0 ([`LICENSE`](LICENSE))。

学習・評価に用いる第三者データセット (pyopenjtalk-plus 辞書, UniDic, JSUT, ROHAN, JMDict, Wikipedia 等) はそれぞれ別ライセンス (BSD / MIT / CC-BY-4.0 / CC-BY-SA-4.0 等) が適用されます。特に Share-Alike 系ライセンスは重み配布時に影響するため、重み公開前に一次資料の再確認が必要です。
