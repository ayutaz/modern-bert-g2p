# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト目的

ModernBERTベースの日本語G2P（Grapheme-to-Phoneme）モデルの新規開発。以下の3ティアの敵を明示的に上回ることを目標とする:

1. **Tier 1 (must-beat)**: OpenJTalk / pyopenjtalk (JVS-3000 kana CER 1.03%)
2. **Tier 2 (should-beat)**: haqumei (JSUT Basic5000 PER 1.17%, ROHAN KER 1.64%)
3. **Tier 3 (stretch)**: フロンティアLLM (Claude Opus 4.6 0.52%, Gemini 3.1 Pro 0.62% on JVS-3000)

現在はコード実装前の調査フェーズが完了した状態。`docs/research/` に技術ドキュメントが揃っている。

## リポジトリの現状

- **コード未実装** — ソースコード、テスト、CI等は未着手
- **調査完了** — 6本の技術ドキュメントが `docs/research/` に存在
- **既存ファイル**:
  - `docs/requirements.md` — **要求定義書 (要件ID FR/NFR/CR/AC付き、Phase 0前に確定要)**
  - `docs/research/01_overview.md` — サマリーと開発戦略
  - `docs/research/02_existing_systems.md` — 既存G2Pシステムのサーベイ
  - `docs/research/03_datasets_and_benchmarks.md` — 評価データ・ベンチマーク・ライセンス
  - `docs/research/04_papers_and_references.md` — 論文と一次情報
  - `docs/research/05_technical_design.md` — ModernBERT ベースのモデル設計
  - `docs/research/06_implementation_roadmap.md` — 7フェーズの実装ロードマップ
  - `docs/research/07_nn_only_benchmarks.md` — 純粋NN日本語G2Pのベンチマーク徹底調査 (追加深掘り)
  - `docs/research/08_market_landscape.md` — 市場に存在する日本語対応NN-G2Pモデル 網羅的インベントリ

## 設計の核心思想 (実装時の判断基準)

以下は調査で確立した設計原則。実装時に迷ったらこの原則に戻る:

1. **単一 neural モデルで OpenJTalk を置き換えない**。主要OSS TTS (Style-Bert-VITS2, VITS Japanese, GPT-SoVITS, Bert-VITS2, Misaki) は全て pyopenjtalk をコアに採用、NN は補助的にしか使わない。同じハイブリッド路線を踏襲する。
2. **NHK Kurihara Interspeech 2024 の TJ-G2P + BAS が架構の直接的な青写真**。ModernBERT を BAS 相当のアクセント連続変異補正に使うのが最も証拠に基づいた設計。
3. **Hida ICASSP 2022 のマルチタスク (G2P + 多音字 + APBP + ANPP)** を multi-head で実装する。主観MOS 3.67 (対 オラクル 3.69) の near-oracle 品質を目指す根拠。
4. **SentencePieceトークナイザーの警告に注意**: SB Intuitions 自らが「token classification タスクで性能が悪い」と modernbert-ja HFカードで明記。naive per-token classificationは避け、seq2seq か MeCab pretokenize か char-level BERT のどれかを Phase 2 で head-to-head比較して選ぶ。
5. **haqumei は "rule 天井" であり "NN 天井" ではない (v1.3 反映)**。徹底解剖の結果、PER 1.17% の 80-90% は pyopenjtalk-plus 辞書由来、NN 由来は 0-5% のみ (Kanalizer が英単語遭遇時のみ発火)。**同じ pyopenjtalk-plus 辞書を採用しつつ、ModernBERT を BAS/多音字/略語判定/アクセント推定に投入する設計で、haqumei が持たない改善軸で戦える**。詳細: `docs/research/02_existing_systems.md §A.3`

## 主要な数値目標

| 指標 | ベースライン | 目標 | Stretch |
|---|---|---|---|
| JSUT Basic5000 PER (haqumei-eval と同一プロトコル) | haqumei 1.17% | < 1.0% | < 0.5% |
| JVS-3000 kana CER | OpenJTalk 1.03% | < 0.9% | < 0.62% (Gemini越え) |
| ROHAN KER (haqumei-eval と同一プロトコル) | haqumei 1.64% | < 1.5% | < 1.0% |
| JSUT モーラアクセント精度 (**haqumei 非公表、我々が公表**) | Hida 97.33% | > 97.5% | > 98% |
| 多音字 hard-set PER | (haqumei 数値なし → 我々が公表) | haqumei との差 ≥ 0.5pt | ≥ 1.0pt |
| 英字略語 hard-set PER | (同上) | haqumei との差 ≥ 0.5pt | ≥ 1.0pt |
| 固有名詞 hard-set PER | (同上) | haqumei との差 ≥ 0.5pt | ≥ 1.0pt |

## ベースモデル選定 (推奨: sbintuitions/modernbert-ja-130m)

- **主軸**: `sbintuitions/modernbert-ja-130m` (132M params, MIT, 8k context)
- **Ablation 対象**: 30m / 70m / 310m, `llm-jp/llm-jp-modernbert-base`, `tohoku-nlp/bert-base-japanese-char-v2`
- **理由**: サイズと精度のバランス、MIT ライセンスで商用可、4サイズ展開で系統的ablation可能

## データセット & 評価

### 主要評価 (必須3本柱)

- **JVS-3000** (Koriyama Interspeech 2026 benchmark) — kana CER
- **JSUT Basic5000** (jsut-label) — PER
- **ROHAN 4600** — KER

### 主要学習データソース

- pyopenjtalk-plus 辞書 (~800K エントリ)
- UniDic全エントリ (~1M)
- Wikipedia日本語版 ふりがな抽出 (~500K sentences)
- 青空文庫 ふりがな付きテキスト (~200K sentences)

### ライセンス警告

- 商用配布を想定する場合、`pyopenjtalk-plus` / UniDic / JSUT / ROHAN / JMDict / Wikipedia は各々異なる条項 (BSD / MIT / CC-BY-4.0 / CC-BY-SA-4.0)。実装前に一次資料で再確認。特に **Share-Alike (SA)** 系のライセンスは重み配布時に注意。

## 開発フロー (7フェーズ)

- **P0** (1週): ベースライン再現 (haqumei, pyopenjtalk)
- **P1** (2週): 学習データ生成
- **P2** (2週): シングルタスク G2P baseline + トークナイザー選定
- **P3** (3週): マルチタスク学習 (G2P + polyphone + APBP + ANPP + BAS)
- **P4** (2週): ハイブリッド化 (辞書primary + ModernBERT correction)
- **P5** (2週): スケール & Ablation
- **P6** (1週): 評価・公開

詳細は `docs/research/06_implementation_roadmap.md` 参照。

## 参照優先順位

コーディング中に事実確認が必要な時の優先順位:

1. `docs/research/` 内の該当ドキュメント (プロジェクト内合意)
2. 各ドキュメントに引用されている一次情報 URL (論文 arxiv、HFカード、公式GitHub)
3. Web検索 (二次情報)

## 実装時の技術的注意事項

- **音素表記**: JULIUS音素セット + モーラアクセントH/L + アクセント句境界マーカ '/' を canonical とする (pyopenjtalk と互換)
- **トークナイザー起因の失敗モード**: Phase 2 で判断が確定するまで、seq2seq / MeCab-pretokenize / char-level の3並列パイロットを維持する
- **カテゴリ別 sample reweighting**: 数詞 / 固有名詞 (漢字/カタカナ) / 助数詞語 / 外来語 / 英単語混在 / 英字略語 には `sample_weight = 2.0` を推奨
- **Hard-set**: **7カテゴリ (多音字/助数詞/固有名詞/カタカナ外来語/数詞・単位/英単語混在文/英字略語) 各200文** の人手キュレーションを Phase 1 で作成し、以降のすべての評価で使用
- **多言語混在対応 (MUST)**: 実世界の日本語文には英単語・略語・記号連結語 (iPhone, PDF, AI, Wi-Fi, e-mail等) がデフォルトで混在する。以下を実装:
  - 英単語混在 → Kanalizer流の音写 or CMUdict→日本語音素マッピング (Phase 2で head-to-head比較)
  - 英字略語 → アルファベット読み (AI→エーアイ) と単語読み (NASA→ナサ) を辞書 + 文脈で判定
  - 英数字・単位 (10km, 3GB, 2025年, 10:30, 3.14) の正規化ロジック
- **Reproducibility**: 全ての ablation は同じ seed / 同じ splits で実行し、結果表を1つに統合する

## Pure-NN 日本語G2P の実証的失敗パターン (07 で判明した追加事実)

07 の追加調査で確定した以下の事実は、hybrid 戦略の証拠を強化する。実装中に "pure-NN で行けるのでは" という誘惑が湧いた時に読み返すこと:

- **PnG BERT (Yasuda & Toda 2022)**: pure-NN BERT で JSUT G2P whole-segment accuracy 45.5% のみ。TTS accent MOS 2.51 vs rule-derived Tacotron 3.04 で敗北
- **Kakegawa TJ-G2P vs OpenJTalk (JSUT400)**: pure-NN 11.85% PPL CER vs OpenJTalk 10.82% (rule) で敗北。助数詞語では pure-NN 15.43% → hybrid化で 1.42% (11倍改善は辞書由来、NN由来ではない)
- **CharsiuG2P (byT5)**: 多言語aggregate PER 0.089 でも日本語 specific 数値は公式未公開。第三者測定で日本語 PER 10.5% (hybrid の 1桁下)
- **CC-G2PnP (2026年最新pure-NN)**: 「Dict-DNN hybrid を上回る」主張は敵対的検証で 0-3 refute
- **XPhoneBERT / fo-BERT / UR-BERT はG2Pモデルではない** — G2P benchmark として引用しない

## Refuted (反証済み) の主張 — 参照禁止

deep-research の3票敵対的検証で棄却された主張。文献引用時に混同しないよう注意:

- **却下 (01調査)**: 「NHK Kurihara 2024 が Japanese G2P を pure neural では unsolvable と framing し、[ ] タグ付き hybrid architecture を提案」— この記述は該当論文には存在しない。dual Transformer (T5 + BERT) の実際の architecture description に忠実に基づくこと。
- **却下 (07調査, 0-3)**: 「PnG BERT (arxiv 2212.08321) は pure-NN が lexicon dictionary の Japanese word coverage に本質的に及ばないと明示的に framing」— 論文本文にこの強い主張は無い。数値の敗北は事実だが、著者は "coverage-limited" とは言っていない
- **却下 (07調査, 0-3)**: 「CC-G2PnP は Dict-DNN hybrid ベースラインを 6D-Eval で両指標で上回る」— 論文本文の主張だが verify で棄却。pure-NN が hybrid を打ち破った事例は依然存在しない
- **却下 (07調査, 1-2)**: PnG BERT の training labels が Kuromoji+Neologd pseudo-labels であるという主張。数値 45.5% 自体は verified だが labels origin の詳細断定は refute

## ワークフロー的な補足

- 新規コードを書くときは `superpowers:brainstorming` を必ず先に呼び、要件を明確化してから TDD で進める
- 既存コードを触るときは `superpowers:systematic-debugging` で仮説→検証を明示化する
- 実装完了時は `superpowers:verification-before-completion` で証拠を集めるまで「完了」宣言を保留する
- 現時点でコードは存在しないため、Phase 0 の作業を始める場合は `superpowers:writing-plans` で先に実装計画を書き起こす
