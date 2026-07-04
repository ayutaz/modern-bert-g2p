# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト目的

ModernBERT (encoder-only pure-NN) の日本語 G2P (Grapheme-to-Phoneme) 応用可能性を検証する **研究プロジェクト**。推論パスに rule / dictionary lookup を含めない **pure-NN, no dict inference** を厳守する。Primary target は公開されている pure-NN Japanese G2P モデル群を JSUT / JVS / ROHAN 標準ベンチで系統的に上回ること。

### 3ティアの competitive landscape

1. **Primary (must-beat, pure-NN のみ)**: 越えるべき先行 pure-NN 4 モデル
   - **CharsiuG2P** (ByT5-small, ~300M, 多言語) — 自著 held-out で日本語 PER 10.51% (IPA-vs-dict word-list; monolingual-JA-only ByT5 は 66.89% に崩壊)
   - **PnG BERT (Yasuda & Toda 2022)** — 事前学習 validation の whole-word G2P accuracy 45.5% のみ公表 (JSUT 別途 PER は未公表)。TTS accent MOS 2.51 vs rule-labeled Tacotron 3.04 で敗北
   - **Kakegawa TJ-G2P (Kurihara & Sano 2024 の再実装)** — JSUT400 (Kurihara が Basic5000 から 400 抽出した ad-hoc split) で PPL CER 11.85% (OpenJTalk 10.82% に敗北)。Kakegawa 2021 原論文は JSUT を触っておらず新聞 5,142 文で Word-Accuracy 評価
   - **CC-G2PnP (Shirahata & Yamamoto 2026, SB Intuitions)** — 内部 6D-Eval (2,722 文) で PnP CER 1.79-1.80%、Phoneme CER 0.48-0.52%。Dict-DNN-NS hybrid が 1.71%/0.40% で勝つ (「hybrid 越え」は refute 済み)
2. **Reference (compare but not target)**: haqumei 1.17% / OpenJTalk 1.03% — hybrid 天井なので pure-NN 130M では絶対値では劣る前提。gap を可視化するが acceptance criteria には入れない
3. **Aspirational (world-first if achieved)**: Frontier LLM (Claude Opus 4.6 0.52%, Gemini 3.1 Pro 0.62% on JVS-3000) を 130M pure-NN で近似できるか。LLM の parse mode 依存度が未定量のため文字通りの越えは非現実的想定

### Pivot の理由 (v2.0, 2026-07-04)

「ルールベースの G2P 処理が入るのであればこのプロジェクトをする必要がない」 — hybrid 設計は結果の 80-95% が辞書由来のため、130M ModernBERT を書く動機自体が消える。pure-NN 世界内の空白領域を JSUT / JVS / ROHAN 標準ベンチで埋める初のデータポイントとして再定義する。

## リポジトリの現状

- **コード未実装** — ソースコード、テスト、CI等は未着手
- **調査完了** — 9本の技術ドキュメントが `docs/research/` に存在
- **既存ファイル**:
  - `docs/requirements.md` — **要求定義書 (要件ID FR/NFR/CR/AC付き、v2.0 で pure-NN pivot 反映済み)**
  - `docs/research/01_overview.md` — サマリーと開発戦略
  - `docs/research/02_existing_systems.md` — 既存G2Pシステムのサーベイ
  - `docs/research/03_datasets_and_benchmarks.md` — 評価データ・ベンチマーク・ライセンス
  - `docs/research/04_papers_and_references.md` — 論文と一次情報
  - `docs/research/05_technical_design.md` — ModernBERT ベースのモデル設計
  - `docs/research/06_implementation_roadmap.md` — 実装ロードマップ (v2.0 で P4 hybrid 削除、pretrain-plus-fine-tune に置換)
  - `docs/research/07_nn_only_benchmarks.md` — 純粋NN日本語G2Pのベンチマーク徹底調査 (v2.0 で「越えるべき baseline」に framing shift)
  - `docs/research/08_market_landscape.md` — 市場に存在する日本語対応NN-G2Pモデル 網羅的インベントリ
  - `docs/research/09_pure_nn_g2p_benchmarks.md` — 越えるべき pure-NN 先行 4 モデルの徹底整理

## 設計の核心思想 (実装時の判断基準)

以下は pure-NN pivot 後の設計原則。実装時に迷ったらこの原則に戻る:

1. **Pure-NN G2P として学習・推論する**。dict lookup / MeCab 出力を **推論パスに含めない**。ただし char-level BERT (P-C) の pre-tokenize は「内部トークナイザー」として許容し、pyopenjtalk / MeCab を推論に呼ぶことはしない。
2. **NHK Kurihara Interspeech 2024 の TJ-G2P + BAS の architecture は継承**。T5 seq2seq (encoder) と char-BERT (BAS) の 2 段を、hybrid の post-processor としてではなく **multi-task の 1 head として encoder に統合** する。
3. **Hida ICASSP 2022 のマルチタスク (G2P + 多音字 + APBP + ANPP)** を multi-head で実装する。主観 MOS 3.67 (対 オラクル 3.69) の near-oracle 品質を目指す根拠。BAS を 5 つ目の head に追加する。
4. **SentencePiece トークナイザーの警告に注意**: SB Intuitions 自らが「token classification タスクで性能が悪い」と modernbert-ja HFカードで明記。naive per-token classification は避け、seq2seq か char-level BERT のどちらかを Phase 2 で head-to-head 比較して選ぶ (2 pilot P-A/P-C 並列)。**P-B (MeCab pretokenize) は v2.0 pivot で drop — MeCab 依存が pure-NN 原則に反するため**。
5. **Pure-NN の実証的失敗パターン (docs/research/07 の 5 事例、docs/research/09 の追加分析) は既知の baseline**。同じ失敗を避けるため、以下 3 点を学習前に固定する:
   - (a) **データ量**: PnG BERT (Aozora 4.9M) / Kakegawa (news 5M) より 5-10 倍以上のコーパス (5M+ sentence-level pairs、char-level では 50M+ token) を確保
   - (b) **マルチタスク supervision (Hida 2022 型)** を pretrain と fine-tune の両方で活用
   - (c) **評価は JSUT / JVS / ROHAN 3 本柱に統一** (先行 pure-NN は指標が異なり比較不能だった)

### 補足: pyopenjtalk-plus (surface, yomi, accent) の扱い

- **教師信号としての利用は許容**: MLM 継続 pretrain / 事前学習 pseudo-label としては使う (PnG BERT が Aozora + morph 解析 pseudo-label で pretrain したのと同じ慣行)
- **推論時の利用は禁止**: 推論パスに pyopenjtalk / MeCab / 辞書 lookup を呼ばない
- **既知の限界**: 辞書由来の教師信号は辞書エラーを NN が継承する rule-leakage である。Frontier LLM が 0.52% を出せるのは web-scale の非辞書コーパスがあるからで、我々の 130M では辞書 pretrain が実質的な上限を作りうる。この点は `docs/research/09` §7 で明示的に self-audit する

## 主要な数値目標

Pure-NN research の tier 設計。**Reference values (haqumei 1.17%, OpenJTalk 1.03%, Claude Opus 4.6 0.52%) は表の脚注で参照値として扱い、acceptance criteria には含めない**。

| 指標 | Pure-NN baseline (先行研究) | Conservative | Stretch | Aspirational |
|---|---|---|---|---|
| JSUT Basic5000 PER | CharsiuG2P 10.51% (自著 held-out, IPA) | < 5.0% | < 3.0% | < 0.6% (Frontier LLM 並み) |
| JVS-3000 kana CER | pure-NN 公開値なし (Phase 2 で自己測定; Koriyama Interspeech 2026 arxiv 2606.22009 が benchmark 定義) | < 5.0% | < 2.0% | < 0.62% (Gemini 3.1 Pro 越え) |
| ROHAN KER | pure-NN 公開値なし (Phase 2 で自己測定) | < 5.0% | < 2.0% | < 1.0% |
| JSUT モーラアクセント精度 | PnG BERT の whole-seg 45.5% からは換算不能 | > 90% | > 96.66% (Hida hybrid 越え) | > 98% |
| Multi-head compatibility (Phase 3) | 該当なし (先行 pure-NN は single-task) | G2P / polyphone / APBP / ANPP / BAS の 5 head を joint training し、G2P 単独 fine-tune 比 PER 相対改善 > 5% | > 15% | > 30% |
| 多音字 hard-set PER | pure-NN 公開なし (我々が公表) | 公表する | pyopenjtalk baseline との差を最小化 | 逆転 |

**脚注 (target ではない参考値)**:
- haqumei JSUT PER 1.17% / ROHAN KER 1.64% — pyopenjtalk-plus 辞書 primary の hybrid 天井
- OpenJTalk JVS-3000 kana CER 1.03% — 純 rule
- Claude Opus 4.6 0.52% / Gemini 3.1 Pro 0.62% — LLM (parse mode 依存度未定量)
- Ohnaka et al. INTERSPEECH 2025 (arxiv 2506.04527) LARGE-TTSaug PER 0.93% — **speech+text encoder** (line-distilbert-base-japanese) なので pure-text encoder のスコープ外だが、NN 側の到達可能上限の informed reference として認識する

## ベースモデル選定 (推奨: sbintuitions/modernbert-ja-130m)

- **主軸**: `sbintuitions/modernbert-ja-130m` (132M params, MIT, 8k context)
- **Ablation 対象**: 30m / 70m / 310m, `llm-jp/llm-jp-modernbert-base`, `tohoku-nlp/bert-base-japanese-char-v2`
- **理由**: サイズと精度のバランス、MIT ライセンスで商用可、4 サイズ展開で系統的 ablation 可能。CC-G2PnP と同じ SB Intuitions 系列の base LM を pure-NN G2P に転用する初の試みとなる (`yomi-linter-modernbert-ja-130m` は G2P-adjacent linter であり text→phoneme ではない)

## データセット & 評価

### 主要評価 (必須3本柱)

- **JVS-3000** (Koriyama Interspeech 2026 benchmark, arxiv 2606.22009) — kana CER
- **JSUT Basic5000** (jsut-label) — PER。**Kurihara 2024 の JSUT400 split (400 文) も並列で報告** し先行 pure-NN との direct 比較を可能にする
- **ROHAN 4600** — KER

### 主要学習データソース

- pyopenjtalk-plus 辞書 (~800K エントリ) — **教師信号として使う。推論には使わない**
- UniDic 全エントリ (~1M) — **教師信号として使う。推論には使わない**
- Wikipedia 日本語版 ふりがな抽出 (~500K sentences)
- 青空文庫 ふりがな付きテキスト (~200K sentences)
- **総量目標: 5M+ sentence-level pairs** (PnG BERT / Kakegawa の 5-10 倍以上を pure-NN 失敗パターン (a) 回避のため確保)

### ライセンス警告

- 商用配布を想定する場合、`pyopenjtalk-plus` / UniDic / JSUT / ROHAN / JMDict / Wikipedia は各々異なる条項 (BSD / MIT / CC-BY-4.0 / CC-BY-SA-4.0)。実装前に一次資料で再確認。特に **Share-Alike (SA)** 系のライセンスは重み配布時に注意。
- 研究プロジェクトとしての公開 (HF Hub artifact + arxiv preprint) が primary output なので、商用配布よりも学術的再現性を優先する。

## 開発フロー (v2.0, hybrid phase 削除)

- **P0** (1 週): baseline 測定 — pure-NN 先行 4 モデル (CharsiuG2P / PnG BERT / Kakegawa TJ-G2P / CC-G2PnP checkpoint) を再測定可能な範囲で JSUT / JVS / ROHAN 上で数値化。haqumei / OpenJTalk / Frontier LLM は reference table として固定
- **P1** (2 週): 学習データ生成 (5M+ pairs 目標)
- **P2** (2 週): pure-NN シングルタスク G2P baseline + トークナイザー 2 pilot (P-A seq2seq / P-C char-level BERT。P-B MeCab-pretokenize は v2.0 pivot で drop)
- **P3** (2 週): マルチタスク学習 (G2P + polyphone + APBP + ANPP + BAS の 5 head)
- **P4** (2 週, **hybrid → pretrain-plus-fine-tune に置換**): MLM 継続 pretrain — pyopenjtalk-plus / UniDic の (surface, yomi, accent) を教師信号として MLM 追加 supervision に活用。**推論に辞書を持ち込まないが、教師データとして使うのは pure-NN の慣行に合致**。Exit: JSUT PER < 5.0% で CharsiuG2P baseline を明確に上回る
- **P5** (3 週, **拡大・主役化**): Scale & Ablation — 30/70/130/310m × 2 tokenizer × 3 data scale の Pareto / scaling law を公開。Frontier LLM approach の architectural 分析、2 pilot ensemble、LLM 蒸留の小規模実験を含む
- **P6** (1 週): 評価・公開 — HF Hub artifact 公開 + arxiv preprint。**preprint Table 1 = pure-NN 4 モデル vs ours の JSUT / JVS / ROHAN 3 本柱 comparison**。Exit criteria: 「pure-NN 4 モデルすべてを 3 本柱の全指標で上回る」

**合計 11-12 週** (P4 hybrid 削除 -2 週、P5 拡大 +1〜2 週、P3 compaction -1 週で正味 1〜2 週短縮)

詳細は `docs/research/06_implementation_roadmap.md` v2.0 参照。

## 参照優先順位

コーディング中に事実確認が必要な時の優先順位:

1. `docs/research/` 内の該当ドキュメント (プロジェクト内合意)
2. 各ドキュメントに引用されている一次情報 URL (論文 arxiv、HFカード、公式GitHub)
3. Web検索 (二次情報)

## 実装時の技術的注意事項

- **音素表記**: JULIUS 音素セット + モーラアクセント H/L + アクセント句境界マーカ '/' を canonical とする。IPA (CharsiuG2P 系) との mapping table を Phase 0 で確定
- **トークナイザー起因の失敗モード**: Phase 2 で判断が確定するまで、seq2seq / char-level の 2 並列パイロット (P-A/P-C) を維持する。P-B (MeCab-pretokenize) は v2.0 pivot で drop
- **カテゴリ別 sample reweighting**: 数詞 / 固有名詞 (漢字/カタカナ) / 助数詞語 / 外来語 / 英単語混在 / 英字略語 には `sample_weight = 2.0` を推奨
- **Hard-set**: **7 カテゴリ (多音字/助数詞/固有名詞/カタカナ外来語/数詞・単位/英単語混在文/英字略語) 各 200 文** の人手キュレーションを Phase 1 で作成し、以降のすべての評価で使用
- **多言語混在対応 (MUST)**: 実世界の日本語文には英単語・略語・記号連結語 (iPhone, PDF, AI, Wi-Fi, e-mail 等) がデフォルトで混在する。以下を実装 (pure-NN 制約下で):
  - 英単語混在 → Kanalizer 流の音写 or CMUdict→日本語音素マッピングを **学習データに合成**、推論時は NN が seq2seq / classification で解く
  - 英字略語 → アルファベット読み (AI→エーアイ) と単語読み (NASA→ナサ) を **NN token classification のみで判定** (辞書 + 文脈 hybrid ではない)
  - 英数字・単位 (10km, 3GB, 2025年, 10:30, 3.14) の正規化ロジックも NN で end-to-end
- **Reproducibility (研究プロジェクトなので強化)**: 全ての ablation は同じ seed / 同じ splits で実行し、結果表を 1 つに統合する。すべての ablation 数値を HF Hub artifact + wandb で開示する

## Pure-NN 日本語G2P の先行研究 (超えるべき baseline)

以下は 07 / 09 で確定した pure-NN 先行研究。「失敗パターン」ではなく「本プロジェクトが越えるべき baseline」として位置づけを反転させた:

- **PnG BERT (Yasuda & Toda 2022, arxiv 2212.08321)**: BERT-base ~110M。事前学習 validation で masked-G2P whole-word accuracy 45.5% (JSUT 別途 PER は未公表)。TTS accent MOS 2.51 vs rule-labeled Tacotron 3.04 で敗北。→ 我々は同 architecture size 帯で JSUT PER として直接測定可能な数値を公表する
- **Kakegawa TJ-G2P**: Kakegawa 2021 原論文は newspaper 5,142 文で Word-Accuracy 89.8-98.0% (rule に 0.2-0.5pt 敗北)。Kurihara & Sano 2024 が JSUT400 で re-implement し PPL CER 11.85% (OpenJTalk 10.82%)。counter words で 15.43% → BAS 統合で 1.42% は **辞書 substitution + BAS 由来の改善で NN 貢献ではない**。→ 我々は counter/proper-noun hard-set を pure-NN のみで解く
- **CharsiuG2P (byT5, Zhu et al. Interspeech 2022, arxiv 2204.03067)**: 多言語 aggregate PER 0.089。日本語自著 held-out (IPA-vs-dict word-list) で PER 10.51%。Monolingual-JA-only ByT5 は 66.89% に崩壊 → 多言語共学習が主に効いている。→ 我々は monolingual JA で 10.51% を明確に下回れば primary target 達成
- **CC-G2PnP (Shirahata & Yamamoto 2026, arxiv 2602.17157, SB Intuitions)**: Conformer-CTC streaming G2PnP。内部 6D-Eval で PnP CER 1.79-1.80%、Dict-DNN-NS hybrid が 1.71% で勝つ。**6D-Eval は private 2,722 文で再現不可** — 我々は JSUT / JVS / ROHAN で置き換えて評価する
- **Ohnaka et al. INTERSPEECH 2025 (arxiv 2506.04527)**: speech+text encoder + line-distilbert-base-japanese で LARGE-TTSaug PER 0.93%。**pure-text ではなく speech-conditioned** なのでスコープ外だが、NN 側の informed reference として認識
- **XPhoneBERT / fo-BERT / UR-BERT はG2Pモデルではない** — G2P benchmark として引用しない

## Refuted (反証済み) の主張 — 参照禁止

deep-research の 3 票敵対的検証で棄却された主張。文献引用時に混同しないよう注意:

- **却下 (01調査)**: 「NHK Kurihara 2024 が Japanese G2P を pure neural では unsolvable と framing し、[ ] タグ付き hybrid architecture を提案」— この記述は該当論文には存在しない。dual Transformer (T5 + BERT) の実際の architecture description に忠実に基づくこと
- **却下 (07調査, 0-3)**: 「PnG BERT (arxiv 2212.08321) は pure-NN が lexicon dictionary の Japanese word coverage に本質的に及ばないと明示的に framing」— 論文本文にこの強い主張は無い。数値の敗北は事実だが、著者は "coverage-limited" とは言っていない
- **却下 (07調査, 0-3)**: 「CC-G2PnP は Dict-DNN hybrid ベースラインを 6D-Eval で両指標で上回る」— 論文本文の主張だが verify で棄却。paper 本文は "streaming G2PnP task" での improvement のみ主張しており、Table 1 で non-streaming Dict-DNN-NS が全指標で勝つ
- **却下 (07調査, 1-2)**: PnG BERT の training labels が Kuromoji+Neologd pseudo-labels であるという主張。数値 45.5% 自体は verified だが labels origin の詳細断定は refute (paper は "morphological analysis" とのみ記述)
- **新規 (v2.0)**: 「pure-NN で hybrid を **pure-text 入力の条件で** 越えた事例が公開ベンチに存在する」という強い主張は 2026-07 時点で存在しない。Ohnaka 2025 が PER 0.93% を出しているが speech+text 入力である点に注意。本プロジェクトは pure-text 入力の pure-NN でこの空白を埋めることを目標とする。09 追加調査で発掘される新事例があれば同セクションに追記する

## ワークフロー的な補足

- 新規コードを書くときは `superpowers:brainstorming` を必ず先に呼び、要件を明確化してから TDD で進める
- 既存コードを触るときは `superpowers:systematic-debugging` で仮説→検証を明示化する
- 実装完了時は `superpowers:verification-before-completion` で証拠を集めるまで「完了」宣言を保留する
- 現時点でコードは存在しないため、Phase 0 の作業を始める場合は `superpowers:writing-plans` で先に実装計画を書き起こす
- **Pure-NN 制約の自己検査**: 実装中 / PR review 時に「推論パスに pyopenjtalk / MeCab / dict lookup を呼んでいないか」を確認する。教師信号として辞書を使うのは OK、推論に呼ぶのは NG

## 方針転換履歴 (v2.0 pivot)

**日付**: 2026-07-04
**バージョン**: v2.0 (Pure-NN Research Pivot)

### 変更の背景

ユーザーからの明示的方針転換: 「ルールベースの g2p の処理が入るのであればそれでいいのでこのプロジェクトをする必要がないです」

v1.4 までの hybrid 設計 (辞書 primary + ModernBERT correction) は、docs/research/02 §A.3 で確定した通り「結果の 80-95% が pyopenjtalk-plus 辞書由来」であり、130M ModernBERT を新規開発する動機自体が消える。同じ辞書を採用しつつ NN で差別化する戦略も、絶対値では haqumei/OpenJTalk に劣り、「差別化軸」 (BAS / 多音字 / 略語判定) の改善が hybrid ceiling の枠内でしか効かないことが判明した。

### 主な変更点

1. **プロジェクト目的の再定義**: production drop-in から研究プロジェクトへ。ターゲットを OpenJTalk / haqumei / Frontier LLM の 3 ティアから、pure-NN 先行 4 モデル (CharsiuG2P / PnG BERT / Kakegawa TJ-G2P / CC-G2PnP) の primary tier + reference / aspirational の 3 段に再構成
2. **設計原則 1 の反転**: 「単一 neural モデルで OpenJTalk を置き換えない」→ 「Pure-NN G2P として学習・推論する (推論に dict lookup / MeCab を呼ばない)」
3. **設計原則 5 の削除**: 「haqumei は rule 天井、同じ辞書を採用しつつ NN で差別化」戦略を削除。代わりに「pure-NN 失敗パターン回避 3 点セット (データ量 5-10 倍、multi-task supervision、3 本柱統一評価)」を新原則 5 として追加
4. **数値目標の再構成**: haqumei / OpenJTalk 絶対値越えを削除。JSUT PER 目標を「haqumei 越え < 1.0%」から「CharsiuG2P baseline 越え < 5.0% (conservative) / < 3.0% (stretch) / < 0.6% (aspirational, Frontier LLM 並み)」に変更。reference values は脚注に降格
5. **Phase 4 hybrid 化の削除**: 旧 P4 (2 週、辞書 primary + NN correction) を廃止。代わりに **P4 = pretrain-plus-fine-tune** を挿入 — pyopenjtalk-plus / UniDic を教師信号として MLM 継続 pretrain に活用 (推論に辞書を持ち込まない)
6. **Phase 5 の拡大**: 2 週 → 3 週。Scale / Ablation を主役に昇格。Frontier LLM architectural 分析、2 pilot ensemble、LLM 蒸留を含める
7. **Phase 6 の焦点変更**: 「haqumei / OpenJTalk 越え」→ 「pure-NN 4 モデルすべてを越える」。pyopenjtalk 互換 wrapper は削除 (production API compat 不要)。arxiv preprint Table 1 = pure-NN 4 モデル比較を固定
8. **失敗パターン framing の反転**: 「Pure-NN 日本語G2P の実証的失敗パターン」→ 「Pure-NN 日本語G2P の先行研究 (超えるべき baseline)」。5 事例の記述は保持、解釈フレームを反転
9. **Refuted 項目の拡張**: 「pure-NN が pure-text 入力で hybrid を越えた事例は 2026-07 時点で存在しない」を追加。Ohnaka 2025 の speech+text 例外を明記

### 関連ファイル更新

- `docs/requirements.md` v2.0: FR-20〜23 (hybrid pipeline), FR-35 (pyopenjtalk 互換), FR-50〜54 (haqumei 越え戦略) を削除。FR-60〜62 (pure-NN 制約) を新規追加
- `docs/research/06_implementation_roadmap.md` v2.0: P4 hybrid 削除、pretrain-plus-fine-tune 置換、P5 拡大
- `docs/research/07_nn_only_benchmarks.md` v2.0: framing shift (「失敗」→「越えるべき baseline」)、内容は保持
- `docs/research/09_pure_nn_g2p_benchmarks.md`: 新規追加。4 pure-NN 先行研究の徹底整理、protocol harmonization、data-supervision as rule-leakage の self-audit

### 認識している risks / open questions

- **OQ-1**: PnG BERT の train recipe 詳細 (data size, epoch, LR) を公開範囲で抽出
- **OQ-2**: CharsiuG2P byT5 の JSUT Basic5000 上の再測定 (我々の環境で)
- **OQ-3**: CC-G2PnP の 6D-Eval と JSUT の overlap 有無、same-split 再測定可能性
- **OQ-4**: Kakegawa TJ-G2P alone の code 公開有無 (Kurihara 論文の付録参照)
- **OQ-5**: Frontier LLM (Claude/Gemini) の pure decode-mode (parse mode 切) の kana CER
- **OQ-6**: 最小の ModernBERT-Ja サイズで pure-NN JSUT PER < CharsiuG2P 10.51% を達成できる scaling law
- **Risk**: pyopenjtalk-plus 教師信号は rule-leakage の性質を持ち、辞書エラーが NN の暗黙 ceiling を作る可能性がある。09 §7 で self-audit を義務化
- **Risk**: hybrid に届かない negative result となる可能性は認識するが、「pure-NN の JSUT / JVS / ROHAN 上の scaling law を初めて公開する」ことに学術的意義があるため preprint publishable と判断する
