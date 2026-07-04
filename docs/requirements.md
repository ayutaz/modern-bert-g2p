# 要求定義書 — ModernBERT日本語G2P

**バージョン:** 2.0 (**Pure-NN Research Pivot**)
**作成日:** 2026-07-04 (v1.0 初版 / v1.1 補強 / v1.2 blocker調査反映 / v1.3 haqumei深掘り反映 / v1.4 プロソディ出力形式拡張 / **v2.0 Pure-NN pivot**)
**根拠:** `docs/research/01_overview.md` 〜 `08_market_landscape.md` の調査結果、追加調査、Phase 0 blocker 並列調査、haqumei徹底解剖、haqumei ProsodyFormat 実測、**v2.0 で pure-NN pivot 設計 spec (2026-07-04) と Review-1 adversarial critique の 4 blocker (B1〜B4) を反映**
**ステータス:** **Phase 0 開始可** (v1.2 で全 blocker 解決)

---

## ⚠️ v2.0 Pure-NN Pivot 宣言 (2026-07-04)

**背景**: ユーザー方針転換 —「ルールベースの g2p の処理が入るのであればそれでいいのでこのプロジェクトをする必要がないです」。**hybrid 経路は結果の 80-95% が辞書由来** (haqumei 徹底解剖より) のため、130M ModernBERT を書く動機自体が消える。

**新 positioning**:
- **Character**: 研究プロジェクト (production drop-in ではない)
- **Target**: pure-NN Japanese G2P の空白領域を JSUT/JVS/ROHAN 標準ベンチで埋める初のデータポイント
- **勝つべき相手 (must-beat, pure-text NN のみ)**: CharsiuG2P / PnG BERT / Kakegawa TJ-G2P / CC-G2PnP
- **参考値 (target ではない)**: haqumei 1.17% / OpenJTalk 1.03% / Frontier LLM 0.52-0.62% / Ohnaka 2025 speech+text 0.93%

**v2.0 の破壊的変更**:
- **削除 (deprecated)**: FR-20〜23 (hybrid 推論), FR-35 (pyopenjtalk 互換テスト), FR-50〜53 (haqumei 越え攻撃戦略の hybrid 系), CR-26 (pyopenjtalk-plus 辞書 primary), CR-43 (pure-NN 禁則)
- **降格 (MUST → SHOULD)**: NFR-05, NFR-06, NFR-17, CR-27, CR-28
- **新規**: FR-60〜62 (pure-NN 制約), NFR-08〜09 (pure-NN 先行研究 baseline), CR-90〜92 (pure-NN inference boundary + データ教師信号の rule-leakage 開示)
- **反転**: CR-43 (単一 pure-NN 禁則 → pure-NN MUST)

**Review-1 adversarial critique の 4 blocker (B1〜B4) への対応**:
- **B1 (protocol 不整合)**: FR-61 で先行研究 4 モデルを我々の環境で同一 metric に換算・再測定することを MUST 化
- **B2 (CC-G2PnP 数値誤り)**: 「6D-Eval SER 8.4%」は誤り。正しくは Table 1 PnP CER 1.79-1.80%, Phoneme CER 0.48-0.52%, Dict-DNN-NS が 1.71/0.40 で勝利。09 ドキュメントで訂正
- **B3 (Frontier LLM 数値の一次確認)**: 0.52/0.62% は Koriyama Interspeech 2026 (arxiv 2606.22009) を直接引用して確定
- **B4 (pyopenjtalk-plus 教師信号 = rule leakage 相当)**: CR-91 で「辞書由来 pretrain 教師 = rule-based signal の学習パイプへの持ち込み」を明示自己申告 (推論 pure-NN との境界を分離)

---

## 0. ドキュメントの読み方

- 要件ID規則:
  - `FR-*` = 機能要件 (Functional Requirement)
  - `NFR-*` = 非機能要件 (Non-Functional Requirement)
  - `CR-*` = 制約 (Constraint)
  - `AC-*` = 受け入れ基準 (Acceptance Criteria)
- **MUST** / **SHOULD** / **MAY** はRFC 2119の意味で使用
- 各要件は該当する調査ドキュメントセクションへの trace linkを持つ

---

## 1. プロジェクト概要

### 1.1 プロジェクト名

ModernBERT日本語G2P (仮称 — 公開時にブランド名確定)

### 1.2 目的 (**v2.0 pure-NN pivot**)

**ModernBERT (encoder-only pure-NN) の Japanese G2P 応用可能性を検証する研究プロジェクト**。出力パスに rule / dict lookup を含めない (pure-NN, no dict inference)。Primary target は公開されている pure-NN Japanese G2P モデル群を JSUT/JVS/ROHAN 標準ベンチで系統的に上回ること。

**3ティアの competitive landscape**:
1. **Primary (must-beat, pure-text NN のみ)**: CharsiuG2P (own-dict-holdout Japanese PER 10.51% / monolingual 66.89%) / PnG BERT (pretraining validation whole-word G2P accuracy 45.5%, JSUT test PER 未公表) / Kakegawa TJ-G2P (Kurihara 2024 の re-implementation で JSUT400 PPL CER 11.85%) / CC-G2PnP (6D-Eval PnP CER 1.79-1.80%, ただし 6D-Eval は SB Intuitions 私有 2,722 文セット、Dict-DNN-NS が 1.71/0.40 で勝利)
2. **Reference (compare but not target)**: OpenJTalk 1.03% / haqumei 1.17% — hybrid 天井なので pure-NN 側の 130M では絶対値では劣る前提。gap を可視化する
3. **Aspirational (world-first if achieved)**: Frontier LLM (Koriyama Interspeech 2026, arxiv 2606.22009) の Claude Opus 4.6 JVS-3000 kana CER 0.52% / Gemini 3.1 Pro 0.62% を 130M pure-NN で近似できるか

**Informed reference (speech-conditioned, competition scope 外)**: Ohnaka et al. INTERSPEECH 2025 (arxiv 2506.04527) の speech+text encoder + line-distilbert-base-japanese が LARGE-TTSaug PER 0.93% — pure-text NN が到達しうる上限の informed reference。本プロジェクトは pure-text のみで戦うため直接競合ではないが、NN が hybrid 並の精度に到達しうるという存在証明として引用可能。

### 1.3 スコープ

**含む** (In-scope):
- 日本語テキスト → 音素列 + モーラアクセント + アクセント句境界 の変換モデル (**pure-NN, no dict inference**)
- 学習・評価パイプラインと再現可能なスクリプト
- 3本柱ベンチマーク (JSUT Basic5000 / JVS-3000 / ROHAN 4600) の公開結果
- pure-NN 先行研究 4 モデル (CharsiuG2P / PnG BERT / Kakegawa TJ-G2P / CC-G2PnP) を同一プロトコルで再測定した比較 table
- Hugging Face Hub / GitHub での公開
- ~~pyopenjtalk互換API (Style-Bert-VITS2等へのドロップイン置換)~~ **(deprecated 2026-07-04 v2.0 pivot)** — 研究プロジェクトなので production API compat は out-of-scope

**含まない** (Out-of-scope):
- TTS音響モデル (voice cloning, prosody generation)
- **中国語・韓国語・その他アジア言語混在文** (英日混在は in-scope だが、他言語は Phase 6 以降)
- **ローマ字入力からのG2P** (「konnichiwa」等の全ローマ字入力は out-of-scope、部分的な英単語混在のみ扱う)
- 独立した多言語 G2P モデル (英単独、中単独 等の G2P 提供)
- 音声認識前段のG2P
- リアルタイム・ストリーミング推論最適化 (Phase 6以降で検討)

**国際化スコープの明確化 (v1.1 追加)**:

| 入力パターン | 対応 | 例 |
|---|---|---|
| 純日本語 (漢字/ひらがな/カタカナ) | ✅ **in-scope** | 「今日は雨だ」 |
| 英単語が日本語文に混在 | ✅ **in-scope** | 「iPhone を買った」 |
| 英字略語が日本語文に混在 | ✅ **in-scope** | 「AI エンジニア」 |
| 英数字・単位が混在 | ✅ **in-scope** | 「10km 走った」 |
| 記号連結語 | ✅ **in-scope** | 「Wi-Fi 環境」 |
| 全英単語 (日本語なし) | ⚠️ **best-effort** | 「Hello World」→ 英→カナ音写のみ |
| 中国語・韓国語混在 | ❌ **out-of-scope** | 「北京語で挨拶」の中国語部分等 |
| ローマ字全文 | ❌ **out-of-scope** | 「konnichiwa」等 |
| 音素記号入力 (JULIUS等) | ❌ **out-of-scope** | 「k o N n i ch i w a」 |

### 1.4 ステークホルダーと想定利用者

| ステークホルダー | 期待 |
|---|---|
| 音声合成研究者 | pure-NN 日本語 G2P の JSUT/JVS/ROHAN 標準ベンチ上の scaling law データポイント |
| 音声/NLP アカデミア | pure-NN G2P 先行研究 (PnG BERT / CharsiuG2P / Kakegawa / CC-G2PnP) との横断比較 table |
| 日本語音声NLPコミュニティ | ModernBERT × G2P の応用可能性検証結果 (positive/negative result 両方) |
| ~~OSS TTS開発者~~ **(deprecated v2.0)** | ~~pyopenjtalk のドロップイン置換~~ — 本プロジェクトは pure-NN 研究であり production TTS drop-in ではない |
| ~~商用TTSサービス提供者~~ **(deprecated v2.0)** | 同上 — Apache-2.0 のオープン重みは配布するが production API 互換保証はしない |

---

## 2. 機能要件 (Functional Requirements)

### 2.1 コアG2P機能

- **FR-01** [MUST]: 任意の日本語テキストを入力として受け付ける。**実世界の日本語文にはデフォルトで多言語要素が混在するため、以下すべてを一体の入力として扱う**:
  - 漢字、ひらがな、カタカナ
  - **英単語がアルファベット表記のまま埋め込まれるケース** (例: 「iPhone を買った」「PDF を開く」)
  - **英字略語** (AI, NASA, HTML, PDF, WHO, GDP等)
  - **英数字混在** (10km, 3GB, 2025年, 10:30, 3.14, Ver.2.0)
  - **記号・ハイフン連結語** (Wi-Fi, e-mail, T-shirt, C++)
  - **カタカナ外来語** (コンピュータ、アプリ、コーヒー)
  - **絵文字・機種依存文字** (Phase 1 の正規化で除去/置換対象)
- **FR-02** [MUST]: 入力テキストに対して JULIUS音素セット準拠の音素列を出力する ([05設計 §4.1])
- **FR-03** [MUST] (**v1.4 拡張**): 各モーラに対してアクセント記号 (H / L) を付与した出力を提供する。**アクセント (単語レベル) + イントネーション (文レベル) を1つのシーケンスに統合する**
- **FR-04** [MUST] (**v1.4 拡張**): アクセント句境界マーカを含む出力形式を提供する。以下の記号を canonical にサポート:
  - `^` : 文頭 (BOS)
  - `$` : 文末 (EOS)
  - `#` : アクセント句境界 (intonation phrase boundary)
  - `[` : アクセント句開始 / pitch 上昇位置 (L→H)
  - `]` : アクセント核 / pitch 下降位置 (H→L)
  - `?` : 疑問文の上昇イントネーション
  - **haqumei の `ProsodyFormat` と互換にする** (tdmelodic 記法, 実物確認済み)
- **FR-04a** [MUST] (**v1.4 追加**): 以下の**3種類の出力フォーマット**をサポート (haqumei と同じ設計)
  - **tdmelodic 風**: `['^', 'ky', 'o', ']', 'o', 'w', 'a', '#', 'a', ']', 'm', 'e', 'd', 'a', '$']`
  - **Prefix (L_/H_)**: 各音素に `L_` / `H_` プレフィックス — 例: `['H_ky', 'H_o', 'L_o', ...]`
  - **Numeric (:0/:1)**: 各音素に `:0` / `:1` サフィックス — 例: `['ky:1', 'o:1', 'o:0', ...]`
- **FR-04b** [MUST] (**v1.4 追加**): **単語単位 API** を提供 (haqumei の `g2p_mapping_prosody` 相当) — 各単語について:
  - `accent_nucleus`: アクセント核位置 (数値、0=平板、1=頭高等)
  - `mora_count`: モーラ数
  - `chain_flag`: アクセント句連結フラグ (-1=独立/新句頭, 0=連結中止, 1=連結継続) — **イントネーション句境界の制御信号**
  - `chain_rule`: 連結規則 ("C1" / "動詞%F2@0" 等)
  - `pos` / `pos_group1〜3`: 品詞細分類 (固有名詞判定等に必要)
  - `pron`: カタカナ発音
- **FR-05** [SHOULD]: X-SAMPA、カタカナ音素表記への変換ユーティリティを提供する
- **FR-05a** [SHOULD] (**v1.4 追加**): 疑問文以外の**プロソディ・イベント**もサポート:
  - 感嘆文 (!) の下降強調
  - 引用境界 (「」内) の pitch reset
  - 明示的な pause 挿入 (、。「,」「.」)

### 2.1a 多言語混在文の扱い (**新規追加**)

日本語文に埋め込まれた多言語要素をどう音素化するかは、精度に直接影響する。以下を **MUST** 要件とする:

- **FR-06** [MUST]: 英単語混在部分について、以下いずれかの戦略で音素列を生成する:
  - **Strategy A**: 英→カタカナ音写 → 日本語音素化 (Kanalizer流。VOICEVOX/kanalizer-model のパスに準拠)
  - **Strategy B**: 英発音記号 (CMUdict / IPA) を経由し、日本語音素にマッピング
  - **Phase 2** で Strategy A / B の両方をパイロット評価し、精度・レイテンシ・カバレッジで最良を確定する
- **FR-07** [MUST] (**v2.0 pure-NN 修正**): 英字略語 (AI, NASA等) を **アルファベット読み** (例: AI → エーアイ) と **単語読み** (例: NASA → ナサ) で区別する。**判定は NN token classification のみで行う (推論時の外部辞書 lookup を含めない)**。ただし学習信号としての辞書由来 (surface, reading) ペアの利用は許容 (CR-91 の rule-leakage 開示範囲内)
- **FR-08** [MUST]: 英数字・単位表現 (10km, 3GB, 2025年, 10:30, 3.14) の正規化・読み上げロジックを実装する
- **FR-09** [MUST]: 記号連結語 (Wi-Fi, e-mail, T-shirt) を1トークンとして扱い、既存辞書または外来語NNで音素化する
- **FR-0A** [SHOULD]: 未知の混在パターン (SNS発の造語、絵文字混在) に対するフォールバック戦略を実装する

### 2.2 マルチタスク・ヘッド

以下のマルチタスク定式化を **MUST** 実装する ([05設計 §3.1] / [04論文 §A.3])

- **FR-10** [MUST]: G2P main head (seq2seq または token classification) — 音素列生成
- **FR-11** [MUST]: 多音字曖昧性解消 head (Polyphone disambiguation)
- **FR-12** [MUST]: アクセント句境界予測 head (APBP: Accent Phrase Boundary Prediction)
- **FR-13** [MUST]: アクセント核位置予測 head (ANPP: Accent Nucleus Position Prediction)
- **FR-14** [MUST]: アクセント sandhi 補正 head (BAS: Accent Sandhi correction、NHK BAS相当)

### 2.3 ~~ハイブリッド推論パイプライン~~ **(全項目 deprecated 2026-07-04 v2.0 pure-NN pivot)**

**v2.0 で全面削除**。理由: pure-NN pivot により推論パスに rule/dict lookup を含めない (FR-60 で新規MUST化)。詳細な削除項目:

- ~~**FR-20**~~ (**deprecated v2.0**): ~~pyopenjtalk (辞書lookup) を primary path として使用する~~
- ~~**FR-21**~~ (**deprecated v2.0**): ~~辞書lookup の uncertainty region 検出~~
- ~~**FR-22**~~ (**deprecated v2.0**): ~~uncertainty region に対する条件付き NN 推論~~
- ~~**FR-23**~~ (**deprecated v2.0**): ~~reconciliation strategy 比較~~

### 2.4 API

- **FR-30** [MUST]: `g2p(text: str) -> phonemes: str` の pyopenjtalk互換シグネチャを提供する
- **FR-31** [MUST]: `g2p(text, kana=True)` でカタカナ出力オプションをサポートする
- **FR-32** [MUST]: `g2p(text, return_accent=True)` でアクセント情報付き出力を提供する
- **FR-33** [SHOULD]: バッチ推論API (`g2p_batch(texts: List[str])`) を提供する
- **FR-34** [MAY] (**v2.0 reframe**): Rust bindings を提供する。~~haqumei 互換 API~~ という production drop-in 目的は削除し、pure-NN 推論の高速化バインディングとして位置づける
- ~~**FR-35**~~ (**deprecated 2026-07-04 v2.0 pivot**): ~~pyopenjtalk 互換テストスイートで辞書ヒット入力の一致率 ≥ 99.0%~~ — pure-NN が dict と一致することは目標ではない。研究プロジェクトなので production drop-in 保証は out-of-scope

### 2.5a ~~haqumei 越えのための攻撃戦略~~ **(FR-50〜53 deprecated 2026-07-04 v2.0 pivot、FR-54 のみ保持)**

**v2.0 の理由**: v1.3 の攻撃戦略は「同じ pyopenjtalk-plus 辞書を採用しつつ NN 補正を上乗せする hybrid 差別化」を前提としていた。pure-NN pivot により hybrid 差別化戦略自体が消えるため、以下 4 項目は deprecated。ただし FR-54 (mora accent 公表) のみ pure-NN でも独立に有効なので保持し、FR-40 系に統合する。

- ~~**FR-50**~~ (**deprecated v2.0**): ~~アクセント連続変異 (BAS/accent sandhi) 補正 → haqumei に対する hybrid 差別化~~ → **保持だがフレーム変更**: BAS は multi-task の 1 head として encoder に統合 (FR-14 で既に MUST 化済)。「haqumei 越え」の framing のみ削除
- ~~**FR-51**~~ (**deprecated v2.0**): ~~多音字曖昧性解消 NN → haqumei の辞書優先度依存への差別化~~ → **保持だがフレーム変更**: 多音字 head は FR-11 で既に MUST 化済。「haqumei 越え」の framing のみ削除
- ~~**FR-52**~~ (**deprecated v2.0**): ~~英字略語判定 NN → haqumei Kanalizer への差別化~~ → **保持だがフレーム変更**: 略語判定は FR-07 で pure-NN として再定義済み
- ~~**FR-53**~~ (**deprecated v2.0**): ~~未知語処理 → haqumei の pau 落としへの差別化~~ → **保持だがフレーム変更**: 未知語処理は FR-01 の in-scope 入力扱いに統合済
- **FR-54** [MUST] (**v2.0 で SHOULD → MUST 昇格 + 意味変更**): **mora accent accuracy を独自公表指標として公表する**。pure-NN の JSUT モーラアクセント accuracy を公開ベンチとして提示することで、pure-NN 系列で情報優位を獲得する。haqumei 未公表領域という framing は削除、pure-NN 先行研究 (PnG BERT / CharsiuG2P / CC-G2PnP) が未公表であることを新規優位性根拠にする

### 2.5 モデル配布

- **FR-40** [MUST]: PyTorch (Hugging Face Transformers) 形式で配布する
- **FR-41** [MUST]: ONNX 形式で配布する (haqumei風のデプロイパスに乗るため)
- **FR-42** [SHOULD]: MODEL_CARD / INFERENCE_GUIDE / LICENSEの標準ドキュメント一式を含める
- **FR-43** [MAY]: 量子化バリアント (int8) を配布する
- **FR-44** [MUST] (**v1.1 追加**): モデルタグは **SemVer準拠** (`v{MAJOR}.{MINOR}.{PATCH}[-{alpha|beta|rc}.{N}]`) で採番する
  - MAJOR = 音素表記 / API シグネチャ 互換破壊時
  - MINOR = 新機能・新オプション追加時 (後方互換)
  - PATCH = 精度改善・バグ修正 (完全後方互換)
  - Phase 3完了時 = `v0.1.0-alpha.1`, **Phase 4' (Pretrain-plus-fine-tune + Scale/Ablation) 完了時** = `v0.2.0-beta.1`, Phase 6公開時 = `v1.0.0`
- **FR-45** [SHOULD] (**v1.1 追加**): CHANGELOG.md を SemVer リリースごとに更新し、精度差分と互換破壊を明記する

### 2.6 Pure-NN 制約 (**v2.0 新規追加**)

- **FR-60** [MUST] (**v2.0 新規**): **推論時に外部辞書 lookup を発生させない (pure-NN 制約)**。推論パスで pyopenjtalk / MeCab / UniDic / JMDict / pyopenjtalk-plus 辞書等を呼ばない。char-level pre-tokenize や morpheme 情報を **学習信号** として使うことは許容 (CR-91 の rule-leakage 開示範囲内で明示自己申告)。ただし推論時の tokenizer 内部処理 (SentencePiece / char-level BERT の pre-tokenize) は "内部トークナイザー" として許容し、これは辞書 lookup とは区別する
- **FR-61** [MUST] (**v2.0 新規、Review-1 B1 対応**): **pure-NN 先行研究 4 モデルを JSUT/JVS/ROHAN 3 本柱の同一 split・同一 metric で再測定する**。既存文献の指標不整合 (CharsiuG2P own-dict-holdout PER vs Kurihara JSUT400 PPL CER vs PnG BERT pretraining validation whole-word acc vs CC-G2PnP 6D-Eval PnP CER) を全部 PER/kana CER/KER に換算し公表する。**引用時は元の metric 名を保持し、換算後の値と併記する** (protocol harmonization は `docs/research/09` の §4 で規定)
- **FR-62** [SHOULD] (**v2.0 新規**): **マルチタスク supervision** (polyphone / APBP / ANPP / BAS) を学習に使うが、推論時にはこれら head の出力を **hybrid で辞書と統合しない**。head の出力は全て NN の順伝播結果として提示する

---

## 3. 非機能要件 (Non-Functional Requirements)

### 3.1 精度目標 (**v2.0 pure-NN 3 段構成に再設計**)

**v2.0 pivot** により pure-NN 世界内の tier を Conservative / Stretch / Aspirational の 3 段構成に。**haqumei / OpenJTalk / Frontier LLM は reference-only** (target ではない)。**pure-NN 先行研究の baseline は Review-1 B1 の通り protocol 不整合のため、FR-61 の再測定完了までは "参照値" として扱う**。

| 指標 | Pure-NN Baseline (protocol 不整合、FR-61 で再測定要) | Conservative (MUST) | Stretch (SHOULD) | Aspirational (MAY) | 参照 |
|---|---|---|---|---|---|
| **NFR-01** (v2.0 pure-NN 化): JVS-3000 kana CER | (pure-NN 公開値なし、Phase 2 で自己測定) | < 5.0% | < 2.0% | < 0.62% (Gemini 3.1 Pro 越え) | [03 §1.1] + [09] |
| **NFR-02** (v2.0 pure-NN 化): JSUT Basic5000 PER | CharsiuG2P own-dict-holdout ~10.51% (**IPA-vs-dict word-list PER, ≠ JSUT sentence PER**, Review-1 B1) | < 5.0% | < 2.0% | < 0.5% (Frontier LLM 並み) | [03 §1.2] + [09] |
| **NFR-03** (v2.0 pure-NN 化): ROHAN 4600 KER | (pure-NN 公開値なし、Phase 2 で自己測定) | < 5.0% | < 2.0% | < 1.0% | [03 §1.3] + [09] |
| **NFR-04**: JSUT モーラアクセント精度 | (Hida 97.33% は hybrid なので reference のみ、pure-NN 公開値なし) | > 90% | > 96.66% (Hida hybrid 越え) | > 98% | [04 §A.3] |
| ~~**NFR-05**~~ (**deprecated 2026-07-04 v2.0**): ~~haqumei と同一プロトコルの JSUT PER~~ → **reference-only 降格**: haqumei 1.17% は参考値として測定するが acceptance criteria には含めない | haqumei 1.17% | (参考のみ) | — | — | reference |
| ~~**NFR-06**~~ (**deprecated 2026-07-04 v2.0**): ~~haqumei と同一プロトコルの ROHAN KER~~ → **reference-only 降格**: 同上 | haqumei 1.64% | (参考のみ) | — | — | reference |
| **NFR-07** (**v1.3 → v2.0 で意味変更**): 本プロジェクト独自公表指標 — JSUT モーラアクセント精度公表 | (pure-NN 先行研究 全 4 モデルが未公表) | 必ず公表する | > 96.66% | > 98% | 情報優位獲得 |
| **NFR-08** (**v2.0 新規**): **Pure-NN 先行 4 モデル (CharsiuG2P / PnG BERT / Kakegawa TJ-G2P / CC-G2PnP) 越え** — FR-61 の再測定結果に基づき、JSUT/JVS/ROHAN 3 本柱の 全 metric で明確に上回る | 4 モデルの max value | 全 metric で下回る (MUST) | 4 モデルの best を 2 倍上回る | 4 モデル全て 5 倍以上上回る | [09] |
| **NFR-09** (**v2.0 新規、Review-1 S1 対応**): **Speech-conditioned NN との gap 可視化** (Ohnaka 2025 arxiv 2506.04527 PER 0.93% と Furigana Whisper JSUT closed-set CER 0.19%/6.43% を informed reference として測定) | Ohnaka 0.93% (speech+text) | (target ではない、gap を報告のみ) | — | — | [09] |

**Reference values (target ではない、v2.0 で明示注記)**: haqumei 1.17% / OpenJTalk 1.03% / Claude Opus 4.6 0.52% (Koriyama Interspeech 2026, arxiv 2606.22009) / Gemini 3.1 Pro 0.62% は表の脚注に "hybrid / rule / LLM 参照値" として明示。Frontier LLM は **parse mode = 暗黙 hybrid** の可能性があるため 130M pure-NN 越えは非現実的想定。

### 3.2 Hard-set 精度 (per-category)

以下 **7カテゴリ** の hard-set (各200文) 上で、pyopenjtalk 単独 baseline を上回る必要がある:

- **NFR-10** [MUST]: 多音字 hard-set の PER が pyopenjtalk baseline を上回る
- **NFR-11** [MUST]: 助数詞語 hard-set の PER が pyopenjtalk baseline を上回る
- **NFR-12** [MUST]: 固有名詞 hard-set (漢字 + カタカナ) の PER が pyopenjtalk baseline を上回る
- **NFR-13** [MUST]: カタカナ外来語 hard-set の PER が pyopenjtalk baseline を上回る
- **NFR-14** [MUST]: 数詞 / 日付 / 時刻 / 単位 hard-set の PER が pyopenjtalk baseline を上回る
- **NFR-15** [MUST]: **英単語混在文 hard-set (英単語+日本語混在文 200文) の PER が pyopenjtalk baseline を上回る** — 「iPhone を買った」「PDF を開く」等
- **NFR-16** [MUST]: **英字略語 hard-set (AI/NASA/HTML/e-mail等の混在文 200文) の PER が pyopenjtalk baseline を上回る**
- ~~**NFR-17**~~ (**deprecated 2026-07-04 v2.0**): ~~7 hard-set 全カテゴリで haqumei baseline を 0.5pt 以上上回る~~ → **reference-only 降格**: haqumei 数値は参考として測定するが acceptance criteria から除外。pure-NN 側の hard-set 目標は NFR-10〜16 で pyopenjtalk baseline 越えのみを MUST とする
- **NFR-18** [MUST] (**v2.0 新規**): 7 hard-set 全カテゴリの pure-NN 数値を pure-NN 先行研究 4 モデル (CharsiuG2P / PnG BERT / Kakegawa / CC-G2PnP) の再測定結果と並べて公開する。**pure-NN 先行が hard-set 別 metric を公表していない = 本プロジェクトが初のデータポイントとして情報優位**

### 3.3 推論性能

- **NFR-20** [MUST]: 1文 (~50文字) の p50 レイテンシがGPU (T4) 上で < 20ms
- **NFR-21** [MUST]: 1文 (~50文字) の p50 レイテンシがCPU (x86 8-core) 上で < 100ms
- ~~**NFR-22**~~ (**deprecated 2026-07-04 v2.0**): ~~ハイブリッド推論の平均レイテンシがフル推論より30%以上高速~~ — hybrid path 削除に伴い連動。pure-NN の単一 forward pass のみを最適化対象とする
- **NFR-23** [MAY]: GPU (RTX 4090) 上で p50 < 10ms

### 3.4 モデルサイズ

- **NFR-30** [MUST]: 最終モデルが `sbintuitions/modernbert-ja-130m` (132M params) をベースとする
- **NFR-31** [SHOULD]: 30M/70M/310m の Ablation バリアントも配布する
- **NFR-32** [MAY]: 量子化 (int8) バリアントで元サイズの1/4以下

### 3.4a Ablation 設計 (**v1.1 追加**、調査B に基づく具体化)

先行研究 (Hida 2022: PD 2軸×7設定, PnG BERT: 6システム, CharsiuG2P: 8/12/16層) の慣行に従い、**one-axis-at-a-time** で実行する。フル格子ではない。

- **NFR-33** [MUST]: **総 Ablation runs 数を 40〜60 runs以内**に制限する (単一 GPU 24GB で 8〜14日で完走可能な範囲)
- **NFR-34** [MUST]: **各設定を最低 3 seed** で回し、frontier候補は 5 seed に増やす (Mosbach 25, Dodge 20 の慣行の縮小版)
- **NFR-35** [MUST]: **軸確定の順序**を以下に固定する (先行軸で結果が良かった1点を固定して後続軸を回す):
  1. **Tokenizer 軸** (Phase 2): SP (P-A seq2seq) / char-level (P-C) の 2設定 × 3 seed = 6 runs (130m 固定、中データ、lr=3e-5。**P-B MeCab-pretokenize は v2.0 pivot で drop**)
  2. **Learning rate 軸** (Phase 2 末): 1e-5, 3e-5, 5e-5 の 3設定 × 3 seed = 9 runs (以降 LR は sweep しない)
  3. **Model size 軸** (Phase 5): 30m / 70m / 130m / 310m の 4設定 × 3 seed = 12 runs
  4. **Data 軸** (Phase 5): 100万 / 200万 / 500万 文 の 3設定 × 3 seed = 9 runs (scaling law)
  5. **LoRA vs full FT** (Phase 5, 310m のみ): 2設定 × 3 seed = 6 runs
- **NFR-36** [MUST]: 各 Ablation run の結果は Weights & Biases もしくは同等の実験管理ツールで永続化し、リポジトリからリンクを張る
- **NFR-37** [SHOULD]: **統計検定**として paired two-tailed t-test で mean/std を報告する
- **NFR-38** [SHOULD]: Ablation 結果を **Pareto frontier図** (x軸=モデルサイズ、y軸=PER) と **scaling law図** (x軸=データ量、y軸=PER) の2枚で公開する

### 3.5 再現可能性

- **NFR-40** [MUST]: すべての Ablation は同じ seed / 同じ splits で実行し、結果表を1つに統合する
- **NFR-41** [MUST]: 学習・評価スクリプトを CI で動作確認可能な状態で公開する
- **NFR-42** [MUST]: 評価データセットの split (train / valid / test) を明示的にリポジトリに記録する

### 3.6 ドキュメント

- **NFR-50** [MUST]: README、MODEL_CARD、INFERENCE_GUIDE、LICENSEの4文書を公開時に整備
- **NFR-51** [SHOULD]: `docs/research/` の技術ドキュメントを最新状態で維持
- **NFR-52** [MAY]: 論文または arxiv preprint を公開

### 3.7 計算コスト (**v1.1 追加**、調査A に基づく)

- **NFR-60** [MUST]: **単一 GPU 24GB** (RTX 4090 / L4 / A5000 / A10G 等) で fine-tune 1 run が完走できる設定を primary configuration とする
  - 想定: seq_len=1024, batch_size=16〜32, bf16 + Flash Attention 2, gradient accumulation=2〜4
  - 想定 GPU 時間: **1 run あたり 5〜10 GPU-hours** (100万文 × 3 epochs, マルチタスク5head)
- **NFR-61** [MUST] (**v1.2 で Vast.ai 実相場に更新**): **全 Ablation (40〜60 runs) の総 GPU コストが US$100 以下**で完結する見積り
  - **primary クラウド: Vast.ai** (2026-07 実測)
  - **primary GPU: RTX 4090 24GB** — 最安 $0.14/hr, p50 $0.35/hr, 50+ offers 利用可能
  - **fallback GPU: RTX 3090 24GB** — 最安 $0.116/hr, p50 $0.149/hr, より安定・大量供給
  - 予算試算: 60 runs × 8h = 480 GPU-hours → RTX 4090 最安 $67 / p50 $170 / RTX 3090 最安 $56
  - CI check: Vast.ai の在庫が枯渇した場合 RunPod Community に自動フォールバック
- **NFR-62** [SHOULD]: DeepSpeed / FSDP / ZeRO 等の分散学習は使用しない (130m は分散不要の閾値以下)
- **NFR-63** [MAY]: 310m variant の Ablation では LoRA (r=16, alpha=32) を primary、full fine-tune を Ablation対照とする (メモリ節約率 20〜30%)
- **NFR-64** [MUST]: **学習を open reproducible にする** — 各 run の hyperparameter, seed, GPU 種別, 実測時間, 実測 VRAM 使用量を `experiments/logs/` に永続化

### 3.8 性能回帰許容範囲 (**v1.1 追加**)

- **NFR-70** [MUST]: PATCHリリース (`v0.M.p → v0.M.(p+1)`) では、3本柱ベンチマーク (JVS/JSUT/ROHAN) のいずれも精度が悪化しないこと
- **NFR-71** [MUST]: MINORリリース (`v0.M.0 → v0.(M+1).0`) では、3本柱の平均が改善すること
- **NFR-72** [MUST]: 特定ベンチマークで **0.3ポイント以上の悪化がある場合は CI で reject** する自動リグレッションテストを設ける
- **NFR-73** [SHOULD]: 各リリースで 7 Hard-set の精度差分を CHANGELOG に記載する

---

## 4. 制約 (Constraints)

### 4.1 ベースモデル

- **CR-01** [MUST]: `sbintuitions/modernbert-ja-130m` を主軸ベースモデルとして使用する ([05設計 §1.1])
- **CR-02** [MUST]: Ablation対照として最低1つ以上の別トークナイザ系モデル (`llm-jp-modernbert-base` または `bert-base-japanese-char-v2`) を head-to-head 比較する
- **CR-03** [MUST]: **SB Intuitions が明記する SentencePiece トークナイザーの token classification 弱点に対応する** ([01 §3.3])
  - Phase 2 で seq2seq (P-A) / char-level BERT (P-C) の 2並列パイロットを維持し、最良を選定 (**P-B MeCab-pretokenize は v2.0 pivot で drop — MeCab 依存が pure-NN 原則に反するため**)

### 4.2 データ

- **CR-10** [MUST] (**v1.2 で更新**): 学習データは以下から構築する ([03 §5.1]):
  - pyopenjtalk-plus 辞書 (MIT + Modified BSD)
  - UniDic 全エントリ (BSD-New)
  - Wikipedia日本語版 ふりがな抽出 (CC-BY-SA-4.0, 法的立場を Model Card 記載)
  - 青空文庫 ふりがな付きテキスト (public domain)
  - **llm-jp-corpus** (Apache-2.0) — v1.2 で追加
  - ⚠️ **JSUT Basic5000 は学習データから完全除外** (v1.2 で CC-BY-SA-4.0 判明のため評価専用)
- **CR-11** [MUST] (**v1.2 で URL 追加**): JVS-3000 nonpara subset (Koriyama Interspeech 2026 benchmark) は評価専用に固定、学習に混入させない
  - 入手経路: `git clone https://github.com/CyberAgentAILab/jvs_nonpara_kana`
  - データ実体: `jvs_nonpara_kana.csv` (3,000文の手動アノテート kana)
  - 評価: 同梱の `eval_cer.py` を直接パイプラインに組み込む (長音記号バリアント正規化ロジック込み)
- **CR-12** [MUST]: 全データをJSONスキーマに正規化し `data/processed/` に保存する ([06 §Phase 1])
- **CR-13** [MUST]: 数詞 / 固有名詞 (漢字/カタカナ) / 助数詞語 / 外来語 のサンプルには `sample_weight = 2.0` を適用する ([03 §7])
- **CR-14** [MUST] (**v1.2 で手法を確定**): **7カテゴリ (多音字/助数詞/固有名詞/カタカナ外来語/数詞・単位/英単語混在文/英字略語) 各200文=計1,400文の hard-set を Phase 1 中にキュレーションする**
  - **primary手法: LLM半自動 + 人手 diff review (C案)** — Claude Opus 4.6 / Gemini 3.1 Pro / pyopenjtalk / UniDic の 3〜4-way diff で「発散セル」のみ人手 review。API費 < ¥5,000, 実効 12〜25h (2〜4営業日)
  - **根拠**: Koriyama Interspeech 2026 で Claude Opus 4.6 が JVS-3000 kana CER 0.52% を達成 → LLM 出力を正解候補として扱う運用が学術的に成立
  - **Phase 0 開始前の必須事前作業**: **カテゴリ別 seed set 20文 × 7カテゴリ = 140文** の gold ラベルを開発者自身で手作業で作成 (LLM 校正精度の Cohen's kappa 測定 baseline)
  - **品質保証**: (a) 3〜4-way diff で発散セルのみ精査、(b) JGLUE 式 majority-vote (LLM 複数モデル)、(c) 最終公開版は音声学専門家 (東工大郡山研 / 東大齋藤研 系) に 100文サンプル監修依頼 (謝金 ¥3〜5万想定)
- **CR-15** [MUST]: 学習データに英単語混在パターンを **最低5万文** 含める (Wikipedia日本語版のtech関連記事、GitHub日本語READMEクローリング等)
- **CR-16** [MUST]: 英単語 → カタカナ音写辞書 (Kanalizer + CMUdict派生等) を Phase 1 で整備し、Strategy A のフォールバック用リソースとして固定する

### 4.3 ライセンス

- **CR-20** [MUST] (**v1.2 で Apache-2.0 に確定**): 公開モデル weights のライセンスは **Apache-2.0** とする。patent grant、下流OSS TTS互換性、ModernBERT系との整合が根拠 (OPEN-05 解決参照)
- **CR-21** [MUST] (**v1.2 で精緻化**): 学習データソースのライセンス条項を尊重する ([03 §6]):
  - JSUT **音声**: CC-BY-4.0 (帰属明示)
  - **JSUT テキスト (Basic5000 含む)**: **CC-BY-SA-4.0** → ⚠️ **eval only に厳格分離、学習に混入禁止**
  - UniDic: BSD派生 (帰属明示)
  - Open JTalk辞書 / pyopenjtalk-plus: 修正BSD / MIT
  - **Wikipedia**: CC-BY-SA-4.0 → **学習に使用可、Model Card に法的立場明記** (先例: Japanese StableLM, LLM-jp-3)
  - **JMDict**: CC-BY-SA-4.0 → ⚠️ **runtime lookup に限定 (学習gradientに含めない、Misaki の先例に準拠)**
  - **JVS-3000 kana annotation** (CyberAgent AI Lab): CC-BY-SA-4.0 → ⚠️ **評価専用 held-out に厳格分離**
  - Aozora Bunko: public domain (制限なし)
  - llm-jp-corpus: Apache-2.0 (制限なし)
- **CR-22** [MUST] (**v1.2 で更新**): Model Card に以下の法的立場を明記する:
  > "Training data includes CC-BY-SA-4.0 sources (Wikipedia-JA). Model weights are released under Apache-2.0 based on the position that trained weights are not a derivative work of training data (CC 2025 primer; Andersen v. Stability 2025; Japan Copyright Act Art. 30-4)."
- **CR-23** [MUST] (**v1.2 追加**): **JMDict は推論時 runtime lookup にのみ使用** — training gradient に含めない設計を実装する (Style-Bert-VITS2 + Misaki 先例)
- **CR-24** [MUST] (**v1.2 追加**): **JSUT Basic5000 と JVS-3000 kana は評価専用**、学習コーパスから自動的に除外する CI check を設ける (data leakage 防止)
- **CR-25** [SHOULD] (**v1.2 追加**): permissive-only スタック (Aozora 1.6M + UniDic + pyopenjtalk-plus + llm-jp-corpus) だけで 100万文コーパスを構築するオプションを常に維持する (Wikipedia依存の代替として)
- ~~**CR-26**~~ (**deprecated 2026-07-04 v2.0**): ~~pyopenjtalk-plus 辞書を hybrid path の primary 辞書に採用する~~ — pure-NN pivot により hybrid path 自体が削除。ただし **pretraining の教師信号としての (surface, yomi) ペア利用は許容** (CR-91 参照, rule-leakage 開示範囲内)
- **CR-27** [SHOULD] (**v1.3 → v2.0 で MUST → SHOULD 降格**): haqumei-eval と同一プロトコルの評価スクリプトは reference-only として実装する。**acceptance criteria から除外**。haqumei 数値との比較は Model Card と preprint の reference 章に掲載するが release gate には含めない:
  - JSUT: prj-beatrice/jsut-label の `basic5000.yaml` を SHA256 pin で使用 (reference)
  - `pau` (無音) を無視した Levenshtein 距離ベース PER 計算 (reference)
  - `phone_level3` を canonical レベルとする (reference)
  - `HaqumeiOptions { use_unidic_yomi: true, normalize_iu: Some(Yuu) }` 相当の前処理を実装 (reference)
  - ROHAN 側は `g2k_per_word` 文字単位 Levenshtein に準拠 (reference)
- **CR-28** [SHOULD] (**v1.3 → v2.0 で MUST → SHOULD 降格**): haqumei と本プロジェクトの評価を並走する CI ジョブは reference のみ。差分表は release gate 判定には使わないが preprint Table 1 の reference 列として掲載

### 4.4 開発フロー

- **CR-30** [MUST]: 実装は `docs/research/06_implementation_roadmap.md` の7フェーズ (P0〜P6) の順序で進める
- **CR-31** [MUST]: 新規コード実装前に `superpowers:brainstorming` で要件を明確化する
- **CR-32** [MUST]: 既存コード変更時は `superpowers:systematic-debugging` で仮説→検証を明示化する
- **CR-33** [MUST]: 実装完了時は `superpowers:verification-before-completion` で証拠を集めるまで「完了」宣言を保留する

### 4.5 反証済み前提の禁則

以下の主張は敵対的検証で棄却されているため、設計・実装の前提として使用してはならない ([04 §G], [07 §6], [CLAUDE.md]):

- **CR-40** [MUST NOT]: 「NHK Kurihara 2024 が Japanese G2P を pure neural では unsolvable と framing」と引用しない
- **CR-41** [MUST NOT]: 「PnG BERT が pure-NN の coverage-limited を明示的に framing」と引用しない — Yasuda & Toda 2022 の paper 本文にこの強い主張は存在しない (evidence brief 検証済)。paper が実際に述べているのは "features … were not dominant in surface-form information with their masking strategy"
- **CR-42** [MUST NOT]: 「CC-G2PnP が Dict-DNN hybrid を 6D-Eval で上回った」と引用しない — CC-G2PnP arxiv 2602.17157 Table 1 は Dict-DNN-NS (PnP CER 1.71, Phoneme CER 0.40, MOS 4.07) が CC-G2PnP-NS (1.80, 0.48, 4.02) に勝利。paper 本文の主張だが数値で refute
- **CR-42b** [MUST NOT] (**v2.0 追加、Review-1 B2 訂正**): ~~「CC-G2PnP 6D-Eval SER 8.4%」と引用しない~~ — この数値は redesign spec の誤り。正しい引用は Table 1 の PnP CER 1.79-1.80% / Phoneme CER 0.48-0.52%
- **CR-42c** [MUST NOT] (**v2.0 追加**): 「pure-NN で hybrid を越えた事例が公開ベンチに存在する」という強い主張を pure-text NN 領域で行わない (2026-07 時点で存在しない、pivot 認識前提)。**ただし Ohnaka 2025 arxiv 2506.04527 の speech+text NN が LARGE-TTSaug PER 0.93% で hybrid parity に迫っている点は informed reference として明示引用可**
- ~~**CR-43**~~ (**v2.0 で反転**): ~~単一 pure-NN モデルで pyopenjtalk を置換する設計を採用しない~~ → **CR-43 (v2.0 新規、意味反転)**: **単一 pure-NN モデルで JSUT/JVS/ROHAN を評価する設計を採用する (MUST)**。[07 §7.1] の5つの実証的失敗パターンは "避けるべき前例" から "越えるべき baseline" に framing shift

### 4.5a Pure-NN inference boundary + データ教師信号の rule-leakage 開示 (**v2.0 新規**)

- **CR-90** [MUST] (**v2.0 新規**): **推論時に外部辞書 lookup を発生させない**。以下を推論パイプラインから排除する:
  - pyopenjtalk / pyopenjtalk-plus の run_frontend / g2p / g2p_mapping_prosody 呼び出し
  - MeCab / UniDic morphological analyzer の呼び出し
  - JMDict / EDICT / 独自辞書の lookup
  - Kanalizer ONNX の呼び出し (haqumei が英単語遭遇時のみ発火する経路)
  - 例外: SentencePiece / char-level tokenizer の内部処理は "辞書 lookup" ではないため許容
- **CR-91** [MUST] (**v2.0 新規、Review-1 B4 対応**): **学習信号としての辞書由来ペアの利用を明示的自己申告する**。以下を Model Card / preprint に開示:
  - pyopenjtalk-plus 辞書 (~800K entries) を pretrain の MLM 教師信号として使用
  - UniDic 全エントリ (~1M) を pretrain 教師信号として使用
  - これらは rule-based tool 由来なので **PnG BERT が hit した同じ failure mode (pseudo-label ceiling) を継承する可能性を明示自己申告**
  - **推論 pure-NN と学習信号 rule-supervised** の区別を README / Model Card / preprint で明示する
- **CR-92** [MUST] (**v2.0 新規**): FR-61 の再測定において、pure-NN 先行 4 モデルを **カテゴリ別** に分類して比較 (Review-1 R1 対応):
  - Category-A: pure-text encoder (本プロジェクト = target)
  - Category-B: pure-text seq2seq (CharsiuG2P, Kakegawa TJ-G2P, Kurihara TJ-G2P baseline)
  - Category-C: pure-text encoder+decoder MLM (PnG BERT)
  - Category-D: ASR-style CTC (CC-G2PnP)
  - Category-E (informed reference のみ): speech+text (Ohnaka 2025, Furigana Whisper, Hu 2025)
  - Category-F (informed reference のみ): prosody-only (Koriyama SSW13 2025 arxiv 2507.03912)

### 4.6 計算資源

- **CR-50** [MUST]: 全 fine-tune は bf16 混合精度 + Flash Attention 2 on で実施する
- **CR-51** [SHOULD]: LoRA (r=16, alpha=32) と full fine-tune の両方を Phase 2 で評価
- **CR-52** [MUST] (**v1.1 でMAYから昇格**): **単一 GPU (24GB VRAM級) で完走できる設定を primary configuration** とする — 分散学習に依存しない構成にロックする
- **CR-53** [MUST] (**v1.1 追加**): **130m モデルは full fine-tune を primary**、LoRA を Ablation対照 (BERT系130Mでは LoRA のメモリ節約率が 20〜30% と限定的なため)
- **CR-54** [MUST] (**v1.1 追加**): **310m モデルは LoRA を primary**、full fine-tune を Ablation対照 (メモリ節約効果が現れる規模)
- **CR-55** [MUST] (**v1.1 追加**): 学習は **RunPod / Lambda Labs / Vast.ai 相当のスポット GPU** でも完走できることを CI で確認する (US$100 以下 の総コスト予算に収めるため)

### 4.7 依存関係 (**v1.1 追加**)

- **CR-60** [MUST]: **Python ≥ 3.10** (ModernBERT は Python 3.10+ で公式サポート)
- **CR-61** [MUST]: **PyTorch ≥ 2.1** (Flash Attention 2 対応), **transformers ≥ 4.48** (ModernBERT公式サポート)
- ~~**CR-62**~~ (**deprecated 2026-07-04 v2.0**): ~~pyopenjtalk ≥ 0.4.0 / pyopenjtalk-plus をハイブリッド推論の primary path として使用~~ — pure-NN pivot により hybrid path 削除。pyopenjtalk-plus は **開発時の教師信号生成用ツール** としてのみ dev dependency に残す (CR-91 の rule-leakage 開示範囲)、runtime dependency からは除外
- **CR-63** [SHOULD]: ONNX Runtime ≥ 1.20 (ONNX 配布時のリファレンス推論エンジン)
- **CR-64** [SHOULD]: CUDA 12.1+ / cuDNN 9+ (Flash Attention 2 の推奨バージョン)
- **CR-65** [MAY]: MPS (Apple Silicon) はベストエフォート対応 (Phase 6 の Nice-to-have)

### 4.8 段階リリース戦略 (**v1.1 追加**)

- **CR-70** [MUST]: 以下の**3段階リリースゲート**を設ける:
  - **α (alpha)**: Phase 3完了時 (`v0.1.0-alpha.1`) — 内部評価のみ、HF Hub にはプライベート公開
  - **β (beta)**: **Phase 4' (Pretrain-plus-fine-tune + Scale/Ablation) 完了時** (`v0.2.0-beta.1`) — HF Hub パブリック公開、限定的な社外テスター (Style-Bert-VITS2 コミュニティ主要メンテナ 3〜5名) の feedback 収集
  - **1.0 (stable)**: Phase 6完了時 (`v1.0.0`) — GitHub Releases + arxiv preprint + Model Card 一式公開
- **CR-71** [MUST]: β リリース時点で **AC-01〜03 の全 MUST 要件を満たす**こと (Ablation の完全性のみが Phase 5 で追加される想定)
- **CR-72** [SHOULD]: 各リリース時に downstream TTS (Style-Bert-VITS2 想定) との統合テストを 1件以上実施し、結果を公開する

### 4.9 悪用防止・コンプライアンス (**v1.1 追加**)

- **CR-80** [MUST]: 学習データに含まれる**固有名詞・個人名の扱い** — 公開 Wikipedia / 青空文庫由来のみを使用し、非公開個人情報 (SNS 由来の実名等) は含めない
- **CR-81** [MUST]: **Model Card に bias disclosure を記載** — 学習データ由来のバイアス (地域方言の非対応、古典表記の限定的対応 等) を明示する
- **CR-82** [MUST]: **なりすましTTS への悪用防止** — Model Card に「TTS音響モデルを含まないため voice cloning には直接利用できない」旨を明記
- **CR-83** [SHOULD]: 個人情報保護法 (APPI) 準拠の観点で、Phase 1 の法務レビューに含める (CR-22 と統合)

---

## 5. 受け入れ基準 (Acceptance Criteria)

### 5.1 Phase Gate — 各フェーズを完了と認める条件

- **AC-P0** [**部分達成 v1.3 実測**]: haqumei の JSUT PER 1.17% を ±0.1% 以内で再現でき、3本柱ベースライン (OpenJTalk / haqumei) の測定が1コマンドで再生成できる
  - ✅ **haqumei JSUT PER 実測: 1.1657% vs 官報 1.17%, Diff 0.0043 pt** (2026-07-03, Python 3.12 + haqumei 0.8.0 on macOS aarch64)。再現手順: `scripts/eval_haqumei_jsut.py` 参照
  - ⏳ pyopenjtalk JVS-3000 kana CER 1.03% の実測再現は未実施 (B-03)
  - ⏳ 3-baseline 1コマンド化 (`scripts/eval_baselines.sh`) は未実装 (B-05)
- **AC-P1**: 統合スキーマの全データが `data/processed/` に格納され、5 hard-set が揃っている
- **AC-P2**: 2並列パイロット (seq2seq / char-level。**P-B MeCab-pretokenize は v2.0 pivot で drop**) の head-to-head 結果に基づき、Phase 3以降の主軸トークナイザ戦略が確定している
- **AC-P3**: JSUT Basic5000 PER < 1.5% かつ JSUT accent-labeled subset mora-accent accuracy > 96.5% を達成する **(v2.0 注: pure-NN で。1.5% は stretch tier に位置し得るが、pretrain-plus-fine-tune の最終段階を想定した Phase 3 exit criteria として保持)**
- ~~**AC-P4**~~ **(v2.0 で書き換え)**: ~~haqumei 越え — JSUT PER < 1.0%, ROHAN KER < 1.5%, JVS-3000 kana CER < 0.9%~~ → **AC-P4 (v2.0 pure-NN 版)**: **pretrain-plus-fine-tune の pure-NN で JSUT PER < 5.0% を達成し、CharsiuG2P baseline (own-dict-holdout 10.51%) を **同一プロトコル (FR-61)** で明確に上回る**。旧 hybrid Phase 4 は削除、pretrain-plus-fine-tune に置換 (詳細は 06_implementation_roadmap.md の v2.0 更新)
- **AC-P5**: 「モデルサイズ vs 精度」「トークナイザー vs 精度」「学習データ量 vs 精度」の Pareto/scaling 曲線を提示できる **(v2.0 で 3 週に拡大、Frontier LLM approach 分析 / ensemble ablation / LLM 蒸留予備実験を追加)**
- **AC-P6** (**v2.0 更新**): Hugging Face Hub / GitHub 公開、~~pyopenjtalk互換API~~ **(pyopenjtalk 互換 wrapper は削除、削除)**、ONNX配布、標準ドキュメント一式が揃っている。**preprint Table 1 に pure-NN 4 モデル (JSUT/JVS/ROHAN × PnG BERT / Kakegawa / CharsiuG2P / CC-G2PnP / ours) 比較 table が固定されている (Category-A/B/C/D 別で)**

### 5.2 最終受け入れ (**プロジェクト完了**の条件)

- ~~**AC-01**~~ **(v2.0 で書き換え)**: ~~3本柱すべてで haqumei / OpenJTalk を明確に上回る (最低0.15%以上の差)~~ → **AC-01 (v2.0 pure-NN 版)**: **pure-NN 先行 4 モデル (CharsiuG2P / PnG BERT / Kakegawa TJ-G2P / CC-G2PnP) を 3 本柱の全指標で上回る** (Category-A/B/C/D 全部を pure-text NN として並べた上で) — NFR-01, NFR-02, NFR-03, NFR-08
- **AC-02** (**v2.0 で stretch 降格**): フロンティアLLM (Gemini 3.1 Pro 0.62%) との JVS-3000 差 < 0.2% は **stretch のみ** (MUST から SHOULD 相当に降格)
- **AC-03**: **7カテゴリ Hard-set全てで pyopenjtalk baseline を上回る** (NFR-10〜16) **(v2.0: pure-NN の順伝播のみで達成する)**
- ~~**AC-04**~~ (**deprecated 2026-07-04 v2.0**): ~~Style-Bert-VITS2 に投入した downstream TTS pronunciation CER が既存 pyopenjtalk投入時より改善する~~ — production TTS drop-in が目標でなくなったため削除
- ~~**AC-05**~~ (**deprecated 2026-07-04 v2.0**): ~~pyopenjtalk のドロップイン置換として15分以内に動作確認~~ — research project positioning のため削除
- **AC-06**: モデル配布 (HF Hub) + 評価スクリプト再現 (GitHub) が第三者に成立している
- ~~**AC-07**~~ (**deprecated 2026-07-04 v2.0**): ~~pyopenjtalk 互換テスト (FR-35) で辞書ヒット入力の一致率 ≥ 99.0%~~ — FR-35 削除に伴い連動
- **AC-08** (**v1.1 追加**): **全 Ablation の総 GPU コストが US$100 以下** で完結している (実測レシート/クラウド利用明細で証跡)
- **AC-09** (**v1.1 追加**): CHANGELOG.md に v0.1.0-alpha → v1.0.0 の全リリース精度差分と互換破壊が記録されている
- **AC-10** (**v1.1 追加**): Model Card に bias / misuse disclosure が記載され、法務レビューを経ている
- **AC-11** (**v2.0 新規**): Model Card / preprint に **CR-91 の rule-leakage 開示 (pyopenjtalk-plus / UniDic の教師信号としての利用) と CR-90 の inference boundary (pure-NN 推論保証) の対比** が明記されている
- **AC-12** (**v2.0 新規**): FR-61 の pure-NN 先行 4 モデル同一プロトコル再測定結果が preprint Table 1 として掲載され、protocol harmonization (Review-1 R4) の §4 が `docs/research/09` に記載されている

---

## 6. 優先順位マトリクス

### 6.1 MoSCoW

**MUST have (P0〜P4 で完了)** (**v2.0 pure-NN 化**):
- **Pure-NN 先行 4 モデル越え** (AC-01) — CharsiuG2P / PnG BERT / Kakegawa TJ-G2P / CC-G2PnP
- **Pure-NN 制約 (推論時 dict lookup なし)** (FR-60〜62, CR-90〜92)
- ~~ハイブリッド推論パイプライン (FR-20〜22)~~ **(deprecated v2.0)**
- HF Hub / GitHub 公開 (FR-40〜41)
- ~~pyopenjtalk互換API + 互換テスト (FR-30〜32, FR-35)~~ **(FR-35 deprecated v2.0; FR-30〜32 は pure-NN の g2p() 関数として保持)**
- **多言語混在文対応 (英単語/略語/英数字/記号連結語) を pure-NN で** (FR-06〜0A, FR-07 は v2.0 で NN-only 化)
- **SemVer準拠のバージョニング** (FR-44) — v1.1追加
- **単一 24GB GPU 完走の primary configuration** (CR-52, NFR-60) — v1.1追加
- **Pretraining-plus-fine-tune (Phase 4 置換)** で JSUT PER < 5.0% (AC-P4 v2.0)

**SHOULD have (P5〜P6 で完了)**:
- モデルサイズ Ablation (30M/70M/130M/310M) — NFR-33〜35 に具体化
- 量子化 (int8) 配布
- LLM stretch target (< 0.62% JVS-3000)
- Pareto frontier / scaling law の可視化公開 (NFR-38)
- 段階リリース (α → β → 1.0) (CR-70〜72) — v1.1追加

**COULD have (Phase 6以降で検討)**:
- Rust bindings (haqumei互換)
- ストリーミング推論
- LLM蒸留による精度向上
- MPS (Apple Silicon) 対応 (CR-65) — v1.1追加

**WON'T have (今回スコープ外)**:
- TTS音響モデル / voice cloning
- リアルタイム音素ストリーミング
- モバイル・エッジ最適化
- **中国語・韓国語・その他アジア言語混在文** — v1.1で明示化
- **ローマ字全文からの G2P** — v1.1で明示化

### 6.2 Kano分析 (品質モデル) (**v2.0 pure-NN 再設計**)

- **Must-be** (無いと不満): Pure-NN 先行 4 モデル越え (CharsiuG2P / PnG BERT / Kakegawa / CC-G2PnP)、pure-NN 制約 (推論 dict-free)、商用可ライセンス
- **One-dimensional** (あるほど満足): モデルサイズが小さい、レイテンシが低い、Hard-set精度が高い、pure-NN scaling law の解像度が高い
- **Attractive** (あると驚喜): LLM並みの精度 (0.62%) を 130M pure-NN で近似、pure-NN 4 モデルを 5 倍以上上回る、日本語 pure-NN G2P 標準ベンチマークリーダーボードを立ち上げる、Ohnaka 2025 の speech+text NN と pure-text NN で parity に迫る

---

## 7. リスクと予防策 (要件レベル)

| リスクID | 内容 | 予防策 | 対応要件 |
|---|---|---|---|
| **RISK-01** | トークナイザー選定の失敗 | Phase 2で2並列パイロット (P-A/P-C) | CR-03 |
| **RISK-02** | データライセンス問題 | Phase 1開始前に法務レビュー | CR-22 |
| **RISK-03** | 計算資源不足 | LoRAとfull fine-tuneの両方を用意、130mから開始 | CR-51, CR-52 |
| **RISK-04** | JVS-3000評価データ未公開 | 論文著者に問い合わせ、代替として自作JVSサブセット | AC-02 |
| **RISK-05** | ModernBERTトークナイザー起因の性能悪化 | Phase 2で確認、必要ならchar-levelにpivot | CR-03 |
| **RISK-06** (**v2.0 で意味反転**) | pure-NN で hybrid parity に到達できない可能性 | negative result も contribution として preprint publication path を確保、pretrain-plus-fine-tune (Phase 4 置換) + scaling ablation で最大限探索、Ohnaka 2025 speech+text の 0.93% を pure-text NN の informed reference として引用 | CR-40〜42, CR-90〜92, AC-P4 v2.0 |
| **RISK-17** (**v2.0 新規**) | pyopenjtalk-plus 教師信号 = rule-based label pseudo-labeling で PnG BERT と同じ failure mode (surface-form 情報が dominant にならない) を継承する | CR-91 で明示自己申告し、pretrain 段階の validation で早期 detection、必要なら教師信号を UniDic のみに縮小 | CR-91, FR-62 |
| **RISK-18** (**v2.0 新規、Review-1 B1 対応**) | 先行研究 4 モデルの protocol 不整合により FR-61 の再測定が困難 (CharsiuG2P 私有 test split, PnG BERT pretraining validation, Kakegawa 未 JSUT, CC-G2PnP 6D-Eval 私有) | 再測定不能なモデルは "own metric + our metric" の両方併記し、我々の JSUT/JVS/ROHAN 上の直接測定を主軸に | FR-61, CR-92 |
| **RISK-07** | Hard-setアノテーション不足 | Phase 1で明示的キュレーション時間確保 | CR-14 |
| **RISK-08** | Loss weight tuning が難航 | Phase 3で grid search を初期から計画 | (Phase 3 内部) |
| **RISK-09** | 英単語混在文の学習データ不足 | Wikipedia tech記事、GitHub日本語README等の追加クローリング | CR-15 |
| **RISK-10** | 英→カタカナ音写のカバレッジ不足 (未知綴りの英単語) | Kanalizer NN + CMUdict + フォールバック規則の3層構造 | FR-06, CR-16 |
| **RISK-11** | 略語のアルファベット読み vs 単語読み判定失敗 | 文脈依存の学習 + 明示的な略語辞書 (AI→エーアイ 等) 200件以上を Phase 1 で整備 | FR-07 |
| **RISK-12** (v1.1) | Ablation 総 runs が単一GPUの実効上限を超え Phase 5 が期限超過 | one-axis-at-a-time で 40〜60 runs に事前制限、frontier候補のみ 5 seeds | NFR-33〜35 |
| **RISK-13** (v1.1) | クラウド GPU スポット価格が想定を超え US$100 予算オーバー | RunPod Community / Vast.ai の複数プロバイダを並行検討、on-demand H100 は使わない | NFR-61, CR-55 |
| **RISK-14** (**v2.0 reframe**) | ~~pyopenjtalk 互換テストが FR-35 の 99.0% を下回る~~ (FR-35/FR-23 削除に伴い旧リスク廃止) → **Phase 4' (Pretrain-plus-fine-tune) の pretrain 段が fine-tune 後 PER を改善しない** | Phase 4' で pretrain 有無 ablation を必ず実施し gold-only fine-tune と併記、寄与ゼロなら教師信号構成を見直す | AC-P4 v2.0, FR-62 |
| **RISK-15** (v1.1) | 悪用防止の観点で TTS ベンダから苦情 | Model Card の bias / misuse disclosure を Phase 6 レビューで法務確認 | CR-81, CR-82 |
| **RISK-16** (v1.1) | 性能回帰 CI (NFR-72) が false positive で PR merge を阻害 | Baseline の統計的分散を Phase 5 で測定し閾値を調整 | NFR-70〜72 |

---

## 8. トレーサビリティ (要件 → 調査ドキュメント)

各要件が根拠とする調査ドキュメント:

| 要件領域 | 根拠 |
|---|---|
| コア設計思想 | `01_overview.md §3`, `05_technical_design.md §1〜3` |
| 精度目標 | `01_overview.md §2`, `03_datasets_and_benchmarks.md §1〜3` |
| ~~ハイブリッド戦略~~ → Pure-NN 先行研究 baseline (**v2.0 pivot**) | `02_existing_systems.md §F`, `04_papers_and_references.md §A.2`, `08_market_landscape.md §9〜10`, **`09_pure_nn_g2p_benchmarks.md` (pivot 後の主 trace)** |
| マルチタスク定式化 | `04_papers_and_references.md §A.3`, `05_technical_design.md §3.1` |
| トークナイザー警告 | `01_overview.md §3.3`, `05_technical_design.md §2` |
| pure-NN禁則 | `07_nn_only_benchmarks.md §7`, `CLAUDE.md` |
| 市場空白領域 | `08_market_landscape.md §9〜10` |
| ライセンス制約 | `03_datasets_and_benchmarks.md §6` |
| フェーズ運用 | `06_implementation_roadmap.md` |
| **計算コスト・Ablation** (v1.1) | HuggingFace `sbintuitions/modernbert-ja-130m` モデルカード, Answer.AI ModernBERT blog, Phil Schmid fine-tune benchmarks, RunPod pricing 2026-07, Hida ICASSP 2022 (arxiv 2201.09427), PnG BERT (arxiv 2212.08321), CharsiuG2P (arxiv 2204.03067) |
| **依存関係バージョン** (v1.1) | ModernBERT 公式サポート要件 (PyTorch 2.1+, transformers 4.48+) |

---

## 9. 未確定事項 (Phase 0 開始前の解決状態)

**v1.2 で 4/5 項目が解決**。残 1 (OPEN-04 ブランド名) は Phase 0〜3 中の任意タイミングで確定可。

- **OPEN-01** [✅ **RESOLVED v1.2**]: JVS-3000 kana アノテーションは **CyberAgent AI Lab** の GitHub リポジトリ `CyberAgentAILab/jvs_nonpara_kana` で完全公開。3,000文の手動アノテート kana + `eval_cer.py` 同梱。CC-BY-SA-4.0 (**評価専用 held-out に厳格分離**して学習に混入させない)。論文: Koriyama, "Benchmarking LLMs for G2P: A Japanese Case Study", Interspeech 2026, arxiv:2606.22009
- **OPEN-02** [✅ **RESOLVED v1.2**]: Wikipedia日本語版 CC-BY-SA-4.0 の Share-Alike は、**モデル重み配布に継承しない解釈が支配的** (CC 2025 公式プライマー、Andersen v. Stability 判決、日本著作権法 30条の4)。先例: **Japanese StableLM / LLM-jp-3 が「日本語 Wikipedia + Apache-2.0 weights」を実施済み**。Model Card に法的立場を明記する形で採用可
- **OPEN-03** [⏳ Phase 2 で確定]: 主軸トークナイザ戦略 (a=seq2seq / c=char-level。**b=MeCab-pretokenize は v2.0 pivot で drop**) は Phase 2 の 2並列パイロット結果を待つ (v1.1 NFR-35 で確定手順を規定済み)
- **OPEN-04** [⏳ Phase 0〜3 任意タイミング]: モデル公開ブランド名 / リポジトリ名。仮称 "ModernBERT日本語G2P" のまま Phase 0 開始可、公開直前に確定
- **OPEN-05** [✅ **RESOLVED v1.2**]: 最終ライセンスは **Apache-2.0** に決定。根拠: (1) Transformer architecture の patent grant 保護 (MIT にはない)、(2) 下流 OSS TTS 全てと互換 (Style-Bert-VITS2 AGPL / GPT-SoVITS MIT / Kokoro-Misaki Apache 等)、(3) ModernBERT (Answer.AI) 系との整合。MIT ベースの sbintuitions/modernbert-ja-130m を fine-tune した派生 weights を Apache-2.0 で配布は MIT permissive の再ライセンス可により合法

---

## 10. 用語集

- **G2P**: Grapheme-to-Phoneme。文字表記→音素列の変換
- **PER**: Phoneme Error Rate。音素列の Levenshtein距離ベースの誤り率
- **kana CER**: kana Character Error Rate。かな文字列レベルのCER
- **KER**: Katakana Error Rate。カタカナ表記のCER
- **APBP**: Accent Phrase Boundary Prediction。アクセント句境界予測
- **ANPP**: Accent Nucleus Position Prediction。アクセント核位置予測
- **BAS**: Accent Sandhi (連続変異) を BERT で予測する module (NHK Kurihara 2024由来)
- **PPL**: Phoneme + Prosodic Label sequence
- **Uncertainty region**: 辞書lookup 結果の信頼度が低いスパン (多音字/OOV/複合語 等)
- **Hard-set**: 特定カテゴリ (多音字等) で難読ケースをキュレーションした評価セット

---

## 11. 変更履歴

| バージョン | 日付 | 変更内容 | 承認者 |
|---|---|---|---|
| 1.0 | 2026-07-03 | 初版。調査01〜08を統合したドラフト | (Phase 0 前に確定要) |
| 1.1 | 2026-07-03 | 追加調査(A計算コスト実測・B先行研究Ablation) を反映して10領域を補強。**国際化スコープの明示** / **バージョニング(SemVer)** / **Ablation具体化(NFR-33〜38)** / **計算コスト規定(NFR-60〜64)** / **性能回帰CI(NFR-70〜73)** / **段階リリース(α→β→1.0)** / **悪用防止・コンプライアンス** / **依存関係バージョン明示** / **pyopenjtalk互換テスト(FR-35)** / **多言語混在対応をMoSCoW MUSTに昇格** | (Phase 0 前に確定要) |
| **1.2** | **2026-07-03** | **Phase 0 blocker 5件の並列調査結果を反映して OPEN-01〜05 のうち4件を解決**。(1) OPEN-01=✅JVS-3000 は CyberAgent AI Lab GitHub で公開、(2) OPEN-02=✅Wikipedia CC-BY-SA-4.0 は先例あり Apache-2.0 weights で配布可 (Japanese StableLM/LLM-jp-3 先例)、(3) OPEN-05=✅**ライセンスを Apache-2.0 に確定**、(4) JSUT テキストが CC-BY-SA-4.0 と判明 → eval only 分離を CR-24 で強制、(5) JMDict は runtime lookup のみ許可 (CR-23)、(6) Vast.ai を primary クラウドに確定 (RTX 4090 primary, RTX 3090 fallback)、(7) Hard-set キュレーションは LLM半自動 (C案) を primary手法に確定 | **Phase 0 開始可** |
| **1.3** | **2026-07-03** | **haqumei v0.8.0 徹底解剖の結果を反映**。(1) **haqumei は "rule 天井"** — PER 1.17% の 80-90% は pyopenjtalk-plus 辞書由来、NN 由来はわずか 0-5% (Kanalizerは英単語遭遇時のみ発火)、(2) 攻撃ポイント FR-50〜54 追加 (BAS / polyphone / 略語判定 / 未知語処理 / mora accuracy 公表)、(3) haqumei-eval と同一プロトコル評価スクリプト実装を CR-27 で強制、(4) pyopenjtalk-plus 辞書採用を CR-26 で primary辞書として確定、(5) haqumei との並走 CI を CR-28 で追加、(6) 弱点カテゴリ (多音字/略語/固有名詞/連続変異) で haqumei に対し 0.5pt 以上の差をつけることを NFR-17 で強制、(7) mora accent accuracy を独自公表指標として NFR-07 に追加 (haqumei 非公表領域で情報優位) | Phase 0 開始可 |
| **1.4** | **2026-07-03** | **プロソディ出力形式を拡張**。haqumei ProsodyFormat 実測動作確認に基づき、(1) FR-03 を「アクセント + イントネーション を1つのシーケンスに統合」に拡張、(2) FR-04 に canonical記号セット (`^` `$` `#` `[` `]` `?`) を明示、(3) FR-04a: 3種類の出力フォーマット (tdmelodic 風 / Prefix `L_H_` / Numeric `:0:1`) の MUST サポートを追加、(4) FR-04b: 単語単位 API (accent_nucleus, chain_flag, chain_rule, pos 等) の MUST サポートを追加、(5) FR-05a: 疑問文 `?` / 感嘆 / 引用境界 / pause 挿入 の SHOULD サポートを追加。**アクセントとイントネーションの両方を同時出力できる要件を明文化** | Phase 0 開始可 |
| **2.0** | **2026-07-04** | **Pure-NN Research Pivot** (**破壊的変更**)。ユーザー方針転換 (rule-based 依存の hybrid では project 意義が消える) を反映。(1) **削除**: FR-20〜23 (hybrid 推論), FR-35 (pyopenjtalk 互換), FR-50〜53 (haqumei 越え攻撃戦略の hybrid 系), CR-26 (pyopenjtalk-plus 辞書 primary), AC-04 (Style-Bert-VITS2 downstream), AC-05 (pyopenjtalk drop-in), AC-07 (pyopenjtalk 互換テスト)。(2) **新規**: FR-60〜62 (pure-NN 制約), NFR-08〜09 (pure-NN 4 モデル越え + speech+text NN reference), NFR-18 (hard-set pure-NN 情報優位), CR-90〜92 (pure-NN inference boundary + rule-leakage 開示 + Category-A〜F 分類), CR-42b (CC-G2PnP 数値訂正), CR-42c (pure-text NN hybrid 越え存在しない claim), AC-11〜12 (rule-leakage 開示 + protocol harmonization), RISK-17〜18 (pseudo-label ceiling + protocol 不整合)。(3) **降格 (MUST → SHOULD / reference-only)**: NFR-05, NFR-06, NFR-17, CR-27, CR-28。(4) **反転**: CR-43 (pure-NN 禁則 → pure-NN MUST)。(5) **Review-1 adversarial critique の 4 blocker (B1〜B4) 対応**: B1=FR-61 で同一プロトコル再測定、B2=CC-G2PnP 6D-Eval SER 8.4% は誤りで Table 1 PnP CER 1.79-1.80%, B3=Frontier LLM は Koriyama Interspeech 2026 (arxiv 2606.22009) の直接引用、B4=CR-91 で pyopenjtalk-plus 教師信号 = rule-leakage の明示自己申告 | Phase 0 開始可 |

---

## 12. 一言まとめ (**v2.0 pure-NN pivot 版**)

**本プロジェクトは「ModernBERT (encoder-only pure-NN) の Japanese G2P 応用可能性を検証する研究プロジェクト」に転換する**。推論パスに rule/dict lookup を含めない (pure-NN 制約)。Primary target は pure-NN 日本語 G2P 先行 4 モデル (CharsiuG2P / PnG BERT / Kakegawa TJ-G2P / CC-G2PnP) を JSUT/JVS/ROHAN 標準ベンチで系統的に上回ること。haqumei / OpenJTalk / Frontier LLM は reference-only。

**v2.0 の pivot 根拠**: v1.3 の hybrid 差別化戦略は「同じ pyopenjtalk-plus 辞書を採用しつつ NN 補正を上乗せする」ものだった。しかしユーザーの正当な指摘 —「ルールベースの g2p の処理が入るのであればそれでいいのでこのプロジェクトをする必要がないです」— により、hybrid 経路は 130M ModernBERT を書く動機自体を失う。**pure-NN pivot により、既存 pure-NN 先行研究が全て hybrid に敗北している空白領域を、標準ベンチ 3 本柱で埋める初のデータポイントを提供する研究プロジェクトに再定義する**。

**v2.0 の設計哲学 5層** (v1.1 の 5層構成を pure-NN 化):

1. **精度目標 (pure-NN 3 段構成)** (NFR-01〜04, NFR-08〜09): Conservative < 5.0% / Stretch < 2.0% / Aspirational < 0.5-0.62% (LLM 並)、pure-NN 4 モデル越えを MUST、Ohnaka 2025 speech+text 0.93% を informed reference
2. **Hard-set 制約 (pure-NN で pyopenjtalk baseline 越え)** (NFR-10〜16, NFR-18): 7カテゴリ 各200文で pure-NN の順伝播のみで pyopenjtalk baseline を上回る
3. **Pure-NN 制約と rule-leakage 開示** (CR-40〜43 意味反転, CR-90〜92, FR-60〜62): 推論 dict-free を MUST、pretraining 教師信号としての辞書利用は許容だが明示自己申告
4. **v1.1 由来 — 計算実現可能性** (NFR-60〜64, CR-52〜55): 単一 24GB GPU / US$100以下 で全 Ablation を完走
5. **v1.1 由来 — 段階リリース・悪用防止** (CR-70〜72, CR-80〜83): α → β → 1.0 と、bias / misuse disclosure

**Negative result publication path**: 仮に pure-NN で hybrid に届かなくても、pure-NN の JSUT/JVS/ROHAN 上の scaling law を初めて公開する意義を preprint に明記する。「pure-NN で hybrid parity 到達失敗の re-confirmation」も学術的 contribution として位置付ける (詳細は `docs/research/09` の §6 で規定)。
