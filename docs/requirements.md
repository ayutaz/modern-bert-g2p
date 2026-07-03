# 要求定義書 — ModernBERT日本語G2P

**バージョン:** 1.0
**作成日:** 2026-07-03
**根拠:** `docs/research/01_overview.md` 〜 `08_market_landscape.md` の調査結果と合意事項
**ステータス:** ドラフト (Phase 0 開始前の確定を要する)

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

### 1.2 目的

日本語 Grapheme-to-Phoneme (G2P) を、既存OSSと同等以上の精度で提供する、fine-tuned ModernBERTベースの公開モデルを開発する。

### 1.3 スコープ

**含む** (In-scope):
- 日本語テキスト → 音素列 + モーラアクセント + アクセント句境界 の変換モデル
- 学習・評価パイプラインと再現可能なスクリプト
- 3本柱ベンチマーク (JSUT Basic5000 / JVS-3000 / ROHAN 4600) の公開結果
- Hugging Face Hub / GitHub での公開
- pyopenjtalk互換API (Style-Bert-VITS2等へのドロップイン置換)

**含まない** (Out-of-scope):
- TTS音響モデル (voice cloning, prosody generation)
- 多言語G2P
- 音声認識前段のG2P
- リアルタイム・ストリーミング推論最適化 (Phase 6以降で検討)

### 1.4 ステークホルダーと想定利用者

| ステークホルダー | 期待 |
|---|---|
| OSS TTS開発者 (Style-Bert-VITS2 / GPT-SoVITS等) | pyopenjtalk のドロップイン置換で TTS品質向上 |
| 音声合成研究者 | 標準ベンチマークでの再現可能なNN G2P baseline |
| 商用TTSサービス提供者 | 商用配布可能なライセンスで採用可能なOSS G2P |
| 日本語音声NLPコミュニティ | 空白領域を埋める最初のデータポイント |

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
- **FR-03** [MUST]: 各モーラに対してアクセント記号 (H / L) を付与した出力を提供する
- **FR-04** [MUST]: アクセント句境界マーカ ('/') を含む出力形式を提供する
- **FR-05** [SHOULD]: X-SAMPA、カタカナ音素表記への変換ユーティリティを提供する

### 2.1a 多言語混在文の扱い (**新規追加**)

日本語文に埋め込まれた多言語要素をどう音素化するかは、精度に直接影響する。以下を **MUST** 要件とする:

- **FR-06** [MUST]: 英単語混在部分について、以下いずれかの戦略で音素列を生成する:
  - **Strategy A**: 英→カタカナ音写 → 日本語音素化 (Kanalizer流。VOICEVOX/kanalizer-model のパスに準拠)
  - **Strategy B**: 英発音記号 (CMUdict / IPA) を経由し、日本語音素にマッピング
  - **Phase 2** で Strategy A / B の両方をパイロット評価し、精度・レイテンシ・カバレッジで最良を確定する
- **FR-07** [MUST]: 英字略語 (AI, NASA等) を **アルファベット読み** (例: AI → エーアイ) と **単語読み** (例: NASA → ナサ) で区別する。判定は文脈+辞書+モデル出力の3層で決定する
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

### 2.3 ハイブリッド推論パイプライン

- **FR-20** [MUST]: pyopenjtalk (辞書lookup) を primary path として使用する
- **FR-21** [MUST]: 辞書lookup の uncertainty region (多音字位置 / 複合語 / 助数詞連結 / OOVカタカナ / 固有名詞) を検出する
- **FR-22** [MUST]: uncertainty region に対してのみ ModernBERT を発火する条件付き推論を実装する
- **FR-23** [SHOULD]: 辞書結果とNN結果を統合する reconciliation strategy を複数比較可能にする
  - Strategy A: 辞書ヒットは常に辞書優先
  - Strategy B: uncertainty threshold で切り替え
  - Strategy C: NN confidence が高ければ辞書を上書き

### 2.4 API

- **FR-30** [MUST]: `g2p(text: str) -> phonemes: str` の pyopenjtalk互換シグネチャを提供する
- **FR-31** [MUST]: `g2p(text, kana=True)` でカタカナ出力オプションをサポートする
- **FR-32** [MUST]: `g2p(text, return_accent=True)` でアクセント情報付き出力を提供する
- **FR-33** [SHOULD]: バッチ推論API (`g2p_batch(texts: List[str])`) を提供する
- **FR-34** [MAY]: Rust bindings (haqumei互換) を提供する

### 2.5 モデル配布

- **FR-40** [MUST]: PyTorch (Hugging Face Transformers) 形式で配布する
- **FR-41** [MUST]: ONNX 形式で配布する (haqumei風のデプロイパスに乗るため)
- **FR-42** [SHOULD]: MODEL_CARD / INFERENCE_GUIDE / LICENSEの標準ドキュメント一式を含める
- **FR-43** [MAY]: 量子化バリアント (int8) を配布する

---

## 3. 非機能要件 (Non-Functional Requirements)

### 3.1 精度目標 (**必達 vs Stretch**)

3ティアの敵を明示的に上回ることを目標とする ([01overview §2])

| 指標 | ベースライン | **必達目標 (MUST)** | Stretch (SHOULD) | 参照 |
|---|---|---|---|---|
| **NFR-01**: JVS-3000 kana CER | OpenJTalk 1.03% | < 1.03% | < 0.62% (Gemini 3.1 Pro) / < 0.52% (Claude Opus 4.6) | [03 §1.1] |
| **NFR-02**: JSUT Basic5000 PER | haqumei 1.17% | < 1.17% | < 0.5% | [03 §1.2] |
| **NFR-03**: ROHAN 4600 KER | haqumei 1.64% | < 1.64% | < 1.0% | [03 §1.3] |
| **NFR-04**: JSUT モーラアクセント精度 | Hida 97.33% | > 97.33% | > 98% | [04 §A.3] |

### 3.2 Hard-set 精度 (per-category)

以下 **7カテゴリ** の hard-set (各200文) 上で、pyopenjtalk 単独 baseline を上回る必要がある:

- **NFR-10** [MUST]: 多音字 hard-set の PER が pyopenjtalk baseline を上回る
- **NFR-11** [MUST]: 助数詞語 hard-set の PER が pyopenjtalk baseline を上回る
- **NFR-12** [MUST]: 固有名詞 hard-set (漢字 + カタカナ) の PER が pyopenjtalk baseline を上回る
- **NFR-13** [MUST]: カタカナ外来語 hard-set の PER が pyopenjtalk baseline を上回る
- **NFR-14** [MUST]: 数詞 / 日付 / 時刻 / 単位 hard-set の PER が pyopenjtalk baseline を上回る
- **NFR-15** [MUST]: **英単語混在文 hard-set (英単語+日本語混在文 200文) の PER が pyopenjtalk baseline を上回る** — 「iPhone を買った」「PDF を開く」等
- **NFR-16** [MUST]: **英字略語 hard-set (AI/NASA/HTML/e-mail等の混在文 200文) の PER が pyopenjtalk baseline を上回る**

### 3.3 推論性能

- **NFR-20** [MUST]: 1文 (~50文字) の p50 レイテンシがGPU (T4) 上で < 20ms
- **NFR-21** [MUST]: 1文 (~50文字) の p50 レイテンシがCPU (x86 8-core) 上で < 100ms
- **NFR-22** [SHOULD]: ハイブリッド推論の平均レイテンシがフル推論より30%以上高速
- **NFR-23** [MAY]: GPU (RTX 4090) 上で p50 < 10ms

### 3.4 モデルサイズ

- **NFR-30** [MUST]: 最終モデルが `sbintuitions/modernbert-ja-130m` (132M params) をベースとする
- **NFR-31** [SHOULD]: 30M/70M/310m の Ablation バリアントも配布する
- **NFR-32** [MAY]: 量子化 (int8) バリアントで元サイズの1/4以下

### 3.5 再現可能性

- **NFR-40** [MUST]: すべての Ablation は同じ seed / 同じ splits で実行し、結果表を1つに統合する
- **NFR-41** [MUST]: 学習・評価スクリプトを CI で動作確認可能な状態で公開する
- **NFR-42** [MUST]: 評価データセットの split (train / valid / test) を明示的にリポジトリに記録する

### 3.6 ドキュメント

- **NFR-50** [MUST]: README、MODEL_CARD、INFERENCE_GUIDE、LICENSEの4文書を公開時に整備
- **NFR-51** [SHOULD]: `docs/research/` の技術ドキュメントを最新状態で維持
- **NFR-52** [MAY]: 論文または arxiv preprint を公開

---

## 4. 制約 (Constraints)

### 4.1 ベースモデル

- **CR-01** [MUST]: `sbintuitions/modernbert-ja-130m` を主軸ベースモデルとして使用する ([05設計 §1.1])
- **CR-02** [MUST]: Ablation対照として最低1つ以上の別トークナイザ系モデル (`llm-jp-modernbert-base` または `bert-base-japanese-char-v2`) を head-to-head 比較する
- **CR-03** [MUST]: **SB Intuitions が明記する SentencePiece トークナイザーの token classification 弱点に対応する** ([01 §3.3])
  - Phase 2 で seq2seq / MeCab-pretokenize / char-level BERT の 3並列パイロットを維持し、最良を選定

### 4.2 データ

- **CR-10** [MUST]: 学習データは以下から構築する ([03 §5.1]):
  - pyopenjtalk-plus 辞書
  - UniDic全エントリ
  - Wikipedia日本語版 ふりがな抽出
  - 青空文庫 ふりがな付きテキスト
  - JSUT Basic5000 (evaluation held-out 500文除く)
- **CR-11** [MUST]: JVS nonpara30 subset の 3,000文 (Koriyama benchmark) は評価専用に固定、学習に混入させない
- **CR-12** [MUST]: 全データをJSONスキーマに正規化し `data/processed/` に保存する ([06 §Phase 1])
- **CR-13** [MUST]: 数詞 / 固有名詞 (漢字/カタカナ) / 助数詞語 / 外来語 のサンプルには `sample_weight = 2.0` を適用する ([03 §7])
- **CR-14** [MUST]: **7カテゴリ (多音字/助数詞/固有名詞/カタカナ外来語/数詞・単位/英単語混在文/英字略語) 各200文のhard-setをPhase 1中に人手キュレーションする**
- **CR-15** [MUST]: 学習データに英単語混在パターンを **最低5万文** 含める (Wikipedia日本語版のtech関連記事、GitHub日本語READMEクローリング等)
- **CR-16** [MUST]: 英単語 → カタカナ音写辞書 (Kanalizer + CMUdict派生等) を Phase 1 で整備し、Strategy A のフォールバック用リソースとして固定する

### 4.3 ライセンス

- **CR-20** [MUST]: 公開モデル weights のライセンスは商用配布可能とする (MIT または Apache-2.0を第一候補)
- **CR-21** [MUST]: 学習データソースのライセンス条項を尊重する ([03 §6]):
  - JSUT音声: CC-BY-4.0 (帰属明示)
  - UniDic: BSD派生 (帰属明示)
  - Open JTalk辞書: 修正BSD
  - Wikipedia: CC-BY-SA-4.0 (**Share-Alike 注意**)
  - JMDict: CC-BY-SA-4.0 (**Share-Alike 注意**)
- **CR-22** [MUST]: Phase 1 開始前に法務レビューを行い、CC-BY-SA-4.0 データを学習に含む場合の重み配布可否を確定する

### 4.4 開発フロー

- **CR-30** [MUST]: 実装は `docs/research/06_implementation_roadmap.md` の7フェーズ (P0〜P6) の順序で進める
- **CR-31** [MUST]: 新規コード実装前に `superpowers:brainstorming` で要件を明確化する
- **CR-32** [MUST]: 既存コード変更時は `superpowers:systematic-debugging` で仮説→検証を明示化する
- **CR-33** [MUST]: 実装完了時は `superpowers:verification-before-completion` で証拠を集めるまで「完了」宣言を保留する

### 4.5 反証済み前提の禁則

以下の主張は敵対的検証で棄却されているため、設計・実装の前提として使用してはならない ([04 §G], [07 §6], [CLAUDE.md]):

- **CR-40** [MUST NOT]: 「NHK Kurihara 2024 が Japanese G2P を pure neural では unsolvable と framing」と引用しない
- **CR-41** [MUST NOT]: 「PnG BERT が pure-NN の coverage-limited を明示的に framing」と引用しない
- **CR-42** [MUST NOT]: 「CC-G2PnP が Dict-DNN hybrid を 6D-Eval で上回った」と引用しない (論文本文の主張だが verify で棄却)
- **CR-43** [MUST NOT]: 単一 pure-NN モデルで pyopenjtalk を置換する設計を採用しない ([07 §7.1] の5つの実証的失敗パターンを根拠に)

### 4.6 計算資源

- **CR-50** [MUST]: 全 fine-tune は bf16 混合精度 + Flash Attention on で実施する
- **CR-51** [SHOULD]: LoRA (r=16, alpha=32) と full fine-tune の両方を Phase 2 で評価
- **CR-52** [MAY]: 単一 GPU (24GB VRAM級) で完走できる設定を primary configuration とする

---

## 5. 受け入れ基準 (Acceptance Criteria)

### 5.1 Phase Gate — 各フェーズを完了と認める条件

- **AC-P0**: haqumei の JSUT PER 1.17% を ±0.1% 以内で再現でき、3本柱ベースライン (OpenJTalk / haqumei) の測定が1コマンドで再生成できる
- **AC-P1**: 統合スキーマの全データが `data/processed/` に格納され、5 hard-set が揃っている
- **AC-P2**: 3並列パイロット (seq2seq / MeCab-pretokenize / char-level) の head-to-head 結果に基づき、Phase 3以降の主軸トークナイザ戦略が確定している
- **AC-P3**: JSUT Basic5000 PER < 1.5% かつ JSUT accent-labeled subset mora-accent accuracy > 96.5% を達成する
- **AC-P4**: **haqumei 越え** — JSUT PER < 1.0%, ROHAN KER < 1.5%, JVS-3000 kana CER < 0.9%
- **AC-P5**: 「モデルサイズ vs 精度」「トークナイザー vs 精度」「学習データ量 vs 精度」の Pareto/scaling 曲線を提示できる
- **AC-P6**: Hugging Face Hub / GitHub 公開、pyopenjtalk互換API、ONNX配布、標準ドキュメント一式が揃っている

### 5.2 最終受け入れ (**プロジェクト完了**の条件)

- **AC-01**: 3本柱すべてで **haqumei / OpenJTalk を明確に上回る** (最低0.15%以上の差) — NFR-01, NFR-02, NFR-03
- **AC-02**: フロンティアLLM (Gemini 3.1 Pro 0.62%) との JVS-3000 差 < 0.2% (stretch: 差 < 0.1% あるいは越え)
- **AC-03**: **7カテゴリ Hard-set全てで pyopenjtalk baseline を上回る** (NFR-10〜16)
- **AC-04**: Style-Bert-VITS2 に投入した downstream TTS pronunciation CER が既存 pyopenjtalk投入時より改善する
- **AC-05**: ドキュメントとサンプルコードで、外部開発者が pyopenjtalk のドロップイン置換として15分以内に動作確認できる
- **AC-06**: モデル配布 (HF Hub) + 評価スクリプト再現 (GitHub) が第三者に成立している

---

## 6. 優先順位マトリクス

### 6.1 MoSCoW

**MUST have (P0〜P4 で完了)**:
- 3本柱ベンチマークでの haqumei越え精度 (AC-01)
- ハイブリッド推論パイプライン (FR-20〜22)
- pyopenjtalk互換API (FR-30〜32)
- HF Hub / GitHub 公開 (FR-40〜41)

**SHOULD have (P5〜P6 で完了)**:
- モデルサイズ Ablation (30M/70M/130M/310M)
- 量子化 (int8) 配布
- LLM stretch target (< 0.62% JVS-3000)

**COULD have (Phase 6以降で検討)**:
- Rust bindings (haqumei互換)
- ストリーミング推論
- 多言語混在 (英日混在文)
- LLM蒸留による精度向上

**WON'T have (今回スコープ外)**:
- TTS音響モデル / voice cloning
- リアルタイム音素ストリーミング
- モバイル・エッジ最適化

### 6.2 Kano分析 (品質モデル)

- **Must-be** (無いと不満): JSUT PER < 1.17%、pyopenjtalk互換API、商用可ライセンス
- **One-dimensional** (あるほど満足): モデルサイズが小さい、レイテンシが低い、Hard-set精度が高い
- **Attractive** (あると驚喜): LLM並みの精度、pyopenjtalkを完全にdrop-in置換できる、日本語標準ベンチマークリーダーボードを立ち上げる

---

## 7. リスクと予防策 (要件レベル)

| リスクID | 内容 | 予防策 | 対応要件 |
|---|---|---|---|
| **RISK-01** | トークナイザー選定の失敗 | Phase 2で3並列パイロット | CR-03 |
| **RISK-02** | データライセンス問題 | Phase 1開始前に法務レビュー | CR-22 |
| **RISK-03** | 計算資源不足 | LoRAとfull fine-tuneの両方を用意、130mから開始 | CR-51, CR-52 |
| **RISK-04** | JVS-3000評価データ未公開 | 論文著者に問い合わせ、代替として自作JVSサブセット | AC-02 |
| **RISK-05** | ModernBERTトークナイザー起因の性能悪化 | Phase 2で確認、必要ならchar-levelにpivot | CR-03 |
| **RISK-06** | pure-NN で prosperity期待 | 反証済み前提を明示的に禁則化 | CR-40〜43 |
| **RISK-07** | Hard-setアノテーション不足 | Phase 1で明示的キュレーション時間確保 | CR-14 |
| **RISK-08** | Loss weight tuning が難航 | Phase 3で grid search を初期から計画 | (Phase 3 内部) |
| **RISK-09** | 英単語混在文の学習データ不足 | Wikipedia tech記事、GitHub日本語README等の追加クローリング | CR-15 |
| **RISK-10** | 英→カタカナ音写のカバレッジ不足 (未知綴りの英単語) | Kanalizer NN + CMUdict + フォールバック規則の3層構造 | FR-06, CR-16 |
| **RISK-11** | 略語のアルファベット読み vs 単語読み判定失敗 | 文脈依存の学習 + 明示的な略語辞書 (AI→エーアイ 等) 200件以上を Phase 1 で整備 | FR-07 |

---

## 8. トレーサビリティ (要件 → 調査ドキュメント)

各要件が根拠とする調査ドキュメント:

| 要件領域 | 根拠 |
|---|---|
| コア設計思想 | `01_overview.md §3`, `05_technical_design.md §1〜3` |
| 精度目標 | `01_overview.md §2`, `03_datasets_and_benchmarks.md §1〜3` |
| ハイブリッド戦略 | `02_existing_systems.md §F`, `04_papers_and_references.md §A.2`, `08_market_landscape.md §9〜10` |
| マルチタスク定式化 | `04_papers_and_references.md §A.3`, `05_technical_design.md §3.1` |
| トークナイザー警告 | `01_overview.md §3.3`, `05_technical_design.md §2` |
| pure-NN禁則 | `07_nn_only_benchmarks.md §7`, `CLAUDE.md` |
| 市場空白領域 | `08_market_landscape.md §9〜10` |
| ライセンス制約 | `03_datasets_and_benchmarks.md §6` |
| フェーズ運用 | `06_implementation_roadmap.md` |

---

## 9. 未確定事項 (Phase 0 開始前に確定を要する)

以下は本要求定義書ドラフト時点で **意思決定待ち** の項目。Phase 0 kickoff前にステークホルダーで確定する:

- **OPEN-01**: JVS-3000 (Koriyama benchmark) の kana アノテーションデータの入手可否と、入手できない場合の代替評価スキーム
- **OPEN-02**: Wikipedia日本語版のCC-BY-SA-4.0 (Share-Alike) を学習データに使う場合、公開する重みが Share-Alikeを継承するか、または Wikipediaを使わずに他ソース (UniDic + pyopenjtalk-plus + 青空文庫のみ) で学習するかの判断
- **OPEN-03**: 主軸トークナイザ戦略 (a=seq2seq / b=MeCab-pretokenize / c=char-level) は Phase 2 の 3並列パイロット結果を待つ。**この時点でstakeholder review を必ず経る**
- **OPEN-04**: モデル公開ブランド名 / リポジトリ名 (仮称: "ModernBERT日本語G2P" の正式命名)
- **OPEN-05**: 最終ライセンスの確定 (MIT vs Apache-2.0)。データ由来のShare-Alike継承有無の判断次第

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

---

## 12. 一言まとめ

**本プロジェクトは「ModernBERT を単一の pyopenjtalk 置換モデルにする」のではなく、「haqumei/pyopenjtalk が崩れる領域 (アクセント連続変異 / 多音字 / OOV) を、市場に存在しない fine-tuned encoder NN で狙い撃ちで補正するハイブリッド」を作る**。要求定義の骨子は、この設計哲学を精度目標 (NFR-01〜04)、Hard-set 制約 (NFR-10〜14)、pure-NN禁則 (CR-40〜43) の3層でロックしている。
