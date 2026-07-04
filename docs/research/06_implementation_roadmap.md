# 06. 実装ロードマップ

01〜05 の調査結果に基づき、開発を **6 つのフェーズ** に分けて時系列で整理する。各フェーズには exit criteria (次に進む条件) を明示する。

> **v2.0 (2026-07-04) 大改訂**: Pure-NN Research Pivot を反映し、旧 Phase 4 (辞書 primary + ModernBERT correction ハイブリッド) を全面削除。旧 Phase 5 を「Scale & Ablation (Core NN Improvement)」に格上げして 3–4 週に拡大した。トークナイザーパイロットは P-B (MeCab pretokenize) を drop し P-A/P-C の 2 pilot に縮小 (MeCab 依存が pure-NN 原則に反するため)。詳細は下記「方針転換」節を参照。

---

## 方針転換 (2026-07-04) — Phase 4 (ハイブリッド化) 削除

### 背景

プロジェクトを **「pure-NN Japanese G2P の研究プロジェクト」** に posture 変更した (詳細は `CLAUDE.md` v2.0, `docs/requirements.md` v2.0, `docs/research/09_pure_nn_g2p_benchmarks.md` v2.0)。

Pivot の核心的理由 (ユーザー原文の趣旨):

> ルールベースの g2p の処理が入るのであればそれでいいのでこのプロジェクトをする必要がないです

- Hybrid 出力の 80–95% は辞書 (pyopenjtalk-plus) 由来という徹底解剖結果 (v1.3 反映) と両立させると、130M ModernBERT を新規に書く動機自体が消える。
- したがって **推論経路には dict / MeCab / rule を一切含めない** pure-NN 制約を MUST 化する。
- 越えるべき相手は "haqumei / OpenJTalk" ではなく "**pure-NN 先行研究 4 モデル** (PnG BERT / Kakegawa TJ-G2P / CharsiuG2P / CC-G2PnP)" に切り替わる。

### 削除・置換ポリシー

- **旧 Phase 4 (ハイブリッド化) は全面削除**。「辞書 primary + NN correction」「uncertainty region 検出」「reconciliation strategy A/B/C」の 4 タスクはすべて pure-NN 制約と両立しない。
- 代わりに **旧 Phase 5 (スケール & Ablation)** を「Core NN Improvement (Scale & Ablation)」に格上げし 2 週 → **3–4 週** に拡大。ここが pure-NN プロジェクトの主戦場となる。
- 旧 Phase 6 (公開) は焦点を変更 — pyopenjtalk 互換 wrapper を削除、代わりに **Pure-NN 標準ベンチ table** (JSUT / JVS / ROHAN × 4 先行モデル vs ours) を preprint Table 1 として fix する。

### Rule-leakage の明示的自己審査

Pure-NN 制約は **推論経路のみ** に適用する。学習段階では以下の rule-derived signals を **教師信号** として使用可能とする (`docs/research/09 §7 (planned)` で審査):

- pyopenjtalk-plus 辞書 (surface, yomi, accent) → MLM pretrain の追加 supervision に使用可
- UniDic 辞書 → MLM pretrain / polyphone labels 生成に使用可
- pyopenjtalk 出力 → APBP / ANPP labels の pseudo-label 生成に使用可

**ただし推論時には pyopenjtalk / MeCab / dict lookup を呼ばない**。この境界の明示は PnG BERT が陥った "pseudo-label ceiling" 問題を回避するために不可欠 (`docs/research/07 §5`)。

### 新旧タイムライン比較

| Phase | 旧 | 新 |
|---|---|---|
| P0 baseline | 1 週 | 1 週 (変更なし) |
| P1 data | 2 週 | 2 週 (変更なし、pyopenjtalk-plus を教師信号目的に限定) |
| P2 single-task baseline | 2 週 | 2 週 (変更なし) |
| P3 multi-task | 3 週 | 2 週 (compaction) |
| **P4 hybrid** | **2 週** | **削除** |
| **P4' (旧 P5) Scale & Ablation** | **2 週** | **3–4 週 (拡大)** |
| P5 (旧 P6) 評価・公開 | 1 週 | 1 週 |
| **合計** | **13 週** | **11–12 週** (P4 削除の 2 週から Ablation 拡大に 1–2 週再投資、正味 1–2 週短縮) |

---

## Phase 0: 基盤とベースライン再現 (目標: 1週間)

### 目的

既存 SOTA を自環境で再現し、以降のすべての改善を測定する **base of truth** を確立する。**Pure-NN pivot 後の位置付け**: haqumei / OpenJTalk は "reference-only" (competitive target ではない)。並列で **pure-NN 先行 4 モデル** (PnG BERT / Kakegawa TJ-G2P / CharsiuG2P / CC-G2PnP) の JSUT/JVS/ROHAN 上の数値も再測定/自己測定し、**越えるべき baseline** として fix する。

### タスク

- [ ] リポジトリ初期化 (`pyproject.toml`, `src/`, `tests/`, `notebooks/`, `data/`, `checkpoints/`, `docs/`)
- [ ] 開発環境: Python 3.11+, PyTorch 2.4+, Transformers 4.48+, CUDA環境
- [ ] pyopenjtalk / pyopenjtalk-plus セットアップと単純動作確認 (**reference 用のみ**)
- [ ] haqumei リポジトリ clone、公開手順で JSUT Basic5000 の PER を再現 (**reference 用のみ**)
- [ ] haqumei で ROHAN 4600 の KER を再現 (**reference 用のみ**)
- [ ] 評価スクリプト実装 (PER, kana CER, KER 計算共通ライブラリ)
- [ ] JVS-3000 (Koriyama benchmark) データセット取得 (論文 arxiv 2606.22009 のGitHub確認)
- [ ] OpenJTalk baseline を JSUT/JVS/ROHAN 3本柱で測定 (**reference-only**)
- [ ] **[NEW] Pure-NN 先行 4 モデルの自己測定**:
  - CharsiuG2P byT5-small (multilingual) を JSUT Basic5000 で PER 実測 (先行研究の 10.5% は own-dict-holdout で JSUT ではない — 我々の環境で再測定必須)
  - PnG BERT は公開実装がなければ再現できないため、原著数値 (whole-word acc 45.5%) を「再現不能・参考値」として明示
  - Kakegawa TJ-G2P alone は Kurihara 2024 の再実装数値 (JSUT400 PPL CER 11.85%) を引用 (実装なし)
  - CC-G2PnP は公開 checkpoint (ayousanz reproduction) で 6D-Eval PnP CER 1.79–1.80% を再測定、JSUT overlap の有無を確認
- [ ] **[NEW] Frontier LLM baseline**: Koriyama arxiv 2606.22009 記載の Claude Opus 4.6 = JVS kana CER 0.52%, Gemini 3.1 Pro = 0.62% を **一次引用で確認** (適宜 API 再測定)
- [ ] 結果を `docs/baselines/` にmarkdownで記録 — **reference / pure-NN prior-art / frontier LLM** の 3 セクション構成

### Exit Criteria

- haqumei の JSUT PER 1.17% を ±0.1% 以内で再現できる (reference 用)
- 3本柱ベンチマークで全 reference ベースラインの数値を1コマンドで再生成できる
- Pure-NN 先行 4 モデルのうち **少なくとも 2 モデル** (CharsiuG2P, CC-G2PnP) を我々の環境で PER 実測できる
- 評価スクリプトが CI で動く

### 参考ソース

- [haqumei README](https://github.com/o24s/haqumei) (reference)
- [pyopenjtalk](https://github.com/r9y9/pyopenjtalk) (reference)
- [arxiv 2606.22009 (Koriyama benchmark)](https://arxiv.org/abs/2606.22009) (frontier LLM 数値の一次情報)
- [CharsiuG2P](https://github.com/lingjzhu/CharsiuG2P) (pure-NN prior-art)
- [CC-G2PnP arxiv 2602.17157](https://arxiv.org/pdf/2602.17157) (pure-NN prior-art)

---

## Phase 1: 学習データ生成 (目標: 2週間)

### 目的

ModernBERT を fine-tune / MLM continued pretrain するための大規模な (テキスト, 音素列, アクセント, カテゴリ) データセットを整備する。**Pure-NN 制約下では、rule-derived な (surface, yomi, accent) triple を教師信号としてのみ使用し、推論経路には持ち込まない**。

### タスク

- [ ] **pyopenjtalk-plus 辞書** から全エントリ (~800K) を抽出、(surface, yomi, accent_type) トリプル化 → **教師信号として MLM pretrain 用の corpus 化**
- [ ] **UniDic全エントリ** (~1M) 抽出。pyopenjtalk-plus とマージし、conflict は UniDic 優先で解決 → 同上
- [ ] **Wikipedia 日本語版** からふりがな付き記事本文抽出 (~500K sentences)
- [ ] **青空文庫** からふりがな付きテキスト抽出 (~200K sentences)
- [ ] **JSUT Basic5000** の 4,500文 (500文 held-out) を学習セットに追加
- [ ] **[NEW] LLM 合成データ (weak supervision)** — Claude/Gemini で JVS-scale データを合成し kana を weak supervision に (licensing 慎重、Phase 4' scale ablation で使用有無を判定)
- [ ] 全データを共通スキーマに正規化:
  ```json
  {
    "text": "私は東京で本を読みました。",
    "phonemes": "watashi_wa_tookyoo_de_hoN_o_yomimashita.",
    "mora_accents": "LHHH_LH_LHHH_H_LH_LH_LHHHHHH",
    "accent_phrases": ["私は", "東京で", "本を", "読みました"],
    "category": "general",
    "source": "wikipedia|aozora|unidic|jsut|pyopenjtalk_plus|llm_synth",
    "signal_type": "gold|silver_dict|bronze_llm"
  }
  ```
- [ ] Hard-set 手動キュレーション (7 カテゴリ × 200 文 = 1,400 文):
  - 多音字漢字を含む文 200
  - 助数詞語 200
  - 固有名詞 (漢字+カタカナ) 200
  - カタカナ外来語 200
  - 数詞/日付/時刻/単位 200
  - 英単語混在文 (iPhone / e-mail 等) 200
  - 英字略語 (AI / NASA / Wi-Fi 等) 200
- [ ] 学習/検証/テスト分割 (JVS/JSUT held-out はテスト専用に固定)
- [ ] データ品質チェック: 音素-テキスト整合、アクセント長=モーラ数、pyopenjtalk出力との一致度
- [ ] **[NEW] Rule-leakage 監査**: `signal_type` によって dict-derived vs speech-derived vs LLM-synth をラベリング。Phase 4' で ablation 対象に。

### Exit Criteria

- 統合スキーマの全データが `data/processed/` に格納
- 各カテゴリの分布ヒストグラムを可視化 (Koriyamaベンチと同等の比率)
- Hard-set 7カテゴリの gold labels が揃っている
- データローダー (streaming, on-the-fly padding) が動く
- **Data scale target**: 少なくとも 2M pairs (PnG BERT / Kakegawa の 5-10 倍以上)。Phase 4' で 0.5M / 1M / 2M / 5M の scaling law を測定できる状態

### 参考ソース

- [pyopenjtalk-plus](https://github.com/tsukumijima/pyopenjtalk-plus)
- [UniDic](https://clrd.ninjal.ac.jp/unidic/) (推定リンク)
- [sarulab-speech/jsut-label](https://github.com/sarulab-speech/jsut-label)

---

## Phase 2: シングルタスク G2P ベースライン (目標: 2週間)

### 目的

「ModernBERT で pure-NN シングルタスク G2P」を実装し、pure-NN 先行研究および reference (OpenJTalk / haqumei) との差分を測定する。マルチタスクの複雑さを入れる前の **必須の sanity check**。

### タスク

- [ ] **モデル1 (P-A, main)**: `sbintuitions/modernbert-ja-130m` encoder + T5系 char-level decoder (seq2seq)
  - LoRA (r=16) と full fine-tune の両方を試行
- [ ] **モデル2 (P-C, 対照)**: `tohoku-nlp/bert-base-japanese-char-v2` encoder + token classification head (char-level BERT)
- [ ] ~~**モデル3 (P-B, 対照)**: MeCab pretokenize + token classification~~ **(v2.0 pivot で drop)** — MeCab は rule-based であり、学習時の pre-tokenize であっても pure-NN 原則 (推論経路に rule/dict を含めない) の純度検証で除外リスクが高いと判断。加えて `docs/research/09 §9.3` は pre-tokenizer としての MeCab 実行を既に「禁止 (char-level or subword-only に統一)」と規定しており、P-B はこの方針と両立しない。詳細は `docs/design/phase2_tokenizer_pilots.md` の Status update を参照
- [ ] 3モデル全てで JSUT Basic5000 PER と JVS-3000 kana CER を測定
- [ ] **[NEW] Pure-NN 先行 4 モデルとの直接比較** (Phase 0 で測定した数値と並置)
- [ ] トークナイザー選択の結論を出す (05_technical_design.md の (a)(b)(c) のどれが最良か)

### Exit Criteria

- **[UPDATED]** Pure-NN 先行 4 モデルのうち少なくとも **CharsiuG2P を明確に上回る** (JSUT PER が CharsiuG2P 自己測定値の 1/2 以下)
- 3モデルの head-to-head 結果と要因分析を `docs/experiments/phase2.md` に記録
- 最良のトークナイザー戦略を選定し、Phase 3 以降の主軸に採用
- **[REMOVED]** 「OpenJTalk baseline を JSUT PER で下回る」— OpenJTalk は reference-only なので目標から降格。ただし数値差分は preprint に記載

### 参考ソース

- [modernbert-ja-130m](https://huggingface.co/sbintuitions/modernbert-ja-130m)
- [bert-base-japanese-char-v2](https://huggingface.co/tohoku-nlp/bert-base-japanese-char-v2)
- [Kurihara Interspeech 2024](https://www.isca-archive.org/interspeech_2024/kurihara24_interspeech.pdf)

---

## Phase 3: マルチタスク学習 (目標: 2週間)

### 目的

Hida et al. ICASSP 2022 と NHK Kurihara Interspeech 2024 を統合し、多タスク同時学習で pure-NN 単体の限界を突破する。**hybrid の post-processor としてではなく、multi-task の 1 head として encoder に統合する** (方針転換に伴う架構解釈の変更)。

> **[UPDATED]** 旧 3 週 → 2 週に圧縮。P4 削除により Phase 4' (Scale & Ablation) の 3–4 週にコンピュートリソースを再配分するため。

### タスク

- [ ] Head A (G2P main), Head B (Polyphone), Head C (APBP), Head D (ANPP), Head E (BAS) を実装 — **全て NN head、推論時に外部 rule を呼ばない**
- [ ] マルチタスクデータセット:
  - JSUT accent-labeled subset を APBP/ANPP/BAS用に整形
  - polyphone labels を UniDic + pyopenjtalk-plus 辞書から自動抽出 (**pretrain の教師信号として、推論経路には含めない**)
- [ ] Loss weights の grid search (α_G2P, α_poly, α_APBP, α_ANPP, α_BAS)
- [ ] カテゴリ別 sample reweighting の効果測定 (`sample_weight = 2.0` for 数詞 / 固有名詞 / 助数詞 / 外来語 / 英単語混在 / 英字略語)
- [ ] Ablation: 各 head の on/off で PER と mora-accent accuracy の変化を測定
- [ ] Hida 2022 の 96.66% mora-accent accuracy を目標に到達 (Hida は hybrid なので、pure-NN で同水準は挑戦的目標)

### Exit Criteria

- **[UPDATED]** JSUT Basic5000 PER < 3.0% (旧 < 1.5% は hybrid 前提の目標。pure-NN では CharsiuG2P 10.5% を大きく下回る 3.0% が conservative goal)
- JSUT accent-labeled subset で mora-accent accuracy > 90% (旧 > 96.5% は Phase 4' 到達目標に降格)
- 各 head の寄与が定量的に説明できる (Hida table 相当を作る)

### 参考ソース

- [Hida ICASSP 2022](https://ar5iv.labs.arxiv.org/html/2201.09427)
- [Kurihara Interspeech 2024](https://www.isca-archive.org/interspeech_2024/kurihara24_interspeech.pdf)

---

## Phase 4' (旧 P5): Core NN Improvement — Scale & Ablation (目標: 3–4週間)

> **[EXPANDED]** 旧 Phase 5 を格上げ。Phase 4 (hybrid) 削除に伴い、pure-NN プロジェクトの **主戦場** となる phase。ここで「pure-NN 130M が pure-NN 先行研究 4 モデルを 3 本柱で越える」を実証する。

### 目的

決定した最良アーキテクチャで、**モデルサイズ・トークナイザー・データ量・pretrain 有無・distillation 有無** の系統的な Ablation を行い、pure-NN 単体で到達可能な最良点を確定する。**scaling law + Frontier LLM gap analysis を preprint の Table 2/3 として fix する**。

### タスク

#### 4'-A. Continued MLM Pretrain (1 週)

- [ ] MLM continued pretrain: pyopenjtalk-plus / UniDic の (surface, kana, accent) を **教師信号として MLM に使う**
- [ ] Domain adaptation: JVS-nonpara-kana / JSUT / ROHAN の 3 domain characteristics を pretrain corpus に反映
- [ ] Pretrain 有無で fine-tune 後 PER を比較 (ablation)

#### 4'-B. モデルサイズ × トークナイザー × データ量 の 3D Ablation (2 週)

- [ ] **モデルサイズ**: modernbert-ja-30m / 70m / 130m / 310m の 4 系統で完全学習
- [ ] **追加 encoder 比較**:
  - `llm-jp/llm-jp-modernbert-base` head-to-head
  - `tohoku-nlp/bert-base-japanese-char-v2` (P-C variant) head-to-head
  - (可能なら) `yomi-linter-modernbert-ja-130m` を prior-art anchor として比較
- [ ] **学習データ量 scaling**: 0.5M / 1M / 2M / 5M pairs の 4 点で PER をプロット (scaling law の切片・傾き係数を fit)
- [ ] **Tokenizer 2 pilot (P-A/P-C)** の 3D grid: {size} × {tokenizer} × {data_scale} (P-B は v2.0 pivot で drop)
- [ ] **Multi-task heads on/off** の完全 ablation table
- [ ] **CTC auxiliary loss** (CC-G2PnP idea) の on/off ablation
- [ ] **[NEW] Ensemble**: P-A/P-C の ensemble で pure-NN 単体を越えられるか
- [ ] **[NEW] Distillation ablation**: Frontier LLM (Claude Opus 4.6 / Gemini 3.1 Pro) からの蒸留で 130M が LLM の 30B に迫れるか (小規模実験、大規模は Phase 6 以降)
- [ ] **[NEW] Frontier LLM approach 分析**: LLM 側の parse mode (暗黙 hybrid 部分) を切った場合の pure decode-mode kana CER を実測。130M pure-NN の理論上限予測に反映

#### 4'-C. 結果統合 (0.5–1 週)

- [ ] 全 ablation 結果を1つの結果表にまとめる (preprint Table 2 相当)
- [ ] **Pareto 曲線**: {model_size, tokenizer, data_scale} vs PER の 3D 可視化
- [ ] **Scaling law fit**: L(N, D) = a * N^-α + b * D^-β + irreducible の 3 係数を報告
- [ ] Failure case 集: pure-NN が hybrid reference に負けている例を 7 カテゴリごとに 5 件ずつ収集

### Exit Criteria

- **[UPDATED, Core]** Pure-NN 先行 4 モデルを JSUT/JVS/ROHAN 3 本柱の **全指標で明確に上回る** (CharsiuG2P 10.5% を 2× 以上、best-of-prior を明確に凌駕)
- Pure-NN pipeline で JSUT Basic5000 PER < 2.0% (stretch: < 1.0%)
- モデルサイズ vs 精度 の Pareto 曲線を提示可能
- トークナイザー vs 精度 を系統的に説明可能
- 学習データ量 vs 精度 の scaling law を提示可能
- Frontier LLM との absolute gap を **数値で説明可能** (「なぜ LLM が 0.52% で我々の 130M は Xxx%」の analysis)
- 全 ablation 結果を1つの結果表にまとめる (論文 Table 相当)

### 参考ソース

- [modernbert-ja モデル群](https://huggingface.co/sbintuitions)
- [llm-jp-modernbert 論文 (arxiv 2504.15544)](https://arxiv.org/pdf/2504.15544)
- [CC-G2PnP (arxiv 2602.17157)](https://arxiv.org/pdf/2602.17157)
- [Ohnaka INTERSPEECH 2025 (arxiv 2506.04527)](https://arxiv.org/abs/2506.04527) — 参考 (speech+text で PER 0.93%、pure-text 130M の到達可能上限予測に使用)

---

## Phase 5 (旧 P6): 評価・公開・統合 (目標: 1週間)

### 目的

Final モデルを 3本柱ベンチマーク + Hard-sets で総合評価し、pure-NN research プロジェクトとして公開可能な状態にする。**Pure-NN NN-G2P benchmark table を preprint の Table 1 として fix する**。

### タスク

- [ ] Final モデルで以下を計測:
  - JVS-3000 kana CER (LLM SOTA および pure-NN prior-art との差分をプロット)
  - JSUT Basic5000 PER (pure-NN 4 モデルおよび haqumei との差分)
  - ROHAN KER (pure-NN prior-art および haqumei との差分)
  - 7カテゴリ Hard-set の per-category PER
  - **[REMOVED]** ~~Downstream TTS pronunciation CER (Style-Bert-VITS2 に投入)~~ — production TTS drop-in は目標外
- [ ] Latency / throughput / model size を計測 (CPU int8, GPU fp16, GPU int8)
- [ ] README, MODEL_CARD, LICENSE (**Apache-2.0**), INFERENCE_GUIDE を整備
- [ ] ONNX export 手順とサンプルコード
- [ ] **[REMOVED]** ~~pyopenjtalk との互換 API wrapper~~ — production API 互換は research project に不要
- [ ] **[NEW]** **Pure-NN benchmark table を preprint Table 1 として構築**:
  | Model | Category | Params | JSUT PER | JVS CER | ROHAN KER |
  |---|---|---|---|---|---|
  | PnG BERT | pure-text encoder | ~110M | (N/A) | (N/A) | (N/A) |
  | Kakegawa TJ-G2P alone | pure-text seq2seq | ? | 11.85%* | (N/A) | (N/A) |
  | CharsiuG2P | pure-text seq2seq (multilingual) | ~40M | (own-holdout 10.51%) | (N/A) | (N/A) |
  | CC-G2PnP | pure-text streaming CTC | ? | (6D-Eval PnP CER 1.79%) | (N/A) | (N/A) |
  | **Ours (130M)** | pure-text encoder + multi-task | 130M | **TBD** | **TBD** | **TBD** |
  | (reference) OpenJTalk | rule | — | 10.82%* | 1.03% | — |
  | (reference) haqumei | hybrid | ~130M | 1.17% | — | 1.64% |
  | (reference) Claude Opus 4.6 | LLM | 30B+ | — | 0.52% | — |
  \* Kurihara 2024 の JSUT400 PPL CER (JSUT Basic5000 との protocol 差に注意)
- [ ] **[NEW]** **Frontier LLM gap analysis**: 130M pure-NN と LLM 30B+ の gap を数値化 (order-of-magnitude で並ぶかを判定)
- [ ] Hugging Face Hub にモデル公開 (Apache-2.0)
- [ ] Blog / arXiv preprint / GitHub リリース

### Exit Criteria

- **[UPDATED]** **Pure-NN 先行 4 モデルすべてを 3 本柱の全指標で上回る** (旧「haqumei / OpenJTalk を明確に上回る」を削除)
- Aspirational: LLM (Claude Opus 4.6 0.52% / Gemini 3.1 Pro 0.62%) との差 が **order-of-magnitude 未満** (10 倍以内、= 5% 以下) に収まる
- **[NEW]** Negative result 経路も許容: 仮に aspirational tier に届かなくても、pure-NN の JSUT/JVS/ROHAN 上の scaling law + Pareto 曲線を初めて公開する意義で preprint publishable
- ドキュメントとサンプルコードが最小限で使える
- モデル配布と評価スクリプトが CI で再現できる

---

## 全体的な進捗管理

### 主要マイルストーン

| Phase | 期間 | 累計 | 主要成果 |
|---|---|---|---|
| P0 | 1週間 | 1週間 | reference ベースライン + pure-NN 先行研究の自己測定 |
| P1 | 2週間 | 3週間 | 学習データ整備 (2M+ pairs、rule-leakage 監査済み) |
| P2 | 2週間 | 5週間 | pure-NN シングルタスク baseline (CharsiuG2P 越え、トークナイザー決定) |
| P3 | 2週間 | 7週間 | pure-NN マルチタスク学習 (compaction) |
| **P4' (旧 P5)** | **3–4週間** | **10–11週間** | **Core NN Improvement (scale + ablation + pretrain + distillation)** |
| P5 (旧 P6) | 1週間 | **11–12週間** | 公開 (Pure-NN benchmark Table 1 + Apache-2.0 HF Hub) |

**総期間目安: 11–12 週間 (約 3 ヶ月)** — 1人フルタイム想定。旧 13 週 から 1–2 週短縮 (Phase 4 削除 -2 週、Phase 4' 拡大 +1–2 週、Phase 3 圧縮 -1 週の相殺)。

### リスクと予防策

| リスク | 予防策 |
|---|---|
| トークナイザー選択の失敗 | Phase 2で2並列パイロット (P-A/P-C。P-B は v2.0 pivot で drop) |
| データライセンス問題 | Phase 1 開始前に法務レビュー (SA 系ライセンスに特に注意) |
| 学習時計算資源不足 | LoRAとfull fine-tuneの両方を用意、130mから開始 |
| 評価データ (JVS-3000) 未公開 | 論文著者に問い合わせ、代替として自作JVSサブセットで評価 |
| ModernBERTトークナイザー起因の性能悪化 | Phase 2で確認、必要ならchar-levelにpivot |
| ハードsetのアノテーション不足 | Phase 1 での明示的キュレーション時間確保 |
| Loss weight tuning が難航 | Phase 3で grid search を初期から計画 |
| **[NEW] Pure-NN が pure-NN prior-art すら越えられない** | Phase 4' で scale ablation により scaling law を実測、5M pairs スケールでも到達できなければ speech+text (Ohnaka 2025 路線) へ pivot 検討 |
| **[NEW] pyopenjtalk-plus 教師信号の rule-leakage 過剰依存** | Phase 4' で pretrain 有無 ablation を必ず実施、gold-only fine-tune 結果も併記 |
| **[NEW] Frontier LLM との gap が縮まらない** | Phase 4' で distillation ablation を計画、届かなければ preprint で "order-of-magnitude gap" を明示的に報告 (negative result 経路) |

### 開発時に監視すべき指標

**毎epoch/評価毎に記録**:
- JSUT Basic5000 PER (主要)
- JVS-3000 kana CER (主要)
- ROHAN KER (主要)
- Mora-accent accuracy (JSUT accent labeled)
- Per-category PER (7 hard-sets)
- Training loss (per-head で分離)
- Validation loss (per-head)
- GPU memory / step time / throughput
- **[NEW]** Pure-NN 先行 4 モデルとの差分 (delta plot)

**Weekly checkpoint**:
- Ablation table update
- Failure case集 (誤り例を最新モデルで5例ずつ確認、docs/failure_analysis.md)
- Pure-NN prior-art 比較チャート更新
- **[NEW]** Frontier LLM gap plot

---

## 継続的な改善アイデア (Phase 5 以降)

- **LLM 蒸留の本格化**: Phase 4' の小規模実験を大規模化。Claude/Geminiで JVS-scale データを合成し、その kana を weak supervision に活用 (licensing慎重)
- **Speech+text 拡張**: Ohnaka 2025 (arxiv 2506.04527) 路線で speech encoder を条件付けに追加、PER 0.93% を目指す
- **Multi-speaker prosody**: 単一G2Pではなく話者ごとのアクセント傾向を条件付け
- **多言語混在**: 英語/中国語/韓国語の混在テキストへの拡張 (Misakiの方向性)
- **Streaming推論**: CC-G2PnP のCTCベース設計を統合してストリーミング化
- ~~**RAG風の辞書拡張**~~ (**削除**: pure-NN 制約と両立しない)

---

## まとめ

- 3ヶ月・**6 フェーズ (旧 7 フェーズから Phase 4 削除)** ・各フェーズ明確な exit criteria
- **Pure-NN 先行研究 4 モデルを 3 本柱で越える** を Phase 4' の閾値に置き、Phase 5 は公開に集中
- Ablation は Phase 4' に集中させ、途中の実装コストを最小化
- 公開は Phase 5 でまとめて、モデル / データ処理 / 評価スクリプト / **Pure-NN benchmark Table 1** を揃える
- **haqumei / OpenJTalk / Frontier LLM は reference-only** — 目標は pure-NN prior-art 越え、aspirational は LLM に order-of-magnitude で並ぶこと
