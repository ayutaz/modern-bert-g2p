# 06. 実装ロードマップ

01〜05 の調査結果に基づき、開発を7つのフェーズに分けて時系列で整理する。各フェーズには exit criteria (次に進む条件) を明示する。

---

## Phase 0: 基盤とベースライン再現 (目標: 1週間)

### 目的

既存 SOTA を自環境で再現し、以降のすべての改善を測定する **base of truth** を確立する。

### タスク

- [ ] リポジトリ初期化 (`pyproject.toml`, `src/`, `tests/`, `notebooks/`, `data/`, `checkpoints/`, `docs/`)
- [ ] 開発環境: Python 3.11+, PyTorch 2.4+, Transformers 4.48+, CUDA環境
- [ ] pyopenjtalk / pyopenjtalk-plus セットアップと単純動作確認
- [ ] haqumei リポジトリ clone、公開手順で JSUT Basic5000 の PER を再現
- [ ] haqumei で ROHAN 4600 の KER を再現
- [ ] 評価スクリプト実装 (PER, kana CER, KER 計算共通ライブラリ)
- [ ] JVS-3000 (Koriyama benchmark) データセット取得 (論文 arxiv 2606.22009 のGitHub確認)
- [ ] OpenJTalk baseline を JSUT/JVS/ROHAN 3本柱で測定
- [ ] 結果を `docs/baselines/` にmarkdownで記録

### Exit Criteria

- haqumei の JSUT PER 1.17% を ±0.1% 以内で再現できる
- 3本柱ベンチマークで全ベースラインの数値を1コマンドで再生成できる
- 評価スクリプトが CI で動く

### 参考ソース

- [haqumei README](https://github.com/o24s/haqumei)
- [pyopenjtalk](https://github.com/r9y9/pyopenjtalk)
- [arxiv 2606.22009 (Koriyama benchmark)](https://arxiv.org/abs/2606.22009)

---

## Phase 1: 学習データ生成 (目標: 2週間)

### 目的

ModernBERT を fine-tune するための大規模な (テキスト, 音素列, アクセント, カテゴリ) データセットを整備する。

### タスク

- [ ] **pyopenjtalk-plus 辞書** から全エントリ (~800K) を抽出、(surface, yomi, accent_type) トリプル化
- [ ] **UniDic全エントリ** (~1M) 抽出。pyopenjtalk-plus とマージし、conflict は UniDic 優先で解決
- [ ] **Wikipedia 日本語版** からふりがな付き記事本文抽出 (~500K sentences)
- [ ] **青空文庫** からふりがな付きテキスト抽出 (~200K sentences)
- [ ] **JSUT Basic5000** の 4,500文 (500文 held-out) を学習セットに追加
- [ ] 全データを共通スキーマに正規化:
  ```json
  {
    "text": "私は東京で本を読みました。",
    "phonemes": "watashi_wa_tookyoo_de_hoN_o_yomimashita.",
    "mora_accents": "LHHH_LH_LHHH_H_LH_LH_LHHHHHH",
    "accent_phrases": ["私は", "東京で", "本を", "読みました"],
    "category": "general",
    "source": "wikipedia|aozora|unidic|jsut|pyopenjtalk_plus"
  }
  ```
- [ ] Hard-set 手動キュレーション:
  - 多音字漢字を含む文200
  - 助数詞語200
  - 固有名詞200 (漢字+カタカナ)
  - 外来語200
  - 数詞/日付/時刻200
- [ ] 学習/検証/テスト分割 (JVS/JSUT held-out はテスト専用に固定)
- [ ] データ品質チェック: 音素-テキスト整合、アクセント長=モーラ数、pyopenjtalk出力との一致度

### Exit Criteria

- 統合スキーマの全データが `data/processed/` に格納
- 各カテゴリの分布ヒストグラムを可視化 (Koriyamaベンチと同等の比率)
- Hard-set 5カテゴリの gold labels が揃っている
- データローダー (streaming, on-the-fly padding) が動く

### 参考ソース

- [pyopenjtalk-plus](https://github.com/tsukumijima/pyopenjtalk-plus)
- [UniDic](https://clrd.ninjal.ac.jp/unidic/) (推定リンク)
- [sarulab-speech/jsut-label](https://github.com/sarulab-speech/jsut-label)

---

## Phase 2: シングルタスク G2P ベースライン (目標: 2週間)

### 目的

「ModernBERT で単純に seq2seq G2P」を実装し、既存ベースラインとの差分を測定する。マルチタスクの複雑さを入れる前の**必須のsanity check**。

### タスク

- [ ] **モデル1 (main)**: `sbintuitions/modernbert-ja-130m` encoder + T5系 char-level decoder
  - LoRA (r=16) と full fine-tune の両方を試行
- [ ] **モデル2 (対照)**: `tohoku-nlp/bert-base-japanese-char-v2` encoder + token classification head (NHK BAS 相当だが G2Pメインとして)
- [ ] **モデル3 (対照)**: `sbintuitions/modernbert-ja-130m` encoder + MeCab pretokenize + token classification (Hida 2022 相当)
- [ ] 3モデル全てで JSUT Basic5000 PER と JVS-3000 kana CER を測定
- [ ] Ablation: encoder サイズ (30M / 70M / 130M / 310M)
- [ ] トークナイザー選択の結論を出す (05_technical_design.md の (a)(b)(c)のどれが最良か)

### Exit Criteria

- Phase 0 の OpenJTalk baseline を JSUT PER で下回る
- 3モデルの head-to-head 結果と要因分析を `docs/experiments/phase2.md` に記録
- 最良のトークナイザー戦略を選定し、Phase 3 以降の主軸に採用

### 参考ソース

- [modernbert-ja-130m](https://huggingface.co/sbintuitions/modernbert-ja-130m)
- [bert-base-japanese-char-v2](https://huggingface.co/tohoku-nlp/bert-base-japanese-char-v2)
- [Kurihara Interspeech 2024](https://www.isca-archive.org/interspeech_2024/kurihara24_interspeech.pdf)

---

## Phase 3: マルチタスク学習 (目標: 3週間)

### 目的

Hida et al. ICASSP 2022 と NHK Kurihara Interspeech 2024 を統合し、多タスク同時学習で near-oracle 品質に近づける。

### タスク

- [ ] Head A (G2P main), Head B (Polyphone), Head C (APBP), Head D (ANPP), Head E (BAS) を実装
- [ ] マルチタスクデータセット:
  - JSUT accent-labeled subset を APBP/ANPP/BAS用に整形
  - polyphone labels を UniDic + pyopenjtalk-plus 辞書から自動抽出
- [ ] Loss weights の grid search (α_G2P, α_poly, α_APBP, α_ANPP, α_BAS)
- [ ] カテゴリ別 sample reweighting の効果測定
- [ ] Ablation: 各 head の on/off で PER と mora-accent accuracy の変化を測定
- [ ] Hida 2022 の 96.66% mora-accent accuracy を目標に到達

### Exit Criteria

- JSUT Basic5000 PER < 1.5%
- JSUT accent-labeled subset で mora-accent accuracy > 96.5%
- 各 head の寄与が定量的に説明できる (Hida table 相当を作る)

### 参考ソース

- [Hida ICASSP 2022](https://ar5iv.labs.arxiv.org/html/2201.09427)
- [Kurihara Interspeech 2024](https://www.isca-archive.org/interspeech_2024/kurihara24_interspeech.pdf)

---

## Phase 4: ハイブリッド化 (辞書 primary + ModernBERT correction) (目標: 2週間)

### 目的

haqumei / Misaki v2 が採用する「辞書優先 + NN補正」を実装し、決定的な精度向上を狙う。

### タスク

- [ ] Inference pipeline実装:
  1. pyopenjtalk (+ pyopenjtalk-plus 辞書) で base yomi/accent を取得
  2. Uncertainty検出 (多音字位置、複合語、助数詞連結、OOVカタカナ、固有名詞)
  3. Uncertainty region のみ ModernBERT を fire
  4. Reconciliation: 辞書 = 高信頼、ModernBERT = 低信頼のヒューリスティック合成
- [ ] Reconciliation strategy の複数比較:
  - Strategy A: 辞書ヒットは常に辞書優先 (最も保守的)
  - Strategy B: uncertainty threshold で切り替え
  - Strategy C: ModernBERTのconfidence が高ければ辞書上書き
- [ ] JSUT / JVS / ROHAN でハイブリッド前後の結果比較
- [ ] Latency比較 (ハイブリッドは条件付き実行なので、フル実行より高速なはず)

### Exit Criteria

- JSUT Basic5000 PER < 1.0% (haqumei の 1.17% を明確に上回る)
- ROHAN KER < 1.5% (haqumei の 1.64% を明確に上回る)
- JVS-3000 kana CER < 0.9% (OpenJTalk の 1.03% を明確に上回る)

### 参考ソース

- [haqumei](https://github.com/o24s/haqumei)
- [Misaki README](https://github.com/hexgrad/misaki)

---

## Phase 5: スケール & Ablation (目標: 2週間)

### 目的

決定した最良アーキテクチャで、モデルサイズ・トークナイザー・データ量の系統的な Ablation を行い、最良点を確定する。

### タスク

- [ ] modernbert-ja-30m / 70m / 130m / 310m の4系統で完全学習し、PER と kana CER をプロット
- [ ] llm-jp-modernbert-base との head-to-head 比較
- [ ] 学習データ量スケーリング曲線 (10%, 30%, 50%, 100%)
- [ ] Multi-task heads on/off の完全ablation table を作成
- [ ] Loanword ONNX モデル (haqumei-kanalizer 相当) の代替として、外来語専用の小さなヘッドを追加する Ablation
- [ ] CTC auxiliary loss (CC-G2PnP idea) の on/off ablation

### Exit Criteria

- 「モデルサイズ vs 精度」のPareto曲線を提示可能
- 「トークナイザー vs 精度」を系統的に説明可能
- 「学習データ量 vs 精度」のscaling law を提示可能
- 全 ablation 結果を1つの結果表にまとめる (論文 Table 相当)

### 参考ソース

- [modernbert-ja モデル群](https://huggingface.co/sbintuitions)
- [llm-jp-modernbert 論文 (arxiv 2504.15544)](https://arxiv.org/pdf/2504.15544)
- [CC-G2PnP (arxiv 2602.17157)](https://arxiv.org/pdf/2602.17157)

---

## Phase 6: 評価・公開・統合 (目標: 1週間)

### 目的

Final モデルを 3本柱ベンチマーク + Hard-sets で総合評価し、公開可能な状態にする。

### タスク

- [ ] Final モデルで以下を計測:
  - JVS-3000 kana CER (LLM SOTAとの差分をプロット)
  - JSUT Basic5000 PER (haqumei との差分)
  - ROHAN KER (haqumei との差分)
  - 5カテゴリ Hard-set の per-category PER
  - Downstream TTS pronunciation CER (Style-Bert-VITS2 に投入して測定)
- [ ] Latency / throughput / model size を計測 (CPU int8, GPU fp16, GPU int8)
- [ ] README, MODEL_CARD, LICENSE, INFERENCE_GUIDE を整備
- [ ] ONNX export 手順とサンプルコード
- [ ] pyopenjtalk との互換API wrapper (`g2p(text) -> phonemes` のシグネチャ)
- [ ] Hugging Face Hub にモデル公開
- [ ] Blog / arXiv preprint / GitHub リリース

### Exit Criteria

- 3本柱で haqumei / OpenJTalk を明確に上回る (0.15%以上の差)
- LLM (Gemini 3.1 Pro 0.62%) との差 < 0.2% (stretch: 差 < 0.1% あるいは越え)
- ドキュメントとサンプルコードが最小限で使える
- モデル配布と評価スクリプトが CI で再現できる

---

## 全体的な進捗管理

### 主要マイルストーン

| Phase | 期間 | 累計 | 主要成果 |
|---|---|---|---|
| P0 | 1週間 | 1週間 | ベースライン再現 |
| P1 | 2週間 | 3週間 | 学習データ整備 |
| P2 | 2週間 | 5週間 | シングルタスク baseline (トークナイザー決定) |
| P3 | 3週間 | 8週間 | マルチタスク学習 |
| P4 | 2週間 | 10週間 | ハイブリッド化 (haqumei越え) |
| P5 | 2週間 | 12週間 | スケール & Ablation |
| P6 | 1週間 | 13週間 | 公開 |

**総期間目安: 13週間 (約3ヶ月)** — 1人フルタイム想定。

### リスクと予防策

| リスク | 予防策 |
|---|---|
| トークナイザー選択の失敗 | Phase 2で3並列パイロット |
| データライセンス問題 | Phase 1 開始前に法務レビュー |
| 学習時計算資源不足 | LoRAとfull fine-tuneの両方を用意、130mから開始 |
| 評価データ (JVS-3000) 未公開 | 論文著者に問い合わせ、代替として自作JVSサブセットで評価 |
| ModernBERTトークナイザー起因の性能悪化 | Phase 2で確認、必要ならchar-levelにpivot |
| ハードsetのアノテーション不足 | Phase 1 での明示的キュレーション時間確保 |
| Loss weight tuning が難航 | Phase 3で grid search を初期から計画 |

### 開発時に監視すべき指標

**毎epoch/評価毎に記録**:
- JSUT Basic5000 PER (主要)
- JVS-3000 kana CER (主要)
- ROHAN KER (主要)
- Mora-accent accuracy (JSUT accent labeled)
- Per-category PER (5 hard-sets)
- Training loss (per-head で分離)
- Validation loss (per-head)
- GPU memory / step time / throughput

**Weekly checkpoint**:
- Ablation table update
- Failure case集 (誤り例を最新モデルで5例ずつ確認、docs/failure_analysis.md)
- ベースライン比較チャート更新

---

## 継続的な改善アイデア (Phase 6以降)

- **LLM 蒸留**: Claude/Geminiで JVS-scale データを合成し、その kana を weak supervision に活用 (licensing慎重)
- **Multi-speaker prosody**: 単一G2Pではなく話者ごとのアクセント傾向を条件付け
- **多言語混在**: 英語/中国語/韓国語の混在テキストへの拡張 (Misakiの方向性)
- **Streaming推論**: CC-G2PnP のCTCベース設計を統合してストリーミング化
- **RAG風の辞書拡張**: 新語 (SNS由来、企業名) を外部知識として動的に注入

---

## まとめ

- 3ヶ月・7フェーズ・各フェーズ明確な exit criteria
- **haqumei を超える** ことを Phase 4 の閾値に置き、それより後は精度上限探索
- Ablation は Phase 5 に集中させ、途中の実装コストを最小化
- 公開は Phase 6 でまとめて、モデル / データ処理 / 評価スクリプトを揃える
