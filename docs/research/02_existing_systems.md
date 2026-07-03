# 02. 既存日本語G2Pシステムのサーベイ

各システムの仕組み・強み・弱み・我々のプロジェクトへの示唆をまとめる。すべて deep-research の一次情報源に基づく。

---

## A. ルールベース / 辞書ベース

### A.1 Open JTalk / pyopenjtalk

- **リポジトリ**: [r9y9/pyopenjtalk](https://github.com/r9y9/pyopenjtalk)
- **中身**: MeCab による形態素解析 → NAIST-jdic (Open JTalkバイナリ辞書) から読み・アクセント型を辞書lookup → HMMベースのcontext-dependent label生成
- **強み**: 決定的、高速、辞書に載っていれば非常に高精度。日本語TTSデファクト。
- **弱み**:
  - 未知語 (新語・固有名詞・外来語) は近似規則にフォールバックし精度が急落
  - 助数詞語や複合語のアクセント連続変異が苦手 (NHK の Kurihara ら 2024 が定量的に指摘: 助数詞語 CER 16.24%)
  - 辞書の更新が事実上停止しており、2020年代の新語に対応できない
- **ベンチマーク数値**:
  - Koriyama Interspeech 2026 JVS-3000: kana CER = 1.03%
  - Kurihara Interspeech 2024 Wikipedia+アクセント辞書: 助数詞語 CER = 16.24%, 固有名詞 phoneme CER = 8.33%
- **示唆**: **ハイブリッドの土台**として最適だが、**そのまま使うと固有名詞・助数詞・複合語で崩れる** — ここが我々の NN で狙い撃ちすべきターゲット。
- ソース: [pyopenjtalk README](https://github.com/r9y9/pyopenjtalk), [Kurihara & Sano Interspeech 2024](https://www.isca-archive.org/interspeech_2024/kurihara24_interspeech.pdf), [Koriyama arxiv 2606.22009](https://arxiv.org/html/2606.22009)

### A.2 pyopenjtalk-plus

- **リポジトリ**: [tsukumijima/pyopenjtalk-plus](https://github.com/tsukumijima/pyopenjtalk-plus)
- **中身**: pyopenjtalk のフォーク。辞書を UniDic ベースに拡張・修正、Bert-VITS2/Style-Bert-VITS2/GPT-SoVITS 用の実運用改良版
- **強み**: 現代語・新語カバレッジが本家より高い、UniDic の読み・アクセント情報を直接活用可能
- **弱み**: 本家と同じくルール/辞書由来の限界を継承
- **示唆**: **学習データ生成の主要ソース**。辞書から (漢字表記, 読み仮名, アクセント型) のトリプルを大量取得可能。

### A.3 haqumei (**現時点のOSS SOTA相当**)

- **リポジトリ**: [o24s/haqumei](https://github.com/o24s/haqumei)
- **中身**: 修正版Open JTalk + pyopenjtalk-plus辞書 + オプションのONNXニューラル外来語モデル (haqumei-kanalizer)
- **公開ベンチマーク結果**:
  - JSUT Basic5000 (prj-beatrice/jsut-label, use_unidic_yomi=true): **PER 1.17%** (S=2117, D=527, I=831, N=297843)
  - ROHAN: **KER 1.64%** (S=1689, D=493, I=288, N=150637)
- **強み**: OSSで最も低いPER/KER。実装が公開されていて再現可能。
- **弱み**:
  - Self-reported 数値 (査読論文ではない)
  - モデルアーキテクチャがハイブリッドで解釈が難しい
- **示唆**: **必ずベースラインとして再現・比較する**。特にどの Errore カテゴリ (S=Substitute, D=Delete, I=Insert) が多いかを分析することで、我々のモデルで狙うべき失敗モードが特定できる。ソース: [haqumei README](https://github.com/o24s/haqumei)

### A.4 その他のルール系

- **MeCab + IPA/UniDic辞書**: 素の形態素解析器。読みは辞書lookup、アクセント情報無し。研究の下流タスクとして直接使用は稀。
- **Julius (音素辞書)**: 主にASR用途で、G2Pモデルとしての完成度は Open JTalk 未満。

---

## B. Neural系オープンソースG2P

### B.1 Misaki (v2 Japanese support)

- **リポジトリ**: [hexgrad/misaki](https://github.com/hexgrad/misaki)
- **中身**: 多言語G2Pエンジン。第二世代日本語対応で **pyopenjtalk + full unidic** を基盤に、pitch accent マーク + フレーズマージを追加
- **将来計画 (READMEのTODO)**:
  - 辞書学習の seq2seq フォールバック (OOV対策)
  - BERT contextual embeddings + logistic regression による homograph曖昧性解消
- **示唆**: Kokoro TTSに採用されている多言語G2Pの標準的な設計。我々のプロジェクトの構造とかなり近い方向性を独立に選択している。
- ソース: [Misaki README](https://github.com/hexgrad/misaki)

### B.2 Style-Bert-VITS2 (JP-Extra)

- **リポジトリ**: [litagin02/Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2/blob/master/docs/Style-Bert-VITS2_en.md)
- **中身**: 音声合成モデル。G2Pレイヤは pyopenjtalk 依存 (`g2kata_tone` / `kata_tone2phone_tone` ユーティリティ)。
- **示唆**: 実用TTSがG2Pに pyopenjtalk をそのまま採用しているため、**改善したG2Pのドロップイン置換先**として使える。
- ソース: [Style-Bert-VITS2 docs](https://github.com/litagin02/Style-Bert-VITS2/blob/master/docs/Style-Bert-VITS2_en.md)

### B.3 Bert-VITS2 / GPT-SoVITS / VITS Japanese

- 全て pyopenjtalk (もしくはそのフォーク) を採用。G2Pレイヤ自体には独自のNNを持たない。
- ソース: [arxiv 2505.17320](https://arxiv.org/html/2505.17320v1) が VITS 系Japanese pipelinesの pyopenjtalk 使用を明記。

### B.4 ESPnet (音素トークナイザ)

- **リポジトリ**: [espnet/espnet](https://github.com/espnet/espnet/blob/master/espnet2/text/phoneme_tokenizer.py)
- **中身**: pyopenjtalkベースの音素トークナイザ実装。TTS/ASR両用途で汎用的。
- **示唆**: 音素ラベル定義と正規化ロジックの実装リファレンス。

### B.5 CC-G2PnP (LY Corporation, ICASSP 2026)

- **論文**: [arxiv 2602.17157](https://arxiv.org/pdf/2602.17157) (Shirahata & Yamamoto)
- **中身**: 単語境界に依存しないストリーミング G2P。CTC decoder でgrapheme-phoneme対応を暗黙学習し、日本語のような**単語区切りが明示的でない言語**に適用可能
- **強み**: MeCabなしでE2E処理できる、ストリーミング対応
- **弱み**: バッチ精度は辞書ベースに劣る (論文中の比較値はKoriyama benchmarkほどタイトではない)
- **示唆**: 我々のプロジェクトはバッチ処理・オフライン推論を前提としているため、直接採用しないが、**アライメント学習の技術要素**として CTC は有用。特に seq2seq ヘッドの補助ロスとして検討価値あり。

### B.6 pkshatech / cl-tohoku / rinna / ku-nlp 系日本語BERT

Hugging Face公開の主要日本語BERTは一般的なMLMタスク向けであり、**G2P特化の公開重みは調査時点で確認できず**。しかし基盤エンコーダとしては以下が候補:

- **[cl-tohoku/bert-base-japanese-char-v2](https://huggingface.co/tohoku-nlp/bert-base-japanese-char-v2)**: 文字レベル日本語BERT。**NHK の BAS モジュールが採用**。トークナイザー由来の失敗モードを回避できる。
- **[sbintuitions/modernbert-ja](https://huggingface.co/sbintuitions/modernbert-ja-310m)**: ModernBERT日本語版。詳細は 05_technical_design.md 参照。
- **[llm-jp/llm-jp-modernbert-base](https://huggingface.co/llm-jp/llm-jp-modernbert-base)**: 別系統のModernBERT日本語版。llm-jp-corpus v4 (~0.69Tトークン) で学習。

---

## C. 学術系の高精度アーキテクチャ

### C.1 NHK の Dual Transformer (Interspeech 2024) — **最重要参照**

- **論文**: [Kurihara & Sano, "Enhancing Japanese Text-to-Speech Accuracy..." Interspeech 2024](https://www.isca-archive.org/interspeech_2024/kurihara24_interspeech.pdf)
- **アーキテクチャ**:
  - **TJ-G2P**: T5ベースの基本G2P。504,545 PPL sentences + 4,000 human-labeled JSUT Basic5000 PPL文で学習
  - **BAS (BERT-based Accent Sandhi)**: `tohoku-nlp/bert-base-japanese-char-v2` を fine-tune。8,369 複合語 + 246 human-labeled複合語で学習。TJ-G2P の出力をアクセント連続変異の観点で補正
- **公開結果 (Wikipedia+アクセント辞書 evaluation)**:

  | metric | OpenJTalk | TJ-G2P | TJ-G2P + BAS |
  |---|---|---|---|
  | 助数詞語 CER | 16.24% | 15.43% | **1.42%** (11.4x reduction) |
  | 固有名詞 phoneme CER | 8.33% | 6.91% | **5.89%** |

- **JSUT複合語 246 items のアクセント連続変異評価**:

  | metric | naive PPL 連結 | BAS |
  |---|---|---|
  | CER | 15.21% | **6.04%** (2.5x reduction) |
  | BLEU | 0.64 | 0.89 |
  | RIBES | 0.96 | 0.99 |
  | NIST | 9.13 | 11.52 |

- **示唆**: **これが我々のアーキテクチャの直接的な青写真**。TJ-G2P の位置に ModernBERT-based seq2seq を置き、BAS を token classification head として組み込めば、既存の証拠に基づく最も強い設計になる。

### C.2 Hida et al., ICASSP 2022 — **マルチタスク定式化の実証**

- **論文**: [Hida et al., "Polyphone disambiguation and accent prediction using pre-trained language models in Japanese TTS front-end", ICASSP 2022 (arxiv 2201.09427)](https://ar5iv.labs.arxiv.org/html/2201.09427)
- **アーキテクチャ**:
  - BiLSTM + CRF ベース
  - BERT暗黙embedding + 明示的形態素素性 (MeCab features) の両方を注入
  - アクセント推定を **APBP (Accent Phrase Boundary Prediction)** + **ANPP (Accent Nucleus Position Prediction)** に分解
- **結果**:
  - **多音字精度: 94.34%** (BERT暗黙+明示素性で +5.7 point 改善)
  - APBP F1: 96.30, sentence-exact: 58.68%
  - **モーラアクセント精度: 96.66%** (in-house), **97.33%** (JSUT公開分)
  - **主観MOS: 3.67 ± 0.07** (対 オラクル記号 3.69 ± 0.07) — near-oracle
- **示唆**: **マルチタスク定式化 (G2P + APBP + ANPP + 多音字) が有効**であることを実証。我々の ModernBERT にも同じマルチタスクヘッド群を載せる根拠となる。
- **注意**: 論文中の 94.34% と 96.66% は in-house NHK data 上の数値で、公開再現は不可。ただし JSUT split の 97.33% は再現可能。

### C.3 Hida et al., 追加論文

- [arxiv 2212.08321](https://arxiv.org/abs/2212.08321) - ICASSP 2022 に続く同グループのアクセント推定関連
- [arxiv 2204.03067](https://ar5iv.labs.arxiv.org/html/2204.03067) - 関連研究

### C.4 Ogura et al., ICASSP 2025 (NICT ASTREC)

- **論文**: [Ogura ICASSP 2025 preprint (NICT)](https://ast-astrec.nict.go.jp/release/preprints/preprint_icassp_2025_ogura.pdf)
- **中身**: アクセント推定に関する新しい定式化。詳細は 04_papers_and_references.md 参照。
- **示唆**: Hida et al. 以降のアクセント推定研究の続き。実装前に一次資料の読解必須。

---

## D. LLMベース G2P (2026年の新SOTA)

### D.1 Koriyama Interspeech 2026 Benchmark — **必読**

- **論文**: [Koriyama, "Benchmarking Large Language Models for Grapheme-to-Phoneme Conversion: A Japanese Case Study" (arxiv 2606.22009)](https://arxiv.org/html/2606.22009)
- **評価セット**: JVS nonpara30 から 3,000 sentences を人手アノテーション。内訳:
  - 漢字固有名詞 6.0%
  - カタカナ固有名詞 8.5%
  - 数詞 14.2%
  - その他一般文 71.3%
- **結果 (parse-mode kana CER)**:

  | System | kana CER |
  |---|---|
  | Claude Opus 4.6 | **0.52%** |
  | Gemini 3.1 Pro | 0.62% |
  | (他LLM省略) | |
  | **OpenJTalk (baseline)** | 1.03% |

- **重要な洞察**: 論文は「kana CER は基本的にphoneme error rate と等価」と明言。したがって PER と直接比較可能。
- **下流TTS評価**: Gemini 3.1 Pro の kana を kana入力型TTSに投げると 2.38% CER、対して E2E Gemini 2.5 Flash TTS は 3.96%、Qwen 3 TTS は 4.31%、CosyVoice 2 は 12.08%。**G2Pの質は今もE2E TTS品質のボトルネック**であることを実証。
- **示唆**:
  - **LLMは真のSOTAだが再現性・コスト・安定性・レイテンシで実用外**。我々のプロジェクトの正当化ロジックは「LLM並みの精度を、自ホストできる encoder サイズで達成」となる。
  - **評価プロトコルの標準として JVS-3000 を必ず含める**。
  - LLM出力を教師データとした distillation の可能性 — LLMを金の物差しとして使い、その kana ラベルで ModernBERT を蒸留する戦略が現実的。
- **時限性の注意**: この論文は 2026-07 時点で ~2週間前公開、Claude/Gemini APIバージョン依存で数値は流動的。

### D.2 商用API (Google/Amazon/Microsoft/LINE等)

- 公開ベンチマーク結果は限定的。Google TTS、Amazon Polly、Azure Speech の日本語G2Pの技術的詳細は非公開。
- LY Corporation (LINE/ヤフー) は前述 CC-G2PnP を ICASSP 2026 で公開。ソース: [arxiv 2602.17157](https://arxiv.org/pdf/2602.17157)
- **示唆**: 商用APIは直接の比較対象外。SOTAは Koriyama benchmark の LLM 数値で代表させる。

---

## E. NeMo などフレームワーク組み込みG2P

- **NVIDIA NeMo**: [G2P documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/tts/g2p.html)
  - 英語含む多言語G2Pのトレーニングパイプラインを提供。日本語向けの決定的な組み込みG2Pモデルは無いが、seq2seq (ByT5 / T5) のfine-tune基盤として活用可能。

---

## F. 全体的な洞察と設計への示唆

1. **主要な OSS TTS は全て pyopenjtalk (Open JTalk) に寄っている**。したがって我々のG2Pは、pyopenjtalk が壊れる部分 (助数詞・複合語・固有名詞・外来語・多音字) に注力すれば、そのままドロップイン置換の商用価値が生まれる。
2. **NHK の TJ-G2P + BAS は最良のTransformer G2Pアーキテクチャの青写真**。特に BAS の設計 (bert-base-japanese-char-v2で連続変異のみ担当) は我々の ModernBERT の設計に直接転用できる。
3. **Hida et al. のマルチタスク (G2P + APBP + ANPP + 多音字) は near-oracle MOS を出せる証拠**。ロス設計はこれに従う。
4. **フロンティア LLM は真のSOTA だが自ホスト外**。したがってプロジェクトの正当化は「LLM 品質を数百MB のエンコーダで実現する」ポジショニング。
5. **haqumei は再現可能な OSS ベースライン**。開発初期に必ず環境構築して JSUT/ROHAN で数値を再現し、その後の改善を測定する基準にする。

---

## 参考: システム比較サマリー表

| System | Type | Latest / Impl. | JSUT-Basic5000 PER | JVS-3000 kana CER | 備考 |
|---|---|---|---|---|---|
| OpenJTalk / pyopenjtalk | Rule + dict | Active | — | 1.03% | デファクト、辞書更新停止 |
| pyopenjtalk-plus | Rule + dict | Active | — | — | 辞書改良フォーク |
| haqumei | Hybrid | Active | **1.17%** | — | ONNX 外来語NN併用、OSS実用最強 |
| Misaki v2 | Hybrid | Active | — | — | pyopenjtalk+unidic+seq2seq TODO |
| NHK TJ-G2P + BAS | T5 + BERT | Interspeech 2024 | 6.04% (compound-only) | — | in-house data基準 |
| Hida BiLSTM+BERT | BiLSTM+CRF+BERT | ICASSP 2022 | 96.66% mora accent | — | in-house data基準 |
| CC-G2PnP | CTC + streaming | ICASSP 2026 | — | — | 単語境界不要 |
| Claude Opus 4.6 | LLM API | 2026-06 | — | **0.52%** | 非再現、API依存 |
| Gemini 3.1 Pro | LLM API | 2026-06 | — | 0.62% | 非再現、API依存 |
| **本プロジェクト (目標)** | ModernBERT + dict + multi-task | — | < 1.0% | < 0.62% | ハイブリッド設計 |
