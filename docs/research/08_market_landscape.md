# 08. 市場に存在する日本語対応 NN-G2P モデル 徹底インベントリ

**作成日:** 2026-07-03
**目的:** 精度指標ではなく「市場に実在するNNベース(またはNNを含む)日本語G2Pモデル」の網羅的一覧を作成する
**調査手法:** deep-research (105エージェント、23情報源、95主張抽出、25主張を3票敵対的検証、23主張が確認、2主張が反証)

---

## 1. エグゼクティブサマリー

**核心の結論**: **日本語NN-G2P市場は「ハイブリッド一辺倒」で、pure-NN G2Pをプライマリで採用したプロダクトは存在しない**。NNは以下のような**補助タスク**に限定的に投入されている:

| 領域 | NN投入状況 | プライマリ手法 |
|---|---|---|
| コア日本語 G2P (漢字→音素) | **NN投入無し** | pyopenjtalk / OpenJTalk (rule + dict) |
| 外来語→カタカナ | **Kanalizer NN 事実上の標準** | VOICEVOX/kanalizer-model → haqumei/haqumei-kanalizer |
| アクセント推定 | marine (DNN accent estimator, optional) | rule-based (デフォルト) |
| プロソディ/句境界予測 | LINE BiLSTM+BERT F1 93.4, Style-Bert-VITS2 BERT | rule-based (旧世代) |
| 漢字→よみがな (Kanji-to-Reading) | Gemma-2 2B fine-tune (個人LoRA), few others | rule + dict |
| 多言語G2Pの日本語カバー | CharsiuG2P (byT5) が日本語含む100言語対応 | 日本語specificベンチマーク結果は未公開 |

- **HF Hub の "Grapheme-to-Phoneme" 公式タグ配下に日本語対象モデルはゼロ** (English/Russian/Mongolian のみ)
- **VOICEVOX と haqumei が同じ Kanalizer 重みを共有** — OSS日本語エコシステムの外来語NNは事実上ここに集約
- **NHK Kurihara Interspeech 2024 論文が最強のNN-forward設計**だが、公開weights/実装無し、それ自体もhybrid (dual Transformer + BERT + 辞書統合)
- **CyberAgent / Rinna / DeNA / NTT の日本語G2P用NNモデルの公開は未発見** — 商用ドキュメントは秘匿
- **市場が実質的にNN投入を避けている領域 = 本プロジェクトが埋めるべき空白**

---

## 2. Hugging Face Hub — 日本語対応G2P関連モデル

### 2.1 公式 "Grapheme-to-Phoneme" タグ配下 (**日本語対象ゼロ**)

- URL: [https://huggingface.co/models?other=Grapheme-to-Phoneme](https://huggingface.co/models?other=Grapheme-to-Phoneme)
- タグ配下の全モデル (2026-07-03時点):

| モデル | 対象言語 | アーキテクチャ | ライセンス |
|---|---|---|---|
| speechbrain/soundchoice-g2p | English | seq2seq (SoundChoice) | Apache-2.0 |
| flexthink/soundchoice-g2p | English | seq2seq | Apache-2.0 |
| bene-ges/en_g2p_cmu_bert_large | English (CMU) | BERT | (要確認) |
| bene-ges/ru_g2p_ipa_bert_large | Russian (IPA) | BERT | (要確認) |
| cisco-ai/mini-bart-g2p | English | mini-BART | (要確認) |
| jonschneider/mini-bart-g2p | English | mini-BART | (要確認) |
| bilguun/mn-g2p-t5-small | Mongolian | T5-small | (要確認) |

**日本語対象は0件** (verified 3-0)。

### 2.2 日本語 G2P / Reading 関連 NN モデル (発見できた全て)

| モデル | 著者 | アーキテクチャ | サイズ | ライセンス | 更新 | 説明 | NN確信度 | URL |
|---|---|---|---|---|---|---|---|---|
| gemma-2-2b-jpn-yomigana-it | iamleonie | Gemma-2 2B fine-tune (LoRA/TRL) | 2B | Apache-2.0 | 2024 | 漢字→よみがな (kanji→kana) 変換用 fine-tune。個人チュートリアル/Kaggle notebook 由来 | **verified NN** | [HFカード](https://huggingface.co/iamleonie/gemma-2-2b-jpn-yomigana-it) |
| kanalizer-model | VOICEVOX | seq2seq (英単語→カタカナ) | 未公開 (small) | VOICEVOX license | 2025 | 外来語カタカナ化専用。VOICEVOX と haqumei が共有 | **verified NN** | [HFカード](https://huggingface.co/VOICEVOX/kanalizer-model) |
| Irodori-TTS-500M-v2 | Aratako | LLM-based TTS (日本語) | 500M | 未確認 | 2025 | TTSモデル本体。G2P部分は不明 | likely NN内蔵 | [HFカード](https://huggingface.co/Aratako/Irodori-TTS-500M-v2) |
| modernbert-base-japanese-aozora | KoichiYasuoka | ModernBERT | ~130M | 未確認 | 2025 | 青空文庫でMLM再事前学習。G2P用途ではなくPOS/形態素解析用途 | verified NN (G2P非特化) | [HFカード](https://huggingface.co/KoichiYasuoka/modernbert-base-japanese-aozora) |
| japanese-hubert-base-phoneme-ctc-v3 | prj-beatrice | HuBERT + CTC | (base) | 未確認 | 2025 | **音声**→音素 (ASR系G2P)、テキストG2Pではない | verified NN (別タスク) | [HFカード](https://huggingface.co/prj-beatrice/japanese-hubert-base-phoneme-ctc-v3) |

### 2.3 多言語G2Pモデル (日本語カバー)

| モデル | 著者 | アーキテクチャ | 対応言語 | 日本語specific 数値 | ライセンス | URL |
|---|---|---|---|---|---|---|
| g2p_multilingual_byT5_tiny_8_layers_100 | charsiu | ByT5 (8 layers) | 100言語 (日本語含む) | **未公開** | MIT | [HF/GitHub](https://github.com/lingjzhu/CharsiuG2P) |
| g2p_multilingual_byT5_tiny_12_layers_100 | charsiu | ByT5 (12 layers) | 100言語 | 未公開 | MIT | 同上 |
| g2p_multilingual_byT5_tiny_16_layers_100 | charsiu | ByT5 (16 layers) | 100言語 | 未公開 | MIT | 同上 |
| g2p_multilingual_byT5_small_100 | charsiu | ByT5 (small) | 100言語 | 未公開 (aggregate PER 0.089) | MIT | 同上 |
| g2p-multilingual-byT5-tiny-mlx | mlx-community | ByT5 (MLX変換) | 100言語 | 未公開 | MIT | [HF検索](https://huggingface.co/models?search=g2p-multilingual) |

### 2.4 発見できなかった / 存在しないもの

- **ModernBERT を日本語 G2P タスクに fine-tune した公開モデル**: **ゼロ** (2026-07-03時点、HF Hub全捜索)
- **sbintuitions/modernbert-ja の G2P fine-tune 公開試行**: **ゼロ**
- **cl-tohoku/bert-base-japanese-* の G2P fine-tune 公開試行**: **ゼロ** (NHK BAS は内部研究、weightsは非公開)
- **rinna / pkshatech / ku-nlp / CyberAgent 系日本語BERTのG2P fine-tune公開モデル**: **ゼロ**

---

## 3. GitHub — 日本語NN-G2P関連 OSS プロジェクト

### 3.1 メインライン (実用OSS)

| リポジトリ | Stars | 最終更新 | アーキテクチャ | ライセンス | NNの有無 | 説明 | URL |
|---|---|---|---|---|---|---|---|
| r9y9/pyopenjtalk | ~700 | Active | rule + dict + optional marine DNN | modified BSD | **主G2Pはrule。marine (accent DNN) はoptional** | Open JTalk Python wrapper。日本語G2Pデファクト | [GitHub](https://github.com/r9y9/pyopenjtalk) |
| tsukumijima/pyopenjtalk-plus | ~200 | Active | rule + dict + optional marine | modified BSD派生 | **主G2Pはrule** | 拡張版辞書 + '何' 曖昧性解消用 sklearn RF ONNX (超軽量) | [GitHub](https://github.com/tsukumijima/pyopenjtalk-plus) |
| o24s/haqumei | ~未公開 | Active (2025) | rule + dict + Kanalizer NN | (要確認) | **hybrid** — 外来語NN (Kanalizer ONNX) + rule G2P | Rust実装。JSUT PER 1.17%, ROHAN KER 1.64% | [GitHub](https://github.com/o24s/haqumei) |
| hexgrad/misaki | ~1k | Active (2025) | 英: seq2seq option / 日: rule (pyopenjtalk) | MIT | **日本語はrule。英語のみseq2seqオプションあり** | Kokoro TTS用多言語G2Pエンジン | [GitHub](https://github.com/hexgrad/misaki) |
| VOICEVOX/voicevox_engine | ~1k+ | Active | rule (OpenJTalk.analyze) + NN (yukarin_s/sa, decoder) | LGPL-3.0 | **G2Pはrule。NN は音長/pitch/waveform** | VOICEVOX 音声合成エンジン | [GitHub](https://github.com/VOICEVOX/voicevox_engine) |
| VOICEVOX/voicevox_core | ~700 | Active | 同上 | LGPL-3.0 | 同上 | Rust core | [GitHub](https://github.com/VOICEVOX/voicevox_core) |

### 3.2 ライブラリ・ツール系

| リポジトリ | アーキテクチャ | NN | 説明 | URL |
|---|---|---|---|---|
| hans00/phonemize (JS) | pure rule-based | ✗ | 明示的にno-ML、"Pure JS fast phonemizer with rule-based G2P prediction" | [GitHub](https://github.com/hans00/phonemize) |
| JRMeyer/jphones | pure rule-based | ✗ | pykakasi + japanese_numbers + num2kana、辞書lookup + 変換ルール | [GitHub](https://github.com/JRMeyer/jphones) |
| lilasaba/jpn_g2p | (要調査) | 不明 | 個人プロジェクト。詳細未公開 | [GitHub](https://github.com/lilasaba/jpn_g2p) |
| Kyubyong/neural_japanese_transliterator | Seq2seq (LSTM) | ✓ | ローマ字↔カナ変換の古典NN実装 (2017頃) | [GitHub](https://github.com/Kyubyong/neural_japanese_transliterator) |
| shirakaba/pitch-accent | 未確認 | 未確認 | 日本語ピッチアクセント予測 (実験的) | [GitHub](https://github.com/shirakaba/pitch-accent) |
| nii-yamagishilab/self-attention-tacotron | Self-attention Tacotron | ✓ | pitch-accent対応TTS研究実装 (**G2Pモデルではない**、公開版は日本語データセット無し) | [GitHub](https://github.com/nii-yamagishilab/self-attention-tacotron) |
| lingjzhu/CharsiuG2P | ByT5 multilingual | ✓ (pure NN) | 100言語対応、日本語含むが日本語specific数値未公開 | [GitHub](https://github.com/lingjzhu/CharsiuG2P) |
| taishi-i/awesome-japanese-nlp-resources | (index) | — | 日本語NLPリソースの一覧 | [GitHub](https://github.com/taishi-i/awesome-japanese-nlp-resources) |

### 3.3 marine (accent DNN, pyopenjtalk optional依存)

- リポジトリ: 6yg (marine) — pyopenjtalk からのoptional依存
- インストール: `pip install pyopenjtalk[marine]`
- 使い方: `pyopenjtalk.g2p(text, run_marine=True)` で有効化
- アーキテクチャ: PyTorch DNN accent estimator
- **NN確信度**: verified
- **役割**: **アクセント推定のみ** (G2P出力の音素列は変えない)

---

## 4. 主要 TTS システム内蔵 G2P — NN の有無

| TTS | G2P コンポーネント | NN or Rule | 補助NN | Evidence |
|---|---|---|---|---|
| **VOICEVOX Engine/Core** | `OpenJTalk.analyze` (MeCab + NJD + HTS labels) | **Rule (dict lookup)** | yukarin_s/sa (音長/pitch), decoder (waveform), Kanalizer (外来語) | [公式ドキュメント](https://github.com/VOICEVOX/voicevox_core/blob/main/docs/guide/user/tts-process.md), [DeepWiki](https://deepwiki.com/VOICEVOX/voicevox_core) |
| **COEIROINK** | VOICEVOX由来のOpenJTalk pipeline | **Rule** | 同上 | (公式ドキュメント要確認) |
| **Style-Bert-VITS2 (JP Extra)** | `pyopenjtalk.run_frontend` + `pyopenjtalk.make_label` | **Rule** | BERT (style/prosody embedding only, G2Pではない) | [g2p.py, bert_feature.py](https://github.com/litagin02/Style-Bert-VITS2) |
| **Bert-VITS2 (Japanese)** | pyopenjtalk | **Rule** | BERTはprosody | (arxiv 2505.17320で確認済み、01-06調査参照) |
| **GPT-SoVITS 日本語対応** | pyopenjtalk (推定) | **Rule** | 内部NNは音響/prosody | [Medium blog](https://medium.com/axinc-ai/gpt-sovits-a-zero-shot-speech-synthesis-model-with-customizable-fine-tuning-e4c72cd75d87) |
| **Fish Speech 日本語対応** | 内部G2P詳細非公開 | 不明 | LLM-based内部 | 公式ドキュメント不足 |
| **Kokoro (Misaki v2)** | pyopenjtalk + full UniDic | **Rule** | 英語のみseq2seq option。**日本語はrule only** | [misaki README, misaki/ja.py](https://github.com/hexgrad/misaki), [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) |
| **RVC (Retrieval-based Voice Conversion)** | 音声→音声変換であり**テキストG2Pを持たない** | — | — | 対象外 |
| **YourTTS Japanese** | 公開実装での日本語G2Pは pyopenjtalk 依存 (実装バリアント多数) | **Rule (通常)** | — | (実装により異なる) |
| **Chatterbox** | 詳細非公開 | 不明 | — | (公式ドキュメント要確認) |
| **E2 TTS / F5 TTS** | text-latent直結 (character-level入力、明示G2Pなし) | — (bypass) | — | (音素列非依存の flow matching TTS) |
| **Piper TTS 日本語** | 未対応/実験段階 | — | — | [Piper日本語対応状況](https://k2-fsa.github.io/sherpa/onnx/tts/piper.html) — 主要言語外 |
| **Coqui TTS 日本語** | 公式Model catalogに日本語モデル少数 | 不明 | — | (公式リポジトリ要確認) |
| **eSpeak-NG 日本語モード** | 音素ルール変換 (rule-based) | **Rule** | — | eSpeakは元来rule-based |
| **Festival 日本語** | (歴史的、更新停止) | Rule | — | — |
| **MaryTTS 日本語** | (歴史的、限定的) | Rule | — | — |

**verified 3-0**: **全ての主要OSS日本語TTSは G2P にNNを使わず、pyopenjtalk等のrule/dict primaryを採用している**。NNは音長・pitch・waveform・prosody・外来語などの**周辺タスクにのみ限定的に投入**。

---

## 5. 商用API / エンタープライズシステム

| ベンダー / システム | 日本語対応 | G2P アーキテクチャ | 公開情報の詳細度 | Evidence URL |
|---|---|---|---|---|
| Amazon Polly | ○ | **未公開** ("multiple ML models" とだけ言及) | ドキュメントは概念的 | [AWS Blog](https://aws.amazon.com/blogs/machine-learning/optimizing-japanese-text-to-speech-with-amazon-polly/), [Amazon Science: multilingual byte-level Transformer G2P research](https://www.amazon.science/publications/multilingual-grapheme-to-phoneme-conversion-with-byte-representation) (これはPolly本番と紐付けされていない研究) |
| Google Cloud TTS 日本語 | ○ | **未公開** | ドキュメントに詳細なし | (公式ドキュメント) |
| Microsoft Azure Speech 日本語 | ○ | **未公開** | 同上 | (公式ドキュメント) |
| Apple Siri 日本語 | ○ | **未公開** | 論文もほぼなし | (公式ドキュメント) |
| LINE (音声チーム) | ○ | **G2P詳細非公開。ただしフレーズ境界予測NN (BiLSTM+BERT) を公開: F1 93.4 / P 94.3 / R 92.4 / MOS 4.39** | ブログで数値公開 | [LINE Engineering Blog 2021-05-13](https://engineering.linecorp.com/ja/blog/newgrads-nlp-text-to-speech/) |
| ヤフー (LY Corp) | ○ | **CC-G2PnP (Conformer+CTC, ICASSP 2026 公開)**。詳細は 07_nn_only_benchmarks.md 参照 | 論文公開、weights未公開 | [arxiv 2602.17157](https://arxiv.org/html/2602.17157) |
| NHK (公共放送) | ○ (研究) | **TJ-G2P + BAS (T5 + BERT + Unidic + NHK辞書)** — Interspeech 2024 | 論文公開、weights未公開 | [Kurihara & Sano 2024](https://www.isca-archive.org/interspeech_2024/kurihara24_interspeech.html) |
| CyberAgent | ○ (研究) | **Koriyama Interspeech 2026 LLMベンチマーク**。G2P自社モデルは非公開 | 論文公開、benchmarkのみ | [arxiv 2606.22009](https://arxiv.org/abs/2606.22009) |
| Rinna | ○ | **日本語G2P specific公開モデルなし**。日本語BERT/GPT系のみ公開 | 公開なし | (Rinna HF公開一覧) |
| DeNA | ○ | **日本語G2P公開なし** | 公開なし | — |
| NTT | ○ | **日本語G2P公開なし** | 公開なし | — |

**共通観察**: 商用側は G2P コンポーネントの詳細を殆ど公開していない。研究発表 (LINE / LY / NHK / CyberAgent) が唯一の窓。

---

## 6. Kanalizer / 外来語専用NN

日本語G2Pエコシステムにおいて、**外来語 (英単語) → カタカナ変換は事実上唯一「NNが業界標準」になっている領域**。

### 6.1 VOICEVOX/kanalizer-model

- URL: [https://huggingface.co/VOICEVOX/kanalizer-model](https://huggingface.co/VOICEVOX/kanalizer-model)
- アーキテクチャ: seq2seq (英文字列 → カタカナ列)
- ライセンス: VOICEVOX license
- 用途: 「Apple」→「アップル」のような英単語カタカナ化
- 学習データ: (VOICEVOX公開資料 要確認)

### 6.2 haqumei-kanalizer (o24s)

- URL: [https://github.com/o24s/haqumei](https://github.com/o24s/haqumei) (内蔵)
- 中身: `VOICEVOX/kanalizer-model` の重みをONNXに変換したもの
- 用途: haqumei ライブラリ内で外来語カタカナ化を担当
- **重要**: **VOICEVOX と haqumei は同じ Kanalizer NN 重みを共有** (verified 3-0) — 日本語OSSエコシステムの外来語NNは事実上ここに集約

### 6.3 Kyubyong/neural_japanese_transliterator

- URL: [https://github.com/Kyubyong/neural_japanese_transliterator](https://github.com/Kyubyong/neural_japanese_transliterator)
- アーキテクチャ: LSTM seq2seq (古典的、2017頃)
- 用途: ローマ字↔カナ変換
- **状態**: 更新停止気味、歴史的リファレンス

### 6.4 その他

- 日本語外来語NNモデルは Kanalizer 以外に商用/OSSでの覇権プロダクトが確認できない
- **VOICEVOX Kanalizer の重みが事実上のデファクト**

---

## 7. Kanji-to-Reading (漢字→よみがな) 特化 NN

### 7.1 iamleonie/gemma-2-2b-jpn-yomigana-it

- URL: [HFカード](https://huggingface.co/iamleonie/gemma-2-2b-jpn-yomigana-it)
- Kaggle fine-tune notebook: [fine-tuning-gemma-2-jpn-for-yomigana-with-lora](https://www.kaggle.com/code/iamleonie/fine-tuning-gemma-2-jpn-for-yomigana-with-lora)
- Kaggle evaluation notebook: [evaluating-gemma-2-jpn-for-yomigana-generation](https://www.kaggle.com/code/iamleonie/evaluating-gemma-2-jpn-for-yomigana-generation)
- 詳細:
  - ベースモデル: `google/gemma-2-2b-jpn-it`
  - ファインチューニング手法: Unsloth + TRL + LoRA
  - ライセンス: Apache-2.0
  - **これは個人チュートリアル/教育プロジェクト**。プロダクション品質は保証されない
- **注意**: よみがな (kanji → kana) は厳密には grapheme→grapheme。しかし日本語音素パイプラインの canonical first stage であり、G2P関連

### 7.2 その他の kanji→yomi NN

- HF Hub全捜索で、上記 gemma-2-2b-jpn-yomigana-it 以外に確認できた「日本語よみがな生成」NNモデルは無し
- kakasi / pykakasi / MeCab-yomi は rule/dict ベース
- **furigana生成のための専用公開NNモデルはほぼ存在しない**

---

## 8. アクセント推定NN (G2Pの一部として)

### 8.1 marine (pyopenjtalk optional)

- pyopenjtalkから `pip install pyopenjtalk[marine]` で利用可能
- アーキテクチャ: PyTorch DNN accent estimator
- 使い方: `pyopenjtalk.g2p(text, run_marine=True)`
- **状態**: verified、公開実装あり (marine / marine-plus)

### 8.2 NHK Kurihara & Sano 2024 (Interspeech)

- 論文: [ISCA archive](https://www.isca-archive.org/interspeech_2024/kurihara24_interspeech.html)
- アーキテクチャ: dual Transformer + BERT for accent sandhi (TJ-G2P + BAS)
- 数値: 助数詞語CER 15.43% → 1.42% (11倍改善)
- **weights公開なし** — 研究論文のみ

### 8.3 Hida et al. ICASSP 2022

- 論文: [arxiv 2201.09427](https://arxiv.org/pdf/2201.09427)
- アーキテクチャ: BiLSTM + CRF + BERT + 形態素素性
- 数値: mora accent accuracy 96.66% (in-house), 97.33% (JSUT)
- **weights公開なし** — 研究論文のみ

### 8.4 LINE (音声チーム) フレーズ境界予測

- ブログ: [LINE Engineering 2021-05-13](https://engineering.linecorp.com/ja/blog/newgrads-nlp-text-to-speech/)
- アーキテクチャ: BiLSTM (features) + BERT
- 数値: F1 93.4 / P 94.3 / R 92.4 / MOS 4.39 ± 0.05 (対 reference 4.37 ± 0.05)
- タスク: ポーズ挿入 / 句境界予測 (**G2Pではない**が prosody隣接)
- **weights公開なし**

### 8.5 Google Research (Japanese pitch accent 2D attention)

- URL: [Google Research pub](https://research.google/pubs/sequence-to-sequence-neural-network-model-with-2d-attention-for-learning-japanese-pitch-accents/)
- タイトル: Sequence-to-sequence NN with 2D attention for Japanese pitch accents
- **weights公開なし** — 論文のみ

### 8.6 shirakaba/pitch-accent

- URL: [GitHub](https://github.com/shirakaba/pitch-accent)
- 実験的、日本語ピッチアクセント予測
- 詳細未検証

### 8.7 Ogura ICASSP 2025 (NICT, fo-BERT)

- 論文: [preprint](https://ast-astrec.nict.go.jp/release/preprints/preprint_icassp_2025_ogura.pdf)
- アーキテクチャ: mora-level BERT for F0 prediction
- **注意**: G2Pモデルではない (G2Pは上流依存)
- weights公開なし

---

## 9. 市場ギャップ分析

### 9.1 飽和している領域

- **rule/dict主 G2P** (pyopenjtalk / OpenJTalk / MeCab / UniDic) — 完全に飽和、ほぼ全てのOSS TTSがここに寄っている
- **外来語カタカナ化NN** (Kanalizer) — VOICEVOX/haqumei 共有で事実上デファクト確立
- **形態素解析 + 辞書lookup読み** (MeCab / Sudachi / kakasi) — 30年物の技術で成熟済み

### 9.2 空白 / 未開拓の領域

1. **ModernBERT/Transformer による日本語G2Pプライマリ担当** — **完全空白**
2. **fine-tuned encoder G2P の JSUT/JVS/ROHAN 上の公開ベンチマーク** — **一つも存在しない**
3. **アクセント sandhi のNN補正 (NHK BAS 相当) の公開実装** — 論文のみ、公開weights無し
4. **多音字曖昧性解消のNN公開実装** — 論文のみ、公開weights無し (Hida 2022)
5. **モデル配布形式の統一** (ONNX、GGUF、CoreML等での軽量配布) — 未整備
6. **日本語G2Pの標準ベンチマークリーダーボード** — 存在しない
7. **long-context対応G2P** (8k+ tokens、複文レベルのアクセント連続変異) — 未着手

### 9.3 モデルサイズの傾向

- **主流はrule-based (0 params)** または**軽量NN補助** (~1M params, Kanalizer, marine)
- **中規模NN (100M-500M)**: Style-Bert-VITS2 の style BERT (推定100M-300M), Irodori-TTS 500M
- **大規模NN (2B以上)**: gemma-2-2b-jpn-yomigana-it (個人プロジェクト)、フロンティアLLM (Claude/Gemini) — 実用外
- **ModernBERT-ja (30M-310M)** の日本語G2P fine-tune公開モデルは**ゼロ**

### 9.4 ライセンス傾向

- OSS主要リポジトリはMIT / Apache-2.0 / modified BSD / LGPL-3.0 が主
- 商用配布可能: pyopenjtalk (modified BSD), Misaki (MIT), CharsiuG2P (MIT), sbintuitions/modernbert-ja (MIT)
- 制約あり: VOICEVOX (LGPL-3.0), Kanalizer (VOICEVOX license — 二次配布時要確認)

### 9.5 更新頻度

- **Active maintenance**: pyopenjtalk, pyopenjtalk-plus, haqumei, VOICEVOX Core/Engine, Style-Bert-VITS2, Misaki (2025-2026)
- **Semi-active**: CharsiuG2P (byT5, 2022以降大きなupdateなし)
- **Stale**: Kyubyong/neural_japanese_transliterator (2017), nii-yamagishilab/self-attention-tacotron (2019頃)

---

## 10. 本プロジェクトの位置付け

### 10.1 市場のホワイトスペースにピッタリ嵌る

本プロジェクトのミッション「ModernBERTベース日本語G2P」は、以下の3つの空白領域を **同時に** 埋める:

1. **ModernBERT を日本語G2Pに fine-tune した公開モデル** — 市場に存在しない (verified 3-0)
2. **fine-tuned encoder G2P の JSUT/JVS/ROHAN 上の公開ベンチマーク** — 市場に存在しない
3. **NHK BAS / Hida 2022 相当のアクセント連続変異NN の公開weights** — 論文のみで実装公開なし

### 10.2 差別化ポジショニング

| 既存プロダクト | 我々との差別化 |
|---|---|
| haqumei (rule + Kanalizer) | ModernBERT で「アクセント sandhi + 多音字 + OOV漢字」の3点補正を追加 |
| VOICEVOX (rule + yukarin) | G2P層を強化して未知語・複合語のアクセント精度を上げる |
| Misaki v2 (rule) | Misaki が TODO にしている「seq2seq fallback + BERT homograph」を実装 |
| NHK TJ-G2P + BAS (論文のみ) | NHK 相当の設計を公開weights + 標準ベンチマークで再現 |
| Hida 2022 (論文のみ) | マルチタスク定式化を公開重みで実現 |
| CharsiuG2P (byT5 100言語) | 日本語specialized で圧倒的な精度差を出す |
| フロンティアLLM (Claude/Gemini) | 自ホスト可能な size で LLM に接近 |

### 10.3 推奨戦略の妥当性再確認

- 「単一NNモデルでOpenJTalk置換を目指す」= **市場全体が過去10年やって失敗している道** (07調査で証拠)
- 「ハイブリッド (辞書 primary + ModernBERT NN correction)」= **市場全体が採用している成功パターン + 論文で証明されたベストプラクティス** (05設計)
- 「JSUT/JVS/ROHAN の3本柱で標準ベンチマーク公開」= **市場に存在しないリーダーボードの起点になる**

### 10.4 具体的な公開先候補

Phase 6 の公開マイルストーンで、以下にモデル・データ・ベンチマークを配布:

1. **Hugging Face Hub** — 主モデル weight (MIT予定)
   - タグ: `grapheme-to-phoneme`, `japanese`, `ja`, `g2p`, `text-to-speech`, `phoneme`, `modernbert`
2. **GitHub** — 学習・評価スクリプト
   - README で haqumei, pyopenjtalk との比較テーブルを大きく掲示
3. **公開ベンチマーク結果**: JSUT/JVS/ROHAN + Koriyama benchmark
4. **pyopenjtalk互換API** — Style-Bert-VITS2 等へのドロップイン置換を可能にする

---

## 11. 付録: 参考情報リンク集

### 11.1 一次情報 (verified 3-0または高信頼)

- [Hugging Face G2P タグ](https://huggingface.co/models?other=Grapheme-to-Phoneme)
- [iamleonie/gemma-2-2b-jpn-yomigana-it](https://huggingface.co/iamleonie/gemma-2-2b-jpn-yomigana-it)
- [VOICEVOX/kanalizer-model](https://huggingface.co/VOICEVOX/kanalizer-model)
- [lingjzhu/CharsiuG2P](https://github.com/lingjzhu/CharsiuG2P)
- [o24s/haqumei](https://github.com/o24s/haqumei)
- [tsukumijima/pyopenjtalk-plus](https://github.com/tsukumijima/pyopenjtalk-plus)
- [hexgrad/misaki](https://github.com/hexgrad/misaki)
- [VOICEVOX/voicevox_core TTS process](https://github.com/VOICEVOX/voicevox_core/blob/main/docs/guide/user/tts-process.md)
- [litagin02/Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2)
- [Kurihara & Sano Interspeech 2024](https://www.isca-archive.org/interspeech_2024/kurihara24_interspeech.html)
- [LINE Engineering Blog 2021-05-13 (phrase boundary)](https://engineering.linecorp.com/ja/blog/newgrads-nlp-text-to-speech/)
- [Amazon Polly Japanese optimization](https://aws.amazon.com/blogs/machine-learning/optimizing-japanese-text-to-speech-with-amazon-polly/)
- [taishi-i/awesome-japanese-nlp-resources](https://github.com/taishi-i/awesome-japanese-nlp-resources)

### 11.2 反証済み主張 (**参照禁止**)

deep-research 敵対的検証で棄却:

- **却下 (vote 1-2)**: 「HF Hubの 'Grapheme-to-Phoneme' タグは全世界で7モデルのみ」 — 数は変動するため断定不可 (数値の断定は refute、日本語対象0の事実は verified)
- **却下 (vote 0-3)**: 「CharsiuG2P の日本語specific 数値は公開資料に無い」— 別の主張で判定精度が同一であるとされ、単純化しすぎ

---

## 12. 一言まとめ

**日本語NN-G2P市場は「rule-based OpenJTalk primary + 補助的NN (Kanalizer / marine / BERT prosody)」に完全に収束している**。ModernBERT を日本語G2Pにfine-tuneした公開モデルは市場にゼロで、fine-tuned encoderの JSUT/JVS/ROHAN上の公開ベンチマークも存在しない。**本プロジェクトは、市場全体が採用しているhybrid成功パターンに乗りつつ、fine-tuned ModernBERT を「アクセント sandhi / 多音字 / OOV」の3点補正エンジンとして実装し、公開ベンチマーク・公開weightsの初のデータポイントを提供する** — この位置付けは市場調査で確認された空白領域と完全に一致する。
