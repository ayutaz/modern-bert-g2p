# 04. 論文・参考文献

deep-research で敵対的検証 (3票中2票以上でrefute判定を通過) を経た信頼できる一次情報のみを列挙する。**Refuted (反証済み) の主張は末尾に別記**。

---

## A. 直接的に採用すべきコア論文 (Tier 1: must-read)

### A.1 Koriyama, "Benchmarking Large Language Models for Grapheme-to-Phoneme Conversion: A Japanese Case Study" (Interspeech 2026)

- **URL**: [https://arxiv.org/abs/2606.22009](https://arxiv.org/abs/2606.22009) / [HTML版](https://arxiv.org/html/2606.22009)
- **要旨**: 30+ の LLM と conventional morphological analyzers (OpenJTalk, MeCab+UniDic) を、3,000文の人手アノテーションJVS-based ベンチマークで比較。kana CER を主指標とする。
- **中核的知見**:
  - Claude Opus 4.6 = 0.52%、Gemini 3.1 Pro = 0.62%、OpenJTalk = 1.03%
  - kana CER は phoneme error rate と本質的に等価
  - Downstream TTS で G2P品質がボトルネックであることを実証
- **我々のプロジェクトへの示唆**: **JVS-3000 を主要評価セットとして採用**する根拠。LLMは技術ターゲット (自ホスト可能な精度限界の目標)。
- **時限性**: 2026-06 公開、~2週間前。API依存で数値は流動的。

### A.2 Kurihara & Sano (NHK), "Enhancing Japanese Text-to-Speech Accuracy..." (Interspeech 2024)

- **URL**: [https://www.isca-archive.org/interspeech_2024/kurihara24_interspeech.pdf](https://www.isca-archive.org/interspeech_2024/kurihara24_interspeech.pdf)
- **要旨**: T5-based TJ-G2P と BERT-based BAS (Accent Sandhi module) の dual Transformer architecture を提案。
- **中核的知見**:
  - TJ-G2P: 504,545 PPL sentences + 4,000 human-labeled JSUT PPL sentences で学習
  - BAS: `tohoku-nlp/bert-base-japanese-char-v2` を 8,369 compound words + 246 human-labeled で fine-tune
  - 助数詞語 CER: 16.24% (OpenJTalk) → 15.43% (TJ-G2P) → **1.42% (TJ-G2P + BAS)**
  - 複合語アクセント連続変異 CER: 15.21% → **6.04%**
- **我々のプロジェクトへの示唆**: **架構の直接的な青写真**。ModernBERT を TJ-G2P と BAS の両ロールで使う設計に転用。

### A.3 Hida et al., "Polyphone disambiguation and accent prediction using pre-trained language models in Japanese TTS front-end" (ICASSP 2022)

- **URL**: [arxiv 2201.09427](https://arxiv.org/abs/2201.09427) / [HTML版](https://ar5iv.labs.arxiv.org/html/2201.09427) / [PDF](https://arxiv.org/pdf/2201.09427)
- **要旨**: BiLSTM+CRF に BERT暗黙embedding + 明示的形態素素性を注入し、多音字 / APBP / ANPP の3タスクを同時解く。
- **中核的知見**:
  - 多音字精度 94.34%
  - APBP F1 96.30
  - モーラアクセント精度 96.66% (in-house), 97.33% (JSUT)
  - Sentence-exact 58.68%
  - **主観MOS 3.67 ± 0.07 (対 オラクル記号 3.69 ± 0.07)** — near-oracle
- **我々のプロジェクトへの示唆**: **マルチタスクヘッド設計の実証根拠**。ModernBERT の上に (G2P + APBP + ANPP + polyphone) の4ヘッドを載せる根拠。

### A.4 Shirahata & Yamamoto (LY Corp), "CC-G2PnP" (ICASSP 2026, arxiv 2602.17157)

- **URL**: [https://arxiv.org/pdf/2602.17157](https://arxiv.org/pdf/2602.17157)
- **要旨**: CTC decoder を用いて grapheme-phoneme alignment を暗黙学習する streaming G2P。単語境界情報不要。
- **中核的知見**:
  - Japanese のような unsegmented language に適用可能
  - MeCab等の morphological analysis に依存しない
- **我々のプロジェクトへの示唆**: **アライメント補助ロス**として CTC を導入する根拠。ただし主目標はバッチ精度なので、直接採用ではなく補助的な位置付け。

---

## B. アクセント推定・関連 (Tier 2)

### B.1 Hida et al., 続編 (arxiv 2212.08321)

- **URL**: [https://arxiv.org/abs/2212.08321](https://arxiv.org/abs/2212.08321)
- **要旨**: ICASSP 2022 の続き。アクセント推定の改良。
- **注意**: 実装前に一次資料の詳細確認必須。

### B.2 Hida 関連 (arxiv 2204.03067)

- **URL**: [https://ar5iv.labs.arxiv.org/html/2204.03067](https://ar5iv.labs.arxiv.org/html/2204.03067)
- **要旨**: 関連研究。多音字曖昧性・アクセント予測系。

### B.3 Ogura et al., ICASSP 2025 (NICT ASTREC preprint)

- **URL**: [https://ast-astrec.nict.go.jp/release/preprints/preprint_icassp_2025_ogura.pdf](https://ast-astrec.nict.go.jp/release/preprints/preprint_icassp_2025_ogura.pdf)
- **要旨**: アクセント推定の新しい定式化 (NICT研究)
- **注意**: 実装前に一次資料の詳細確認必須。

---

## C. TTS 一般・組み込みG2P (Tier 2)

### C.1 arxiv 2505.17320

- **URL**: [https://arxiv.org/html/2505.17320v1](https://arxiv.org/html/2505.17320v1)
- **要旨**: VITS 系日本語TTS で pyopenjtalk が G2P として使われていることを明示。
- **我々のプロジェクトへの示唆**: OSS TTSエコシステムのG2Pレイヤ現状の裏付け。

### C.2 JVS Corpus 論文 (arxiv 2009.09679)

- **URL**: [https://arxiv.org/abs/2009.09679](https://arxiv.org/abs/2009.09679)
- **要旨**: JVSコーパスの公開論文。話者バリエーション音声コーパス。

---

## D. ModernBERT 系 (Tier 1: base model)

### D.1 sbintuitions/modernbert-ja シリーズ

- **モデルカード**:
  - [modernbert-ja-30m](https://huggingface.co/sbintuitions/modernbert-ja-30m) — 37M params (10M ex-emb), 10 layers, 256/1024
  - [modernbert-ja-70m](https://huggingface.co/sbintuitions/modernbert-ja-70m) — 70M params (31M ex-emb), 13 layers, 384/1536
  - [modernbert-ja-130m](https://huggingface.co/sbintuitions/modernbert-ja-130m) — 132M params (80M ex-emb), 19 layers, 512
  - [modernbert-ja-310m](https://huggingface.co/sbintuitions/modernbert-ja-310m) — 315M params (236M ex-emb), 25 layers, 768/3072
- **共通仕様**:
  - SentencePiece unigram + byte-fallback (sarashina2-13b由来), vocab 102,400
  - 8,192 token 最大シーケンス長
  - global_rope_theta = 160,000, local_rope_theta = 10,000
  - GELU, "1 global + 2 local" 128-token sliding window attention
  - **MIT License** (商用可)
  - Released 2025-02
- **重要な公式注意事項** (30M cardより):
  > "token boundaries often do not align with the morpheme boundaries, resulting in poor performance in token classification tasks such as named entity recognition and span extraction"

### D.2 llm-jp/llm-jp-modernbert-base

- **論文**: [arxiv 2504.15544](https://arxiv.org/pdf/2504.15544)
- **HFカード**: [llm-jp/llm-jp-modernbert-base](https://huggingface.co/llm-jp/llm-jp-modernbert-base)
- **要旨**: llm-jp-corpus v4 (~0.69Tトークン) で ModernBERT-base アーキテクチャで学習。stage-1 max_seq_len=1024 → stage-2 8,192。
- **示唆**: sbintuitionsとは別系統のトークナイザー。head-to-head比較の必須対照。

### D.3 tohoku-nlp/bert-base-japanese-char-v2

- **HFカード**: [tohoku-nlp/bert-base-japanese-char-v2](https://huggingface.co/tohoku-nlp/bert-base-japanese-char-v2)
- **要旨**: 文字レベル日本語BERT。**NHK Kurihara 2024 の BAS モジュールで採用**。
- **示唆**: ModernBERTトークナイザーの token classification 弱点を回避する対照モデル。

---

## E. Awesome日本語NLPリソース (Tier 3: index)

### E.1 taishi-i/awesome-japanese-nlp-resources

- **URL**: [https://github.com/taishi-i/awesome-japanese-nlp-resources/blob/main/docs/huggingface.md](https://github.com/taishi-i/awesome-japanese-nlp-resources/blob/main/docs/huggingface.md)
- **要旨**: 日本語NLPの主要リソースを網羅したインデックス。追加のG2P関連モデルを発掘するのに便利。

### E.2 NVIDIA NeMo G2P Documentation

- **URL**: [https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/tts/g2p.html](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/tts/g2p.html)
- **要旨**: G2P学習パイプラインの一般的な設計 (ByT5, T5) のリファレンス。

---

## F. 参考OSS実装 (Tier 1: code reference)

| リポジトリ | 内容 | 用途 |
|---|---|---|
| [r9y9/pyopenjtalk](https://github.com/r9y9/pyopenjtalk) | Open JTalk Python wrapper | ベースライン、辞書lookup |
| [tsukumijima/pyopenjtalk-plus](https://github.com/tsukumijima/pyopenjtalk-plus) | 拡張版辞書 | 学習データ生成 |
| [o24s/haqumei](https://github.com/o24s/haqumei) | OSS SOTA相当のG2P | ベースライン比較の主要対象 |
| [hexgrad/misaki](https://github.com/hexgrad/misaki) | 多言語G2Pエンジン (Kokoro TTS用) | 設計参考 |
| [litagin02/Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2) | Style-Bert-VITS2 TTS | G2Pのドロップイン先 |
| [sarulab-speech/jsut-label](https://github.com/sarulab-speech/jsut-label) | JSUT音素ラベル | 評価用 |
| [prj-beatrice/jsut-label](https://github.com/prj-beatrice/jsut-label) | haqumei用JSUT音素ラベル | 評価用 (haqumei互換) |
| [mmorise/rohan4600](https://github.com/mmorise/rohan4600) | ROHANコーパス | 評価用 |
| [espnet/espnet](https://github.com/espnet/espnet/blob/master/espnet2/text/phoneme_tokenizer.py) | 音素トークナイザ | リファレンス実装 |

---

## G. Refuted (反証済み) の主張 — 参考として明記

deep-research の 3票敵対的検証で棄却 (0-3 refute) された主張。**参照禁止**:

- **却下された主張**: "Kurihara & Sano (Interspeech 2024) が Japanese G2P を pure neural では unsolvable と framing しており、dictionary integration が必要と明示的に主張。特に proper nouns / numerals / counter words を `[ ]` でタグ付けする hybrid architecture を提案"
- **却下ソース候補**: [https://www.isca-archive.org/interspeech_2024/kurihara24_interspeech.pdf](https://www.isca-archive.org/interspeech_2024/kurihara24_interspeech.pdf)
- **理由**: 該当論文は dual Transformer (T5 + BERT) のアーキテクチャを提案しているが、"unsolvable without dictionary" とは主張しておらず、`[ ]` タグの記述も本文中に無い。この主張は誤解に基づく。
- **教訓**: NHK 論文を引用する際は、実際の architecture description に忠実に基づくこと。

---

## H. 未解決だが調査が有益な追加論文候補

以下は deep-research の open questions で言及された未検証領域。実装前に追加調査推奨:

1. **fine-tuned encoder-only ModernBERT の G2P benchmark**: 公開結果を継続監視。
2. **G2P用の tokenizer 選択の head-to-head比較**: 未公開。パイロット実験で自ら測定必要。
3. **haqumei / Misaki / pyopenjtalk の per-カテゴリ (固有名詞, 外来語, 助数詞) 誤り率**: 公開されていない。ロス重み設計のため自作評価が必要。
4. **商用配布可能なG2P学習コーパスのライセンス統合**: 公開ガイドライン無し。実装前に法務相談推奨。

---

## I. 引用スタイル (プロジェクト内)

論文引用時のフォーマット:

- **短形式**: `[Kurihara 2024]`, `[Hida 2022]`, `[Koriyama 2026]`
- **URL付き**: 初出時のみ URL を bracket 内に付す
- **数値引用**: 必ず論文の具体 Table 番号または数値を明示

例:
> NHKの TJ-G2P + BAS は助数詞語 CER を 16.24% (OpenJTalk) から 1.42% に削減した ([Kurihara 2024, Table 4](https://www.isca-archive.org/interspeech_2024/kurihara24_interspeech.pdf))。

---

## J. deep-research のメタデータ

- 調査日: 2026-07-03
- 調査手法: fan-out web searches (6 angles, 30+ initial results) → source fetch (27 URLs) → claim extraction (126 claims) → 3-vote adversarial verification (25 claims verified) → synthesis (8 confirmed findings)
- Refuted: 1 claim (Kurihara "unsolvable" 主張)
- Confidence: 全 findings が "high" confidence
- 制限: LLM ベンチマーク数値は API 依存で流動的、fine-tuned encoder ModernBERT G2Pの公開先行例は現時点で存在しない
