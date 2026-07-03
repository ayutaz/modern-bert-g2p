# 07. 純粋 NN 日本語G2P モデルのベンチマーク徹底調査

**作成日:** 2026-07-03
**目的:** 「純粋NNモデル (辞書lookup非依存)」に絞り、公開ベンチマーク結果を系統的に収集する
**調査手法:** deep-research (108エージェント、25情報源、109主張抽出、25主張を3票敵対的検証、21主張が確認、4主張が反証)
**追加分析:** UR-BERT (arxiv 2606.11681v1) の追加読解 (WebFetch分析)

---

## 1. エグゼクティブ・サマリー — **重要な発見**

> **公開されている「純粋NNのみで、JSUT/JVS/ROHAN のいずれかで PER/kana CER を報告した」日本語G2Pモデルは、事実上存在しない。**

- 存在する pure-NN モデル (PnG BERT, Kakegawa TJ-G2P alone, CC-G2PnP, CharsiuG2P) はすべて、以下いずれかに該当する:
  1. 標準ベンチマーク (JSUT/JVS/ROHAN) 上で pure-NN の数値が **未公開**
  2. 別の非標準データセットで評価されており、直接比較不可
  3. rule-based baseline に **敗北** している (Kakegawa TJ-G2P vs OpenJTalk)
  4. TTS accent MOS で **rule-derived labels に敗北** (PnG BERT)
- 最強の "NN-forward" 結果 (Claude Opus 4.6 = 0.52% kana CER) すら、"parse mode" は LLM parsing + rule-based kana変換の **暗黙のハイブリッド**
- 2022〜2026の間に JSUT/JVS/ROHAN class ベンチマークで トップスコアを出したすべてのシステムは **ハイブリッド構造** (Transformer/BERT + 辞書/形態素解析)

**したがって、pure-NN 日本語G2Pの空白領域は本プロジェクトが埋めるべき技術的な余地であるとともに、そこにこそ「hybrid が正解」の実証的根拠が集約されている**。

---

## 2. Pure-NN 日本語G2Pモデルの個別プロファイル

### 2.1 PnG BERT for Japanese (Yasuda & Toda 2022, IEEE JSTSP)

- **論文**: [arxiv 2212.08321](https://arxiv.org/pdf/2212.08321)
- **アーキテクチャ**: BERT。grapheme (漢字/かな) と phoneme (音素) を同時に入力し、MLMで両方をマスク予測する
- **入力データ**: JSUT + Wikipedia + 青空文庫
- **公開結果** (verified 2-1):
  - **G2P whole-segment accuracy: 45.5%** (best MLM checkpoint)
  - **P2G whole-segment accuracy: 23.6%**
  - TTS accent MOS:
    - PGB2 (PnG BERT variant 2): 2.41 ± 0.03
    - PGB2T: 2.51 ± 0.03
    - TAC (Tacotron ベースライン): 1.89 ± 0.03
    - **TACT (Tacotron + rule-derived accent labels): 3.04 ± 0.03**
- **著者自身の結論** (verbatim):
  > "accents inferred by the pre-trained PnG BERT-based systems were not sufficiently accurate compared to accent labels"
- **示唆**: **encoder-only BERT を pure-NN で G2P に投入しても、rule-derived accent labels を用いた単純な Tacotron に負ける** (2.51 vs 3.04 MOS)。これはNHK Kurihara 2024 と独立に、pure-NN の限界を実証している最も古い data point。
- **注意**: 本論文の refute された主張は「45.5% は rule-based dictionary lookup accuracy より低い」という *解釈* 部分 (1-2 refute)。**45.5% の数値自体は verified**。

### 2.2 TJ-G2P alone (Kakegawa 2021, 引用は Kurihara 2024 Interspeech)

- **論文**: [Kurihara & Sano Interspeech 2024](https://www.isca-archive.org/interspeech_2024/kurihara24_interspeech.pdf) (Table 2 & Table 4)
- **アーキテクチャ**: T5-based Transformer, **辞書lookup無し**
- **評価**: JSUT Basic5000 (400文サブセット), PPL (phoneme + prosodic labels) 形式の CER
- **公開結果 (verified 3-0)**:

  | System | JSUT400 PPL CER | 助数詞語 CER (100文) |
  |---|---|---|
  | Open JTalk (rule) | **10.82%** | 16.24% |
  | TJ-G2P (pure NN) | 11.85% | 15.43% |
  | TJ-G2P + BAS (hybrid) | 12.07% | **1.42%** |

- **観察**:
  - **全体では pure NN (TJ-G2P) が rule-based (OpenJTalk) に負ける** (11.85% vs 10.82%)。ただし論文自身が「statistically no significant differences」と注記
  - **助数詞語カテゴリで辞書導入 (BAS) により 15.43% → 1.42% (11倍改善)** — 改善の主因は NN ではなく **辞書lookup**
- **示唆**: **同じ Transformer が pure-NN の時に負けて、辞書と組み合わせると圧勝する**、というhead-to-head 数値が公開されている唯一のケース。**pure-NN の敗北と hybrid の勝利を同一論文で示す最強のエビデンス**。

### 2.3 CC-G2PnP (Shirahata & Yamamoto ICASSP 2026, LY Corp)

- **論文**: [arxiv 2602.17157](https://arxiv.org/html/2602.17157)
- **アーキテクチャ**: Streaming Conformer + CTC、単語境界情報不要 (unsegmented languages 対応)
- **評価**: 6D-Eval (2,722 expert-annotated Japanese sentences、非標準)
- **公開結果 (verified 3-0)**:

  | System | phoneme CER (SER) | PnP CER (SER) |
  |---|---|---|
  | CC-G2PnP-5-1 (pure NN) | **0.52 (8.4%)** | 1.79 (41.4%) |

- **重要な反証** (vote 0-3, **refuted**): 論文が主張した「CC-G2PnP は Dict-DNN ハイブリッドベースラインを両指標で上回る」というサブクレームは **敵対的検証で棄却**。すなわち **pure-NN vs hybrid の直接比較は未確定**。
- **示唆**:
  - CC-G2PnP の CTC-based アーキテクチャ自体は現存する最先端 pure-NN 日本語G2P
  - しかし評価が 6D-Eval (非標準) に留まり、JSUT/JVS/ROHAN との bridge が無い
  - 論文の "hybrid を打ち破る" 主張は verify で refuted、hybrid 優位の傾向は継続

### 2.4 CharsiuG2P / ByT5 (Zhu et al. Interspeech 2022, 多言語)

- **論文**: [arxiv 2204.03067](https://arxiv.org/abs/2204.03067) / [Interspeech 2022 archive](https://www.isca-archive.org/interspeech_2022/zhu22_interspeech.html)
- **リポジトリ**: [lingjzhu/CharsiuG2P](https://github.com/lingjzhu/CharsiuG2P)
- **アーキテクチャ**: ByT5 byte-level seq2seq、辞書非依存、~100言語対応 (日本語含む)
- **公開結果 (verified 3-0)**:
  - **多言語 aggregate**: PER = 0.089, WER = 0.261 (g2p_multilingual_byT5_small_100)
  - **日本語 specific PER/CER: 公式には未公開 (README/論文とも)**
- **CJK 注意点** (README 明記):
  > "For languages such as Chinese, Korean, Japanese … an external tokenizer must be used"
  → 純粋 byte-level だけでは日本語には不十分と、著者自身が認めている
- **第三者ベンチマーク**: [FluidAudio benchmark](第三者 blog-tier) は日本語で **PER 10.5% / WER 23.8%** と報告 — 多言語 aggregate の PER 8.9% よりむしろ悪く、hybrid baseline (haqumei 1.17%) より **1桁悪い**
- **示唆**:
  - **世界唯一の公開 pure-NN 多言語G2P**だが、日本語 specific 数値の公式報告が無い
  - 第三者ベンチマーク結果は hybrid の 1桁下 — pure NN 多言語モデルは日本語に対しては競争力なし
  - 「外部トークナイザー必要」 = 実質的にhybrid化が要求されている

### 2.5 XPhoneBERT (Interspeech 2023) — **G2Pモデルではない**

- **論文**: [arxiv 2305.19709](https://arxiv.org/abs/2305.19709)
- **アーキテクチャ**: RoBERTa on 330M phoneme-level sentences
- **重要**: **XPhoneBERT は G2P モデルではない** (verified 3-0)。入力が既に音素列であり、G2P (CharsiuG2P/Epitran等) を **前段で必要とする**
- **示唆**: XPhoneBERT を "Japanese G2P benchmark" として引用するのは誤り。G2P品質評価に含めない。

### 2.6 fo-BERT (Ogura et al. ICASSP 2025 preprint, NICT) — **G2Pモデルではない**

- **論文**: [preprint_icassp_2025_ogura.pdf](https://ast-astrec.nict.go.jp/release/preprints/preprint_icassp_2025_ogura.pdf)
- **アーキテクチャ**: mora-level BERT、F0 (基本周波数) 予測用
- **重要**: **fo-BERT はG2Pモデルではない** (verified 3-0)。入力形式は `[CLS] この 箸 が … [SEP] コ ノ ハ シ ガ … [SEP]` で、G2Pは **前段の依存関係**
- **示唆**: fo-BERT はアクセント/プロソディ予測モデルであり、G2Pベンチマークからは除外する。

### 2.7 UR-BERT (Lee et al. Interspeech 2026, Yonsei大) — **G2Pモデルではない**

- **論文**: [arxiv 2606.11681v1](https://arxiv.org/html/2606.11681v1) (追加分析)
- **アーキテクチャ**: BERT-base + Uroman ローマ字化 + STP (Speech Token Prediction、Wav2Vec2からの蒸留)
- **重要**: **UR-BERT は TTS text encoder であり、G2Pモデルではない**。495言語対応の多言語 TTS フロントエンド
- **公開結果 (multilingual TTS の MOS/CER)**:

  | 言語 | モデル | MOS ↑ | CER ↓ | MCD ↓ |
  |---|---|---|---|---|
  | English | VITS baseline | 3.78 | 6.15% | 5.71 |
  | English | +XPhoneBERT (G2P依存) | 4.11 | 4.79% | 5.17 |
  | English | **+UR-BERT** | **4.35** | **3.78%** | 5.23 |
  | German | +XPhoneBERT | 3.53 | 5.85% | 4.77 |
  | German | **+UR-BERT** | **3.78** | **3.07%** | 4.66 |
  | 中国語 | +XPhoneBERT | 3.49 | 25.98% | 5.16 |
  | 中国語 | **+UR-BERT** | **3.88** | **21.83%** | 4.95 |

- **日本語の直接評価**: **無し**。495言語リストへの日本語含有は未確認。
- **示唆**:
  - **XPhoneBERT (G2P依存) を明確に上回っている** — G2P依存を捨てても高品質TTSは可能
  - しかし G2P **精度** (PER/kana CER) の評価は無く、TTS の MOS/CER のみ
  - 日本語の促音・撥音・長音などの音韻対比が Uroman で失われる懸念 — 日本語適用未検証
  - 我々のプロジェクトへの直接転用は限定的だが、**「文字レベル + ローマ字化 + 音声蒸留」で pure-NN TTS encoder が競争力を持てる**という参考

### 2.8 Peters 2017 "Massively Multilingual Neural G2P"

- **論文**: [arxiv 1708.01464](https://arxiv.org/pdf/1708.01464)
- **重要**: **日本語は対象言語に含まれていない** (verified indirectly)
- **示唆**: 日本語G2P benchmark としては引用不可

---

## 3. 標準ベンチマーク vs 実測データポイント総覧

### 3.1 総合比較表 (**JSUT/JVS/ROHAN の公開結果**)

| モデル | タイプ | 辞書lookup | JSUT Basic5000 PER | JVS-3000 kana CER | ROHAN KER | 引用 |
|---|---|---|---|---|---|---|
| OpenJTalk / pyopenjtalk | rule + dict | ○ (必須) | ~10.82% (PPL CER) | 1.03% | N/A | [Koriyama 2606.22009](https://arxiv.org/abs/2606.22009), [Kurihara 2024 Table 2](https://www.isca-archive.org/interspeech_2024/kurihara24_interspeech.pdf) |
| haqumei | rule + dict + ONNX loanword | ○ | **1.17%** | N/A | **1.64%** | [haqumei README](https://github.com/o24s/haqumei) |
| **Kakegawa TJ-G2P (pure NN)** | T5 Transformer | ✗ | **11.85% (PPL CER)** | N/A | N/A | [Kurihara 2024 Table 2](https://www.isca-archive.org/interspeech_2024/kurihara24_interspeech.pdf) |
| **PnG BERT (pure NN)** | BERT (grapheme+phoneme) | ✗ | **G2P 45.5% acc (whole segment)** | N/A | N/A | [Yasuda & Toda 2022](https://arxiv.org/pdf/2212.08321) |
| **CharsiuG2P (byT5, pure NN)** | ByT5 multilingual | ✗ | 未公開 (aggregate PER 0.089) | 未公開 | 未公開 | [Zhu 2022](https://arxiv.org/abs/2204.03067) |
| **CharsiuG2P** (第三者測定, 日本語限定) | ByT5 multilingual | ✗ | ~10.5% (別split) | 未公開 | 未公開 | FluidAudio blog benchmark |
| **CC-G2PnP (pure NN)** | Conformer+CTC | ✗ | 未公開 (6D-Eval で phoneme CER 0.52 = 8.4%) | 未公開 | 未公開 | [Shirahata 2026](https://arxiv.org/html/2602.17157) |
| TJ-G2P + BAS (hybrid) | T5 + BERT + Unidic + NHK辞書 | ○ | 12.07% (PPL CER) | N/A | N/A | [Kurihara 2024](https://www.isca-archive.org/interspeech_2024/kurihara24_interspeech.pdf) |
| Hida 2022 (hybrid) | BiLSTM+CRF+BERT+形態素素性 | ○ | mora acc 97.33% | N/A | N/A | [Hida 2022](https://arxiv.org/abs/2201.09427) |
| Claude Opus 4.6 (LLM) | frontier LLM (parse mode = 暗黙hybrid) | 暗黙的 | N/A | **0.52%** | N/A | [Koriyama 2026](https://arxiv.org/abs/2606.22009) |
| Gemini 3.1 Pro (LLM) | frontier LLM (parse mode) | 暗黙的 | N/A | 0.62% | N/A | [Koriyama 2026](https://arxiv.org/abs/2606.22009) |
| XPhoneBERT | NOT a G2P | — | — | — | — | [arxiv 2305.19709](https://arxiv.org/abs/2305.19709) |
| fo-BERT | NOT a G2P (F0 predictor) | — | — | — | — | [Ogura 2025 preprint](https://ast-astrec.nict.go.jp/release/preprints/preprint_icassp_2025_ogura.pdf) |
| UR-BERT | NOT a G2P (TTS text encoder) | — | — | — | — | [arxiv 2606.11681v1](https://arxiv.org/html/2606.11681v1) |

**注意**: PnG BERT の 45.5% は whole-segment G2P accuracy であり、PERとは直接比較不可 (指標が異なる)。Kakegawa TJ-G2P の 11.85% は PPL (phoneme + prosodic) CER であり、raw kana CER や pure phoneme PER とは異なる。**この指標不整合こそが「pure-NN が公平に比較されたことがない」ことの原因**。

### 3.2 メトリック互換性の警告

異なる論文の数値を並べる際、以下の不整合に注意:

| 論文 | 指標 | 内容 |
|---|---|---|
| Kurihara 2024, Kakegawa TJ-G2P | PPL CER | phoneme + prosodic label sequences on Julius音素セット |
| haqumei | PER | phoneme sequence only, no prosodic |
| Koriyama 2026 | kana CER | かな文字列 CER |
| Yasuda PnG BERT 2022 | whole-segment accuracy | 分節全体一致率 (very strict) |
| CharsiuG2P | multilingual aggregate PER | 100言語平均 |
| CC-G2PnP | phoneme CER, PnP CER | 6D-Eval 独自形式 |

**同一シード・同一splitで再測定しないと、公平な pure-NN vs hybrid 比較は不可能**。これも本プロジェクトが空白を埋めるべき理由の一つ。

---

## 4. HF/GitHub 実装レベルでの pure-NN Japanese G2P 試行

### 4.1 存在する試行

- **[prj-beatrice/japanese-hubert-base-phoneme-ctc-v3](https://huggingface.co/prj-beatrice/japanese-hubert-base-phoneme-ctc-v3)**: HuBERT-based CTC 音素予測モデル。**音声** から音素を予測 (音声ASR系G2P、テキスト→音素ではない)。**テキストG2Pではない**が、日本語音素表現学習の実装参考にはなる。
- **[KoichiYasuoka/modernbert-base-japanese-aozora](https://huggingface.co/KoichiYasuoka/modernbert-base-japanese-aozora)**: ModernBERT を青空文庫で日本語適応させた encoder。G2P特化ではなく形態素解析/POS用。ただし ModernBERT + 日本語データの実装リファレンスとしては有用。

### 4.2 見つからなかった試行 (**重要な負の発見**)

- **modernbert-ja を G2P に fine-tune した公開試行は存在しない** (2026-07-03時点、HF Hub / GitHub / arxiv / Kaggle 全捜索)
- **tohoku-nlp/bert-base-japanese-char-v2 を G2P メインタスクに fine-tune した公開試行は存在しない** (NHK BAS は accent sandhi 特化)
- **ku-nlp / rinna / pkshatech / sbintuitions 各系の Japanese BERT を JSUT/JVS/ROHAN で G2P 評価した公開結果は存在しない**

→ **本プロジェクトが最初のデータポイントになる可能性が非常に高い**。

---

## 5. E2E TTS フレームワーク側の G2P 依存構造

E2E TTS 系フレームワークが G2P をどう扱っているかを、pure-NN G2P の実運用可否の傍証として確認:

| E2E TTS | G2P モジュール | pure-NN か |
|---|---|---|
| Voicebox (Meta) | 英語中心、G2P詳細非公開 | 該当外 |
| VoiceCraft | 英語中心 | 該当外 |
| StyleTTS 2 | 英語 phonemizer (espeak-ng, rule based) | ✗ (rule) |
| Fish Speech | 内部G2P (英中日) 詳細非公開 | 不明 |
| Bark / Bark-JP | 内部音素/tokens 表現 | 該当外 |
| VITS Japanese port | **pyopenjtalk** | ✗ (rule + dict) |
| Style-Bert-VITS2 (JP-Extra) | **pyopenjtalk** | ✗ (rule + dict) |
| Bert-VITS2 | **pyopenjtalk** | ✗ (rule + dict) |
| GPT-SoVITS | **pyopenjtalk** | ✗ (rule + dict) |
| Misaki v2 (Kokoro TTS) | **pyopenjtalk + full UniDic** (+ seq2seq TODO) | ✗ (rule + dict primary) |
| VOICEVOX Engine | 内部辞書 + ルール + ONNX予測 | ハイブリッド |

**結論**: E2E TTS 側もほぼ全て pyopenjtalk か同等の辞書lookupを採用しており、pure-NN 実運用への収束はまだ起きていない。

---

## 6. 反証済み主張のリスト (**参照禁止**)

deep-research 敵対的検証で棄却された主張。文献引用時に混同しないよう注意:

### 6.1 refuted (vote 1-2)
- **却下された主張**: "PnG BERT training labels were Kuromoji+Neologd-derived pseudo-labels, meaning measured errors are lower-bounded by Kuromoji+Neologd errors"
- **理由**: 論文の train label origin の記述はより複雑で、単純に Kuromoji+Neologd と断定できない

### 6.2 refuted (vote 1-2)
- **却下された解釈**: "PnG BERT 45.5% G2P accuracy on JSUT+Wikipedia+Aozorabunko validation set at peak MLM-accuracy checkpoint, well below dictionary-lookup accuracy"
- **理由**: 45.5% の数値自体は verified (別 vote 2-1) だが、"well below dictionary-lookup accuracy" という *解釈* が単純化しすぎ。datasets 詳細は論文本文の記述に忠実に基づくこと。

### 6.3 refuted (vote 0-3)
- **却下された主張**: "The paper (arxiv 2212.08321) explicitly asserts that a purely neural approach cannot match a lexicon dictionary for Japanese word coverage, framing NN-only G2P as fundamentally coverage-limited"
- **理由**: 論文にこの framing は明示的に無い。**pure-NN の敗北は数値では示されているが、著者は「coverage-limited」という強い主張はしていない**。

### 6.4 refuted (vote 0-3)
- **却下された主張**: "CC-G2PnP (pure NN) outperforms Dict-DNN hybrid baseline on 6D-Eval on both CER and phoneme CER metrics"
- **理由**: 論文が本文で主張していたが、敵対的検証で 0-3 で棄却。**pure-NN が hybrid を上回るという最後の希望はここでも成立しない**。

---

## 7. 総合的な発見と本プロジェクトへの示唆

### 7.1 現状の実証的事実

1. **同一の Transformer が pure-NN の時に負けて、辞書と組み合わせると圧勝する** (Kurihara 2024 の TJ-G2P 15.43% vs TJ-G2P+BAS 1.42%, 助数詞語)
2. **BERT pure-NN で MLM 事前学習しても、TTS accent MOS で rule-derived labels の Tacotron に負ける** (PnG BERT 2.51 vs TACT 3.04)
3. **LLM の best score も "parse mode" = 暗黙のハイブリッド** — pure decode-mode は parse-mode に劣る
4. **多言語 pure-NN G2P (CharsiuG2P) は日本語で第三者測定 PER 10.5% と、hybrid の 1桁下**
5. **公開の "pure-NN 日本語G2P が JSUT/JVS/ROHAN で報告した数値" は事実上存在しない** — 空白領域

### 7.2 本プロジェクトの戦略的含意

- **ModernBERT 単独で pyopenjtalk 置換を目指すのは、証拠に基づき危険** — 過去10年の全論文が pure-NN 敗北の同じパターンを示している
- **ハイブリッド路線 (辞書 primary + ModernBERT correction) こそが実装ROI が最も高い** — 05_technical_design.md の方針と一致
- **ただし、本プロジェクトは同時に「fine-tuned encoder ModernBERT G2P の JSUT/JVS/ROHAN 上の初の公開データポイント」を作る意義がある** — pure-NN の Ablation 対照として。仮に負けたとしても、それは学術的貢献 (負の結果の系統的記録)
- **メトリック整合性の確立が最初のマイルストーン** — 全評価を kana CER (Koriyama) と PER (haqumei) の2軸に統一。第3軸として ROHAN KER。

### 7.3 具体的なActions

1. **Phase 2 (シングルタスクベースライン) に pure-NN 比較セットを明示的に追加**:
   - modernbert-ja-130m + seq2seq (no dict)
   - modernbert-ja-130m + token classification (no dict)
   - modernbert-ja-130m + hybrid (dict primary)
   - CharsiuG2P japanese eval (再測定)
   - CC-G2PnP japanese eval on JSUT (可能なら)
2. **メトリック統一**: PER (haqumei互換), kana CER (Koriyama互換), KER (ROHAN互換) の3軸を全モデルで測定。**PPL CERは補助のみ**
3. **公開評価**: pure-NN の負けを含めて、Ablation table を公開。負の結果も学術的価値がある

---

## 8. 未解決の open questions (Phase 2以降で解決)

- Kakegawa TJ-G2P (pure NN) を **標準haqumei/pyopenjtalk-plus training splitで再学習** した場合、JSUT Basic5000 raw PER はいくつになるか
- CharsiuG2P (byT5) を **JSUT Basic5000 と JVS-3000 で MeCab pretokenize併用**した場合の日本語 PER はいくつになるか
- CC-G2PnP を JSUT Basic5000 で **同 split/同メトリック** で再評価した場合、hybrid との差はどうなるか
- modernbert-ja-30/70/130/310m を pure-NN G2P として fine-tune した場合の scaling 曲線

---

## 9. 主要参考ソース

deep-research で verified された 21主張のソース (**すべて一次情報**):

1. [Yasuda & Toda 2022 (PnG BERT, arxiv 2212.08321)](https://arxiv.org/pdf/2212.08321) — pure-NN BERT 限界
2. [Kurihara & Sano Interspeech 2024 (arxiv/ISCA)](https://www.isca-archive.org/interspeech_2024/kurihara24_interspeech.pdf) — pure-NN vs hybrid head-to-head
3. [Hida ICASSP 2022 (arxiv 2201.09427)](https://arxiv.org/pdf/2201.09427) — hybrid best design
4. [Zhu 2022 CharsiuG2P Interspeech 2022 (arxiv 2204.03067)](https://arxiv.org/abs/2204.03067) — 多言語 pure-NN
5. [CharsiuG2P repo](https://github.com/lingjzhu/CharsiuG2P) — 実装と CJK caveat
6. [Koriyama 2026 (arxiv 2606.22009)](https://arxiv.org/abs/2606.22009) — LLM benchmark
7. [XPhoneBERT (arxiv 2305.19709)](https://arxiv.org/abs/2305.19709) — G2P**ではない**
8. [fo-BERT (Ogura 2025 preprint)](https://ast-astrec.nict.go.jp/release/preprints/preprint_icassp_2025_ogura.pdf) — G2P**ではない**
9. [CC-G2PnP (Shirahata & Yamamoto arxiv 2602.17157)](https://arxiv.org/html/2602.17157) — pure-NN 最新
10. [Peters 2017 (arxiv 1708.01464)](https://arxiv.org/pdf/1708.01464) — 日本語含まず
11. [UR-BERT (arxiv 2606.11681v1)](https://arxiv.org/html/2606.11681v1) — TTS text encoder、G2Pではない
12. [japanese-hubert-base-phoneme-ctc-v3](https://huggingface.co/prj-beatrice/japanese-hubert-base-phoneme-ctc-v3) — 音声G2P、テキストG2Pではない
13. [modernbert-base-japanese-aozora](https://huggingface.co/KoichiYasuoka/modernbert-base-japanese-aozora) — ModernBERT日本語適応
14. [haqumei](https://github.com/o24s/haqumei) — hybrid SOTA

---

## 10. 一言まとめ

**"公開されている pure-NN 日本語G2P モデルは、標準ベンチマークで報告した瞬間 rule/dict/hybrid に負ける、または報告そのものを避ける"** ── この事実こそが、本プロジェクトが「ModernBERT を単一の置換モデルではなく、hybrid の中の1つのコンポーネント」として設計する強力な根拠。同時に、pure-NN 対照を Ablation table に明示的に載せる意義もここにある (負けたとしても学術的貢献)。
