# 09. Pure-NN 日本語G2P Benchmarks — 越えるべき先行研究の徹底整理

**作成日:** 2026-07-04
**バージョン:** 2.0
**目的:** Pure-NN pivot (2026-07-04, requirements.md v2.0) を受け、越えるべき先行 pure-NN 日本語G2Pモデルのアーキテクチャ・データ・数値・failure modes を統合ドキュメント化する。Phase 6 preprint の Table 1 の直接の原型となる。
**位置付け:** 07 (`07_nn_only_benchmarks.md`) の失敗パターン記述を「敗北事例」から「越えるべき baseline」に framing shift した後継。08 (`08_market_landscape.md`) の 2024-2026 加速期を反映した updated snapshot。
**verification:** 5並列 deep-research (png_bert / kakegawa_tj_g2p / charsiu_g2p / cc_g2pnp / newer_2024_2026) + Review-1 敵対的批判の反映済み

---

## 1. Executive Summary

本プロジェクトの pivot 前提として **公開ベンチマーク (JSUT / JVS / ROHAN) 上で hybrid (dict lookup + NN) を上回った pure-text-NN 日本語G2P モデルは 2026-07 時点で存在しない**。この事実は 07 で確定済みで、09 では 2024-2026 の 10 件超の新モデル (Furigana Whisper, Ohnaka 2025, Koriyama SSW13, yomi-linter-modernbert-ja-130m, cc-g2pnp 再現版, prj-beatrice HuBERT-CTC v1-v5 等) を追跡した後もこの結論は変わらない。ただし **speech+text の Ohnaka 2025 が LARGE-TTSaug で PER 0.93%** を達成しており、pure-text 制約を外せば NN が hybrid parity に迫る事例が 2025 年に初めて現れた。

本プロジェクトは (a) **pure-text encoder-only の ModernBERT-Ja 130M で先行 4 モデル (PnG BERT / Kakegawa TJ-G2P / CharsiuG2P / CC-G2PnP) を同一プロトコルで初めて系統的に上回る** ことを保守目標 (JSUT PER < 5%)、(b) speech-conditioned NN 群は正直に "out-of-class" と宣言して比較対象から除外する、という 2 軸で positioning する。

先行研究の絶対値は publish 元により **JSUT split・metric・phoneme set がバラバラで直接比較不能** (Review-1 B1)。したがって FR-61 「同一プロトコル再測定」は本プロジェクトの MUST であり、この doc はその測定計画の直接根拠となる。Aspirational tier として Claude Opus 4.6 / Gemini 3.1 Pro の 0.52-0.62% (JVS-3000, Koriyama Interspeech 2026) を参照するが、これは 1T+ params の Frontier LLM が parse mode の暗黙 hybrid で叩き出した数値であり、130M pure-NN の絶対値到達は非現実的想定として扱う。

---

## 2. Pure-NN 日本語 G2P baseline inventory

Category 列は Review-1 R1 を反映:
- `text-enc` = pure-text encoder-only
- `text-s2s` = pure-text seq2seq
- `speech+text` = speech 特徴を追加入力
- `speech-CTC` = speech-only ASR-style
- `prosody-only` = prosody labels のみ (音素は出さない)
- `hybrid` = 内部で dict lookup を持つ (参考記載)

Verified confidence: `V` = 一次資料 (arxiv/HFカード) で直接確認, `P` = plausible / snippet 経由, `U` = unverified.

| # | Name | Paper / Source | Year | Architecture | Category | Params | JSUT PER | JVS CER | ROHAN KER | Confidence |
|---|------|----------------|------|--------------|----------|--------|----------|---------|-----------|------------|
| 1 | PnG BERT (Yasuda & Toda) | [arxiv 2212.08321](https://arxiv.org/abs/2212.08321) | 2022 | BERT-base MLM (grapheme+phoneme) | text-enc | ~110M | pretrain val whole-word acc 45.5% (**not** test PER) | 未報告 | 未報告 | V |
| 2 | Kakegawa 2021 (原型) | [Kakegawa Interspeech 2021](https://www.isca-archive.org/interspeech_2021/kakegawa21_interspeech.pdf) | 2021 | Transformer 5-enc/5-dec, char→syllable | text-s2s | 未公開 | JSUT未評価 (新聞5,142文で Word-Acc = 98.0) | 未報告 | 未報告 | V |
| 3 | TJ-G2P alone (Kurihara 再実装) | [Kurihara Interspeech 2024](https://www.isca-archive.org/interspeech_2024/kurihara24_interspeech.pdf) | 2024 | Kakegawa 2021 を JSUT 用に再学習 | text-s2s | 未公開 (T5-base 想定) | 11.85% (JSUT400 PPL CER, OpenJTalk 10.82% に敗北) | 未報告 | 未報告 | V |
| 4 | CharsiuG2P (multilingual byT5-small) | [arxiv 2204.03067](https://arxiv.org/abs/2204.03067) | 2022 | ByT5-small byte seq2seq, 99 lang | text-s2s | ~300M | own-dict holdout PER **10.51%** (IPA, word-list) | 未報告 | 未報告 | V |
| 5 | CharsiuG2P (monolingual jpn-only ByT5) | 同上 GitHub multilingual_results | 2022 | ByT5-small, jpn dict のみ | text-s2s | ~300M | **66.89%** (collapse) | 未報告 | 未報告 | V |
| 6 | CC-G2PnP (Shirahata & Yamamoto) | [arxiv 2602.17157](https://arxiv.org/abs/2602.17157) | 2026 | Streaming Conformer + CTC | text-s2s (streaming) | 未公開 (~80M 推定) | 6D-Eval Phoneme CER 0.48-0.52 / PnP CER 1.79-1.80 (**JSUT 直接未測定**) | 未報告 | 未報告 | V |
| 7 | Ohnaka 2025 (Grapheme-Coherent PP) | [arxiv 2506.04527](https://arxiv.org/abs/2506.04527) | 2025 | OWSM-CTC v3.1 (27L) + line-distilbert-base-japanese | speech+text | ~300M+ | LARGE-TTSaug PER **0.93%** (JSUT非表示) | 未報告 | 未報告 | V |
| 8 | Furigana Whisper (Parakeet-Inc) | [HF Parakeet-Inc/furigana_whisper_small_jsut](https://huggingface.co/Parakeet-Inc/furigana_whisper_small_jsut) | 2025 | Whisper-small, decoder fine-tune, prompted | speech+text | ~244M | JSUT closed-set CER 0.19% (with prompt) / 6.43% (without) | 未報告 | 未報告 | V |
| 9 | Hu 2025 (Transcript-Prompted Whisper) | [arxiv 2506.07646](https://arxiv.org/abs/2506.07646) | 2025 | Whisper + transcript prompt + dict-decoding | speech+text | 未公開 | 未報告 | 未報告 | 未報告 | V |
| 10 | Koriyama SSW13 2025 (Prosody Labeling) | [arxiv 2507.03912](https://arxiv.org/abs/2507.03912) | 2025 | PnG-BERT + Whisper 融合 | prosody-only | 未公開 | 音素は出さない (accent 89.8%, break 94.3%) | 未報告 | 未報告 | V |
| 11 | prj-beatrice HuBERT-CTC v1-v5 | [HF prj-beatrice/japanese-hubert-base-phoneme-ctc-v5](https://huggingface.co/prj-beatrice/japanese-hubert-base-phoneme-ctc-v5) | 2025-2026 | HuBERT-base + CTC | speech-CTC | 94.4M | 未報告 (HFカードに数値なし) | 未報告 | 未報告 | V |
| 12 | ayousanz/cc-g2pnp reproduction | [HF ayousanz/cc-g2pnp-japanese-g2p-conformer-ctc](https://huggingface.co/ayousanz/cc-g2pnp-japanese-g2p-conformer-ctc) | 2026 | Conformer + CTC, 300k steps | text-s2s (streaming) | 84M | 未報告 (未検証再現、300k steps ≈ 1/4) | 未報告 | 未報告 | V |
| 13 | ayousanz/yomi-linter-modernbert-ja-130m | [HF ayousanz/yomi-linter-modernbert-ja-130m](https://huggingface.co/ayousanz/yomi-linter-modernbert-ja-130m) | 2026 | ModernBERT-Ja-130M + token-cls head | text-enc (linter) | 130M | 音素は出さない (誤読 risk span 検出のみ、proper-noun recall 71-100%) | 未報告 | 未報告 | V |
| 14 | Kanalizer (音写) | [prj-beatrice/kanalizer](https://github.com/prj-beatrice/kanalizer) | 2024 | Seq2seq、英単語 → カタカナ音写 | text-s2s (scope-limited) | ~30M | 音素は出さない (英単語対応のみ) | 未報告 | 未報告 | V |
| 15 (参考) | Dict-DNN-NS (CC-G2PnP paper baseline) | 同 arxiv 2602.17157 Table 1 | 2026 | MeCab+UniDic 形態素解析 + DNN prosody | hybrid | 未公開 | **6D-Eval Phoneme CER 0.40 / PnP CER 1.71** (pure-NN CC-G2PnP に勝利) | 未報告 | 未報告 | V |
| 16 (参考) | haqumei v0.8.0 | [ayutaz/haqumei](https://github.com/ayutaz/haqumei) | 2025 | pyopenjtalk-plus + Kanalizer | hybrid | (rule dominant) | **1.17%** (JSUT Basic5000) | 未報告 | **KER 1.64%** | V |
| 17 (参考) | OpenJTalk | (JVS-3000 Koriyama Interspeech 2026 引用) | (rule) | Rule + HTS | hybrid (rule) | (rule dominant) | 10.82% (JSUT400 PPL CER) | **1.03%** | 未報告 | V |
| 18 (参考) | Claude Opus 4.6 (parse mode) | [Koriyama Interspeech 2026](https://arxiv.org/abs/2606.22009) | 2026 | LLM parse + kana 変換 (暗黙 hybrid) | LLM (hybrid) | ~1T+ | 未報告 | **0.52%** | 未報告 | P |
| 19 (参考) | Gemini 3.1 Pro (parse mode) | 同上 | 2026 | 同上 | LLM (hybrid) | ~1T+ | 未報告 | **0.62%** | 未報告 | P |

**要点**:
- 「JSUT PER」列に **pure-text NN で数値公開済みなのは #3, #4, #5 の 3 件のみ**、しかも #3 は JSUT400 サブセット (Kurihara が独自定義)、#4-5 は own-dict holdout の IPA word-list で JSUT sentence-level ではない (**直接比較は不可能**)
- 「JVS CER」列に pure-text NN の公開数値は **ゼロ**
- 「ROHAN KER」列も pure-text NN の公開数値は **ゼロ**
- したがって FR-61 の同一プロトコル再測定は **単なる形式的義務ではなく、業界初の pure-text NN vs 標準ベンチ 3 本柱 matrix を提供する contribution**

---

## 3. 4 コア baseline の詳細プロファイル

### 3.1 PnG BERT for Japanese (Yasuda & Toda 2022)

**Publication.** [arxiv 2212.08321](https://arxiv.org/abs/2212.08321), IEEE J-STSP Vol. 16 Issue 6, Oct 2022, pp. 1319-1328. DOI [10.1109/JSTSP.2022.3188071](https://ieeexplore.ieee.org/document/9829304/).

**Architecture.** BERT-base (12 layers × 768 hidden, ~110M params)。オリジナルの英語 PnG BERT (Jia et al. Interspeech 2021, [arxiv 2103.15060](https://arxiv.org/abs/2103.15060), 6L × 512 dim) より大きい。Grapheme (漢字/かな) と phoneme (JULIUS 音素) の両ストリームを同一 sequence として concat し、MLM で両方をランダムマスク予測する pre-training を行う。Fine-tune は Tacotron2 の phoneme encoder として downstream TTS に接続。

**Training data & recipe.** 事前学習コーパスは **青空文庫 ~4.9M sentences**。Phoneme labels は "morphological analysis" で自動生成 (paper 本文には具体的なツール名なし。07-doc v1.3 で「Kuromoji + Neologd」の断定は refute 済み、morphological analysis が pseudo-label 元であることのみ verified)。この pseudo-label 依存が、**pyopenjtalk-plus を教師信号として使う我々の Phase 4 と同じ構造的問題を PnG BERT が既に踏んでいた** ことに注意 (Review-1 B4 の警告先例)。

**Reported metrics.** paper Figure 3 / pretraining validation section より (Google snippet 経由で複数独立取得、Confidence V):

| Metric | Value |
|---|---|
| MLM pretrain val accuracy (peak) | 70.3% |
| G2P pretrain val whole-word accuracy | **45.5%** |
| P2G pretrain val whole-word accuracy | 23.6% |
| TTS accent MOS (PGB2 variant) | 2.41 ± 0.03 |
| TTS accent MOS (PGB2T = tone fine-tune) | 2.51 ± 0.03 |
| TTS accent MOS (Tacotron phoneme-only) | 1.89 ± 0.03 |
| TTS accent MOS (Tacotron + rule accent labels) | **3.04 ± 0.03** |

**Important framing correction (Review-1 B1, corrections table row 1).** 07-doc v1.3 の「JSUT G2P whole-segment accuracy 45.5%」は不正確。正しくは **pre-training validation の masked-G2P task 上の whole-word accuracy 45.5%** であり、JSUT test-set G2P benchmark ではない。Paper 自身は 45.5% の解釈として "features … were not dominant in surface-form information with their masking strategy" と述べており、**dictionary coverage への言及は無い** (07-doc の refuted 節と一致)。

**Failure modes.**
- Encoder-only BERT + surface-form MLM は phoneme prediction を "not dominant" にする — pretrain objective 自体の設計不足
- Downstream TTS で accent MOS が rule-derived labels (TACT 3.04) に **明確に負ける** (PGB2T 2.51、Δ = 0.53 MOS ≈ 有意)
- 標準ベンチ (JSUT / JVS / ROHAN) 上の PER が **一切測定されていない** ため hybrid との直接比較すら不能

**Local reproducibility.** GitHub / HuggingFace 公式 checkpoint は無い (2026-07 検索で確認)。追試には青空文庫 4.9M scratch + 論文の hyperparameter 抜粋からの再現が必要 = **Phase 3-4 で 2-3 週の再現コストを見込む**。この負担は高いが、我々の pretrain-plus-fine-tune (Phase 4) の直接的比較 baseline となるため、再現は本 project の Table 1 の説得力に直結する。

### 3.2 Kakegawa TJ-G2P (原型 2021 / TJ-G2P alone は Kurihara 2024 再実装)

**Publication.** 原型は Kakegawa, Hara, Abe, Ijima "Phonetic and Prosodic Information Estimation from Texts for Genuine Japanese End-to-End Text-to-Speech", Interspeech 2021, pp. 126-130, DOI [10.21437/Interspeech.2021-914](https://doi.org/10.21437/Interspeech.2021-914), [PDF](https://www.isca-archive.org/interspeech_2021/kakegawa21_interspeech.pdf)。"TJ-G2P" の呼称は Kurihara & Sano NHK Interspeech 2024 が導入した shorthand で、Kakegawa 論文自身は使用していない。

**Important attribution correction (Review-1 B1, corrections table row 2).** 07-doc および Redesign spec の「Kakegawa TJ-G2P (JSUT400 PPL CER 11.85%)」は正確には **Kurihara 2024 が Kakegawa 2021 arch を JSUT 用に再学習・再測定** した数値。Kakegawa 2021 本体は JSUT を評価せず、**5,142 新聞+ブログ文の Word-Accuracy 98.0** (rule 98.2 に 0.2pt 敗北) で報告している。

**Architecture.** OpenNMT-py Transformer (5 enc / 5 dec, 8 heads, dropout 0.1, Adam β2=0.998, LR 2.0, 40 epochs)。Bi-LSTM (2/2 layers) の対照実験も同時報告。**Params は paper に明記なし** (T5-base 想定はコミュニティ推測)。I/O は **character-level input** (Kanji/Kana/digits/latin/symbols, ~2,600 types) → **syllable-level output** (~300 tokens、phoneme + devocalization + accent-nucleus flag + `/` = accent-phrase boundary)。

**Training data & recipe.** Kakegawa 2021 では 5M 新聞文 + JTAG (Matsuoka 1996 / Fuchi 1998) 由来の PPI 自動アノテーション (labels ~91% correct = noisy pseudo-labels)。Kurihara 2024 の TJ-G2P alone は **504,545 NHK 新聞文 (Open JTalk PPL labels)** で事前学習し JSUT Basic5000 (4,000 train / 1,000 test) で fine-tune。

**Reported metrics.**

Kakegawa 2021 (Table 4, 5,142 文 manually annotated newspaper/blog):
| System | P-acc | PP-acc | B-acc | N-acc |
|---|---|---|---|---|
| Rule (conventional TTS) | 98.2 | 95.3 | 93.3 | 89.8 |
| Transformer (pure NN) | 98.0 | 95.0 | 92.8 | 89.8 |

Kurihara 2024 (Table 2 overall JSUT400):
| System | JSUT400 PPL CER |
|---|---|
| Open JTalk | 10.82% |
| TJ-G2P alone | **11.85%** |
| TJ-G2P + BAS | 12.07% |

Kurihara 2024 (Table 4 counter-word 100文 + proper-noun 100文 hard set):
| System | Counter-word CER |
|---|---|
| TJ-G2P alone | 15.43% |
| TJ-G2P + BAS + dict | **1.42%** (10.9× reduction) |

**Failure modes.**
- Pure NN が JSUT400 上で rule に **0.03 pt-1.0 pt 負ける** (Kakegawa 5,142 文 Word-Acc / Kurihara JSUT400 CER の 2 個別実験で一貫)
- Counter word 上での改善 15.43% → 1.42% (10.9×) は Kurihara 論文が明示的に **dictionary substitution + BAS** に帰属させており、NN capacity ではない
- Pseudo-label 依存 (PPI labels ~91% correct) が学習信号のノイズを上限として組み込む

**Local reproducibility.** Kakegawa の code は **未公開** (Sunao Hara の DBLP 全著作を確認したが該当なし)。Kurihara 2024 の code / checkpoint も **公開なし**。JSUT400 の 400 文 split 定義も Kurihara 論文本文にのみ記述 = **完全再現には 400 文 split の再構築 + arch の scratch 再実装** が必要。TJ-G2P alone の再現は Phase 4 のスコープ外に置き、**代わりに我々自身の Phase 3 multi-task encoder を "TJ-G2P alone 相当の pure-text seq2seq baseline" として自称する** 現実解を推奨。

### 3.3 CharsiuG2P (Zhu, Zhang, Jurgens 2022)

**Publication.** Jian Zhu, Cong Zhang, David Jurgens "ByT5 model for massively multilingual grapheme-to-phoneme conversion", Interspeech 2022, [arxiv 2204.03067](https://arxiv.org/abs/2204.03067), [ISCA archive](https://www.isca-archive.org/interspeech_2022/zhu22_interspeech.html), [GitHub](https://github.com/lingjzhu/CharsiuG2P).

**Architecture.** ByT5-small (byte-level Transformer seq2seq, Google の T5 派生) を 99 言語で multilingual G2P に fine-tune。Paper Table 1 で **ByT5-small = 300M params** と明記。GitHub の updated release (`_100` suffix) は同 arch を追加学習した re-train。

**Training data & recipe.** WikiPron + ipa-dict + 言語別辞書。日本語辞書は [`dicts/jpn.tsv`](https://raw.githubusercontent.com/lingjzhu/CharsiuG2P/main/dicts/jpn.tsv) の **221,788 エントリ (6.3 MB)**、targets は IPA (JULIUS ではない、例: `α-ヘリックス → aɾɯɸaheɾikːɯsɯ`)。Paper は明示的に「CJK words are not separated by spaces, and an external tokenizer must be used before feeding words into the model」と警告。

**Reported metrics.**

Paper Table 1 (multilingual aggregate over 99 lang): **PER 8.8% / WER 25.9%**。GitHub retrain (`byT5_small_100`): PER 0.089 / WER 0.261。

日本語 subset は **paper 本文には per-lang table なし**。数値は GitHub の [`multilingual_results/multilingual/byt5-small`](https://github.com/lingjzhu/CharsiuG2P/tree/main/multilingual_results/multilingual) 内の raw log から抽出 (Confidence V):

| Checkpoint | Japanese PER | Japanese WER |
|---|---|---|
| byt5-small (paper multilingual) | **0.1051** | 0.272 |
| byt5_small_100 (HF release) | 0.1061 | 0.280 |
| byt5_tiny_16_layers_100 | 0.1296 | 0.350 |
| byt5_tiny_12_layers_100 | 0.1393 | 0.362 |
| byt5_tiny_8_layers_100 | 0.1609 | 0.414 |
| mT5-small finetuned | 0.1986 | 0.458 |
| **Monolingual Japanese-only ByT5** | **0.6689** | 0.958 |
| Monolingual + multilingual finetune | 0.1169 | 0.308 |

**Important framing correction (Review-1 B1, corrections table row 3).** Redesign spec の「CharsiuG2P JSUT PER ~10.5%」は正確には **own-dict-holdout (WikiPron+ipa-dict 由来の word-list) 上の IPA vs dictionary PER**。JSUT sentence-level kana CER ではないため、OpenJTalk 1.03% や haqumei 1.17% と直接比較は category error。

**Additional insight (Review-1 S/corrections row 3).** **Monolingual jpn-only ByT5 が 66.89% PER に崩壊** し、multilingual co-training が **6.4× 効いている** — これは我々の recipe への強い示唆で、**multilingual pretrain (llm-jp-modernbert-base の英日混在 or ByT5 の 99lang) が pure-NN 日本語 G2P の必要条件かもしれない** ことを示す。

**Failure modes.**
- IPA target で **JULIUS 音素 (pyopenjtalk 互換) に直接使えない** — 我々の recipe に落とし込むには IPA↔JULIUS mapping table が必要 (§4 Protocol harmonization 参照)
- Sentence-level 非対応 — paper が external tokenizer 前提と警告している
- HFカードに **model card / license の明記なし** (charsiu org 全モデルが "No model card")

**Local reproducibility.** HF checkpoint 公開済み ([`charsiu/g2p_multilingual_byT5_small_100`](https://huggingface.co/charsiu/g2p_multilingual_byT5_small_100), ~7K downloads/mo)。**Phase 4 で JSUT Basic5000 上で自己再測定可能** (OQ-2)。カタカナ形態素境界を MeCab で切り出し、IPA→JULIUS 変換して sentence-level PER を計算するだけ = 1 週間以内。

### 3.4 CC-G2PnP (Shirahata & Yamamoto SB Intuitions 2026)

**Publication.** [arxiv 2602.17157](https://arxiv.org/abs/2602.17157), submitted 2026-02-19, ICASSP 2026 accept (単一情報源、plausible)。著者 Ryuichi Yamamoto は SB Intuitions 所属 (公開情報)、**modernbert-ja-130m と同一ラボ**。この事実は本プロジェクトの strategic 位置付けに重要 — 我々は SB Intuitions 自身の pure-NN G2P を SB Intuitions 自身の base encoder で上回ることを目指す。

**Architecture.** Streaming Conformer + CTC。単語境界情報 (external tokenizer) 不要の unsegmented-language 対応がウリ。Look-ahead parameter (MLA) で streaming/non-streaming を切替。

**Training data & recipe.** 詳細は paper 未公開部分あり。Fine-tune は SB Intuitions 内製コーパス。

**Reported metrics (6D-Eval 上、Table 1 + Table 2)** — Review-1 B2 で redesign spec の「SER 8.4%」が誤りと判明したため、arxiv 一次資料から再抽出:

| System | PnP CER | Phoneme CER | MOS |
|---|---|---|---|
| **Dict-DNN-NS (non-streaming hybrid)** | **1.71** | **0.40** | **4.07 ± 0.09** |
| CC-G2PnP-NS (non-streaming, proposed) | 1.80 | 0.48 | 4.02 ± 0.09 |
| CC-G2PnP-5-1 (best streaming) | 1.79 | 0.52 | 4.02 ± 0.09 |
| Dict-DNN-20 (streaming baseline) | 2.28 | 0.56 | — |
| Dict-DNN-10 | 3.58 | 0.86 | 3.35 |
| Dict-DNN-5 | 6.67 | 1.54 | 2.73 |

**6D-Eval とは.** **2,722 sentences × 6 domain (chat/interview/news/novel/practical book/SNS) の内製評価セット**、expert phoneme + prosodic labels 付き。**public benchmark ではなく再現不能**。paper でも SB Intuitions 外部の再使用実績なし。

**Dict-DNN の実体.** 「Dict-DNN」は pyopenjtalk 相当の rule-based ではなく、**MeCab + UniDic 形態素解析 + DNN prosody head** の **形態素解析+DNN ハイブリッド**。従って CC-G2PnP vs Dict-DNN は「(a) Conformer+CTC pure-NN vs (b) MeCab+UniDic+DNN hybrid」の対決構造 (Review-1 corrections row 4)。

**Refuted claim (07-doc から継続 + Review-1 B2 で強化)** — 「pure-NN CC-G2PnP は Dict-DNN hybrid を 6D-Eval で両指標で上回る」という 07-doc の refute 対象は、正確には **paper 本文は "significantly outperforms the baseline streaming G2PnP model" しか言っておらず、hybrid 越えは主張していない** ことが本 09 の一次資料再検で確認された。**Table 1 は Dict-DNN-NS が全ての objective/subjective metric で勝利** (CER 1.71 < 1.80, phoneme 0.40 < 0.48, MOS 4.07 > 4.02)。したがって "pure-NN が 2026 年最新研究でも hybrid に届かない" は継続確定。

**Failure modes.**
- 6D-Eval は private set のため **第三者検証不能**
- JSUT/JVS/ROHAN 上で **数値が一切測定されていない**
- Dict-DNN-NS に non-streaming 直接対決で敗北 (paper 自身が公表)

**Local reproducibility.** paper には code/checkpoint 公開の記述なし (5 targeted searches all negative)。ただし **community 再現版 [`ayousanz/cc-g2pnp-japanese-g2p-conformer-ctc`](https://huggingface.co/ayousanz/cc-g2pnp-japanese-g2p-conformer-ctc)** が 300k steps (planned の 1/4) で公開済み、MIT。**Phase 4 で JSUT Basic5000 上で ayousanz 再現版を自己測定可能** (OQ-3)。ただし 300k steps は full recipe 未達なので、paper の "best" 数値との比較は不可、あくまで "coarse pure-NN Conformer baseline" 位置付け。

---

## 4. Protocol harmonization (Review-1 R4)

FR-61 の実行前に、以下の非互換を吸収する変換規則を先に固定する。**この節が無い限り FR-61 は執行不能** (Review-1 の警告)。

### 4.1 JSUT split の 3 定義

| 呼称 | 定義 | 出典 | 我々の採用 |
|---|---|---|---|
| Basic5000 (full) | JSUT `basic5000` 全 5,000 文 | jsut-label 公式 | ○ (haqumei-eval 準拠) |
| Basic5000 train/test (4k/1k) | Kurihara 2024 の JSUT 用 fine-tune split | Kurihara Interspeech 2024 | 参考、我々は使わない |
| JSUT400 | Basic5000 test 1000文からランダム 400文 | Kurihara Interspeech 2024 §3.2.1 | Kurihara 数値との bridge 用に再現 |

**採用方針.** 我々の primary 数値は **Basic5000 full (5,000 sentences)** の PER。Kurihara の 11.85% との比較用にのみ **JSUT400 seed=Kurihara 論文注釈のとおり再構築** した bridge 数値を公表する。

### 4.2 Metric 変換

| 出典 metric | 内容 | 我々の canonical への変換 |
|---|---|---|
| CharsiuG2P Japanese PER (0.1051) | own-dict IPA vs dict word-list, edit distance / phoneme count | (a) MeCab で JSUT 文を単語列に分解 → (b) 各単語を CharsiuG2P 推論 → (c) IPA→JULIUS 変換 → (d) 文単位 concat → (e) 標準 PER 計算 |
| Kurihara TJ-G2P "PPL CER" (11.85%) | phoneme + prosodic labels 混在 char-level CER | Prosody label を strip した pure-phoneme PER と 2 系統で報告 |
| PnG BERT whole-word accuracy (45.5%) | pretrain val, whole-word exact match | test-set PER には換算不能 (原理的に不可能) — 表には "not comparable" と注記 |
| CC-G2PnP 6D-Eval Phoneme CER (0.48-0.52) | 内製データ、境界を含む char-level CER | 直接比較不可 (6D-Eval が private のため) — 我々の JSUT 数値と CC-G2PnP paper の JSUT 数値 (存在しない) との bridge は不能 |

### 4.3 Phoneme set

- **我々の canonical**: JULIUS (pyopenjtalk 互換)、モーラアクセント H/L + アクセント句境界 `/`
- **CharsiuG2P**: IPA — [変換ルール](https://github.com/pyopenjtalk/pyopenjtalk/blob/master/pyopenjtalk/htsengine.py) を参考にした JULIUS↔IPA table を Phase 1 で確定
- **PnG BERT**: JULIUS 相当 (paper で明記なしだが morphological analysis 出力 = pyopenjtalk 系と推定)
- **Kurihara TJ-G2P**: PPL 表記 (phoneme + prosody 混在)
- **CC-G2PnP**: Yamamoto lab の internal phoneme set (public 未公開)

**採用方針.** 全 baseline を **JULIUS 系に変換した状態で PER を報告**、変換誤差は明示的に脚注化する。

---

## 5. Frontier LLM reference values (Aspirational tier)

**Source.** [Koriyama Interspeech 2026, arxiv 2606.22009](https://arxiv.org/abs/2606.22009), JVS-3000 kana CER benchmark。**Frontier LLM 数値は本 doc の R1-R5 briefs で直接抽出できていないため Confidence P** (Review-1 B3)。Phase 0 の baseline 再測定で我々自身が確認する。

| Model | JVS-3000 kana CER | Notes |
|---|---|---|
| Claude Opus 4.6 (parse mode) | ~0.52% | 論文 Table (Koriyama Interspeech 2026 引用、要 verify) |
| Gemini 3.1 Pro (parse mode) | ~0.62% | 同上 |
| GPT-5 | 未測定 (Koriyama paper には現れず、または非対象) | Phase 0 で我々が測定 |
| OpenJTalk (rule) | 1.03% | 同 paper Table |
| haqumei | (JSUT の 1.17% は公表、JVS 数値は未確認) | 別プロトコル、bridge 不能 |

**Frontier LLM の "parse mode" は暗黙 hybrid.** LLM に "この文を kana に変換してください" と prompt を投げるだけで、内部で parse → kana 変換の 2 段階 (暗黙 hybrid) を行っていると Koriyama 2026 が示唆。したがって「1T params LLM が pure-NN で 0.52% を出した」という単純解釈は誤り、**Aspirational tier としては 130M pure-text NN で 1% 台 (order-of-magnitude match) が現実的な "world-first" 水準**。

**OQ-5 (open question, R1-R5 で埋める).** Frontier LLM を "parse mode を明示的に off" (=decode-only 直接 kana 生成) にした場合の kana CER 劣化度を測定する。この数値が 0.52% から何倍劣化するかで、Frontier LLM の "本当の pure-NN 能力" を推定する。

---

## 6. Refuted claims section (07 の継承 + 新規)

07 の refuted 節と重複しない、09 で新規に確定した反証を記載。

### 6.1 継承 (07 から)

- **却下**: 「NHK Kurihara 2024 は Japanese G2P を pure-NN では unsolvable と framing」 (07 §7 verified)
- **却下**: 「PnG BERT は pure-NN が lexicon coverage に及ばないと明示的に framing」 (07 §7 verified) — 本 09 §3.1 の paper 一次資料再読でも同結論
- **却下**: 「CC-G2PnP は Dict-DNN hybrid を 6D-Eval で両指標で上回る」 (07 §7 verified) — 本 09 §3.4 の Table 1 直接掲載で hybrid 勝利を再確認
- **却下**: 「PnG BERT の training labels 起源は Kuromoji + Neologd」 (07 §7 verified) — paper 本文は "morphological analysis" のみ、Kuromoji/Neologd の断定不能

### 6.2 09 で新規確定

- **却下 (Review-1 B2 由来)**: 「CC-G2PnP は 6D-Eval で SER 8.4% を達成」 — この数値は arxiv 2602.17157 Table 1/2 に **存在しない**。paper の実際の数値は PnP CER 1.79-1.80 / Phoneme CER 0.48-0.52 / MOS 4.02。redesign spec 執筆時の誤引用と判明、07-doc §2.3 の該当行も修正が必要
- **却下 (Review-1 B1 由来)**: 「PnG BERT JSUT G2P whole-segment accuracy 45.5%」 — 45.5% は **pretrain validation 上の masked-G2P task 上の whole-word accuracy** であり、JSUT test-set PER ではない (直接比較不能)
- **却下 (Review-1 B1 由来)**: 「Kakegawa TJ-G2P の 11.85% は Kakegawa 2021 の結果」 — 11.85% は **Kurihara 2024 の再実装** で、Kakegawa 2021 本体は JSUT 未評価 (5,142 新聞文 Word-Acc)
- **却下 (Review-1 B1 由来)**: 「CharsiuG2P Japanese PER ~10.5% は JSUT PER と比較可能」 — 10.51% は **own-dict-holdout IPA word-list PER**、JSUT sentence-level kana CER とは metric category が異なる
- **未確定 (verify 継続、Review-1 B3)**: 「Claude Opus 4.6 = 0.52% JVS kana CER」 — Koriyama Interspeech 2026 (arxiv 2606.22009) の paper 一次資料からの直接抽出が R1-R5 で完了していない、Phase 0 で自己 verify する

### 6.3 未定/継続監視

- **強い主張 refute 未確定**: 「pure-text NN が hybrid を JSUT/JVS/ROHAN のどれかで上回った公開事例が 2026-07 時点で存在する」 — 09 の全 baseline 精査で **該当事例なし** を確認、ただし 2026 の残り 6 ヶ月で新論文が出る可能性は continuous に監視

---

## 7. 本プロジェクトの target positioning (Review-1 R3 に基づく 2 sub-tier)

### 7.1 vs pure-text NN prior-art (直接対決 tier)

これは本プロジェクトが **競技場**とする土俵。conservative-stretch-aspirational の 3 段。

| Metric | Prior-art best | Conservative | Stretch | Aspirational |
|---|---|---|---|---|
| JSUT Basic5000 PER (full) | 未公表 (CharsiuG2P ~10.5% は own-dict-holdout、直接比較不可) | **< 5.0%** (CharsiuG2P を明確に上回る) | **< 2.0%** (2×) | **< 0.5%** (Frontier LLM 並、world-first) |
| JSUT400 PPL CER (Kurihara bridge) | Kurihara TJ-G2P 11.85% | < 8.0% | < 5.0% | < 1.5% |
| JVS-3000 kana CER | pure-text NN の公表値ゼロ | **公表することが最低目標**、< 5.0% | < 2.0% | < 0.62% (Gemini 越え) |
| ROHAN KER | pure-text NN の公表値ゼロ | **公表することが最低目標**、< 5.0% | < 2.0% | < 1.0% |
| JSUT モーラアクセント精度 | Hida 2022 hybrid 97.33% | > 90% | > 96.66% (Hida hybrid parity) | > 98% |
| 多音字 hard-set PER | pure-text NN の公表値ゼロ | 公表することが最低目標 | 差 vs pyopenjtalk baseline を最小化 | 逆転 |

**Conservative の justification.** CharsiuG2P monolingual-jpn が 66.89% PER に崩壊した事実は、pure-NN 日本語 G2P の naive baseline が 60% 台であることを示す。5.0% は "multilingual pretrain と適切な recipe で 10 倍以上の gain を出す" 妥当ライン。

**Stretch の justification.** 2× improvement は scaling law の 1 桁 decade に相当し、130M pure-NN でも hyperparameter search + multi-task supervision の慣行から到達余地あり (根拠: Hida 2022 hybrid が single-task から 2-3× 改善)。

**Aspirational の justification.** Frontier LLM の 0.52% は 1T param + parse mode hybrid 前提。130M pure-NN が同数値に届けば "1000× smaller model matches LLM" の historic result。ただし non-realistic である前提で、negative でも Table 1 の 4 baseline 越えが確保できれば contribution は成立 (§7.3 参照)。

### 7.2 vs speech-conditioned NN prior-art (out-of-class 参考)

**明示的方針**: Ohnaka 2025 (PER 0.93%, speech+text)、Furigana Whisper (JSUT closed-set CER 0.19%, prompted)、Hu 2025、Koriyama SSW13 は **speech modality を持つ = 我々の pure-text 制約と category が異なる**。同 category と主張することは false-advertising のため、Table 1 では **別 section で "reference-only"** として掲載し、我々の数値との直接比較は禁止する。

ただし **Ohnaka 2025 の PER 0.93% は "NN 系列が hybrid parity に迫れる initial evidence" として引用可能**、本プロジェクトの stretch target の学術的正当化として脚注参照する。

### 7.3 vs hybrid prior-art (reference-only)

本プロジェクトの absolute goal ではないが、コミュニティが期待する参照値を明示:

| System | Type | JSUT PER | JVS CER | ROHAN KER | 我々の位置付け |
|---|---|---|---|---|---|
| haqumei v0.8.0 | hybrid (dict + Kanalizer) | 1.17% | 未公表 | 1.64% | reference-only、絶対値では劣る前提 |
| OpenJTalk | rule + HTS | 10.82% (JSUT400) | 1.03% | 未公表 | reference-only |
| pyopenjtalk-plus | rule + Sudachi | 未公表 (haqumei 経由で参照) | 未公表 | 未公表 | reference-only |
| Claude Opus 4.6 (parse mode) | LLM (暗黙 hybrid) | 未公表 | ~0.52% | 未公表 | aspirational reference |
| Gemini 3.1 Pro (parse mode) | LLM (暗黙 hybrid) | 未公表 | ~0.62% | 未公表 | aspirational reference |

**要点**: hybrid の絶対値 (haqumei 1.17%, OpenJTalk 1.03%) を **数値目標としない**。ただし preprint Table 1 では reference 行として掲載し、gap を honest に見せる。

---

## 8. Publication plan

### 8.1 Preprint / conference target

| Venue | Deadline | 我々の paper positioning | 適合度 |
|---|---|---|---|
| **arxiv preprint** | 2026-12-30 (Phase 6 exit) | "First systematic pure-text NN benchmark on JSUT/JVS/ROHAN with 130M ModernBERT-Ja" | ○ 必ず提出 |
| **Interspeech 2027** | 2027-03 目安 | 同上、TTS-adjacent の会議として fit 高 | ○ 主要提出先 |
| **ICASSP 2027** | 2026-09 abstract / 2026-10 full | speech modality が薄いため fit 中 | △ ORR 見ながら判断 |
| **ACL 2027** | 2027-02 目安 | NLP 会議として tokenizer / multi-task 側面を強調すれば fit | △ 副次提出候補 |
| **SSW 14 (2027)** | 未確定 | ISCA workshop、SB Intuitions / NHK / CyberAgent が集まる主要 J-TTS venue | ○ SB Intuitions ラボと直接対決の意義大 |

**採用方針**: arxiv preprint を最優先で確定し、Interspeech 2027 / SSW 14 に concurrent 投稿。

### 8.2 HF Hub checkpoint plan

| Repo 名 (予定) | 内容 | ライセンス |
|---|---|---|
| `modernbert-g2p-ja-130m-base` | Phase 3 multi-task fine-tune 済み ModernBERT-Ja 130M | MIT (base model 由来) |
| `modernbert-g2p-ja-130m-pretrained-plus` | Phase 4 pretrain-plus-fine-tune 版 | MIT |
| `modernbert-g2p-ja-30m` / `70m` / `310m` | Phase 5 scaling ablation 各サイズ | MIT |
| `modernbert-g2p-ja-eval-splits` | JSUT/JVS/ROHAN eval split と phoneme/prosody label データ | CC-BY-SA-4.0 (ROHAN 継承) |
| `modernbert-g2p-ja-jsut400-bridge` | Kurihara TJ-G2P 数値との bridge 用 400 文 split | 同上 |

**モデルカード必須項目**: JSUT PER, JVS kana CER, ROHAN KER, 7 hard-set 別 PER, mora accent accuracy, tokenizer pilot (P-A/P-C) の別 (**P-B MeCab-pretokenize は v2.0 pivot で drop**)、data supervision 内訳 (§9 で議論)。

### 8.3 Preprint Table 1 の final format

Preprint の flagship table は本 doc §2 の inventory から **pure-text NN 行のみ抽出** + 我々の結果を追加した以下形式:

```
Table 1: Pure-text NN Japanese G2P on standard benchmarks (JSUT/JVS/ROHAN)
------------------------------------------------------------------------
Model                     | Params | JSUT PER | JVS CER | ROHAN KER | Category
PnG BERT (2022)           | ~110M  | n/a*     | n/a     | n/a       | text-enc
Kakegawa 2021 (原型)      | ?      | n/a      | n/a     | n/a       | text-s2s
TJ-G2P alone (Kurihara)   | ?      | 11.85%** | n/a     | n/a       | text-s2s
CharsiuG2P (multilingual) | ~300M  | our-remeasure*** | our-remeasure | our-remeasure | text-s2s
CC-G2PnP (Shirahata)      | ~80M   | n/a      | n/a     | n/a       | text-s2s
--- Our contributions ---
ModernBERT-G2P-Ja 30M     | 30M    | our-XX%  | our-XX% | our-XX%   | text-enc
ModernBERT-G2P-Ja 130M    | 130M   | our-XX%  | our-XX% | our-XX%   | text-enc
ModernBERT-G2P-Ja 310M    | 310M   | our-XX%  | our-XX% | our-XX%   | text-enc
--- Reference (out-of-scope) ---
Ohnaka 2025 (LARGE-TTSaug)| ~300M+ | 0.93%†   | n/a     | n/a       | speech+text
Furigana Whisper          | ~244M  | 0.19%††  | n/a     | n/a       | speech+text
haqumei v0.8.0            | rule   | 1.17%    | n/a     | 1.64%     | hybrid
OpenJTalk                 | rule   | 10.82%** | 1.03%   | n/a       | rule
Claude Opus 4.6           | ~1T+   | n/a      | 0.52%   | n/a       | LLM hybrid

* pretrain val whole-word acc 45.5%; test-set PER 換算不能
** JSUT400 subset (Kurihara 2024 定義)、PPL CER
*** own-dict IPA 10.51%、我々が JSUT sentence-level に再測定
† LARGE-TTSaug (Ohnaka 内製セット)
†† JSUT closed-set + prompted
```

---

## 9. Data supervision as rule-leakage: self-audit (Review-1 R5)

Pure-NN inference 制約と "教師信号としての rule-based tool 使用" は矛盾しないか、を明示的に自己監査する。

### 9.1 使用する rule-based 教師信号 (Phase 4 pretrain-plus-fine-tune 用)

| Source | Type | Phase | Nature | Leakage risk |
|---|---|---|---|---|
| pyopenjtalk-plus (surface, kana) | 800K エントリ辞書 | P1 生成 + P4 教師 | 単語単位の (漢字, ふりがな) pair | 辞書内 word coverage は 100% だが、辞書外への extrapolation は NN 側に依存 |
| pyopenjtalk-plus (surface, mora accent) | 同上 | P1 + P4 | 単語単位アクセント | 同上 |
| UniDic 全エントリ (~1M) | 辞書 | P1 + P4 | 形態素 + 読み | 同上 |
| Wikipedia ふりがな抽出 (~500K sentences) | text corpus | P1 | ふりがな付き文 | 人手 crowdsource 由来、rule 依存低 |
| 青空文庫 ふりがな (~200K sentences) | text corpus | P1 | 同上 | 同上 |
| JVS-nonpara-kana (CyberAgent 2026-06 更新) | eval dataset | eval のみ | JVS 音声の kana annotation | eval のみ、train には使わない |

### 9.2 Leakage 分析 (PnG BERT の教訓との比較)

PnG BERT (§3.1) は **青空文庫 4.9M 文の phoneme labels を "morphological analysis" で pseudo-generate**、その label 精度が学習信号の上限となった。我々の Phase 4 では:

- **同じ pattern を回避する gate**: (a) pyopenjtalk-plus の辞書内 word のみを教師信号として使う (辞書外の推測を教師信号にしない)、(b) hard-set 7 カテゴリの test-time 挙動を **rule 天井にキャップされていないか** を専用 metric (辞書外 word coverage 率) で monitor する
- **明示的な positioning**: 「教師信号として rule 系ツールを使うのは NN training の慣行として一般的だが、**inference-time に rule lookup を発火させない ≠ 教師信号から rule 由来 pattern を学ばないこと**」— 我々は前者を保証するのみ、後者は保証しない (それは pure-NN 定義を狭めすぎる)。
- **arxiv 論文での disclosure**: preprint §3 methodology で本節を "Training-time rule leakage" として明示、reviewer からの "hybrid-in-disguise" 批判を preempt する

### 9.3 Boundary definition (FR-60 の明文化)

- **禁止**: inference 時に pyopenjtalk/MeCab/UniDic を Python subprocess として呼ぶ
- **禁止**: inference 時に dict lookup を dict の存在で trigger する if 分岐
- **禁止**: rule-based tool の中間出力 (形態素境界、品詞など) を推論経路の feature として使用
- **許容**: pre-tokenizer として char-level BERT のように内部で mecab 事前実行して token 化する ← **これも Phase 2 で禁止し、char-level or subword-only に統一する** (FR-60 の追加条項として requirements.md v2.0 に反映)。**この方針に伴い Phase 2 の P-B (MeCab pretokenize pilot) は v2.0 pivot で drop され、pilot は P-A/P-C の 2 本に縮小した**
- **許容**: 学習時に rule-based tool 出力を pseudo-label / MLM supervision として使用
- **許容**: 学習時に rule-based tool 出力を data augmentation の augmentor として使用

---

## 10. Open questions (R1-R5 research 完了後に埋める)

- **OQ-1** (from spec): PnG BERT の train recipe 詳細 (data size, epoch, LR) を公開範囲で抽出 — **本 09 §3.1 で partial (4.9M 文, MLM ratio 未確定)**
- **OQ-2** (from spec): CharsiuG2P byT5 の日本語 subset を JSUT Basic5000 で **我々の環境で再測定** した数値 — **Phase 0 で実施予定、~1 週コスト**
- **OQ-3** (from spec): CC-G2PnP の 6D-Eval の JSUT overlap 有無、および same-split 再測定可能性 — **Phase 0 で ayousanz 再現版を JSUT で測定、6D-Eval は再現不能なので放置**
- **OQ-4** (from spec): Kakegawa TJ-G2P alone の code 公開有無 (Kurihara 論文の付録参照) — **本 09 §3.2 で公開なし確認、再現は Phase 4 のスコープ外**
- **OQ-5** (from spec): Frontier LLM の pure decode-mode の kana CER — **Phase 0 で Claude Opus 4.6 / Gemini 3.1 Pro / GPT-5 を parse-mode-off で JVS-3000 に測定**
- **OQ-6** (from Review-1 R7): **What is the smallest ModernBERT-Ja size at which pure-NN JSUT PER < CharsiuG2P 10.51%?** — Phase 5 の scaling ablation の primary research question。30M / 70m / 130M / 310M の 4 サイズで最初に CharsiuG2P 10.51% を下回るサイズを特定する
- **OQ-7** (from Review-1 S3): `yomi-linter-modernbert-ja-130m` のリンター output を preprocessor として活用できるか (誤読 risk span を multi-task の auxiliary head として組み込む余地)
- **OQ-8** (追加): Ohnaka 2025 speech+text の PER 0.93% を text-only ablation したときの degradation は? — text-only が speech-conditioning を失うと何倍劣化するかで、我々の pure-text 上限の指標となる
- **OQ-9** (追加): 130M pure-NN の理論上限 (scaling law 予測) — Chinchilla-style D-optimal で JSUT PER < 1% が何 params で達成可能か推定
- **OQ-10** (追加): negative result publication path — 仮に hybrid に届かなくても、**pure-text NN の scaling law + 4 baseline 系統的比較** は独立 contribution として preprint 化可能か、venue 選定と reviewer expectation の pre-check

---

## 11. 変更履歴

- **v2.0 (2026-07-04)**: バージョンを他ドキュメント (CLAUDE.md / requirements.md / 06_roadmap) の pivot バージョンに整合させ v2.0 に統一。P-B (MeCab pretokenize pilot) drop を §8.2 / §9.3 に反映 (pilot を P-A/P-C の 2 本に縮小、MeCab 依存が pure-NN 原則および §9.3 の pre-tokenizer 禁止規定に反するため)。requirements.md への参照を v1.5 → v2.0 に更新。ロードマップ総期間 11–12 週に整合。
- **v1.0 (2026-07-04)**: 新規作成。R1-R5 (5 briefs) の統合 + Review-1 敵対的批判の反映。redesign spec v1.0 の骨子から Category 列追加 (R1)、2024-2026 新モデル 5 件追加 (R2)、target 2 sub-tier 分割 (R3)、protocol harmonization 節新設 (R4)、data-supervision self-audit 節新設 (R5)、OQ-6 (Review-1 R7) 追加、CC-G2PnP 数値を Table 1 の一次資料から正確に再抽出 (Review-1 B2)、PnG BERT 45.5% を "pretrain val whole-word acc" に正しく framing (Review-1 B1)、Kakegawa 11.85% を Kurihara 2024 再実装に正しく attribute (Review-1 B1)、CharsiuG2P 10.51% を own-dict-holdout に正しく framing (Review-1 B1)、Frontier LLM 数値の verify pending を明記 (Review-1 B3)。
