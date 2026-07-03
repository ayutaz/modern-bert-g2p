# 01. サマリーと開発戦略の提言

**作成日:** 2026-07-03
**目的:** ModernBERTベースの日本語G2Pモデルを、OpenJTalk等ルールベースおよび既存NNモデルより高精度化する
**調査手法:** deep-researchワークフロー(110エージェント、27情報源、126主張抽出、25主張を3票敵対的検証、24主張が確認済み)

---

## 1. エグゼクティブ・サマリー

2026年時点の日本語G2Pは新たなSOTAレジームに突入している。

- **フロンティアLLM (Claude Opus 4.6 / Gemini 3.1 Pro)** が JVS-3000ベンチマークで kana CER 0.52–0.62% を達成し、OpenJTalk (1.03%) を明確に上回った。ただし独自API依存で再現不可・不安定。ソース: [arxiv 2606.22009](https://arxiv.org/html/2606.22009)
- **辞書ハイブリッド (haqumei)** が JSUT Basic5000で PER 1.17%、ROHANで KER 1.64% を報告。これがオープンウェイトの現実的な精度床。ソース: [github.com/o24s/haqumei](https://github.com/o24s/haqumei)
- **NHKの dual Transformer (Interspeech 2024)** が最も信頼できる Transformer G2P アーキテクチャの青写真。T5ベースの TJ-G2P と、tohoku-nlp の `bert-base-japanese-char-v2` を用いた Accent Sandhi モジュール (BAS) の組み合わせで、複合語の CER を 15.21% → 6.04%、助数詞語の CER を 16.24% → 1.42% まで削減。ソース: [Kurihara & Sano, Interspeech 2024](https://www.isca-archive.org/interspeech_2024/kurihara24_interspeech.pdf)
- **Hida et al. (ICASSP 2022)** の BiLSTM+BERT+明示的形態素素性は、多音字精度 94.34%、モーラアクセント精度 96.66% を達成し、主観MOS 3.67 (対 オラクル記号 3.69) まで到達。マルチタスク定式化が有効であることを実証。ソース: [arxiv 2201.09427](https://ar5iv.labs.arxiv.org/html/2201.09427)

---

## 2. 目標を「勝つべき相手」で明確化

新モデルを正当化するには、以下の三段階の敵を明示的に上回る必要がある。

| ティア | 敵 | 数値目標(参考) | 備考 |
|---|---|---|---|
| 敵1 (must-beat) | OpenJTalk / pyopenjtalk | JVS-3000 kana CER < 1.03% | ルールベースの上限 |
| 敵2 (should-beat) | haqumei (辞書ハイブリッド) | JSUT Basic5000 PER < 1.17%, ROHAN KER < 1.64% | 現在の実用OSS上限 |
| 敵3 (stretch) | フロンティアLLM (Claude/Gemini) | JVS-3000 kana CER < 0.62% | 自ホスト・再現可能に到達すれば技術的貢献 |

- **敵1と敵2を上回れば** 実用OSSとして最強クラス。TTSやAudio系のプロダクトから採用され得る。
- **敵3に接近すれば** 学術的にも新規性がある(fine-tunedエンコーダで初めての試み — 現時点で公開結果は存在しない)。

---

## 3. 推奨戦略: 「Hybrid Multi-Task ModernBERT」

### 3.1 アーキテクチャの基本方針

**単一のモノリシックなneural seq2seqでOpenJTalkを置き換えるのは非効率**。証拠:

1. Misaki v2、Style-Bert-VITS2、VITS Japanese、GPT-SoVITS、Bert-VITS2 — 主要OSS TTSは全て pyopenjtalk (=Open JTalk) をコアに採用し、NNは補助的にしか使わない。ソース: [Misaki README](https://github.com/hexgrad/misaki), [Style-Bert-VITS2 docs](https://github.com/litagin02/Style-Bert-VITS2/blob/master/docs/Style-Bert-VITS2_en.md)
2. NHK Kurihara & Sano は Transformer G2P (TJ-G2P) 単体では複合語 CER 15.43%、助数詞語 CER 15.43% — OpenJTalk 並みしか出せない。BERT-based BAS を後段に置いて初めて破壊的な改善が出る。ソース: [Kurihara Interspeech 2024](https://www.isca-archive.org/interspeech_2024/kurihara24_interspeech.pdf)
3. haqumei が示すように、辞書 + 局所的なNN補助 (loanword用のONNXモデル) の構造で1%台のPERは達成可能。ソース: [haqumei README](https://github.com/o24s/haqumei)

したがって推奨されるのは以下のハイブリッド構造:

```
入力文
  ↓
[1] Mecab/pyopenjtalk による形態素解析・辞書lookup
  ↓
基本読み + 基本アクセント(ルール)
  ↓
[2] ModernBERT (dual head)
  ├─ Head A: 多音字曖昧性解消 (token classification)
  ├─ Head B: アクセント連濁/連続変異 (sequence tagging)
  ├─ Head C: OOV漢字読み推定 (seq2seq、辞書ヒット無時のみ発火)
  └─ Head D: アクセント句境界予測 (APBP) + アクセント核位置予測 (ANPP)
  ↓
最終出力: 音素列 + モーラアクセント + アクセント句境界
```

### 3.2 なぜModernBERTか

- **RoPE + local-global sliding window attention (128トークン局所窓)** — 短距離依存(モーラ間のアクセント連結)と、文レベルの長距離依存(複合語境界、周辺文脈による多音字選択)の両方を効率的に扱える。ソース: [modernbert-ja-310m HFカード](https://huggingface.co/sbintuitions/modernbert-ja-310m)
- **8,192 token 長コンテキスト** — 助数詞語・複合語のアクセント連濁が文長を超えて相互作用するケースを網羅できる。
- **Flash Attention + unpadding** — バッチ内可変長を効率的に処理でき、パディング多用な短文G2Pデータで実効スループットが向上する。
- **sbintuitions/modernbert-ja の MITライセンス、30M/70M/130M/310M の4サイズ展開** — 実用性(30M/70M)から研究性能上限(310M)まで比較実験が可能。ソース: [modernbert-ja-30m](https://huggingface.co/sbintuitions/modernbert-ja-30m), [modernbert-ja-70m](https://huggingface.co/sbintuitions/modernbert-ja-70m), [modernbert-ja-130m](https://huggingface.co/sbintuitions/modernbert-ja-130m), [modernbert-ja-310m](https://huggingface.co/sbintuitions/modernbert-ja-310m)

### 3.3 決定的な設計上の警告

**SB Intuitions は自らのHFカードで「SentencePieceトークナイザーの境界が形態素境界と一致しないため、named entity recognitionやspan extractionなどのtoken classificationタスクで性能が悪い」と明記している**。ソース: [modernbert-ja-30m HFカード](https://huggingface.co/sbintuitions/modernbert-ja-30m)

これは日本語G2Pにも直接効いてくる。素朴に「1トークン=1音素ラベル」で系列ラベリングを行うと、`102,400語彙SentencePiece unigram+byte-fallback` トークナイザーの分割が漢字1文字を跨いだり、複合語を1トークンに畳んだりすることで、アライメントが破綻する。

**対処選択肢** (05_technical_design.md で詳論):

- (a) seq2seqとして定式化 (LoRA+T5ヘッド、ByT5的なdecode)
- (b) MeCab/Sudachi でpre-tokenizeしてから ModernBERT に投入 (embeddingの再学習が必要になる可能性)
- (c) 文字レベルモデル (`tohoku-nlp/bert-base-japanese-char-v2` — NHK が BAS で採用済) に切り替え、ModernBERT の長コンテキスト・効率性を諦める
- (d) `llm-jp-modernbert` を代替ベースとして評価 (トークナイザーが異なる)

**推奨は (a) + (d) のハイブリッド評価**。まず `modernbert-ja-130m` を seq2seq ヘッドで、次に `llm-jp-modernbert-base` を token classification ヘッドで学習し、head-to-head比較する。

---

## 4. データ戦略(要点)

- **主学習セット**: pyopenjtalk-plus 辞書 (UniDic派生) から (surface, yomi, accent) のトリプルを大量生成
- **監督増強**: JSUT Basic5000 の人手アノテーション (jsut-label)、ROHAN 4600、NHK PPL相当のWikipedia+アクセント辞書
- **アクセントラベル**: `tsukumijima/pyopenjtalk-plus` の accent 情報 + Kanade辞書 (公開されていれば)
- **評価**:
  - JVS-3000 (Koriyama Interspeech 2026) の kana CER でLLMと比較
  - JSUT Basic5000 の PER で haqumei と比較
  - ROHAN 4600 の KER で haqumei と比較
  - 独自 hard-set (難読漢字、外来語、助数詞語、固有名詞) でロバスト性を評価

**ライセンス注意**: `pyopenjtalk-plus`は Open JTalk 由来で修正BSD、UniDic は BSD派生、JSUT は CC-BY-4.0、ROHAN は CC-BY-4.0。ただし商用配布時は各辞書の帰属明示が必要。詳細は 03_datasets_and_benchmarks.md 参照。

---

## 5. 開発ロードマップの要旨

| フェーズ | 期間目安 | 成果物 |
|---|---|---|
| P0. 基盤 | 1週間 | pyopenjtalk / haqumei で JVS-3000 + JSUT + ROHAN 評価パイプライン整備、SOTAの再現 |
| P1. データ | 2週間 | pyopenjtalk-plus 辞書からの学習コーパス生成、jsut-label と ROHAN のラベル正規化、Wikipedia辞書 gold-set の抽出 |
| P2. ベースライン | 2週間 | `modernbert-ja-130m` を seq2seq で学習 (LoRA/full-FT両方)、単一タスクG2P、PER目標 < 3% |
| P3. マルチタスク | 3週間 | 多音字head + APBP + ANPP + BASの4タスク同時学習、PER目標 < 1.5% |
| P4. ハイブリッド化 | 2週間 | 辞書lookup優先、NN補正 (haqumeiパイプライン模倣)、PER目標 < 1.0% |
| P5. スケール & Ablation | 2週間 | 30M/70M/130M/310M 比較、`llm-jp-modernbert` 対照、char-level対照 |
| P6. 評価と公開 | 1週間 | JVS-3000 / JSUT Basic5000 / ROHAN 4600 の3ベンチマーク結果を論文/READMEに整備 |

詳細は 06_implementation_roadmap.md 参照。

---

## 6. 主要な未解決問題(オープンクエスチョン)

deep-research 検証プロセスで表面化した「調査では答えが出なかったが、実装前に決めるべき」問い:

1. **fine-tuned エンコーダ ModernBERT G2P の公開ベンチマーク結果は存在しない**。これから作るモデルは、この分野の最初のデータポイントになる可能性が高い。
2. **どのトークナイザー戦略が最良か head-to-head 比較データが無い**。SentencePiece unigram を保持しseq2seqにするか、MeCab pretokenizeで再学習するか、char-levelにフォールバックするか — 実装前に3系統のパイロット学習が必須。
3. **haqumei / Misaki / pyopenjtalk の per-カテゴリ (固有名詞・外来語・助数詞) 誤り率は公開されていない**。Koriyama benchmarkの分類 (漢字固有名詞6%, カタカナ固有名詞8.5%, 数詞14.2%) をトレース分析して、データ拡張とロス重み付けの優先順位を決める必要がある。
4. **商用配布可能な学習データ組成が未確定**。pyopenjtalk-plus / UniDic 派生読みの二次配布可否、ROHANアノテーション、NHK PPL コーパスがそれぞれ異なるライセンス条件を持つため、商用展開を想定するなら初期段階でライセンス監査が必要。

---

## 7. 参考リポジトリ(直接コード参照するもの)

- [r9y9/pyopenjtalk](https://github.com/r9y9/pyopenjtalk) — Open JTalk Python wrapper。デファクト標準。
- [tsukumijima/pyopenjtalk-plus](https://github.com/tsukumijima/pyopenjtalk-plus) — 改良版辞書。学習コーパス生成のベース。
- [o24s/haqumei](https://github.com/o24s/haqumei) — 現時点の実用SOTA相当のOSSハイブリッドG2P。PER/KERベースラインの直接比較対象。
- [hexgrad/misaki](https://github.com/hexgrad/misaki) — 多言語G2Pエンジン。第二世代日本語対応で pyopenjtalk + full unidic採用。TODOに seq2seq fallback + BERT契約語曖昧性解消の計画あり。
- [sarulab-speech/jsut-label](https://github.com/sarulab-speech/jsut-label) — JSUT用の音素ラベル。評価に必須。
- [prj-beatrice/jsut-label](https://github.com/prj-beatrice/jsut-label) — haqumei が使用しているバリアント。要調査。
- [mmorise/rohan4600](https://github.com/mmorise/rohan4600) — ROHAN評価コーパス。
- [espnet/espnet](https://github.com/espnet/espnet/blob/master/espnet2/text/phoneme_tokenizer.py) — 音素トークナイザー実装のリファレンス。

---

## 8. 一言まとめ

「単一ニューラルモデルでOpenJTalkを置き換える」のではなく、「辞書ベースの高精度パイプラインに ModernBERT で **アクセント連続変異、多音字、OOV** の3点を狙い撃ちで補正する」ハイブリッド設計が、証拠に基づく最短ルート。単純seq-labelingは SB Intuitions 自身が警告するトークナイザー起因の失敗モードに直撃するため、初期選定を誤らないこと。
