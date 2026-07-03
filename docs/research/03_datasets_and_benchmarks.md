# 03. データセットとベンチマーク

日本語G2Pを評価・学習するための公開資源、評価指標、SOTA数値を整理する。

---

## 1. 評価用ベンチマーク (公開・再現可能)

### 1.1 JVS-3000 (Koriyama Interspeech 2026 benchmark)

- **説明**: JVS corpus の nonpara30 subset から抽出された 3,000文を、人手で kana アノテーション。日本語G2Pの新しい標準ベンチマーク。
- **構成**:
  - 漢字固有名詞: 6.0%
  - カタカナ固有名詞: 8.5%
  - 数詞: 14.2%
  - 一般文: 71.3%
- **評価指標**: **parse-mode kana CER** (kana character error rate) — 論文は「essentially equivalent to phoneme error rate」と明言
- **公開SOTA**:
  | System | kana CER |
  |---|---|
  | Claude Opus 4.6 | **0.52%** |
  | Gemini 3.1 Pro | 0.62% |
  | OpenJTalk | 1.03% |
- **入手方法**: JVS本体は [https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus)。Koriyama benchmarkのkanaアノテーションは論文著者による公開待ち (arxiv 2606.22009 の GitHub リンクを確認)
- **論文**: [arxiv 2606.22009](https://arxiv.org/abs/2606.22009)
- **注意**: 論文公開 2週間程度 (2026-06)。数値は proprietary API 依存で時間的に流動的。

### 1.2 JSUT Basic5000 + jsut-label

- **JSUT本体**: [Sonobe et al., "JSUT corpus: free large-scale Japanese speech corpus for end-to-end speech synthesis"](https://sites.google.com/site/shinnosuketakamichi/publication/jsut) — 10時間の日本語女性1話者音声、CC-BY-4.0
- **Basic5000**: 5,000文の一般テキスト読み上げ
- **jsut-label系**:
  - **[sarulab-speech/jsut-label](https://github.com/sarulab-speech/jsut-label)**: 音素・アクセントラベルを付加した公式アノテーション
  - **[prj-beatrice/jsut-label](https://github.com/prj-beatrice/jsut-label)**: haqumei が使用しているバリアント。use_unidic_yomi=true で PER評価に使用可能
- **評価指標**: PER (Phoneme Error Rate)
- **公開SOTA (haqumei self-reported)**:
  - **PER = 1.17%** (S=2117, D=527, I=831, N=297843)
- **利用可否**: JSUT音声は CC-BY-4.0、アノテーションは各リポジトリのライセンス条項を要確認 (通常 CC-BY-4.0 継承)
- **参考リンク**: [prj-beatrice/jsut-label](https://github.com/prj-beatrice/jsut-label), [sarulab-speech/jsut-label](https://github.com/sarulab-speech/jsut-label)

### 1.3 ROHAN 4600

- **リポジトリ**: [mmorise/rohan4600](https://github.com/mmorise/rohan4600)
- **説明**: 4,600文の日本語コーパス。読み上げ音声合成の評価用に設計。豊富な音素バランス。
- **評価指標**: KER (Katakana Error Rate)
- **公開SOTA (haqumei self-reported)**:
  - **KER = 1.64%** (S=1689, D=493, I=288, N=150637)
- **利用可否**: CC-BY-4.0
- **示唆**: JSUT より短文・音素バランス志向。**未知の音素連鎖に対するロバスト性の測定**に有用。

### 1.4 JVS Corpus (発話者バリエーション)

- **公式**: [https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus)
- **説明**: 100話者×平均30分の音声コーパス
- **G2P上の意義**: nonpara30 subset が JVS-3000 benchmark の元データ。学習用途と評価用途で明確に分離すべき。
- **参考**: [arxiv 2009.09679](https://arxiv.org/abs/2009.09679) (JVSコーパス関連論文)

### 1.5 ITAコーパス

- **説明**: 音声合成用に音素バランスを重視した読み上げコーパス
- **G2P学習・評価データとしては**: 公開の (テキスト, 音素, アクセント) ラベルセットの規模は限定的。JSUT/ROHANの補助データとして扱うのが実用的。

---

## 2. 学習データソース (辞書/コーパス)

### 2.1 UniDic

- **説明**: 現代日本語書き言葉辞書。形態素・読み・アクセント情報を含む。
- **主要バリアント**:
  - unidic-cwj (書き言葉)
  - unidic-csj (話し言葉)
  - unidic-lite (軽量版、pyopenjtalk-plus等が使用)
- **G2P用途**: 各エントリの (surface, yomi, accent_type) を抽出することで、大規模な (書字, 読み, アクセント) トリプルの学習セットが作れる。
- **ライセンス**: BSD 3-clause 派生 (要確認)。**商用配布時は帰属明示必須**。
- **入手**: [https://clrd.ninjal.ac.jp/unidic/](https://clrd.ninjal.ac.jp/unidic/) (推定 - 実装前に一次リンク確認必須)

### 2.2 Open JTalk 辞書 (NAIST-jdic)

- **説明**: Open JTalk バイナリ辞書。読み・アクセント情報を含む。
- **G2P用途**: pyopenjtalk 経由で (surface, yomi, accent) を取得可能。
- **ライセンス**: 修正BSD (要確認)
- **注意**: 更新停止気味。現代語で不足あり。

### 2.3 pyopenjtalk-plus 辞書

- **リポジトリ**: [tsukumijima/pyopenjtalk-plus](https://github.com/tsukumijima/pyopenjtalk-plus)
- **説明**: pyopenjtalk辞書の拡張フォーク。UniDic派生読みをより広く含む。
- **G2P学習の主要データソース**。

### 2.4 Sudachi辞書

- **説明**: Works Applications製の日本語辞書。short/medium/long unit の3レイヤ分割単位。
- **G2P用途**: 補助的な形態素解析結果として使用可能。読み・アクセント情報は限定的。

### 2.5 JMDict / EDRDG

- **説明**: 日本語-英語辞書。漢字と読みのペアを含む。
- **G2P用途**: (漢字, 読み) ペアの静的辞書として使用可能。アクセント情報無し。
- **ライセンス**: Creative Commons Attribution-ShareAlike 4.0

### 2.6 Wikipedia日本語版

- **G2P用途**: 明示的な読み情報 (Wikipedia記事タイトルのふりがな、記事本文の {読み: ふりがな}) を抽出可能。固有名詞学習に有効。
- **注意**: 読みが明示的でない固有名詞が多く、大規模抽出には別途辞書照合が必要。

### 2.7 青空文庫

- **G2P用途**: ふりがな付きテキストが豊富。特に旧字体・古語・稀な固有名詞の (漢字, 読み) ペアが取得可能。
- **注意**: 現代語の学習には偏り (古典寄り) がある。

---

## 3. 評価指標

### 3.1 PER (Phoneme Error Rate)

- **定義**: 音素列レベルの Levenshtein 距離 / 正解音素数 = (S + D + I) / N
- **哲学**: TTS のフロントエンドとしての本質的な指標。
- **例**: haqumei JSUT Basic5000 = 1.17% (S=2117, D=527, I=831, N=297843)

### 3.2 kana CER (Character Error Rate on kana output)

- **定義**: kana変換した文字列の CER
- **哲学**: Koriyama Interspeech 2026 が採用。PER と近似的に等価 (論文明言)。
- **例**: OpenJTalk = 1.03%, Claude Opus 4.6 = 0.52%

### 3.3 KER (Katakana Error Rate)

- **定義**: カタカナ表記の CER
- **哲学**: haqumei が ROHAN で採用。kana CER とほぼ同等視できる。
- **例**: haqumei ROHAN = 1.64%

### 3.4 モーラアクセント精度 (Mora-Accent Accuracy)

- **定義**: モーラ単位のHigh/Low判定の一致率
- **例**: Hida et al. ICASSP 2022 = 96.66% (in-house), 97.33% (JSUT)
- **意義**: TTSの自然さの直接的指標。PERが低くてもモーラアクセントが間違えば聴感が破綻する。

### 3.5 APBP (Accent Phrase Boundary Prediction) F1

- **定義**: アクセント句境界の F1 スコア
- **例**: Hida et al. = 96.30 F1
- **意義**: 文レベルのプロソディの正確性を測る。

### 3.6 多音字精度 (Polyphone Accuracy)

- **定義**: 多音字漢字 (「行」「方」等) の正しい読みの選択率
- **例**: Hida et al. = 94.34%
- **意義**: 文脈依存の読み分けの精度指標。

### 3.7 Sentence-Exact Accuracy

- **定義**: 文全体が完全に一致する割合
- **例**: Hida et al. APBP sentence-exact = 58.68%
- **意義**: ユーザ体感に近い厳しい指標。

### 3.8 MOS (Mean Opinion Score) — subjective

- **定義**: 生成された音声の主観品質評価 (1-5点)
- **例**: Hida et al. = 3.67 ± 0.07 (対 オラクル記号 3.69 ± 0.07)
- **意義**: G2P改善が最終的にTTS品質に反映されるかの確認。

### 3.9 Downstream TTS Pronunciation CER

- **定義**: G2P出力をTTSに投げて生成された音声を ASR し、書き起こしの CER を測定
- **例**: Koriyama 2026 で Gemini 3.1 Pro kana → kana-TTS = 2.38%
- **意義**: G2Pの下流影響を実測できる。

---

## 4. 評価プロトコル推奨案

### 4.1 主評価セット (必須)

| セット | 指標 | 目標 |
|---|---|---|
| JVS-3000 (Koriyama benchmark) | kana CER | < 0.62% (Geminiに勝つ), stretch: < 0.52% (Claudeに勝つ) |
| JSUT Basic5000 (jsut-label) | PER | < 1.17% (haqumei越え) |
| ROHAN 4600 | KER | < 1.64% (haqumei越え) |
| JSUT Basic5000 accent-labeled subset | mora-accent accuracy | > 97.33% (Hida et al. JSUT越え) |

### 4.2 副評価セット (**7カテゴリ Hard-set**)

日本語G2Pがカバーすべき7つの困難カテゴリ。**多言語混在** (英単語・略語) は実世界の日本語文に高頻度で出現するため必須カテゴリ:

- **難読漢字hard-set**: 多音字が多い文200程度 (「行」「方」「生」等)
- **カタカナ外来語hard-set**: カタカナ表記の外来語を含む短文200程度
- **助数詞hard-set**: 「3人」「五本」等の助数詞語を含む文200程度 (NHK Kurihara benchmarkに準拠)
- **数詞・単位hard-set**: 数字を含む文200程度 (「2025年」「3.14」「10km」「3GB」等)
- **固有名詞hard-set**: 難読姓、地名、企業名 200程度 (漢字+カタカナ両方)
- **英単語混在文hard-set** (**新規**): 「iPhone を買った」「PDF を開く」「Zoom で会議」等、英単語がアルファベット表記のまま埋め込まれた実文200程度
- **英字略語hard-set** (**新規**): AI, NASA, HTML, PDF, WHO, e-mail, Wi-Fi, T-shirt 等の混在文200程度。**アルファベット読み** (AI→エーアイ) と **単語読み** (NASA→ナサ) の両方を含む

### 4.3 未知語ロバスト性テスト

- **手法**:
  - 学習セットに含まれない Wikipedia 固有名詞 500件
  - 2024-2026 の新語 (辞書更新の遅れをシミュレート) 100件
  - 意図的に typo/変則表記を含めた 100件
- **指標**: PER + 出力の質的検査

### 4.4 効率評価

- **推論スループット**: sentences/sec on T4 GPU, RTX 4090, CPU
- **モデルサイズ**: MB
- **メモリ消費**: peak GPU memory during inference
- **レイテンシ**: 1文あたりの p50/p95 (msec)

---

## 5. データ組成の推奨レシピ

### 5.1 学習セット (推奨)

| ソース | 量 (目安) | 用途 |
|---|---|---|
| pyopenjtalk-plus 辞書全エントリ | ~800K | 単語レベル (surface, yomi, accent) 教師 |
| UniDic全エントリ | ~1M | 単語レベル (surface, yomi, accent) 教師 |
| Wikipedia日本語版 (ふりがな抽出) | ~500K sentences | 文レベル multi-task 教師 |
| 青空文庫 (ふりがな付き) | ~200K sentences | 稀な漢字・古語の cover |
| JSUT Basic5000 - 500 held-out | 4,500 sentences | 高品質人手ラベル |
| **Wikipedia日本語版 tech関連記事** | **~50K sentences** | **英単語混在文カバー** (新規) |
| **GitHub日本語README + Qiita/Zenn記事** | **~50K sentences** | **英単語+略語+記号混在文カバー** (新規) |
| **CMUdict + Kanalizer出力** | **~100K (英単語, カタカナ音写) ペア** | **英→カタカナ音写辞書** (新規) |
| **英字略語辞書** | **200件以上** | **AI→エーアイ 等の明示ラベル** (新規) |
| JVS nonpara - held-out除外 | 学習除外 | 評価専用 |

### 5.2 データクリーニング項目

- 全角/半角統一
- 数字表記のnormalization ('二〇二四' → '2024')
- URL/絵文字/機種依存文字の除去
- 音素表記のnormalization (X-SAMPA / IPA / JULIUS音素セットのどれかに統一)
- アクセント表記のnormalization (モーラ単位のH/L, 数値型アクセント核など)
- **英単語の判定・分節** (連続する latin 文字を1トークンとして扱う vs 略語分解)
- **英字略語 vs 単語判定** (大文字連続 + 短い → 略語候補、混合 case → 通常単語)
- **記号連結語の維持** (Wi-Fi, e-mail, T-shirt を分割しない)

### 5.3 音素表記の推奨

- **JULIUS音素セット** (pyopenjtalk がデフォルトで出力する形式) を第一候補にする。既存のTTS pipeline とドロップイン互換。
- **X-SAMPA** も選択肢 (多言語対応)
- **IPA**: 学術的だが、実装 friction が高い
- **カタカナ音素** (haqumei が採用): 実用性は高いが、外来語の細かい音素区別を失う

推奨: **JULIUS音素セット + モーラアクセントH/L + アクセント句境界マーカ '/'** の複合出力。Kurihara Interspeech 2024 と同じ形式。

### 5.4 データ拡張

- **辞書拡張による合成**: pyopenjtalk-plus 辞書の (surface, yomi) を文脈テンプレートに埋め込んで大量の文レベル教師を合成
- **多音字強化**: 多音字を含む文を人手で拡張し、正解ラベルを付与
- **LLM蒸留 (実験的)**: Claude/Geminiに kana変換させ、その出力を weak supervision として活用 (**licensingとhallucination risk に注意**)
- **音便・連濁パターン注入**: 「山田」→「やまだ」ではなく「やまた」等の変則読みを明示的に含める

---

## 6. ライセンスサマリー (実装前確認必須)

| リソース | ライセンス | 商用可 | 帰属 |
|---|---|---|---|
| JSUT音声 | CC-BY-4.0 | ○ | ○ |
| jsut-label (sarulab) | (要確認) | 要確認 | 要確認 |
| JVS音声 | (要確認) | 要確認 | 要確認 |
| ROHAN | CC-BY-4.0 | ○ | ○ |
| pyopenjtalk (r9y9) | 修正BSD | ○ | ○ |
| pyopenjtalk-plus | (要確認) | 要確認 | 要確認 |
| UniDic | BSD 3-clause 派生 | ○ | ○ |
| Open JTalk辞書 (NAIST-jdic) | 修正BSD | ○ | ○ |
| JMDict | CC-BY-SA-4.0 | ○ (Share-Alike) | ○ |
| Wikipedia日本語版 | CC-BY-SA-4.0 | ○ (Share-Alike) | ○ |
| 青空文庫 | 各作品ごと (多くPD) | ケースバイケース | ○ |
| sbintuitions/modernbert-ja | MIT | ○ | ○ |
| llm-jp-modernbert | (要確認 — 論文 arxiv 2504.15544) | 要確認 | 要確認 |
| Kokoro / Misaki | MIT (Kokoro), MIT (Misaki) | ○ | ○ |

**実装前に必ず一次ソースで再確認**。特に (a) 学習した重みの二次配布、(b) 商用API展開、(c) 学習ログや中間データの公開、が異なる条項に該当する場合がある。

---

## 7. 参考: Koriyama benchmark の詳細内訳

Koriyama Interspeech 2026 の 3,000文の特性 (arxiv 2606.22009):

- 音節数レンジ、文長分布などの詳細は論文の Table 参照
- **特筆すべきは 14.2% の数詞含有率** — 日本語G2Pにおいて数字の読み方 (千・万・億、助数詞連結、序数、日付、時刻、金額) が主要な課題であることを示す
- 固有名詞 (漢字6% + カタカナ8.5% = 計14.5%) が別軸の主要課題

**示唆**: 我々のロス設計で **数字カテゴリと固有名詞カテゴリに weighted loss** を適用することが、Koriyama benchmark で勝つための直接的な戦略になる。
