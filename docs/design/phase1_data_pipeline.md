# Phase 1 データパイプライン設計書

- Version: 1.0 (2026-07)
- Status: DESIGN / pre-implementation
- Owner: modern-bert-g2p project
- Companion docs:
  - `docs/research/03_datasets_and_benchmarks.md`
  - `docs/research/06_implementation_roadmap.md` §Phase 1
  - `data/hard_set_seed/schema.json`
  - `scripts/eval_haqumei_jsut.py`

---

## Executive summary

Phase 1 の目的は、ModernBERT-based マルチタスク G2P (Phase 3-5) を fine-tune するための正規化済み日本語コーパスを整備することである。5 種の一次ソース (pyopenjtalk-plus 辞書, UniDic-cwj 3.1.1, JMDict, Wikipedia 日本語版 HTML dump, 青空文庫) を、`scripts/eval_haqumei_jsut.py` と bit-一致する canonical 形式 (JULIUS 音素 + モーラ H/L + アクセント句境界 index) の Parquet shard に落とす。テスト系 (JSUT Basic5000 / JVS-3000 / ROHAN 4600 / hard-set 1400) との contamination は 4 層防御 (text-hash / char n-gram bloom / phoneme n-gram bloom / 文埋め込み cosine) でゼロを維持する。ライセンスは行単位で `license_tag` に保存し、release checkpoint は デフォルトで SA-free variant を採用する。

---

## 1. Data sources

| # | Source | URL | License (data) | 推定行数 | 抽出方式 | Status |
|---|---|---|---|---|---|---|
| S1 | pyopenjtalk-plus 辞書 | `https://github.com/tsukumijima/pyopenjtalk-plus` | BSD-3-Clause (mecab-naist-jdic 由来) + plus 追加分は同レポ `LICENSE` を参照 | 約 800K entries | ソース CSV を直接パース (`.dic` バイナリではなく) | to fetch |
| S2 | UniDic-cwj 3.1.1 (`lex.csv`) | `https://clrd.ninjal.ac.jp/unidic/` | **CC-BY-4.0 (辞書データ)**。コード側は GPL-2.0 / LGPL-2.1 / BSD の triple。行データは CC-BY-4.0 でタグする | 約 872K entries | `lex.csv` の 30 列から `lemma / kana / pron / aType / aConType` を採用 | to fetch |
| S3 | JMDict (EDRDG) | `https://www.edrdg.org/jmdict/edict_doc.html` | EDRDG License (CC-BY-SA-4.0-like, 帰属必須) | 約 200K entries | `JMdict_e.xml` の `<k_ele>` と `<r_ele>` を組み合わせ (漢字-かな) | to fetch |
| S4 | Wikipedia JA HTML dump | `https://dumps.wikimedia.org/other/enterprise_html/runs/` (`jawiki-NS0-YYYYMMDD-ENTERPRISE-HTML.json.tar.gz`) | CC-BY-SA-4.0 + GFDL | 期待抽出量 **50-150K sentences** (ルビ付き文のみ) | HTML 内 `<ruby><rb>...</rb><rt>...</rt></ruby>` の直接抽出 | to fetch |
| S5 | 青空文庫 | `https://github.com/aozorabunko/aozorabunko` (git mirror) | 作品ごと (Public Domain 主体、注釈は Aozora-CC-BY のケースあり) | 約 100-200K sentences | XHTML `<ruby>` 抽出 + 作品ヘッダから license 分岐 | to fetch |

**Snapshot pinning**: 全 source をダウンロード直後に `data/raw/<source>/<version>/` に凍結、`data/raw/manifest.yaml` に URL + SHA-256 + 取得日を記録する。Phase 1 完了時点で immutable。

---

## 2. License analysis

日本語 G2P 学習に使う 5 ソースはすべてライセンスが異なり、行単位で `license_tag` を保存し、export 時に filter する運用を採る。

**S1 pyopenjtalk-plus**: `tsukumijima/pyopenjtalk-plus` は upstream `r9y9/pyopenjtalk` の BSD-3-Clause を継承するが、`plus` の追加分と辞書スクリプトは同レポの `LICENSE` を一次資料として確認する (README は簡略化されており、実際の権利表記に差がある可能性)。tag は `BSD-3-Clause`。

**S2 UniDic-cwj 3.1.1**: 大規模辞書データ (lex.csv) は 2024 年以降 **CC-BY-4.0** で配布されている (NINJAL 公式ページの License Statement を参照)。過去バージョン 2.x のトリプルライセンス表記は主にコード (mecab plug-in) を対象としており、辞書エントリ自体は CC-BY-4.0 として扱う。`unidic-lite` (Apache-2.0) は別 artifact であり、こちらは軽量サブセットを使う場合のみ選択肢。tag は `CC-BY-4.0`。

**S3 JMDict**: EDRDG のライセンスは CC-BY-SA-4.0 互換だが厳密には EDRDG 独自条項 (帰属 + share-alike 相当)。tag は `EDRDG` として区別し、SA-safe export では除外可能にする。

**S4 Wikipedia JA**: CC-BY-SA-4.0 + GFDL のデュアルライセンス。学習データとして使うこと自体に制約はないが、"派生物" 判定が問題になる。tag は `CC-BY-SA-4.0`。

**S5 青空文庫**: 作品ごとに扱いが異なる。多くは著作権失効の Public Domain だが、注釈や工作員コメント (aozora contributors によるルビ・訳文) は `Aozora-CC-BY` が付くケースがある。作品ヘッダ (工作員コメント block) を parser で読み、注釈由来かどうかを line-level で判定する。判定不能行は `Unknown` とし、release checkpoint から除外する。

### 2.1 学習済み重みの再配布に関する立場 (§10.5 相当)

Wikipedia (CC-BY-SA-4.0) や JMDict (EDRDG) の学習データを含む重みを Apache-2.0 で配布可能とする立場は、Japanese StableLM / LLM-jp / RWKV / Gemma 等が採用する「モデル重みは訓練データの派生物ではない」という解釈に依拠する。この解釈は Creative Commons / FSFE / OSI が 2024-2025 に相次いで position statement を公表しているが、法的判例は未確定である。**本プロジェクトの release 方針**:

- **Default public checkpoint**: SA-free variant (`checkpoint_sa_free.parquet` から学習) を採用し、S3 + S4 を除外した subset で pretrain / fine-tune する。この checkpoint のみを Hugging Face Hub に公開する。
- **Research checkpoint**: S3 + S4 を含む full-data checkpoint は internal のみで保持し、公開は法務レビュー後の別リリースとする。ただし `training_data_report.md` で行数の transparency を確保する。
- **Commercial-safe checkpoint**: Public-Domain + MIT + BSD-3-Clause + CC-BY-4.0 のみ (S1 + S2 + S5-PD)。商用 fine-tune 用途に使う。

この 3 系統を Phase 5 の ablation で `sample_weight` 分布と精度差の観点から比較する。

---

## 3. Ingestion pipeline (per source)

**S1 pyopenjtalk-plus / S2 UniDic (辞書系)**: 単語単位で `surface / pron (kana) / accType (数値)` を直取り。1 単語 = 1 サンプルとして格納。`text = surface`、`phonemes` はカタカナ pron を JULIUS 音素列に 1-to-1 変換 (`kana2julius` table を Phase 1 実装で用意する)。`mora_accents` は accType (0 = 平板、1..N-1 = nucleus モーラ位置) から H/L 系列に展開: 平板 → `L H H ... H`、nucleus n → `L H H ... H (n-th) L L ... L`。単語エントリは文レベル context を持たないため `accent_boundaries = []`。

**S3 JMDict**: `<k_ele>` (漢字表記) と `<r_ele>` (読み) の cross product を取り、`re_restr` があれば限定する。アクセントラベルは JMDict に含まれないので `mora_accents` は空、`accent_boundaries = []`。この行は G2P main head の学習には使うが、アクセント系 head (APBP / ANPP / BAS) には mask する。

**S4 Wikipedia HTML dump**: `wikiextractor` は wikitext ベースなのでルビ回収率が < 10% と低い。代わりに **enterprise HTML dump** (2023 年以降月次) を使う。抽出フロー:
1. `.html.tar.gz` を stream 展開
2. 各 article HTML から `<ruby><rb>...</rb><rt>...</rt></ruby>` (Parsoid 出力) を BeautifulSoup + `lxml` で抽出
3. 段落単位で (surface span, kana annotation) タプルを収集し、段落全体の kana を合成
4. Ruby coverage ratio = ruby annotated char / total non-hiragana char を段落単位で計算、`< 0.8` の段落は `quality_flag: "low"` で隔離

期待抽出量は **50-150K sentences** (元設計の 500K は wikitext template 依存で過剰見積り。HTML dump からは 1 桁下)。

**S5 青空文庫**: git mirror を clone、各 XHTML の作品ヘッダを regex で切り出し、公開状態 (作品保護期間、注釈者情報) を判定。`<ruby>` タグから (surface, kana) を回収、旧字仮名は `kanjize` + `mojimoji` で正規化するが、baseline は raw を保持 (正規化は Phase 2 の switch で有効化)。作品単位で license を `Public-Domain` / `Aozora-CC-BY` / `Unknown` にタグ。

**文レベル mora accent (S4/S5 共通)**: `<ruby>` からは文レベル prosody は得られないため、pyopenjtalk-plus で強制 alignment を掛け、`extract_fullcontext` の `f2` (mora accent 1/2) と `xx_f2` から H/L と accent 句境界 index を生成する (noisy teacher label)。これは Phase 4 の hybrid architecture と同じ辞書 pseudo-label を訓練データに埋め込む形になるが、NHK Kurihara 2024 が BAS head を pseudo-label で訓練しているのと同じ設計思想である。

---

## 4. Normalization spec

canonical 形式は `scripts/eval_haqumei_jsut.py` と `data/hard_set_seed/schema.json` に既に確定している。Phase 0.5 で `src/modernbert_g2p/normalize.py` を切り出し、`eval_haqumei_jsut.py` から import する形に refactor する。これにより pipeline 側と評価側で再実装ドリフトが発生しない。

**正規化順序 (Phase 0.5 で単体テストを golden-diff 化)**:

1. NFKC 正規化 (全角/半角統一)
2. 半角記号・数字は残す (英大文字は case-fold しない — 略語判定で参照するため)
3. `IuPronunciation.Yuu` 規則の適用 (`iu` → `yuu`, haqumei と同じ実装を import)
4. Devoicing lowercase: `A E I O U` → `a e i o u` (`N` は保持)
5. `pau` 除去 (評価側の `ignore={"pau"}` と一致させるため、保存時に既に除去)

**Golden-diff test (Phase 0.5 前提条件)**: haqumei 0.8.0 で JSUT Basic5000 の先頭 1000 文を処理し、`normalize.py` の出力と bit-一致することを CI で確認する。差分が 1 サンプルでも出れば pipeline を止める。

**音素セット**: JULIUS 母音 `a i u e o`、子音 `k g s z j t d ch ts n h b p m y r w f v`、特殊 `N q sil pau`。長音は前母音 repeat (`kouko → k o u k o o`)。

**モーラ H/L の長さ不変式 (F6 対応)**: `phonemes` の長さと `mora_accents` の長さは一致しない (CV / CyV / N / q が 1 mora)。schema に **`mora_of_phoneme: list[int]`** を新設し、phoneme index → mora index の写像を明示保存する。これにより下流の BAS head training が `q` (促音) / 長音境界で silently misalign することを防ぐ。

**アクセント句境界**: `accent_boundaries: list[int]` は 0-indexed の mora 位置配列 (文字列 `/` は保存しない、schema 決定に従う)。

---

## 5. Deduplication & quality filter

### 5.1 Deduplication

**Layer A (exact)**: `(source, text_hash, phoneme_hash)` の tuple を SHA-1 で hash 化。異 source 間で重複した場合は 1 サンプル残し、他 source を `dup_of` フィールドで参照 (license 情報を失わないため)。

**Layer B (near-dup, S4/S5 のみ)**: MinHash + LSH (`datasketch`) で char 5-gram Jaccard ≥ 0.8 を検出。同時に phoneme 5-gram Jaccard ≥ 0.9 を **AND 条件**で課すことで、表記違い・音は同じ (定型引用) を dedup、音は違うが同じ話題は残す。閾値と recall は Phase 1 の pilot **10K サンプル** (元設計の 1K からスケールアップ、F の minor 指摘) で precision > 0.95 かつ recall > 0.80 を確認する。

```
pseudocode:
  mh = MinHash(num_perm=128)
  for gram in char_5grams(text): mh.update(gram)
  candidates = lsh.query(mh)
  for c in candidates:
      if jaccard_text(text, c.text) >= 0.8 and jaccard_phoneme(ph, c.ph) >= 0.9:
          mark c as near-dup of shorter side
```

### 5.2 Quality filter

- 文長: 4 mora 以上 300 mora 以下 (単語辞書 entry は 1 mora からも受け入れ)
- 文字 whitelist: CJK Unified Ideographs (U+4E00-9FFF + Extension A/B) + ひらがな + カタカナ (半角含む) + Basic Latin + 常用記号 (`、。「」『』・ー〜（）()／/,.-:;?!`)
- Punctuation policy: 文末 `。` `.` は保持 (音素は `pau → 除去`)、連続記号 (`!?`, `……`) は 1 個に折り畳む
- Mora 一致 assertion: `len(mora_accents) == count_moras(phonemes)`。不一致は自動修復せず廃棄
- `mora_of_phoneme` の consistency: `max(mora_of_phoneme) + 1 == len(mora_accents)` を assertion

---

## 6. Split strategy & contamination prevention

### 6.1 Splits

| Split | Row 数 | 用途 |
|---|---|---|
| train | 全データの約 96% | 学習 |
| val | 20K サンプル固定 | early stopping / hyperparam |
| dev-test | 10K サンプル固定 | Phase 2-5 ablation 公開比較 |
| held-out | 別リポジトリで管理 | Phase 6 直前まで一度も見ない |

### 6.2 Contamination check (4 層防御)

JSUT Basic5000 / JVS-3000 (台本、`Hiroshiba/jvs_r9y9` fork を一次資料とする) / ROHAN 4600 / hard-set 1400 の全文を blocklist 化し、train + val + dev-test から除外する。

**Layer 1 — Text-hash blocklist**: NFKC + 空白折り畳み + 半角化した後 SHA-1。`blocklist_text.sha1` に格納。文単位で `sha1 in blocklist` reject。False-positive を厭わない設定。

**Layer 2 — Char n-gram bloom**: char 12-gram (文長 12 未満は 8-gram fallback) を bloom filter に格納。1 サンプルに 3 個以上 hit した場合 reject。表記ゆれ (改行位置、記号違い) を吸収する第 2 防衛線。

**Layer 3 — Phoneme n-gram bloom**: JSUT/ROHAN の `phone_level3` を分割した音素 20-gram を別 bloom filter に格納。pipeline の phoneme 出力に対し 3-hit で reject。数字表記違いなど text が変わっても音が同じ contamination を検出。

**Layer 4 — Sentence embedding cosine (F5 対応)**: char-Jaccard ≥ 0.6 で Layer 2 を通過しなかった候補に対し、`cl-nagoya/ruri-large` (Apache-2.0) で sentence embedding を計算し、cosine ≥ 0.92 を **human-review queue に投入** (auto-reject しない)。synonym-rewrite 型の contamination を検出。閾値は Phase 1 の 500 サンプル手作業で precision > 0.85 を確認、over-filter リスクを避けるため人手承認前提。

**CI 統合**: `scripts/check_contamination.py` を CI に組み込み、snapshot 生成ごとに自動実行、hit 件数と例文を `contamination_report.md` に記録。Layer 1-3 は 0 件維持が release criteria。Layer 4 は human-review queue が空であることが release criteria。

### 6.3 Hard-set contract

Hard-set 1400 (7 カテゴリ × 200) を Phase 1 に生成 (現在 `data/hard_set_seed/` に 21 サンプル済)。全 `text` を Layer 1 blocklist に追加し、train/val/dev-test から除外。hard-set の gold label 作成は human curation を必須とする (LLM 生成は禁止、hallucination risk 排除)。

---

## 7. Sample weighting (Phase 2 の実験変数として扱う)

CLAUDE.md の指示は `sample_weight = 2.0` の推奨だが、Hida ICASSP 2022 は uniform weighting を採用しており、2.0 の根拠となる文献は存在しない (F7)。したがって **Parquet schema には `category: list[str]` のみを保存し、`sample_weight` はスキーマから外す**。Phase 2 の pilot で weights ∈ {1.0, 1.5, 2.0, 3.0} × category-hard subset の grid search を実施し、empirical に選定する。学習時に `Dataset.map(compute_weight_at_load_time)` で on-the-fly 計算する。

`category` の推定ヒューリスティクス (multi-label 可):

- `numeric_unit`: `\d+(\.\d+)?(km|kg|GB|MB|円|人|年|月|日|:|時|分)` 相当の regex
- `proper_noun_kanji`: 辞書 POS が固有名詞、かつ表記が漢字を含む
- `proper_noun_katakana`: 同条件でカタカナ表記
- `counter`: 助数詞 (`個|匹|本|台|冊|名|回|…`) が直前数字と共起
- `loanword`: カタカナ連続 3 モーラ以上 かつ 辞書 POS = 名詞・一般
- `english_mixed`: Basic Latin 単語を 1 つ以上含み、全体が英字のみでない
- `english_abbreviation`: 全大文字 Latin 連続 2-6 字 (`AI`, `PDF`, `NASA`)

---

## 8. Output schema (per column)

| Column | Type | Description |
|---|---|---|
| `id` | string | Source prefix + 連番 (例 `wiki-0000123456`) |
| `source` | enum | `pyopenjtalk_plus | unidic | jmdict | wikipedia | aozora` |
| `text` | string | 正規化後の surface (§4 の 5 手順適用済み) |
| `phonemes` | list[string] | JULIUS 音素列 (§4)、`pau` 除去済み、devoicing lowercase 済み |
| `mora_accents` | list[string] | `"H" \| "L"` の per-mora 系列。長さ = mora 数 |
| `accent_boundaries` | list[int] | 0-indexed mora 位置での句境界配列 (単一句なら空) |
| `mora_of_phoneme` | list[int] | phoneme index → mora index (F6 修正、BAS head 用) |
| `categories` | list[string] | §7 のカテゴリ (multi-label)。sample_weight は保存しない |
| `license_tag` | enum | `BSD-3-Clause \| CC-BY-4.0 \| EDRDG \| CC-BY-SA-4.0 \| Public-Domain \| Aozora-CC-BY \| Unknown` |
| `quality_flag` | enum | `ok \| low \| suspect` |
| `provenance` | struct | `{"snapshot": str, "ruby_coverage": float, "line_offset": int}` |
| `dup_of` | string \| null | Layer A 重複時に primary サンプルの `id` |

Schema は `data/hard_set_seed/schema.json` を親として draft-2020-12 で拡張、`$ref` で共通部分を再利用する。

---

## 9. Storage decision

**Primary**: **Parquet-per-shard** のみ。source ごとに 1 shard = 100K rows、Snappy 圧縮、`pyarrow` mmap 読み込み。学習時は `datasets.load_dataset("parquet", data_files=..., streaming=True)`。

**JSONL は生成しない**: 元設計の "Parquet + JSONL.gz 並置" は 1.5-2× のディスク重複と維持コストが発生する (F9)。代わりに `scripts/parquet_to_jsonl.py` を用意し、hard-set curation UI や grep デバッグが必要な時に on-demand で JSONL 生成する。

**HF Datasets**: wrapper として `datasets.Dataset.from_parquet(...)` を必要時のみ使用。別ファイル `.arrow` は作らない (I/O 帯域律速回避)。

**Directory layout**:
```
data/
  raw/<source>/<version>/          # snapshot pinned, immutable
  interim/<source>/                # 中間 (ruby extraction, kana2julius 前)
  curated/parquet/<source>/*.parquet  # ← primary、学習で読むのはここのみ
  curated/manifest.yaml            # SHA-256, row 数, license 集計
  contamination_report.md
  training_data_report.md
```

---

## 10. Open questions / deferrals to Phase 1 execution

以下は本設計書では確定せず、Phase 1 実装時に決着する:

- **Q1 (F4 前提)**: `normalize.py` を Phase 0.5 の作業として haqumei から抽出可能か、それとも Phase 1 冒頭にずらすか。golden-diff test の CI 化タイミング。
- **Q2**: pyopenjtalk-plus の `plus` 追加分ライセンスが upstream と一致するかどうか、レポの `LICENSE` を一次確認する必要。
- **Q3**: Wikipedia HTML dump の実際のルビ回収量 (50K か 150K か) は 1 shard 展開して pilot で確定する。150K を大きく下回る場合、Phase 2 で LLM-based ruby augmentation (別倫理レビュー要) を検討。
- **Q4**: 青空文庫の注釈者由来ルビの license 判定 heuristics (作品ヘッダ parser の精度) は 100 作品の手作業 audit で確定する。
- **Q5**: Layer 4 (sentence embedding) の human-review queue 運用手順 (承認/棄却の記録方法、reviewer の資格)。
- **Q6**: MinHash pilot (10K サンプル) の precision/recall 目標に達しない場合の閾値調整戦略。
- **Q7**: JVS-3000 台本の一次資料としては `Hiroshiba/jvs_r9y9` fork を採用するが、original `takamichi/jvs_corpus` との文差分を audit する必要 (元設計の `nnsvs/jvs-corpus` は誤引用、修正済み)。
- **Q8**: 学習データ総量が 500M token を下回った場合の augmentation 戦略 (dictionary combinatorial expansion, template-based synthesis)。
- **Q9**: `mora_of_phoneme` の生成ロジックを pyopenjtalk `f2` から自動導出するか、独自 mora-splitter で post-hoc に付与するかの選択。前者は pipeline 依存度が上がる。
- **Q10**: 3 系統 checkpoint (SA-free / research / commercial-safe) の Phase 5 ablation 予算 (GPU-hour) の見積り。

以上を Phase 1 実装計画 (`superpowers:writing-plans` を本文書を親として起動) で個別 task に分解し、TDD で 1 モジュールずつ埋めていく。
