# Phase 0 Kickoff Checklist — ModernBERT日本語G2P

**作成日:** 2026-07-03
**目的:** Phase 0 (ベースライン再現、1週間) 開始前および Phase 0 中に事前調査・検証すべき項目の全体リスト。
**根拠:** `requirements.md` v1.3 と、Phase 0 blocker 並列調査 (V/D/L/A) の結果を統合。
**運用:** 各項目に status (✅完了 / 🔧進行中 / ⏳P0-P1中に処理 / ⏸P2以降で処理 / ⚠️要確認)、責任者、期限を付ける。

---

## 使い方

- 🔧 と ⚠️ の項目は Phase 0 Week 1 の kickoff meeting で必ずレビュー
- ✅ の項目は `requirements.md` のどの要件で対応しているかリンク
- Phase gate (AC-P0〜P6) を判定するときに、このチェックリストの残項目を再確認

---

## 1. Vast.ai インフラ (V-01〜V-08)

primary クラウド: Vast.ai。credit balance $1,478.79 (2026-07-03 実測)。

| # | 項目 | status | 詳細 |
|---|---|---|---|
| V-01 | RTX 4090 24GB 相場実測 | ✅ | 最安 $0.136/hr, p50 $0.354/hr, 50+ offers 利用可 (2026-07-03 API直接測定) |
| V-02 | RTX 3090 / A100 / H100 相場 | ✅ | RTX 3090: min $0.116/hr, A100 40GB: min $0.402/hr, H100 SXM: min $1.60/hr |
| V-03 | interruptible instance 中断時 checkpoint 保存パターン | 🔧 | Phase 0 中に実測: 5〜10h の学習 job を bid instance で走らせて中断復旧が成立するか。checkpoint 頻度と復旧スクリプト書式を確定 |
| V-04 | Docker image (PyTorch 2.1+ / CUDA 12.1+ / FA2 プリインストール) | 🔧 | Phase 0 中に確定: HuggingFace 公式 `huggingface/transformers-pytorch-gpu` or Vast.ai template `pytorch-2.1.0-cuda12.1-cudnn9-devel` を primary に |
| V-05 | 永続ストレージ (instance disk vs 外部 volume) の料金と容量 | ⚠️ | Vast.ai の disk_space 相場 (offers 内で $/GB確認)。100GB規模のデータを保持するコスト |
| V-06 | HF Hub からのデータ pull 帯域制限とコスト | ⚠️ | Vast.ai の inet_down は Mbps 表示。大量データ pull で追加課金される可能性を確認 |
| V-07 | region/GPU 型番の再現性確保 | ⚠️ | 同一実験の再現に同じ machine_id を指定できるか。cattr で `machine_id` filter 可 |
| V-08 | checkpoint 外部保存 (HF Hub / S3 / W&B) の帯域テスト | ⚠️ | 5GB checkpoint を Vast.ai → HF Hub にアップロードする時間 |

**関連要件**: NFR-60〜64, CR-52〜55

---

## 2. データセット取得・実物検証 (D-01〜D-10)

| # | 項目 | status | 詳細 |
|---|---|---|---|
| D-01 | JVS-3000 kana アノテーション入手 | ✅ | **完了 2026-07-03**: `git clone https://github.com/CyberAgentAILab/jvs_nonpara_kana` 成功、3,000文の `jvs_nonpara_kana.csv` + `eval_cer.py` を local 配置済み。CC-BY-SA-4.0、eval only 分離 |
| D-02 | JSUT Basic5000 + jsut-label 実物ダウンロード確認 | ✅ | **完了 2026-07-03**: `git clone https://github.com/prj-beatrice/jsut-label` 成功、5,000文 (`text_kana/basic5000.yaml`) を local 配置済み。**重要**: `text_level2` フィールドを使うこと (haqumei-eval と同じ) |
| D-03 | ROHAN 4600 実物確認 | 🔧 | `git clone https://github.com/mmorise/rohan4600`。license 再確認 (`Rohan4600_transcript_utf8.txt` は haqumei-eval 側に同梱、上流 URL 未検証) |
| D-04 | pyopenjtalk-plus 辞書バージョン確定 | 🔧 | `tsukumijima/pyopenjtalk-plus` の最新 tag を pin。haqumei が使用している tag と一致させる |
| D-05 | UniDic 版数選定 | 🔧 | UniDic-CWJ 2.3.0 vs 3.1.0 のどちらか、haqumei-eval が使用するのと同じ版に統一 |
| D-06 | Wikipedia 日本語版 dump の日付固定 | 🔧 | `jawiki-20260601-articles.xml.bz2` 相当。ふりがな抽出パイプライン (`wikiextractor` or 独自) を Phase 1 で選定 |
| D-07 | 青空文庫 ふりがな付きテキスト抽出 | 🔧 | 既存 OSS: `aozora-corpus-generator`, `aozorabunko-clean` を候補調査。抽出結果を CI で pin |
| D-08 | 英日混在コーパス最低5万文のソース確定 | 🔧 | Wikipedia tech記事 / Qiita 記事 / Zenn 記事 の各々の入手 URL とライセンス。Phase 1 で確定 |
| D-09 | CMUdict × Kanalizer 100K ペア生成手順 | 🔧 | VOICEVOX/kanalizer-dataset (118k, CC0相当) を採用可能。CMUdict-0.7b + Kanalizer inference で生成 |
| D-10 | 英字略語辞書 200件以上のソース | 🔧 | 既存OSS辞書 (Wikipedia の「日本語で使われる英字略語」ページ、日本国語大辞典 略語表) を Phase 1 で集約 |

**関連要件**: CR-10〜16, CR-26〜28

---

## 3. ベースライン再現 (B-01〜B-05, AC-P0 の中核)

Phase 0 Week 1 で完了する必要があるタスク。

| # | 項目 | status | 詳細 |
|---|---|---|---|
| B-01 | haqumei 実物動作確認 | ✅ | **完了 2026-07-03**: `uv pip install haqumei==0.8.0` で macOS aarch64 wheel 成功。Python API `Haqumei(use_unidic_yomi=True, normalize_iu=IuPronunciation.Yuu).g2p_batch(texts)` で全動作確認 |
| B-02 | **haqumei で JSUT Basic5000 PER 1.17% を実測再現** | ✅ | **完了 2026-07-03**: 実測 **PER 1.1657%** (S=2107, D=540, I=825, N=297843) vs 官報 **1.17%** (S=2117, D=527, I=831, N=297843)。**Diff 0.0043 pt** で ±0.1% 以内、**AC-P0 PASSED**。再現スクリプト: `scratchpad/haqumei_repro/eval_jsut_v3.py`。重要な学び: (a) `text_level2` を使う (`text_level0` ではない、153文差)、(b) 大文字母音 A/E/I/O/U (無声化) は小文字化して比較、(c) N (撥音) は保持 |
| B-03 | **pyopenjtalk で JVS-3000 kana CER 1.03% を実測再現** | 🔧 | `jvs_nonpara_kana` の `eval_cer.py` に pyopenjtalk 出力を渡して測定。**±0.1% 以内一致** が AC-P0 要件 |
| B-04 | フロンティア LLM (Gemini 3.1 Pro / Claude Opus 4.6) の JVS-3000 CER 再現 | ⚠️ | Google AI Studio / Anthropic API の課金アカウント準備。Stretch target なので Phase 6 直前でも可 |
| B-05 | 全 3 baseline の測定を 1コマンドで再生成 | 🔧 | `scripts/eval_baselines.sh` を Phase 0 の成果物として作成 |

**関連要件**: AC-P0, NFR-01〜03, NFR-05〜06

---

## 4. トークナイザー 3並列パイロット (T-01〜T-05)

Phase 2 で決着させる。**Phase 0 では準備のみ**。

| # | 項目 | status | 詳細 |
|---|---|---|---|
| T-01 | ModernBERT SentencePiece の漢字トークン化パターン確認 | ⏳P2 | 漢字 1文字が 1 token になるか、複合語がどう分割されるか |
| T-02 | MeCab + UniDic 環境構築 | ⏳P2 | `fugashi[unidic]` の pip install で確定 |
| T-03 | `tohoku-nlp/bert-base-japanese-char-v2` の下流精度取得 | ⏳P2 | HF card から公表値取得 (v1.2 で CC-BY-SA-4.0 判明 → weights 配布は避け対照モデルとしてのみ使用) |
| T-04 | char-level BERT の 8k context 対応可否 | ⏳P2 | ModernBERT の RoPE + FA2 が char-level にも適用できるか |
| T-05 | seq2seq decoder 設計候補選定 | ⏳P2 | byT5-small / T5-base / 自作 の3案から Phase 2 で確定 |

**関連要件**: NFR-33〜35, CR-03, OPEN-03

---

## 5. 評価パイプライン canonical 定義 (E-01〜E-06)

| # | 項目 | status | 詳細 |
|---|---|---|---|
| E-01 | JULIUS 音素セットの canonical 定義 | 🔧 | Julius phone-set: `/usr/share/julius-4.5/model/phone_m/ja_1phone`。pyopenjtalk と一致するか確認 |
| E-02 | PER 実装 reference | 🔧 | `jiwer` / `torchmetrics.WordErrorRate` / haqumei-eval の Rust 実装 の3候補を head-to-head 比較 |
| E-03 | kana CER の canonical 定義 (Koriyama 2026 の測定方法) | 🔧 | `jvs_nonpara_kana/eval_cer.py` (長音記号バリアント正規化ロジック込み) を採用 |
| E-04 | mora accuracy 実装 (Hida 2022 と同じ算出式) | 🔧 | Hida arxiv:2201.09427 の実装が公開されているか。無ければ論文アルゴリズムを実装 |
| E-05 | ROHAN KER canonical 実装 | ✅ | haqumei-eval の `g2k_per_word` 文字単位 Levenshtein を採用 (CR-27) |
| E-06 | pyopenjtalk 互換テスト 1,000文サンプル選定基準 | 🔧 | 辞書ヒット率が高い一般文 (Wikipedia 平均文長) を bootstrap sample。CI で pin |

**関連要件**: FR-35, NFR-70〜73, CR-27

---

## 6. マルチタスク・ヘッド設計 (M-01〜M-04)

Phase 3 で確定。Phase 0 では要件のみ確認。

| # | 項目 | status | 詳細 |
|---|---|---|---|
| M-01 | 5-head loss weight 初期値 | ⏸P3 | Hida 2022 の値を継承 (G2P:1.0, polyphone:0.5, APBP:0.3, ANPP:0.3, BAS:0.5) |
| M-02 | Curriculum learning 順序 | ⏸P3 | G2P → APBP → ANPP → BAS の順で warmup |
| M-03 | BAS head 入出力 spec | ⚠️P3 | NHK Kurihara 2024 の実装詳細未公開。arxiv preprint 再読 |
| M-04 | Polyphone クラス定義 | ⏸P3 | 多音字辞書のスコープを Phase 1 で構築 |

**関連要件**: FR-10〜14, FR-50〜54

---

## 7. 法務・ライセンス (L-01〜L-06)

| # | 項目 | status | 詳細 |
|---|---|---|---|
| L-01 | Wikipedia CC-BY-SA-4.0 の重み継承判定 | ✅ | Andersen v. Stability 判決 + CC 2025 primer + 日本著作権法 30条の4 で Apache-2.0 配布可 (先例: Japanese StableLM, LLM-jp-3) |
| L-02 | JMDict CC-BY-SA-4.0 の同上判定 | ✅ | runtime lookup のみに限定 (Misaki 先例)、gradient に含めない → SA リスク完全消去 |
| L-03 | 最終ライセンス確定 | ✅ | **Apache-2.0** 確定 (patent grant / 下流 OSS TTS 互換 / ModernBERT 系整合) |
| L-04 | JSUT/ROHAN の CC-BY-4.0 帰属明示 | 🔧 | Model Card テンプレートに `## Data Attribution` セクションを追加、Phase 6 直前で確定 |
| L-05 | pyopenjtalk-plus 辞書 (BSD派生) の帰属明示 | 🔧 | 同上、Model Card に統合 |
| L-06 | 個人情報保護法 (APPI) 準拠レビュー | ⏸P6 | 法務レビュー、Phase 6 直前 |

**関連要件**: CR-20〜25, CR-80〜83

---

## 8. アノテーション体制 (A-01〜A-04)

| # | 項目 | status | 詳細 |
|---|---|---|---|
| A-01 | Hard-set 1,400 文キュレーション体制 | ✅ | LLM 半自動 + 人手 diff review (C案)、2〜4営業日、API 費 <¥5,000 |
| A-02 | アノテーションガイドライン | 🔧 | JULIUS 音素セット + モーラアクセント H/L + アクセント句境界 '/' の canonical rule を Phase 1 冒頭でドキュメント化 |
| A-03 | アノテーションツール選定 | 🔧 | Google Sheets or Doccano (C案では LLM 出力の diff review が主なので Sheets で十分) |
| A-04 | ダブルチェック体制 | 🔧 | Phase 1: 音声学専門家 (東工大郡山研 / 東大齋藤研) に 100 文サンプル監修依頼 (謝金 ¥3〜5万) |

**関連要件**: CR-14, NFR-10〜17

---

## 9. ダウンストリーム統合テスト (I-01〜I-03, AC-04)

| # | 項目 | status | 詳細 |
|---|---|---|---|
| I-01 | Style-Bert-VITS2 バージョン特定 | 🔧 | `litagin02/Style-Bert-VITS2` の最新 tag (2026-07 時点) を pin。JP-Extra を primary 対象 |
| I-02 | 統合テスト用音響モデル準備 | 🔧 | SBV2 pretrained model (JVS speaker 等) を HF Hub から入手 |
| I-03 | TTS pronunciation CER 測定方法 | 🔧 | Whisper-large-v3 を用いた自動 CER 測定 (subjective MOS ではなく automatic) |

**関連要件**: AC-04, FR-30〜35

---

## 10. インフラ・監視・実験管理 (X-01〜X-05)

| # | 項目 | status | 詳細 |
|---|---|---|---|
| X-01 | Weights & Biases のプラン | 🔧 | free tier で 100 experiments/月 → Ablation 60 runs 内で足りるか。paid ($50/月) に切替オプション |
| X-02 | ONNX 推論エンジンの精度差測定 | ⏸P6 | PyTorch → ONNX 変換後の PER 差分測定 (0.1pt 以内を要件) |
| X-03 | Model Card template 選定 | 🔧 | HuggingFace 標準 template を採用。CC-BY-SA-4.0 学習データ由来の法的立場テンプレを含む |
| X-04 | GPU-less regression CI 設計 | 🔧 | GitHub Actions の CPU runner で 100文 mini-eval を実行、精度差 0.3pt で reject (NFR-72) |
| X-05 | CHANGELOG generation 自動化 | ⏳P4 | `release-please` (GitHub Actions) を採用。SemVer 準拠のリリースフロー |

**関連要件**: NFR-70〜73, FR-44〜45, CR-70〜72

---

## 11. Phase 0 Week 1 の必達タスク (優先順)

以下は Phase 0 の 1週間 で完了する必要がある:

1. **[Day 1-2]** Vast.ai の interruptible instance 動作確認 (V-03) と Docker image 選定 (V-04)
2. **[Day 2-3]** データセット実物取得 (D-01, D-02, D-03) → local に配置
3. **[Day 3-4]** haqumei / pyopenjtalk baseline 再現 (B-01, B-02, B-03)
4. **[Day 4-5]** 全 3 baseline 測定を 1 コマンド化 (B-05) → `scripts/eval_baselines.sh`
5. **[Day 5-7]** Hard-set 用の gold seed set 140文 (A-01 の事前作業) を開発者自身で作成
6. **[Week 1 末]** AC-P0 判定: haqumei PER 1.17% と pyopenjtalk kana CER 1.03% がそれぞれ ±0.1% 以内で再現できたら Phase 1 へ進む

---

## 12. 変更履歴

| 版 | 日付 | 変更内容 |
|---|---|---|
| 1.0 | 2026-07-03 | 初版。requirements.md v1.3 と Phase 0 blocker 調査結果を統合してチェックリスト化 |

---

## 13. トレーサビリティ

| チェックリスト項目 | 対応する要求定義 |
|---|---|
| V-01〜V-08 | NFR-60〜64, CR-52〜55 |
| D-01〜D-10 | CR-10〜16, CR-26〜28 |
| B-01〜B-05 | AC-P0, NFR-01〜03, NFR-05〜06 |
| T-01〜T-05 | NFR-33〜35, CR-03, OPEN-03 |
| E-01〜E-06 | FR-35, NFR-70〜73, CR-27 |
| M-01〜M-04 | FR-10〜14, FR-50〜54 |
| L-01〜L-06 | CR-20〜25, CR-80〜83 |
| A-01〜A-04 | CR-14, NFR-10〜17 |
| I-01〜I-03 | AC-04, FR-30〜35 |
| X-01〜X-05 | NFR-70〜73, FR-44〜45 |
