# Phase 2: トークナイザー3並列パイロット head-to-head 設計書

**Status**: DRAFT v2 (design-only, 実装前 / 敵対的 critique 反映済み)
**Location**: `docs/design/phase2_tokenizer_pilots.md`
**Depends on**: `CLAUDE.md` 設計思想 1〜4, `docs/research/05_technical_design.md §1-2`, `docs/research/07_nn_only_benchmarks.md §2.1-2.4`, Phase 0 baseline (haqumei JSUT PER 1.1657%, pyopenjtalk JVS-3000 kana CER 1.0874%, haqumei ROHAN KER 1.6397%)

---

## 1. Executive summary

Phase 2 の中核判断は「G2P タスクをどのトークン粒度で解くか」である。SB Intuitions 自身が modernbert-ja HF カードで警告する「SentencePiece boundary が morpheme と一致せず token classification 性能が悪い」問題を回避するため、**seq2seq (P-A) / MeCab pretokenize + `[MORPH]` boundary token 挿入 (P-B) / char-level BERT (P-C)** の 3 パイロットを同一データ・同一 split・複数 seed で head-to-head 比較する。決定は N=3 seed 上の bootstrap CI 付き JSUT PER + hard-set per-category + multi-head 適合性スコアカードで多段判定する。exit 基準は「pure-NN で haqumei を上回る」ではなく (`docs/research/07_nn_only_benchmarks.md` の失敗パターンから明確に非現実的)、「haqumei との gap を 50% 以上閉じるか、既存 pyopenjtalk-plus 辞書 fallback を merge した hybrid で haqumei を絶対値で下回る」の 2 択とする。Winner 1 pilot が Phase 3 の multi-task 拡張のメインラインに昇格、残る 2 pilot は再現可能な recipe を残して archive する。

---

## 2. 共通評価プロトコル (三 pilot 共通)

Phase 2 全 pilot は下記を **バイト単位で一致** させる。差が出るのは §3 に列挙する pilot-固有の入出力表現のみ。

| 項目 | 値 |
|---|---|
| Seed set | `{20260704, 20260705, 20260706}` (N=3, Phase 2 kickoff 週の日付列を種) |
| Train/Val/Test split | Phase 1 統合コーパスを `85/5/10` で分割、index を `data/splits/phase2_v1/{train,val,test}.idx` に固定 |
| Statistics 型式 | 3 seed の PER 平均 + bootstrap 95% CI (10,000 resample、`scipy.stats.bootstrap`) |
| Hard-set 1400 | `data/hard_set_v1/` = 7 カテゴリ × 200 (Phase 1 exit 依存、§10 open Q ①) |
| 主要ベンチ | JSUT Basic5000 PER, JVS-3000 kana CER, ROHAN 4600 KER (Phase 0 パス流用) |
| 音素表記 | JULIUS 音素セット + モーラアクセント H/L + アクセント句境界 `/` (canonical, `05_technical_design.md §4`) |
| Compute env | uv + Python 3.12, bf16, Flash-Attn on, CUDA 12.x |
| GPU (dev) | RTX 4090 24GB × 1 (spot A6000 48GB fallback) |
| GPU (final run) | H100 80GB × 1 |
| Val eval 頻度 | 全 2000 step、best val PER で checkpoint 保存 |
| Log | wandb + tensorboard、`reports/phase2/{pilot}/{seed}/` にダンプ |

**Canonical form 変換**: 3 pilot は生出力を canonical (JULIUS phonemes + H/L + `/`) に変換してから PER 計算する。変換ルールは §3 の pilot 別出力仕様表で決定的に定義され、pilot 間で有利/不利を作らないよう **単体テストで境界一致を検証** する (train 開始前の必須ゲート)。

---

## 3. パイロット詳細設計

### 3.1 P-A: seq2seq (推奨主軸)

**動機**: NHK Kurihara Interspeech 2024 TJ-G2P と同型。SentencePiece 境界問題を「decoder が音素側で解決する」ことで根本回避。`07_nn_only_benchmarks.md §2.2` の Kakegawa TJ-G2P alone が pure-NN で OpenJTalk に敗北 (11.85% vs 10.82%) している事実は認識、P-A はそれを Phase 4 hybrid merge で挽回する前提。

```
raw text
  |-- SentencePiece (modernbert-ja tokenizer, vocab 102400)
  v
  ModernBERT encoder (130M, hidden 512, 22 layers, 8k ctx)
  v
  Transformer decoder (6 L, hidden 512, 8 heads, FFN 2048)
  v
  phoneme vocab head (vocab ~ 68)
```

**Input encoding**: raw text → SentencePiece → 512 subword (Phase 2 は文単位、8k は Phase 5 で解禁)。
**Output encoding**: JULIUS phoneme token に `H`, `L`, `/`, `<bos>`, `<eos>`, `<pad>` を追加して phoneme vocab 68。音素と直後のモーラアクセントはインタリーブ (例: `w a t a H sh i L / w a L`)。`/` はアクセント句境界。
**Head params**: decoder 約 22M、total trainable ≒ 154M。
**Loss**: cross-entropy + label smoothing 0.1、ignore_index = `<pad>`、teacher forcing 100%。カテゴリ reweighting は `sample_weight = 2.0` を数詞/固有名詞/助数詞/外来語/英単語/略語に適用。
**Training recipe**:

| Hyperparameter | 値 |
|---|---|
| Batch size (effective) | 256 pair (grad accum) |
| LR (encoder / decoder+head) | 3e-5 / 1e-4 |
| Warmup steps | 2000 |
| Total steps | 60k (≈ 3 epoch on 1.5M pair, §7 で GPU-h 見直し済み) |
| LR schedule | linear decay to 0 |
| Weight decay | 0.01 (encoder), 0.0 (decoder norm/embed) |
| Grad clip | 1.0 |
| Optimizer | AdamW β=(0.9, 0.999) |
| Dropout | 0.1 encoder / 0.1 decoder |
| Precision | bf16 |

**Sub-ablation** (P-A 内 2-arm): (a) decoder from-scratch (推奨) / (b) `retrieva-jp/t5-base-japanese` decoder init。ライセンスは §10 open Q ④ で要検証 (CC-BY-SA-4.0 の可能性、Share-Alike 感染懸念 — その場合は init 抜きで from-scratch のみに縮小)。
**Inference**: beam=4, coverage penalty 0.2, length norm α=0.6。

### 3.2 P-B: MeCab pretokenize + `[MORPH]` boundary token 挿入 + ModernBERT + token classification

**動機**: Hida ICASSP 2022 の multi-head 構造に最も近い。ただし critique §3 で指摘された通り、単に MeCab pretokenize + subword mean-pool しても encoder 内部は SentencePiece のまま attend するため、SB Intuitions 警告に対する mitigation として **不十分**。本設計では **形態素境界を明示的な学習可能特殊 token `[MORPH]` として SP 列に挿入**し、encoder 内 self-attention に境界情報を注入する路線を採用する。

**Input encoding (確定案)**:

1. raw text → MeCab (UniDic-3.1.1 + pyopenjtalk-plus 追加語彙、Phase 0 で導入済み) で morpheme 列を得る
2. 隣接 morpheme の間に特殊 token `[MORPH]` を挿入 (`[CLS] tok_1 [MORPH] tok_2 [MORPH] ... [SEP]`)
3. `[MORPH]` は modernbert-ja tokenizer に **新規追加**、embedding は Xavier init。resize_token_embeddings は encoder fine-tune 時に一緒に更新
4. 各 morpheme の representation は「その morpheme を構成する subword の mean pool + 直後の `[MORPH]` の hidden の concat (dim 1024)」

なお critique で「(b) morpheme-id embeddings を subword embeddings に加算」も検討したが、実装工数がほぼ同等で解釈可能性が劣る (どの層で morpheme 情報が使われたか可視化しづらい) ため、Phase 2 では option (a) のみを走らせる。

**Output encoding**: 各 morpheme 位置に対して、対応する canonical phoneme + H/L 列を予測。アクセント句境界 `/` は「前 morpheme 末で APBP head が B ラベル → canonical 化時に `/` 挿入」というルールで決定的に投影する (APBP head は Phase 2 では BIO の linear head を 1 つだけ載せ、Phase 3 の本格 multi-head 学習は後回し)。

**Head 選択肢** (Phase 2 内で 2-arm sub-ablation):

| 変種 | Head | max mora / morph | Head params |
|---|---|---|---|
| B1 flat cls | `Linear(1024, 8 × 68)` (mora-slot × phoneme vocab) | 8 | ~560K |
| B2 tiny decoder | 2-layer LSTM decoder (hidden 256), max 8 step | 8 | ~1.6M |

`max mora / morph = 8` は Phase 1 で JSUT + Wikipedia 統合コーパスの morpheme-per-mora 分布の 99.5% 分位から決定 (critique §5 の "5 は arbitrary" 指摘への対応)。8 超過の morpheme は Phase 前処理で強制分割し、分割位置を align table に保存 (evaluation 時に再結合)。

**Total trainable**: 132M (encoder) + head params + `[MORPH]` embedding 1 個 (dim 512)。

**Loss**: per-morpheme cross-entropy (B1) or seq2seq CE (B2)、label smoothing 0.05。**辞書一致 morpheme に対しては loss weight 0.3** で down-weight (辞書が正解を持っている前提、NN は不確実領域に集中)。APBP head は BIO の 3-class CE、`α_APBP = 0.2`。

**Training recipe**: LR 5e-5 (encoder), 3e-4 (head), warmup 1500, total 45k steps, batch effective 512。他は §3.1 と同じ。

**Inference**: MeCab pretokenize → `[MORPH]` 挿入 → ModernBERT → head → 形態素毎 phoneme + APBP を canonical form へ変換。

### 3.3 P-C: char-level BERT (対照 baseline)

**動機**: NHK BAS が `tohoku-nlp/bert-base-japanese-char-v2` を採用した実績、境界問題は自明に無い。ModernBERT を使わない対照として、Phase 2 の判断根拠を強化。critique §minor で指摘された通り、char-v2 は**単一文字 WordPiece + `##`-continuation for rare CJK 構成**で、大部分の日本語文字は 1 char = 1 WordPiece token として扱われる (要実装時に単体テストで確認)。

```
raw text
  |-- char tokenizer (vocab ~6144, tohoku-nlp char-v2)
  v
  BERT encoder (base, 12 L, hidden 768, 110M)
  v
  per-char linear (or CRF) head -> phoneme label(s)
```

**Input encoding**: char tokenizer で 1 文字 1 token (`##` continuation を除き)、max 512 char。長文は Phase 2 では文分割で処理、Phase 5 で 8k 相当の window 拡張を検討。
**Output encoding**: 各 char 位置に対して phoneme fixed-slot 8 の multi-label (1 char が最大 8 phoneme を出せる)。`H/L` は phoneme label と別 axis で per-mora 予測 (linear head 2 class × 8 slot)。`/` は「文字位置ごとに B/I/O BIO tag 1 head」で予測し canonical form 化時に挿入。
**Fixed slot = 8** は critique §5 (5 は不足) への対応 (承知 s y o u ch i = 6、面倒 m e N d o u = 6 の実測を含めた 99.5% 分位から決定)。

**副パイロット構成** (Phase 2 内):

- **C1**: linear head + CE loss
- **C2**: linear + CRF (BIO 風の phoneme sequence 制約)

**Total trainable**: 110M + 250K (C1) / 400K (C2)。
**Loss**: char-level cross-entropy、ignore_index for pad/special、label smoothing 0.05。
**Training recipe**: LR 5e-5, warmup 1000, total 40k steps, batch 256, weight decay 0.01。
**Inference**: greedy (C1) or Viterbi (C2)、canonical form に再構成。

### 3.4 出力仕様の pilot 別対応表 (critique §4 対応)

`/`, `H/L`, phoneme の 3 系列を canonical form へ落とし込む決定的アルゴリズムを 3 pilot で明示する。

| Pilot | phoneme emission | mora H/L emission | `/` (accent phrase boundary) emission |
|---|---|---|---|
| P-A | decoder が phoneme token を autoregressive 出力、`H/L` は phoneme と同じ vocab に混ぜてインタリーブ | 上に含む (単一系列) | decoder が `/` を専用 token として出力 |
| P-B | morpheme head が per-morpheme phoneme 列を出力し concat | phoneme vocab 内に `H/L` を含めた multi-label (B1) / seq2seq (B2) | 独立 APBP head (BIO 3-class) が morpheme 末に B ラベル → `/` を canonical 化時に挿入 |
| P-C | per-char fixed slot=8 の phoneme label | per-char × 8 slot の H/L 2-class head | per-char BIO 3-class head (文字境界に B ラベル → `/` 挿入) |

Canonical form 変換の完全一致は Phase 2 kickoff 前に単体テスト (`tests/test_canonicalize.py`) で 100 パターン以上検証する。ここでズレると PER 差の解釈が不可能になる。

---

## 4. メトリクス

| Metric | 計測手段 | Threshold / 用途 |
|---|---|---|
| PER canonical (JSUT Basic5000) | `src/modernbert_g2p/metrics/per.py` | 主指標。haqumei 1.1657% との gap を測る |
| kana CER (JVS-3000) | `metrics/cer.py` | Tier 1 (OpenJTalk 1.03%) との比較 |
| KER (ROHAN 4600) | `metrics/ker.py` | Tier 2 (haqumei 1.6397%) との比較 |
| Per-category PER (hard-set 7×200) | 上記の subset 版 | §5 tier-3 の disqualifier |
| Throughput sentences/sec, bs=32 | fp16 推論、100 iter warmup + 500 iter mean | tie-break、pilot 別 |
| Latency ms/mora, bs=1 | 同上、autoregressive P-A のコストを公平に評価 | critique §6 対応 |
| Output length dist (mora/sentence) | eval 実行時に併記 | throughput 数値の解釈用 |
| GPU memory peak | `torch.cuda.max_memory_allocated()` | デプロイ制約 |
| Param count | `sum(p.numel() ...)` | 配布サイズ |
| Multi-head compat score | §11 スコアカード | Phase 3 適合性 (critique §7 対応) |
| Bootstrap 95% CI on PER | 3 seed × 10,000 resample | §5 決定に必要 |
| Val loss curve | wandb | 過学習検出 |

3 pilot × 4 データ (JSUT/JVS/ROHAN/hard-set) × 上記主要 8 指標 = 96 セルの統合表を `reports/phase2_pilot_table.md` に一枚化する。

---

## 5. 決定ルール (勝者選定基準)

**Lexicographic order** で以下を評価:

### Tier 1 (足切り) — pure-NN Phase 2 版

Phase 2 は pure-NN で haqumei に勝つことを求めない (`07_nn_only_benchmarks.md` で pure-NN が hybrid に負けることは実証済み、hybrid 化は Phase 4 で行う設計思想 1 に一致)。以下 **どちらか片方** が満たされれば通過:

- (a) JSUT PER 平均 (3 seed) が haqumei との gap を **50% 以上閉じている** (`haqumei_per - pilot_per >= (haqumei_per - target_ideal) * 0.5`、target_ideal = 0.5%、すなわち PER ≤ 0.83% ≒ haqumei との差 −0.33pt 以上を pure-NN として達成)
- (b) 上記に加えて、**pyopenjtalk-plus 辞書 fallback を merge した mini-hybrid** (辞書ヒット word は辞書優先、OOV / polyphone のみ pilot 出力を採用する Phase 4 予告版) を評価した hybrid PER が **haqumei 1.1657% を絶対値で下回る**

Tier 1 を 3 pilot 全て clear できなければ Phase 2 を延長し、データ生成量拡張または pilot 設定の再ablation に戻る (critique §1 対応)。

### Tier 2 (PER 差)

3 pilot 間の **JSUT PER 平均差が bootstrap 95% CI で non-overlap かつ Δ ≥ 0.2%** ならその差で勝者決定。critique §2 対応: N=1 seed × Δ=0.05% は noise なので N=3 seed × Δ=0.2% + CI non-overlap を要求する。

### Tier 3 (tie-break)

PER 差が Δ<0.2% or CI overlap の時のみ以下を順に:

1. Inference throughput (sentence/sec on H100 bs=32) — ただし出力長分布を disclose (critique §6)
2. Latency ms/mora (bs=1) — autoregressive P-A への公平化
3. Param count 少ない方
4. **License**: MIT/Apache-2.0 優先。char-v2 が CC-BY-SA-4.0 のため P-C はここで penalty (critique §minor で openly 明示)
5. 実装工数 / 保守性 (seq2seq が最も枯れた技術)

### Tier 4 (disqualifier) — Phase 1 依存

**Hard-set 7 カテゴリのうち 1 でも haqumei 比 +0.5pt 以上悪化していれば失格**。ただし Phase 1 の hard-set 1400 完成が未達なら (現状 21 sample、§10 open Q ②)、Tier 4 は **advisory** に降格し、決定は Tier 1-3 のみで行う (critique §9 対応)。

決定は Phase 2 週次レビューで tech-lead 承認、結果を `docs/decisions/phase2_pilot_winner.md` に ADR 形式で残す。

---

## 6. リスク分析 (pilot 別 top-3 failure mode)

### 6.1 P-A (seq2seq)

| # | 失敗モード | 兆候 | mitigation |
|---|---|---|---|
| 1 | Decoder hallucination (辞書外語で音素捏造) | ROHAN 上で無関係 phoneme sequence | coverage penalty 追加, Phase 4 で dictionary constrained decoding |
| 2 | Autoregressive で inference latency が haqumei 相当に届かない | latency ms/mora で判明 | non-autoregressive decoder への sub-ablation (Phase 3 stretch), batch beam 実装 |
| 3 | 長文で exposure bias | val PER が train と乖離 | scheduled sampling (10% で self-generated token feed) |

### 6.2 P-B (MeCab + `[MORPH]` + ModernBERT)

| # | 失敗モード | 兆候 | mitigation |
|---|---|---|---|
| 1 | `[MORPH]` token が cold init のため embedding が学習不足 (SB Intuitions 警告への mitigation が期待どおり効かない) | train loss が下がりきらない, `[MORPH]` embedding norm が更新後も小さい | encoder LR を 1e-4 に上げる sub-run、`[MORPH]` embedding のみ warmup 期間中に LR 10× 適用 |
| 2 | MeCab 分割誤りが upstream error として固定化 | 固有名詞 hard-set で高 PER | pyopenjtalk-plus 辞書の追加語彙を MeCab dict にマージ (Phase 0 導入済み) |
| 3 | max mora / morpheme = 8 超過の助数詞連結 | truncation warn | Phase 前処理で強制分割し align table 保存、evaluation で再結合 |

### 6.3 P-C (char-level BERT)

| # | 失敗モード | 兆候 | mitigation |
|---|---|---|---|
| 1 | ModernBERT の 8k long-context を捨てるため、複合語アクセント連続変異が劣化 | ROHAN KER が Kurihara BAS 6.04% に届かず 10%+ | Phase 3 で BAS head を別 encoder (ModernBERT) 側で追加、dual-encoder 化 |
| 2 | 1 char → 複数 phoneme (漢字) の multi-label が学習困難 | 助数詞/固有名詞 hard-set が特に悪い | mini seq2seq LSTM per char を副 head で載せる sub-run (C2 拡張) |
| 3 | tohoku char-v2 の CC-BY-SA-4.0 が重み配布時に Share-Alike 感染 | 配布ライセンス設計で衝突 | Apache-2.0 の代替 char-BERT (KyotoU 系, `ku-nlp/bert-base-japanese-char-wwm`) を Phase 5 予備で評価 |

---

## 7. Compute 予算

Critique §8 で指摘された算数エラー (60k step × 2 step/s = 8.3h ≠ 32h) を修正し、bs=256 effective + bf16 + 130M encoder + 6L decoder on RTX 4090 の実効スループットを **0.6 step/s** と再見積もり (encoder 132M full fine-tune + decoder 22M で、grad accum 込みで通信ボトルネックあり)。3 seed 分を計上。

| Pilot | 1-seed train step | step/s | 1-seed train (GPU-h) | Seeds | Sub-ablation runs | Eval + misc (GPU-h) | 合計 (GPU-h) |
|---|---|---|---|---|---|---|---|
| P-A seq2seq | 60k | 0.6 | 28 | 3 | ×2 (from-scratch / T5-init) | 8 | 176 |
| P-B MeCab + `[MORPH]` + ModernBERT | 45k | 0.8 | 16 | 3 | ×2 (B1 flat / B2 mini-dec) | 10 (MeCab preprocess incl.) | 106 |
| P-C char BERT | 40k | 1.0 | 11 | 3 | ×2 (C1 linear / C2 CRF) | 6 | 72 |
| 共通 (data sanity, seed dry-run, canonical-form 単体テスト) | | | | | | 25 | 25 |
| **合計 (4090 換算)** | | | | | | | **379 GPU-h** |

### 7.1 GPU 別スケーリング

| GPU | 想定倍率 (vs 4090) | Phase 2 合計 (GPU-h) | 参考価格 (spot) | 概算 total cost |
|---|---|---|---|---|
| RTX 4090 24GB | 1.0× | 379 | $0.4/h | $152 |
| A6000 48GB (spot) | 0.9× | 421 | $0.8/h | $337 |
| H100 80GB | 2.5× (bf16 FA2) | 152 | $2.5/h | $380 |

Phase 2 の割当 2 週間で 24/7 稼働なら 336 GPU-h 相当 (4090 換算)。H100 で final run のみ回し、dev サイクルは 4090 or A6000 spot で 300 GPU-h 消化、H100 で 50 GPU-h の再現実行を最終レポート用に確保する運用を推奨。総 cost 見込みは **$300 程度**。

---

## 8. Phase 2 exit criteria

**全て** を満たしたら Phase 2 完了、Phase 3 (multi-task head 学習) 開始:

1. 3 pilot × 3 seed 全て学習完了、上記 96 セルの統合表が `reports/phase2_pilot_table.md` に埋まっている (bootstrap 95% CI 付き)
2. §5 の Tier 1〜Tier 4 に従って winner 1 つが選出され、`docs/decisions/phase2_pilot_winner.md` に ADR 形式 (Context / Decision / Consequences) で記録
3. 敗者 2 pilot は `experiments/phase2/{P-A,P-B,P-C}/reproduce.md` に **同一 seed で再現可能な** recipe が残されている
4. Tier 1 (a) または (b) のいずれかを winner pilot が満たしている
5. Tier 4 が active な場合 (Phase 1 hard-set 1400 完成済み)、winner pilot が 7 カテゴリ全てで haqumei 比 +0.5pt 以上の悪化を出していない
6. Multi-head 適合性スコアカード (§11) が 3 pilot 分埋まっており、winner の Phase 3 拡張性が Tier 3 以上

---

## 9. Impact analysis: char-level BERT (P-C) が勝った場合

CLAUDE.md の「主軸 = sbintuitions/modernbert-ja-130m」は仮説であり、証拠が矛盾したら update する。P-C 勝利は最も設計思想への衝撃が大きいため、事前に対応パスを規定する。

### 9.1 直接的な帰結と CLAUDE.md 差分

- **主軸ベースモデルの変更**: 主軸を `tohoku-nlp/bert-base-japanese-char-v2` (または Apache-2.0 代替) に差し替え。CLAUDE.md 「ベースモデル選定」節を書き換え:
  - Before: `主軸: sbintuitions/modernbert-ja-130m`
  - After: `主軸: char-level BERT (Phase 2 実証で P-C 勝利)、補助: modernbert-ja-130m (Head E BAS / Head C APBP など context 依存 head 専用の dual-encoder 側)`
- **ModernBERT の位置付け**: 補助 encoder (dual-encoder 構成の long-context 担当) として保持。Phase 3 で BAS/APBP head は ModernBERT 側、G2P main head は char-BERT 側という分業に切替え
- **設計思想 4 の update**: 「seq2seq / MeCab-pretokenize / char-level の 3 並列パイロットで最良を選ぶ」→ 「Phase 2 実証結果に基づき char-level BERT を採用。ModernBERT SentencePiece boundary 問題は `[MORPH]` token 挿入でも埋められなかったことで実証」を追記
- **Refuted 節への追加**: 「Phase 0 時点の推奨『ModernBERT を pure に主軸として採用する』提案は Phase 2 実証で refuted」を明記
- **主要な数値目標**: base model 依存しないため影響なし
- **ライセンス**: char-v2 の CC-BY-SA-4.0 は重み配布 (Share-Alike 感染) の懸念があるため、Phase 2 完了直後に代替 char-BERT の探索を開始。候補: `ku-nlp/bert-base-japanese-char-wwm` (Apache-2.0)、または独自事前学習 (Phase 6 stretch)
- **8k long-context の放棄**: char-v2 は max 512。Phase 5 で対応が必要な文書レベル評価は「文単位分割 + 隣接文 context を concat」で疑似的に確保、または dual-encoder の ModernBERT 側で吸収

### 9.2 意思決定支援 tie-break

P-C 勝利判定は critique §minor の「license bias openly」対応として、Tier 3 tie-break の (4) を通過した時のみ有効。すなわち **PER 差が Δ ≥ 0.2% で明確に P-C 勝利** の時のみ base model 差し替えを実施、Δ < 0.2% CI overlap なら「ModernBERT で長期投資可能性を優先」して P-A or P-B を選ぶ。ADR に明記する。

### 9.3 P-A / P-B 勝利時の impact

- **P-A 勝利**: CLAUDE.md 現行仮説を実証。Phase 3 は seq2seq decoder に multi-head を coupling する設計に移行 (NHK Kurihara TJ-G2P + BAS の直接踏襲)
- **P-B 勝利**: ModernBERT を保持しつつ MeCab 前処理を canonical にする。Phase 3 は Hida ICASSP 2022 の multi-head 構造にほぼそのまま乗る。SB Intuitions 警告への回答として「MeCab pretokenize + `[MORPH]` 特殊 token で解消可能」を実証したことになり、外部発信 (blog/technical report) のトピックにできる

---

## 10. Multi-task heads timing (held constant vs post-selection)

Critique §7 対応: Phase 3 で追加する multi-head (polyphone / APBP / ANPP / BAS) の適合性を Phase 2 単一タスク PER 選定と分けて計測しないと、Phase 3 で pilot 変更が必要になるリスクがある。

**基本方針**: Phase 2 は G2P main のみで PER 選定するが、**すべての pilot に対して Phase 3 拡張の compatibility スコアカードを埋める** ことを exit criteria (§8-6) にする。スコアカードは §11 で定義。

| Head 追加時期 | Pilot 別実装コスト評価 |
|---|---|
| Phase 2 (今回) | G2P main のみ。P-B に APBP head だけ副として載せる (canonical form の `/` 決定に必須) |
| Phase 3 (次期) | Polyphone / ANPP / BAS を full multi-task で追加、Hida 2022 loss weighting に従う |

つまり Phase 2 では multi-head を **held constant** (G2P main only) で pilot 比較し、Phase 3 選択の適合性は compatibility scorecard で **post-selection** の判断材料とする、というハイブリッド戦略。

---

## 11. Multi-head compatibility scorecard

Phase 3 で追加する 4 head (polyphone / APBP / ANPP / BAS) を pilot 毎に「どの hidden にどう載せるか」で 3 段階評価 (High=枯れた設計、Medium=非自明だが可能、Low=構造変更が必要)。

| Head | P-A seq2seq | P-B MeCab+`[MORPH]` | P-C char BERT |
|---|---|---|---|
| Polyphone disambiguation | Medium (encoder hidden の該当漢字位置に載せる、decoder 側から見えるよう cross attn 経路が必要) | **High** (per-morpheme representation にそのまま多クラス head) | Medium (per-char で漢字位置のみに mask、char 単位 candidate と 8-slot cls が競合しがち) |
| APBP (BIO) | Medium (decoder autoregressive 側で `/` 生成、encoder 側で BIO も並走) | **High** (per-morpheme BIO の Phase 2 で既に載っている) | High (per-char BIO は char-BERT の常道) |
| ANPP (nucleus position) | Medium (accent phrase 内 nucleus は decoder が離散的にしか出せない) | High (per-morpheme accent-nucleus position は Hida 2022 設定と同型) | Medium (per-char nucleus は解像度過剰) |
| BAS (accent sandhi correction) | Low (autoregressive decoder との統合が非自明、後段 rerank になる) | **High** (per-morpheme H/L 出力を BAS head で補正) | High (NHK 実装と同型) |
| 総合 compatibility | Medium | **High** | Medium-High |

**Note**: P-A の Low/Medium は critique §7 で指摘された通り「seq2seq decoder に 4 head を hang するのは非自明」の懸念を反映。P-B 勝利なら Phase 3 拡張は最もスムーズ、P-A 勝利なら Phase 3 で decoder ← encoder cross の設計工数が追加、P-C 勝利なら BAS/APBP は容易だが polyphone/ANPP で工夫要。

---

## 12. Seed strategy

- **Seed セット**: `{20260704, 20260705, 20260706}` (Phase 2 kickoff 週の連続日付)
- **各 pilot × 各 sub-ablation** で 3 seed 全て実行 (P-A: 2 sub × 3 seed = 6 run、P-B: 2 sub × 3 seed = 6 run、P-C: 2 sub × 3 seed = 6 run)。合計 18 primary run + eval
- **PER 平均 + bootstrap 95% CI** を報告 (`scipy.stats.bootstrap`, 10,000 resample, method='BCa')
- **CI non-overlap かつ Δ ≥ 0.2%** を §5 Tier 2 の勝者判定条件とする
- **seed 依存性の記録**: 各 seed 単独 PER も併記し、seed variance が 0.1pt を超えたら Phase 2 レビューで再検討 (追加 seed or データ拡張)
- **推論時 seed**: beam search / sampling を使う P-A は inference も seed 固定 (`torch.manual_seed(seed)`)

---

## 13. Open questions (Phase 2 kickoff 前に確定必要)

1. **Phase 1 統合コーパスの正確な件数**: 上記 GPU-h 見積もりは 1.5M pair 前提。実際が 3M を超えると budget 超過 → step 数を 60k → 45k に圧縮する fallback 案を用意
2. **Hard-set 1400 の完成時期**: 現状 21 sample のみ (`data/hard_set_seed/`)。Phase 2 開始時点で 7 カテゴリ × 200 = 1400 が揃っていないと §5 Tier 4 が advisory 降格 (critique §9)。**Phase 1 exit の必須項目に昇格させる**
3. **推論 GPU の確定**: RTX 4090 / A6000 / H100 の 3 種で throughput 測定を義務化。H100 spot 確保可否は要確認、確保できない場合は A6000 と 4090 のみで比較
4. **T5-ja init の重みライセンス** (P-A sub-run): `retrieva-jp/t5-base-japanese` は critique §minor で指摘の通り **CC-BY-SA-4.0** の可能性。もしそうなら Share-Alike 感染懸念のため P-A sub-run から除外し、from-scratch のみに縮小。要 HF card 確認
5. **`llm-jp/llm-jp-modernbert-base` の scope** (critique §10): CLAUDE.md で ablation 対象。Phase 2 では P-A/P-B の tokenizer sub-run として並列するか、Phase 5 に持ち越すかを kickoff で確定。**Phase 2 スケジュール枠から見ると Phase 5 持ち越しを推奨**、ただし Phase 2 データ生成時点で llm-jp tokenizer での tokenize 統計 (unk 率, 平均 subword/文字数) だけは前倒しで測定しておく
6. **`bert-base-japanese-char-v2` の tokenizer 挙動確認** (critique §minor): 「単一文字 WordPiece + `##`-continuation for rare CJK」の想定を単体テストで実測。CJK Extension B 以降の稀漢字がどれだけ `##` 化するかで per-char slot=8 の妥当性が変わる

---

## 14. Deliverable references

- `docs/design/phase2_tokenizer_pilots.md` (本書)
- `reports/phase2_pilot_table.md` (96 cells + CI + throughput + latency)
- `docs/decisions/phase2_pilot_winner.md` (ADR)
- `experiments/phase2/{P-A,P-B,P-C}/reproduce.md` (3 seed × 2 sub-ablation の再現 recipe)
- `tests/test_canonicalize.py` (§3.4 canonical form 一致テスト)
- `data/splits/phase2_v1/{train,val,test}.idx` (固定 split index)
- `data/hard_set_v1/` (Phase 1 依存、Phase 2 exit の Tier 4 で使用)
