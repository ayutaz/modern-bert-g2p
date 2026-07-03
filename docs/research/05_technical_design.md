# 05. ModernBERT を使ったモデル設計の推奨案

01_overview.md の戦略に基づく、具体的な技術設計提案。すべての判断は 02〜04 の一次情報 (arxiv / HFカード / GitHub README) に根拠を持つ。

---

## 1. ベースモデルの選定

### 1.1 第一候補: sbintuitions/modernbert-ja-130m

- **理由**:
  - サイズ (132M params, 80M ex-emb) は fine-tune の compute 効率と性能上限のバランスが良い
  - 8,192 token 長コンテキストで、複合語・アクセント句境界の長距離依存を扱える
  - MIT License、商用配布可能
  - 4サイズ (30M/70M/130M/310M) 展開で、Ablation実験時に同じトークナイザー・同じ事前学習で系統的に比較可能
- **ソース**: [modernbert-ja-130m HFカード](https://huggingface.co/sbintuitions/modernbert-ja-130m)

### 1.2 対照モデル (head-to-head で必須比較)

| モデル | 目的 | ソース |
|---|---|---|
| `sbintuitions/modernbert-ja-30m` | 実用サイズ下限、モバイル/CPU推論可能性 | [HFカード](https://huggingface.co/sbintuitions/modernbert-ja-30m) |
| `sbintuitions/modernbert-ja-70m` | 実用サイズ中位 | [HFカード](https://huggingface.co/sbintuitions/modernbert-ja-70m) |
| `sbintuitions/modernbert-ja-310m` | 精度上限探索 | [HFカード](https://huggingface.co/sbintuitions/modernbert-ja-310m) |
| `llm-jp/llm-jp-modernbert-base` | 別トークナイザー系での性能比較 | [論文](https://arxiv.org/pdf/2504.15544) |
| `tohoku-nlp/bert-base-japanese-char-v2` | 文字レベルbaseline (NHK BAS で採用実績) | [HFカード](https://huggingface.co/tohoku-nlp/bert-base-japanese-char-v2) |

### 1.3 棄却したベースモデル

- **cl-tohoku/bert-base-japanese-v3** — modernbert-ja より旧世代、long-context非対応
- **rinna/japanese-roberta-base** — 同上
- **pkshatech/GLuCoSE-base-ja** — sentence embedding特化、G2Pに不向き

---

## 2. トークナイザーに関する critical decision

### 2.1 問題

SB Intuitions 自らのHFカードで:
> "token boundaries often do not align with the morpheme boundaries, resulting in poor performance in token classification tasks such as named entity recognition and span extraction"

これは G2P の token classification 定式化に直撃する。理由:

- SentencePiece unigram + byte-fallback (vocab 102,400) は BPEに近い挙動で、漢字1文字を含む長いスパンを1トークンに畳むことがある
- G2Pでは1漢字≒複数モーラのマッピングが自然だが、SentencePiece のスパンと合わないと、ラベルを何処に置くべきかが自明でなくなる
- 逆に文字境界と揃った pretokenize を強制すると、事前学習と subword 分布が乖離して転移学習の効きが弱まる

### 2.2 対処選択肢とパイロット計画

| 選択肢 | 概要 | pros | cons | 実装工数 |
|---|---|---|---|---|
| **(a) seq2seq (推奨主軸)** | encoder + T5/BART decoder。encoderで文脈把握、decoderで音素列生成 | tokenizer境界問題を回避、Kurihara NHK と同構造 | inference時autoregressive で遅い | 中 |
| **(b) MeCab pretokenize + encoder** | 文をMeCab分割→形態素毎に ModernBERT に投入→token classificationで読みラベル付与 | tokenizer境界問題を根本解決、Hida ICASSP 2022 構造に近い | 事前学習との subword 分布乖離、embed再学習が必要な可能性 | 高 |
| **(c) char-level BERT に切り替え** | `bert-base-japanese-char-v2` で token classification | 境界問題無、NHK BAS で実証 | 長コンテキスト・効率性を捨てる、ModernBERT特徴を活用できない | 低 |
| **(d) llm-jp-modernbert 併用** | 別系トークナイザーで対照 | 独立した base | 同じ問題を抱える可能性 | 中 |

### 2.3 実装推奨: (a) + (b) の並列実装 + (c) の対照

- **メインライン**: sbintuitions/modernbert-ja-130m encoder + T5/BARTベースの char-level decoder (seq2seq)
- **並列パイロット**: MeCab pretokenize + modernbert-ja-130m + token classification head
- **対照ベースライン**: bert-base-japanese-char-v2 + token classification (NHK BAS 相当)
- **比較指標**: JSUT Basic5000 PER + JVS-3000 kana CER + 訓練時間 / GPU時間 / 推論スループット

パイロット結果次第でメインラインを (a) → (b) に切り替えることを許容する。

---

## 3. タスク定式化

### 3.1 全体構造 (推奨: multi-head hybrid)

```
入力文 (raw text)
  ↓
[前処理レイヤ]
  ├─ 全角/半角統一
  ├─ 数字/記号正規化
  └─ pyopenjtalk による形態素解析 → base_yomi + base_accent (辞書ベース初期推定)
  ↓
[ModernBERT Encoder]
  入力: raw text tokens (SentencePiece)
  補助入力 (side-channel):
    - base_yomi の embedding (辞書lookupの初期推定)
    - MeCab POS tags
    - kanji/kana/latin判定 の binary mask
  ↓
[Multi-task Head Layer]
  ├─ Head A: G2P main (seq2seq または token classification)
  │   → 音素列 + モーラ表記
  ├─ Head B: Polyphone disambiguation
  │   → 各漢字位置ごとの読みラベル (辞書候補内での分類)
  ├─ Head C: APBP (Accent Phrase Boundary Prediction)
  │   → BIO tagging for accent phrase boundary
  ├─ Head D: ANPP (Accent Nucleus Position Prediction)
  │   → 各アクセント句内のnucleus位置ラベル
  └─ Head E: BAS (Accent Sandhi correction)
      → base_accent の局所的補正 (NHK BAS 相当)
  ↓
[後処理 + Reconciliation Layer]
  ├─ 辞書lookup優先マージ (haqumei風)
  ├─ 出力音素列の音韻的妥当性チェック
  └─ 最終出力: [音素列, モーラアクセントH/L, アクセント句境界]
```

### 3.2 Loss設計

```
L_total = α_G2P * L_G2P + α_poly * L_poly + α_APBP * L_APBP + α_ANPP * L_ANPP + α_BAS * L_BAS + α_ctc * L_ctc_aux
```

- `L_G2P`: seq2seq cross-entropy (label smoothing 0.1)
- `L_poly`: token classification cross-entropy on polyphone positions only
- `L_APBP`: BIO tagging cross-entropy
- `L_ANPP`: nucleus position multi-class or regression
- `L_BAS`: token classification cross-entropy on accent labels
- `L_ctc_aux`: (optional, from CC-G2PnP idea) CTC auxiliary loss for alignment stability

**初期の loss weights の推奨** (実験前の hypothesis):

- α_G2P = 1.0, α_poly = 0.5, α_APBP = 0.3, α_ANPP = 0.3, α_BAS = 0.5, α_ctc = 0.1

Ablation: 各 head を on/off して寄与を測定 (Hida 2022 と同様の実験プロトコル)。

### 3.3 カテゴリ別 loss reweighting

Koriyama benchmark の分布 (数詞14.2%, 固有名詞14.5%) を踏まえ、これらのカテゴリで学習時のsample重みを上げる:

```python
if sample_category in ['numeral', 'proper_noun_kanji', 'proper_noun_katakana', 'loanword', 'counter_word']:
    sample_weight = 2.0
else:
    sample_weight = 1.0
```

---

## 4. 音素表記の推奨

### 4.1 選定

**JULIUS音素セット + モーラアクセントH/L + アクセント句境界マーカ**

例:
```
入力: "私は東京で本を読みました"
出力: "w a t a sh i _ w a _ t o o ky o o _ d e _ h o N _ o _ y o m i m a sh i t a"
モーラアクセント: "LHHH_LH_LHHH_H_LH_LH_LHHHHHH"
アクセント句境界: "私は/東京で/本を/読みました"
```

### 4.2 理由

- pyopenjtalk / Open JTalk の default 出力形式で、既存 TTSパイプラインとドロップイン互換
- Kurihara NHK 2024 も同系統の PPL 記法
- カタカナ音素より情報量が多く、外来語の細かい音素区別 (`ti/chi`, `fa/fu-a`) を保持できる

### 4.3 代替 (実験対象)

- **カタカナ音素** (haqumei互換) — haqumei との直接比較に必要
- **X-SAMPA** — 多言語 pipeline 用
- **IPA** — 学術的完全性

**推奨戦略**: JULIUS音素セットを canonical form とし、変換ユーティリティで X-SAMPA / カタカナ音素の出力にも対応。

---

## 5. アーキテクチャの具体的な実装スケッチ

### 5.1 主モデル (推奨主軸: seq2seq)

```python
# Pseudocode

from transformers import AutoModel, AutoTokenizer
import torch.nn as nn

class ModernBertG2P(nn.Module):
    def __init__(self, encoder_name="sbintuitions/modernbert-ja-130m",
                 phoneme_vocab_size=64,  # JULIUS音素セット + 特殊トークン
                 mora_accent_labels=3,   # H, L, ε
                 apbp_labels=3,          # B, I, O
                 poly_max_readings=8):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        d = self.encoder.config.hidden_size  # 512 for 130m

        # Head A: G2P seq2seq (autoregressive decoder)
        self.g2p_decoder = TransformerDecoder(d_model=d, num_layers=6, vocab_size=phoneme_vocab_size)

        # Head B: Polyphone disambiguation (per-token classification, only active on polyphone positions)
        self.poly_head = nn.Linear(d, poly_max_readings)

        # Head C: APBP (per-token BIO tagging)
        self.apbp_head = nn.Linear(d, apbp_labels)

        # Head D: ANPP (per-token nucleus position label)
        self.anpp_head = nn.Linear(d, 2)  # binary: is_nucleus?

        # Head E: BAS (per-token accent H/L correction)
        self.bas_head = nn.Linear(d, mora_accent_labels)

    def forward(self, input_ids, attention_mask,
                decoder_input_ids=None,
                poly_positions=None, apbp_labels=None, anpp_labels=None, bas_labels=None):
        enc_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        outputs = {}
        if decoder_input_ids is not None:
            outputs['g2p_logits'] = self.g2p_decoder(enc_out, decoder_input_ids)
        outputs['poly_logits'] = self.poly_head(enc_out)
        outputs['apbp_logits'] = self.apbp_head(enc_out)
        outputs['anpp_logits'] = self.anpp_head(enc_out)
        outputs['bas_logits'] = self.bas_head(enc_out)
        return outputs
```

### 5.2 補助モデル (対照: token classification)

```python
class ModernBertG2PTokenCls(nn.Module):
    """
    MeCab-pretokenized approach: each morpheme is one 'super-token'
    with a per-position phoneme label (multi-label).
    """
    def __init__(self, encoder_name="sbintuitions/modernbert-ja-130m",
                 max_morae_per_morpheme=10,
                 phoneme_vocab_size=64):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        d = self.encoder.config.hidden_size
        # Per-morpheme, predict up to N phonemes
        self.per_morpheme_head = nn.Linear(d, max_morae_per_morpheme * phoneme_vocab_size)
        # ... (rest similar to above)
```

### 5.3 Hybrid inference layer (推論時のみ)

```python
def hybrid_g2p_inference(text, model, dict_lookup=pyopenjtalk):
    """
    Recipe:
    1. First pass with pyopenjtalk to get base readings + accents
    2. Detect uncertainty regions (polyphone kanji, OOV, compound words)
    3. Run ModernBERT only on those regions
    4. Merge results with dictionary as authority for known words
    """
    base_result = dict_lookup.g2p(text, return_accent=True)
    uncertainty_mask = detect_uncertainty(base_result, text)
    if not uncertainty_mask.any():
        return base_result

    bert_result = model(text)
    merged = merge_with_priority(
        dict_result=base_result,
        bert_result=bert_result,
        priority='dict_unless_uncertain',
        uncertainty_mask=uncertainty_mask
    )
    return merged
```

これは haqumei と Misaki v2 が採用している思想と同じ。

---

## 6. 学習ハイパラの初期推奨

| 項目 | 推奨値 | 根拠 |
|---|---|---|
| バッチサイズ (effective) | 512 | modernbert-ja の 8k context に対して安定 |
| 学習率 | 5e-5 (encoder), 1e-4 (heads) | BERT fine-tune 標準 |
| Warmup steps | 1000 | 3 epoch = 30k steps 想定時 |
| LR schedule | Linear decay | 標準 |
| Epochs | 3-5 | 過学習防止 |
| Weight decay | 0.01 | 標準 |
| Gradient clipping | 1.0 | 標準 |
| Optimizer | AdamW (β1=0.9, β2=0.999) | 標準 |
| Mixed precision | bf16 | ModernBERT対応、安定 |
| Flash Attention | on | ModernBERT公式サポート |
| Label smoothing | 0.1 | seq2seqの過信抑制 |
| LoRA (option) | r=16, alpha=32 | 実験の高速反復用 |

---

## 7. 推論時の要件

### 7.1 レイテンシ目標

- **1文 (~50文字)** の p50 レイテンシ:
  - CPU (x86 8-core): < 100ms
  - GPU (T4): < 20ms
  - GPU (RTX 4090): < 10ms

### 7.2 モデル圧縮の検討

- Quantization (int8): PyTorch dynamic quantization → CPU推論高速化
- ONNX export: haqumei と同じデプロイパスに乗る
- 30M/70M モデルでの精度-速度トレードオフ評価

---

## 8. 監視すべき失敗モード (Ablation で計測)

以下は Koriyama / NHK / Hida のいずれかの論文で明示的に指摘された課題 + 実世界の日本語文で頻出する多言語混在:

1. **助数詞語のアクセント連続変異** (Kurihara: OpenJTalk 16.24% CER)
   - 例: "3人 (さんにん)", "5個 (ごこ)" のアクセント型
2. **複合語のアクセント連続変異** (Kurihara: naive TJ-G2P 15.21% → BAS 6.04%)
   - 例: "東京都" ("とうきょう" + "と" の連結アクセント)
3. **固有名詞の phoneme error** (Kurihara: OpenJTalk 8.33%)
   - 例: 難読姓 "小鳥遊 (たかなし)", 地名 "国府津 (こうづ)"
4. **多音字** (Hida: 94.34% accuracy)
   - 例: "行" → いく/おこなう/こう, "生" → なま/せい/しょう
5. **カタカナ表記外来語** (Koriyama benchmark 8.5%含有)
   - 例: 稀な英単語カタカナ化, 音写のバリエーション
6. **数詞・日付・時刻・単位** (Koriyama benchmark 14.2%含有)
   - 例: "2025年", "3.14", "10:30", "10km", "3GB"
7. **英単語混在文** (**新規、実世界頻出**)
   - 例: "iPhoneを買った", "PDFを開く", "Zoomで会議した"
   - 課題: 英単語のスパン検出 → カタカナ音写 → 日本語音素化
8. **英字略語** (**新規、実世界頻出**)
   - 例: AI (エーアイ), NASA (ナサ), HTML (エイチティーエムエル), e-mail (イーメール), Wi-Fi (ワイファイ)
   - 課題: アルファベット読み vs 単語読みの文脈依存判定

## 8a. 多言語混在文の処理パイプライン (**新規**)

実世界の日本語文にはデフォルトで英単語・略語・記号連結語が混在するため、以下の処理を **前処理 + モデル + 後処理** の3層で行う:

### 8a.1 前処理

- **latin span detection**: 連続 latin 文字を1つのスパンとして識別
- **classify**: 大文字連続 (`AI`, `NASA`) = 略語候補 / mixed case (`iPhone`, `MacBook`) = 通常単語 / kebab-case (`Wi-Fi`, `e-mail`) = 連結語
- **normalize**: 全角latin → 半角latin、全角数字 → 半角数字

### 8a.2 モデル内部 (ModernBERT)

- 混在文をSentencePiece tokenizer に投入すると、latin 文字は byte-fallback で処理される
- BERTのcontextual embeddings が、文脈から適切な発音パターンを学習
- Head E (BAS) が周辺文字との連続変異を補正

### 8a.3 後処理: 英単語 → カタカナ音写

3段階のフォールバック戦略:

1. **辞書lookup優先**: pyopenjtalk-plus + 外来語辞書 (Kanalizer学習データ由来) にヒットすればそれを使用
2. **Kanalizer NN**: 未知綴りは VOICEVOX/kanalizer-model 相当のseq2seqでカタカナ化
3. **規則フォールバック**: それも失敗した場合、CMUdict → 日本語音素マッピング規則の deterministic 変換

### 8a.4 英字略語の判定

```
if word is all-uppercase and length <= 5:
    if in abbreviation_dict:  # AI, NASA, HTML等の明示的リスト
        use listed reading (アルファベット読み or 単語読み)
    else:
        default to alphabet reading (エーアイ etc.)
elif word is mixed case:
    treat as loanword → Kanalizer
```

### 8a.5 例

| 入力 | 期待音素列 (簡易表記) |
|---|---|
| "iPhoneを買った" | ai-fo-N o kat-ta |
| "PDFを開く" | pii-dii-e-fu o hi-ra-ku |
| "AIエンジニア" | ee-ai eN-ji-ni-a |
| "NASAが発表した" | na-sa ga hap-pyo-o-shi-ta |
| "Wi-Fi環境" | wa-i-fa-i kaN-kyo-o |
| "10kmランニング" | jup-pu-ki-ro raN-ni-N-gu |

各カテゴリで別々の hard-set を作り、per-category PER を報告する。

---

## 9. 一言まとめ (設計哲学)

- **単一 seq2seq で辞書を置き換えない**。辞書を primary、ModernBERT を secondary correction とする hybrid。
- **NHK Kurihara の TJ-G2P + BAS を直接踏襲**する。ModernBERT でBAS相当を担う設計。
- **Hida のマルチタスク (多音字 + APBP + ANPP)** を multi-head で実装する。
- **トークナイザー起因の失敗モードは開発初期から警戒**。seq2seq / MeCab-pretokenize / char-level の3並列パイロットで最良を選ぶ。
- **評価は JVS-3000 / JSUT Basic5000 / ROHAN 4600 の3本柱**で、per-カテゴリで hard-set も測定。
