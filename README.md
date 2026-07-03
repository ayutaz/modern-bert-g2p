# modernbert-g2p

ModernBERT ベースの日本語 G2P (Grapheme-to-Phoneme) モデルを新規開発するプロジェクトです。既存の主要 OSS G2P (OpenJTalk / pyopenjtalk / haqumei) を **同一プロトコル** で計測しつつ上回ることを目標とします。

現在は Phase 0 (ベースライン再現 & リポジトリ scaffold) の段階で、モデル学習コードはまだ実装されていません。

---

## 3 ティア目標

| Tier | 敵 | 指標 (代表値) | 我々の目標 |
|---|---|---|---|
| 1 (must-beat) | OpenJTalk / pyopenjtalk | JVS-3000 kana CER 1.03% | JVS-3000 CER < 0.9% |
| 2 (should-beat) | haqumei 0.8.0 | JSUT Basic5000 PER 1.17% / ROHAN KER 1.64% | PER < 1.0% / KER < 1.5% |
| 3 (stretch) | Frontier LLM (Claude Opus 4.6, Gemini 3.1 Pro) | JVS-3000 kana CER 0.52 – 0.62% | 同水準 (< 0.62%) |

数値目標の全文は [`docs/requirements.md`](docs/requirements.md) の FR-EVAL / AC 節を参照。

---

## Quick start

### 1. ベースライン再現 (Phase 0 AC-P0)

```bash
# Python 3.12 で venv を作る (haqumei は 3.14 未対応)
uv venv --python 3.12 .venv
source .venv/bin/activate

# 依存導入
uv pip install -e ".[dev]"

# JSUT-label を取得
git clone --depth 1 https://github.com/prj-beatrice/jsut-label.git

# haqumei の JSUT Basic5000 PER (公式 1.17%) を再現する
JSUT_YAML=./jsut-label/text_kana/basic5000.yaml \
  python scripts/eval_haqumei_jsut.py
```

期待出力: `PER 1.1657%` (公式 1.17% との差 0.005 pt) — AC-P0 要件 (差 ≤ 0.1%) を満たす。

`scripts/eval_baselines.sh` は Phase 0 の間に追加予定 (haqumei / pyopenjtalk / OpenJTalk を一括で測るランナー)。

### 2. テスト実行

```bash
pytest -q
```

`tests/test_metrics.py` は canonical な PER / kana CER / KER 実装のリグレッションを守ります。

---

## ディレクトリ構造

```
modern-bert-g2p/
├── src/modernbert_g2p/          # ソースコード
│   ├── __init__.py
│   └── metrics/                 # PER / CER / KER canonical 実装
│       ├── per.py
│       ├── cer.py
│       └── ker.py
├── tests/                       # pytest スイート
│   └── test_metrics.py
├── scripts/                     # ベースライン測定・データ生成の CLI
│   └── eval_haqumei_jsut.py     # ← Phase 0 AC-P0 検証
├── docs/
│   ├── requirements.md          # 要求定義書 (FR/NFR/CR/AC)
│   ├── phase0_kickoff_checklist.md
│   └── research/                # 8 本の技術ドキュメント (調査完了済)
├── data/                        # (gitignored) raw / processed データ
├── results/                     # (gitignored) 評価結果・チェックポイント
├── pyproject.toml
└── README.md
```

---

## 依存関係

`pyproject.toml` で 3 つに分離しています。

- **ランタイム** (`dependencies`): `pyyaml`, `haqumei==0.8.0`, `pyopenjtalk` — 評価とベースライン測定に必要な最小構成。
- **学習** (`optional-dependencies.training`): `torch>=2.1`, `transformers>=4.48`, `wandb`, `omegaconf`, `fugashi[unidic]` — Phase 2 以降で ModernBERT を学習する時のみ。`uv pip install -e ".[training]"`
- **開発** (`optional-dependencies.dev`): `pytest>=8`, `ruff`, `mypy`, `ipython` — CI と静的検査。

Python サポート範囲: `>=3.10,<3.13` (haqumei 0.8.0 は 3.13/3.14 未対応)。

---

## 設計の核心思想 (要点)

1. **単一 neural モデルで OpenJTalk を置き換えない。** 主要 OSS TTS (Style-Bert-VITS2, VITS Japanese, GPT-SoVITS, Bert-VITS2, Misaki) は全て pyopenjtalk をコアに採用し、NN は補助的にしか使わない。同じハイブリッド路線を踏襲する。
2. **NHK Kurihara Interspeech 2024 の TJ-G2P + BAS が架構の直接的な青写真**。ModernBERT を BAS 相当のアクセント連続変異補正に使う。
3. **haqumei は "rule 天井" であり "NN 天井" ではない**。PER 1.17% の 80–90% は pyopenjtalk-plus 辞書由来、NN 由来は 0–5% のみ。同じ辞書を採用しつつ、ModernBERT を BAS / 多音字 / 略語判定 / アクセント推定に投入する設計で戦う。
4. **Pure-NN で日本語 G2P を解く試みは実証的に失敗してきた** (PnG BERT, Kakegawa TJ-G2P, CC-G2PnP)。Phase 4 で辞書 primary + ModernBERT correction のハイブリッド化に必ず着地する。

詳細な設計原則と反証済み主張のリストは [`CLAUDE.md`](CLAUDE.md) を参照。

---

## Model Card

学習済みモデルの公開時に `docs/model_card.md` を追加します (Phase 6)。テンプレは Hugging Face 標準の Model Card に準拠し、以下を必須項目とします:

- 使用データセットとそのライセンス
- 3 ティア敵の同一プロトコル比較表
- 多音字 / 略語 / 固有名詞 / 数詞 hard-set のカテゴリ別 PER
- Intended use / out-of-scope use / bias & limitations
- 重み再配布ライセンスと依存辞書のライセンス継承

---

## 参照ドキュメント

- [`docs/requirements.md`](docs/requirements.md) — 要求定義書 (FR/NFR/CR/AC 付き)
- [`docs/phase0_kickoff_checklist.md`](docs/phase0_kickoff_checklist.md) — Phase 0 実行チェックリスト
- [`docs/research/01_overview.md`](docs/research/01_overview.md) — サマリーと開発戦略
- [`docs/research/02_existing_systems.md`](docs/research/02_existing_systems.md) — 既存 G2P サーベイ (§A.3 に haqumei 徹底解剖)
- [`docs/research/05_technical_design.md`](docs/research/05_technical_design.md) — ModernBERT ベースのモデル設計
- [`docs/research/06_implementation_roadmap.md`](docs/research/06_implementation_roadmap.md) — 7 フェーズの実装ロードマップ

---

## License

Apache License 2.0. 詳細は今後追加する `LICENSE` ファイルを参照。

学習/評価に用いる第三者データセット (pyopenjtalk-plus 辞書, UniDic, JSUT, ROHAN, JMDict, Wikipedia 等) はそれぞれ別ライセンス (BSD / MIT / CC-BY-4.0 / CC-BY-SA-4.0 等) が適用されます。特に Share-Alike 系ライセンスは重み配布時に影響するため、Phase 6 前に一次資料を再確認してください。
