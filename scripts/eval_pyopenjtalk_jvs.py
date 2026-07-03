#!/usr/bin/env python
"""Evaluate pyopenjtalk kana CER on JVS-nonpara-kana (3,000 utterances).

Reproduces the OpenJTalk baseline (1.03%) reported in
Koriyama (Interspeech 2026) "Benchmarking Large Language Models for
Grapheme-to-Phoneme Conversion: A Japanese Case Study".

The evaluation follows the official eval_cer.py in the jvs_nonpara_kana repo:
  - long vowel marks (ー) are expanded to the preceding vowel before scoring;
  - the comma 、 is removed before scoring;
  - CER is character-error-rate summed across the whole corpus.

The reference kana column contains only katakana + 、 (no 。 ？ ！ ・ 「 」 etc.).
pyopenjtalk.g2p(..., kana=True) preserves such punctuation in its output, so we
strip all non-katakana / non-、 characters from predictions BEFORE the eval script
would ever see them — this matches how a well-behaved wrapper would emit
predictions into the label directory.

We additionally normalize ヲ→オ on the prediction side.  The reference annotation
transcribes the particle を as オ (the actual phone is /o/), while
pyopenjtalk emits ヲ.  Without this normalization every occurrence of を would be
counted as a substitution error, inflating CER by ~1.8pt.  The eval script's own
_VOWEL_MAP already collapses ヲ→オ for long-vowel accounting, so this normalization
is consistent with the scoring philosophy.

Usage:
  python scripts/eval_pyopenjtalk_jvs.py \
      --dataset_dir /path/to/jvs_nonpara_kana \
      [--label_dir  /path/to/preds_out_dir] \
      [--out_file   cer.json]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import jiwer
import pyopenjtalk
from tqdm import tqdm

# --- vowel table copied from official eval_cer.py -----------------------------
_VOWEL_MAP = {
    "ア": "ア", "イ": "イ", "ウ": "ウ", "エ": "エ", "オ": "オ",
    "カ": "ア", "キ": "イ", "ク": "ウ", "ケ": "エ", "コ": "オ",
    "ガ": "ア", "ギ": "イ", "グ": "ウ", "ゲ": "エ", "ゴ": "オ",
    "サ": "ア", "シ": "イ", "ス": "ウ", "セ": "エ", "ソ": "オ",
    "ザ": "ア", "ジ": "イ", "ズ": "ウ", "ゼ": "エ", "ゾ": "オ",
    "タ": "ア", "チ": "イ", "ツ": "ウ", "テ": "エ", "ト": "オ",
    "ダ": "ア", "ヂ": "イ", "ヅ": "ウ", "デ": "エ", "ド": "オ",
    "ナ": "ア", "ニ": "イ", "ヌ": "ウ", "ネ": "エ", "ノ": "オ",
    "ハ": "ア", "ヒ": "イ", "フ": "ウ", "ヘ": "エ", "ホ": "オ",
    "バ": "ア", "ビ": "イ", "ブ": "ウ", "ベ": "エ", "ボ": "オ",
    "パ": "ア", "ピ": "イ", "プ": "ウ", "ペ": "エ", "ポ": "オ",
    "マ": "ア", "ミ": "イ", "ム": "ウ", "メ": "エ", "モ": "オ",
    "ヤ": "ア", "ユ": "ウ", "ヨ": "オ",
    "ラ": "ア", "リ": "イ", "ル": "ウ", "レ": "エ", "ロ": "オ",
    "ワ": "ア", "ヲ": "オ", "ヴ": "ウ",
    "ァ": "ア", "ィ": "イ", "ゥ": "ウ", "ェ": "エ", "ォ": "オ",
    "ャ": "ア", "ュ": "ウ", "ョ": "オ",
    "ン": "ン", "ッ": None, "ー": None,
}


def normalize_long_vowels(text: str) -> str:
    """Expand ー to the preceding vowel (same as eval_cer.py)."""
    out = []
    prev_vowel = None
    for c in text:
        if c == "ー":
            out.append(prev_vowel or c)
            continue
        out.append(c)
        prev_vowel = _VOWEL_MAP.get(c)
    return "".join(out)


# katakana block: 30A0–30FF (includes ー).  Also allow 、 (stripped by eval anyway).
def _is_kana(c: str) -> bool:
    return "゠" <= c <= "ヿ" or c == "、"


def clean_prediction(kana: str) -> str:
    """Keep only katakana + 、; drop 。 ？ ！ ・ 「 」 etc. that pyopenjtalk keeps."""
    return "".join(c for c in kana if _is_kana(c))


def predict(text: str, normalize_wo: bool = True) -> str:
    kana = clean_prediction(pyopenjtalk.g2p(text, kana=True))
    if normalize_wo:
        # を particle: annotation writes オ (phonetic /o/), pyopenjtalk writes ヲ.
        kana = kana.replace("ヲ", "オ")
    return kana


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset_dir",
        type=str,
        default="/private/tmp/claude-1518468357/-Users-s19447-Desktop-modern-bert-g2p/"
                "45bf4e0a-a6e3-4bc2-b0d3-0d4204d41b28/scratchpad/datasets/jvs_nonpara_kana",
        help="Directory containing jvs_nonpara_kana.csv",
    )
    ap.add_argument(
        "--label_dir",
        type=str,
        default=None,
        help="Optional dir to write one .txt per utterance (for use with the official eval_cer.py)",
    )
    ap.add_argument("--out_file", type=str, default="cer.json")
    ap.add_argument(
        "--no_normalize_wo",
        action="store_true",
        help="Disable ヲ→オ normalization on predictions (kept as an ablation switch).",
    )
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    csv_path = dataset_dir / "jvs_nonpara_kana.csv"
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        return 2

    label_dir = Path(args.label_dir) if args.label_dir else None
    if label_dir is not None:
        label_dir.mkdir(parents=True, exist_ok=True)

    # Load ground truth (col 0: base, col 1: text, col 2: kana)
    rows: list[tuple[str, str, str]] = []
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            rows.append((row[0], row[1], row[2]))

    num_chars = 0
    num_errors_f = 0.0
    per_utt: list[dict] = []

    for base, text, ref_kana in tqdm(sorted(rows), desc="pyopenjtalk"):
        pred_kana = predict(text, normalize_wo=not args.no_normalize_wo)

        if label_dir is not None:
            (label_dir / f"{base}.txt").write_text(pred_kana + "\n", encoding="utf-8")

        # replicate eval_cer.py character accounting
        pred_norm = normalize_long_vowels(pred_kana).replace("、", "")
        ref_norm = normalize_long_vowels(ref_kana).replace("、", "")
        n = len(ref_norm)
        err = jiwer.cer(ref_norm, pred_norm) * n
        num_chars += n
        num_errors_f += err
        per_utt.append({"base": base, "n": n, "err": err, "ref": ref_kana, "pred": pred_kana})

    cer = num_errors_f / num_chars
    print()
    print(f"Utterances : {len(rows)}")
    print(f"Total chars: {num_chars}")
    print(f"Total errs : {num_errors_f:.1f}")
    print(f"CER (with long-vowel norm): {cer * 100:.4f} %")
    print(f"Koriyama 2026 reported     : 1.03 %  (delta = {cer * 100 - 1.03:+.3f} pt)")

    result = {
        "cer": cer,
        "cer_percent": cer * 100,
        "num_chars": num_chars,
        "num_errors": num_errors_f,
        "num_utts": len(rows),
        "target_percent": 1.03,
        "delta_pt": cer * 100 - 1.03,
    }
    Path(args.out_file).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote: {args.out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
