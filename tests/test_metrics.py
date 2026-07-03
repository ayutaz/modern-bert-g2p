"""Unit tests for canonical PER / CER / KER metrics.

Guardrails against silent divergence from the haqumei-eval protocol that we
verified end-to-end in :file:`scripts/eval_haqumei_jsut.py`.
"""

from __future__ import annotations

import pytest

from modernbert_g2p.metrics import compute_cer, compute_ker, compute_per


# ---------- PER ----------


def test_compute_per_perfect_match():
    hyp = ["k", "o", "N", "n", "i", "ch", "i", "w", "a"]
    ref = ["k", "o", "N", "n", "i", "ch", "i", "w", "a"]
    out = compute_per(hyp, ref)
    assert out["per"] == 0.0
    assert out["s"] == 0
    assert out["d"] == 0
    assert out["i"] == 0
    assert out["n"] == len(ref)
    assert out["accuracy"] == 100.0


def test_compute_per_all_wrong():
    hyp = ["z", "z", "z"]
    ref = ["a", "b", "c"]
    out = compute_per(hyp, ref)
    assert out["s"] == 3
    assert out["d"] == 0
    assert out["i"] == 0
    assert out["n"] == 3
    assert out["per"] == pytest.approx(100.0)
    assert out["accuracy"] == 0.0


def test_compute_per_ignore_pau():
    # `pau` on either side must be dropped before scoring.
    hyp = ["pau", "a", "k", "a", "pau"]
    ref = ["a", "k", "a"]
    out = compute_per(hyp, ref)
    assert out["s"] == 0
    assert out["d"] == 0
    assert out["i"] == 0
    assert out["per"] == 0.0


def test_compute_per_ignore_custom_set():
    # Custom ignore set overrides default.
    hyp = ["sil", "a", "sil"]
    ref = ["a"]
    out = compute_per(hyp, ref, ignore={"sil"})
    assert out["per"] == 0.0
    # And without the custom ignore, `sil` counts as insertions.
    out2 = compute_per(hyp, ref, ignore=set())
    assert out2["i"] == 2


def test_compute_per_devoicing_normalize():
    # A/E/I/O/U lowercased -> match; N preserved.
    hyp = ["k", "A", "N"]
    ref = ["k", "a", "N"]
    out_norm = compute_per(hyp, ref, devoicing_normalize=True)
    assert out_norm["per"] == 0.0
    out_raw = compute_per(hyp, ref, devoicing_normalize=False)
    assert out_raw["s"] == 1  # 'A' vs 'a'


def test_compute_per_devoicing_preserves_N():
    # 'N' must never lowercase, even under devoicing_normalize.
    hyp = ["N"]
    ref = ["n"]
    out = compute_per(hyp, ref, devoicing_normalize=True)
    assert out["s"] == 1


def test_compute_per_deletion_and_insertion():
    hyp = ["a", "b"]
    ref = ["a", "b", "c"]
    out = compute_per(hyp, ref)
    assert out["d"] == 1
    assert out["s"] == 0
    assert out["i"] == 0

    hyp2 = ["a", "b", "c", "d"]
    ref2 = ["a", "b", "c"]
    out2 = compute_per(hyp2, ref2)
    assert out2["i"] == 1
    assert out2["s"] == 0
    assert out2["d"] == 0


def test_compute_per_empty_reference():
    # Empty reference -> N=0, PER defined as 0.0, insertions counted.
    out = compute_per(["a", "b"], [])
    assert out["n"] == 0
    assert out["per"] == 0.0
    assert out["i"] == 2


# ---------- CER ----------


def test_compute_cer_basic():
    out = compute_cer("コンニチワ", "コンニチワ")
    assert out["per"] == 0.0
    assert out["n"] == 5


def test_compute_cer_hiragana_katakana_folding():
    # Hiragana input, Katakana reference should normalize to zero error.
    out = compute_cer("こんにちわ", "コンニチワ")
    assert out["per"] == 0.0


def test_compute_cer_substitution():
    out = compute_cer("コンニチハ", "コンニチワ")
    assert out["s"] == 1
    assert out["per"] == pytest.approx(20.0)


def test_compute_cer_no_normalize():
    # Without normalization, hiragana vs katakana all differ.
    out = compute_cer("こんにちわ", "コンニチワ", char_normalize=False)
    assert out["s"] == 5


def test_compute_cer_nfkc_folds_fullwidth():
    # Full-width digit vs half-width digit -> same under NFKC.
    out = compute_cer("A１", "A1")
    assert out["per"] == 0.0


# ---------- KER ----------


def test_compute_ker_basic():
    out = compute_ker("コンニチワ", "コンニチワ")
    assert out["per"] == 0.0
    assert out["n"] == 5


def test_compute_ker_hiragana_folded():
    out = compute_ker("こんにちわ", "コンニチワ")
    assert out["per"] == 0.0


def test_compute_ker_strips_phrase_boundary_marker():
    # Accent phrase delimiter must be stripped before scoring.
    out = compute_ker("コンニチ/ワ", "コンニチワ")
    assert out["per"] == 0.0


def test_compute_ker_substitution():
    out = compute_ker("コンニチハ", "コンニチワ")
    assert out["s"] == 1
    assert out["n"] == 5
    assert out["per"] == pytest.approx(20.0)
