"""Unit tests for the 4-layer contamination filter and its BloomFilter core."""

from __future__ import annotations

import builtins
from pathlib import Path

import pytest

from modernbert_g2p.data.contamination import (
    BloomFilter,
    ContaminationFilter,
    load_held_out_texts,
    normalize_l1,
)

FIXTURES = Path(__file__).parent / "fixtures"
HARD_SET_SAMPLES = (
    Path(__file__).resolve().parents[2] / "data" / "hard_set_seed" / "samples.jsonl"
)


def test_bloom_no_false_negatives():
    bf = BloomFilter(capacity=500, fp_rate=1e-2)
    keys = [f"member-{i}".encode() for i in range(500)]
    for k in keys:
        bf.add(k)
    for k in keys:
        assert k in bf
    assert bf.size == 500
    assert bf.bit_size >= 500


def test_bloom_fp_rate_bounded():
    bf = BloomFilter(capacity=1000, fp_rate=1e-2)
    for i in range(1000):
        bf.add(f"present-{i}".encode())
    n_probes = 10000
    false_positives = 0
    for i in range(n_probes):
        if f"unseen-key-{i}".encode() in bf:
            false_positives += 1
    assert false_positives < 2 * n_probes * 1e-2


def test_bloom_rejects_bad_params():
    with pytest.raises(ValueError):
        BloomFilter(capacity=0, fp_rate=0.1)
    with pytest.raises(ValueError):
        BloomFilter(capacity=100, fp_rate=0.0)
    with pytest.raises(ValueError):
        BloomFilter(capacity=100, fp_rate=1.5)


def test_normalize_l1_collapses_case_ws_and_punct():
    assert normalize_l1("Foo　 Bar") == "foo bar"
    assert normalize_l1("Hello, World!") == "hello world"
    assert normalize_l1("こんにちは、世界。") == "こんにちは世界"


def test_l1_exact_hit():
    cf = ContaminationFilter()
    cf.add_held_out(["こんにちは、世界"])
    hit, layer = cf.is_contaminated("こんにちは、世界")
    assert hit
    assert layer == "L1"


def test_l1_normalized_hit_case_ws_punct():
    cf = ContaminationFilter()
    cf.add_held_out(["Hello, World!"])
    hit, layer = cf.is_contaminated("HELLO   world")
    assert hit
    assert layer == "L1"


def test_l1_hit_after_normalization_of_query_punct():
    cf = ContaminationFilter()
    cf.add_held_out(["こんにちは、世界"])
    hit, layer = cf.is_contaminated("こんにちは、世界！")
    assert hit
    assert layer == "L1"


def test_l2_char_ngram_near_duplicate():
    cf = ContaminationFilter(char_ngram=8, bloom_capacity=10_000, bloom_fp_rate=1e-4)
    cf.add_held_out(["the quick brown fox jumps over the lazy dog"])
    hit, layer = cf.is_contaminated("suddenly the quick brown fox jumped over a fence")
    assert hit
    assert layer == "L2"


def test_l2_character_different_text_not_flagged():
    cf = ContaminationFilter(char_ngram=8, bloom_capacity=10_000, bloom_fp_rate=1e-4)
    cf.add_held_out(["全く関係のないホールドアウトの文章です"])
    hit, layer = cf.is_contaminated("コンピュータサイエンスの研究")
    assert not hit
    assert layer == ""


def test_l3_phoneme_ngram_hit():
    cf = ContaminationFilter(phoneme_ngram=6, bloom_capacity=10_000, bloom_fp_rate=1e-4)
    held_out_phonemes = ["k", "o", "N", "n", "i", "ch", "i", "w", "a"]
    cf.add_held_out(
        ["全く別の日本語の文章"],
        phonemes=[held_out_phonemes],
    )
    hit, layer = cf.is_contaminated(
        "PowerPointで発表する予定です",
        phonemes=held_out_phonemes,
    )
    assert hit
    assert layer == "L3"


def test_l3_skipped_without_phonemes_arg():
    cf = ContaminationFilter(phoneme_ngram=6, bloom_capacity=10_000, bloom_fp_rate=1e-4)
    cf.add_held_out(
        ["別の文章"],
        phonemes=[["k", "o", "N", "n", "i", "ch", "i", "w", "a"]],
    )
    hit, layer = cf.is_contaminated("PowerPointで発表する予定です")
    assert not hit


def test_l4_gracefully_disabled_when_st_missing(monkeypatch):
    orig_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sentence_transformers" or name.startswith("sentence_transformers."):
            raise ImportError("sentence_transformers unavailable in test")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    cf = ContaminationFilter()
    cf.add_held_out(["全く関係のないホールドアウトの文章です"])
    hit, layer = cf.is_contaminated("コンピュータサイエンスの研究")
    assert not hit
    assert layer == ""
    assert cf.stats()["l4_skipped"] > 0


def test_l4_hit_via_injected_fake_encoder():
    cf = ContaminationFilter(embedding_threshold=0.85)

    def fake_encoder(texts):
        return [[1.0, 0.0, 0.0] for _ in texts]

    cf._encoder = fake_encoder
    cf.add_held_out(["全く関係のないホールドアウトの文章です"])
    hit, layer = cf.is_contaminated("コンピュータサイエンスの研究")
    assert hit
    assert layer == "L4"
    assert cf.stats()["l4_hits"] == 1


def test_l4_miss_when_cosine_below_threshold():
    cf = ContaminationFilter(embedding_threshold=0.85)

    def fake_encoder(texts):
        out: list[list[float]] = []
        for t in texts:
            out.append([1.0, 0.0, 0.0] if "held" in t else [0.0, 1.0, 0.0])
        return out

    cf._encoder = fake_encoder
    cf.add_held_out(["held-out sentinel document"])
    hit, layer = cf.is_contaminated("完全に別の文章")
    assert not hit
    assert layer == ""


def test_load_held_out_all_none_returns_empty():
    texts, phonemes = load_held_out_texts(None, None, None, None)
    assert texts == []
    assert phonemes is None


def test_load_held_out_jsut_returns_non_empty():
    texts, phonemes = load_held_out_texts(FIXTURES / "tiny_jsut.yaml", None, None, None)
    assert len(texts) == 3
    assert phonemes is not None
    assert len(phonemes) == 3
    assert phonemes[0][0] == "m"


def test_load_held_out_rohan_strips_parens():
    texts, phonemes = load_held_out_texts(None, FIXTURES / "tiny_rohan.txt", None, None)
    assert len(texts) == 3
    assert "注" not in texts[0]
    assert "(" not in texts[0] and ")" not in texts[0]
    assert phonemes is None


def test_load_held_out_hard_set_returns_non_empty():
    if not HARD_SET_SAMPLES.exists():
        pytest.skip("hard_set samples.jsonl not present in this checkout")
    texts, phonemes = load_held_out_texts(None, None, None, HARD_SET_SAMPLES)
    assert len(texts) > 0
    assert phonemes is not None
    assert any(p for p in phonemes)


def test_load_held_out_jvs_skips_comments(tmp_path):
    jvs = tmp_path / "jvs.txt"
    jvs.write_text(
        "# comment line\nこんにちは\n\nもう一つの文\n",
        encoding="utf-8",
    )
    texts, phonemes = load_held_out_texts(None, None, jvs, None)
    assert texts == ["こんにちは", "もう一つの文"]
    assert phonemes is None


def test_load_held_out_missing_paths_ignored(tmp_path):
    fake = tmp_path / "does_not_exist.yaml"
    texts, phonemes = load_held_out_texts(fake, None, None, None)
    assert texts == []
    assert phonemes is None


def test_stats_shape_and_counters():
    cf = ContaminationFilter()
    cf.add_held_out(["こんにちは、世界"])
    cf.is_contaminated("こんにちは、世界")
    cf.is_contaminated("全然違うクエリ")
    st = cf.stats()
    for key in ("l1_hits", "l2_hits", "l3_hits", "l4_hits", "l4_skipped",
               "held_out_count", "queries"):
        assert key in st
    assert st["queries"] == 2
    assert st["held_out_count"] == 1
    assert st["l1_hits"] >= 1


def test_is_contaminated_layer_priority_l1_before_l2():
    cf = ContaminationFilter(char_ngram=8, bloom_capacity=10_000, bloom_fp_rate=1e-4)
    cf.add_held_out(["exactly this sentence is contaminated"])
    hit, layer = cf.is_contaminated("exactly this sentence is contaminated")
    assert hit
    assert layer == "L1"


def test_add_held_out_with_mixed_phoneme_lengths_works():
    cf = ContaminationFilter(phoneme_ngram=6, bloom_capacity=10_000, bloom_fp_rate=1e-4)
    cf.add_held_out(
        ["text a", "text b", "text c"],
        phonemes=[
            ["a", "b"],
            ["k", "o", "N", "n", "i", "ch", "i"],
            [],
        ],
    )
    assert cf.stats()["held_out_count"] == 3
