"""Tests for the two pilot tokenizer wrappers (P-A / P-C).

Heavy runtime dependencies (``transformers``) are guarded with
``pytest.importorskip`` inside each test that requires them. Tests that only
exercise the pure-Python target-side encoding run without any ML stack.
"""

from __future__ import annotations

from typing import Any

import pytest

from modernbert_g2p.models.canonical import (
    CanonicalForm,
    build_default_vocab,
    p_a_to_canonical,
    p_c_to_canonical,
)
from modernbert_g2p.models.tokenization import (
    BaseTokenizer,
    PATokenizer,
    PCTokenizer,
    get_tokenizer,
)
from modernbert_g2p.models.tokenization.base import BIO_LABEL_TO_ID, ID_TO_BIO_LABEL
from modernbert_g2p.models.tokenization.p_a_tokenizer import MORAIC_PHONEMES, split_moras


class _FakeHFTokenizer:
    """Minimal HF-tokenizer-shaped stub for offline tests."""

    def __init__(self, cls_id: int = 1, sep_id: int = 2, pad_id: int = 0, unk_id: int = 3) -> None:
        self.cls_token_id = cls_id
        self.sep_token_id = sep_id
        self.pad_token_id = pad_id
        self.unk_token_id = unk_id
        self._vocab: dict[str, int] = {}
        self._added: dict[str, int] = {}
        self._next_id = 100

    def get_vocab(self) -> dict[str, int]:
        merged = dict(self._vocab)
        merged.update(self._added)
        return merged

    def add_special_tokens(self, mapping: dict[str, Any]) -> int:
        added = 0
        for tok in mapping.get("additional_special_tokens", []):
            if tok not in self._added:
                self._added[tok] = self._next_id
                self._next_id += 1
                added += 1
        return added

    def convert_tokens_to_ids(self, token: str) -> int:
        if token in self._added:
            return self._added[token]
        if token in self._vocab:
            return self._vocab[token]
        return self.unk_token_id

    def __call__(self, text: str, **kwargs: Any) -> dict[str, list[int]]:
        add_specials = kwargs.get("add_special_tokens", True)
        truncation = kwargs.get("truncation", False)
        max_length = kwargs.get("max_length", 512)
        chars = list(text)
        ids: list[int] = []
        for c in chars:
            if c not in self._vocab:
                self._vocab[c] = self._next_id
                self._next_id += 1
            ids.append(self._vocab[c])
        if add_specials:
            ids = [self.cls_token_id] + ids + [self.sep_token_id]
        if truncation and len(ids) > max_length:
            ids = ids[:max_length]
        result: dict[str, list[int]] = {"input_ids": ids}
        if kwargs.get("return_attention_mask", True):
            result["attention_mask"] = [1] * len(ids)
        return result


def _make_canonical_watashi_wa() -> CanonicalForm:
    """Reference form for 私は = w a / t a / sh i / w a with 4 moras."""
    return CanonicalForm(
        phonemes=("w", "a", "t", "a", "sh", "i", "w", "a"),
        mora_accents=("L", "H", "L", "L"),
        accent_boundaries=(3,),
    )


def test_all_tokenizers_import_lazily() -> None:
    """Importing the tokenization package must not drag torch in at import time.

    Guarded via subprocess so prior tests do not pollute sys.modules for this
    assertion.
    """
    import subprocess
    import sys as _sys

    code = (
        "import sys; from modernbert_g2p.models import tokenization; "
        "sys.exit(0 if 'torch' not in sys.modules else 1)"
    )
    result = subprocess.run([_sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, f"tokenization package eagerly imported torch: {result.stdout} {result.stderr}"
    for cls in (PATokenizer, PCTokenizer):
        assert cls.name in {"p_a", "p_c"}


def test_all_tokenizers_satisfy_base_protocol() -> None:
    """Both tokenizer instances structurally satisfy BaseTokenizer."""
    fake = _FakeHFTokenizer()
    pa = PATokenizer(hf_tokenizer=fake)
    pc = PCTokenizer(hf_tokenizer=fake)
    for tk in (pa, pc):
        assert isinstance(tk, BaseTokenizer)
        assert hasattr(tk, "encode_input")
        assert hasattr(tk, "encode")
        assert hasattr(tk, "encode_target")
        assert hasattr(tk, "decode_output")
        assert isinstance(tk.pad_token_id, int)


def test_split_moras_hand_examples() -> None:
    """split_moras chunks by moraic nucleus and attaches trailing consonants."""
    assert split_moras(["w", "a", "t", "a"]) == [["w", "a"], ["t", "a"]]
    assert split_moras(["ky", "o", "u"]) == [["ky", "o"], ["u"]]
    assert split_moras(["N"]) == [["N"]]
    assert split_moras([]) == []
    assert split_moras(["s", "s"]) == [["s", "s"]]


def test_moraic_set_matches_canonical() -> None:
    assert "a" in MORAIC_PHONEMES
    assert "N" in MORAIC_PHONEMES
    assert "q" in MORAIC_PHONEMES
    assert "cl" in MORAIC_PHONEMES
    assert "k" not in MORAIC_PHONEMES
    assert "sh" not in MORAIC_PHONEMES


def test_bio_label_id_mapping_is_bijective() -> None:
    assert BIO_LABEL_TO_ID == {"O": 0, "B": 1, "I": 2}
    assert ID_TO_BIO_LABEL == {0: "O", 1: "B", 2: "I"}


class TestPATokenizer:
    def test_instantiates_with_defaults(self) -> None:
        tk = PATokenizer(hf_tokenizer=_FakeHFTokenizer())
        assert tk.name == "p_a"
        assert tk.max_input_length == 512
        assert tk.max_target_length == 256
        assert tk.pad_token_id == tk.vocab.pad_id == 0

    def test_encode_input_returns_ids_and_mask(self) -> None:
        tk = PATokenizer(hf_tokenizer=_FakeHFTokenizer())
        out = tk.encode_input("こんにちは")
        assert "input_ids" in out
        assert "attention_mask" in out
        assert len(out["input_ids"]) == len(out["attention_mask"])
        assert all(m == 1 for m in out["attention_mask"])

    def test_encode_input_truncates_to_max_length(self) -> None:
        tk = PATokenizer(hf_tokenizer=_FakeHFTokenizer(), max_input_length=10)
        out = tk.encode_input("あいうえおかきくけこさしすせそ")
        assert len(out["input_ids"]) <= 10

    def test_encode_target_returns_list_of_ints(self) -> None:
        tk = PATokenizer(hf_tokenizer=_FakeHFTokenizer())
        cf = CanonicalForm(phonemes=("w", "a"), mora_accents=("H",), accent_boundaries=())
        out = tk.encode_target(cf)
        assert isinstance(out, list)
        assert all(isinstance(x, int) for x in out)

    def test_encode_target_prepends_bos_appends_eos(self) -> None:
        tk = PATokenizer(hf_tokenizer=_FakeHFTokenizer())
        cf = CanonicalForm(phonemes=("w", "a"), mora_accents=("H",), accent_boundaries=())
        out = tk.encode_target(cf)
        assert out[0] == tk.vocab.bos_id
        assert out[-1] == tk.vocab.eos_id
        assert len(out) >= 2

    def test_encode_target_interleaves_phoneme_hl(self) -> None:
        tk = PATokenizer(hf_tokenizer=_FakeHFTokenizer())
        vocab = tk.vocab
        cf = CanonicalForm(phonemes=("w", "a"), mora_accents=("H",), accent_boundaries=())
        out = tk.encode_target(cf)
        expected = [
            vocab.bos_id,
            vocab.id_of("w"),
            vocab.id_of("a"),
            vocab.high_id,
            vocab.eos_id,
        ]
        assert out == expected

    def test_encode_target_inserts_boundary_before_mora(self) -> None:
        tk = PATokenizer(hf_tokenizer=_FakeHFTokenizer())
        vocab = tk.vocab
        cf = CanonicalForm(
            phonemes=("k", "a", "n", "i"),
            mora_accents=("L", "H"),
            accent_boundaries=(1,),
        )
        out = tk.encode_target(cf)
        expected = [
            vocab.bos_id,
            vocab.id_of("k"),
            vocab.id_of("a"),
            vocab.low_id,
            vocab.boundary_id,
            vocab.id_of("n"),
            vocab.id_of("i"),
            vocab.high_id,
            vocab.eos_id,
        ]
        assert out == expected

    def test_encode_target_roundtrip_via_p_a_to_canonical(self) -> None:
        tk = PATokenizer(hf_tokenizer=_FakeHFTokenizer())
        cf = _make_canonical_watashi_wa()
        ids = tk.encode_target(cf)
        recovered = p_a_to_canonical(ids, tk.vocab)
        assert recovered == cf

    def test_encode_target_roundtrip_multiple_patterns(self) -> None:
        tk = PATokenizer(hf_tokenizer=_FakeHFTokenizer())
        patterns = [
            CanonicalForm(("a",), ("H",), ()),
            CanonicalForm(("k", "a"), ("L",), ()),
            CanonicalForm(("N",), ("H",), ()),
            CanonicalForm(("k", "y", "o", "u"), ("H", "L"), ()),
            CanonicalForm(("s", "a", "k", "u", "r", "a"), ("L", "H", "L"), ()),
            CanonicalForm(
                ("t", "o", "u", "k", "y", "o", "u"),
                ("L", "H", "H", "H"),
                (2,),
            ),
            CanonicalForm(("q", "t", "a"), ("L", "H"), ()),
            CanonicalForm(("cl", "k", "a"), ("H", "L"), ()),
            CanonicalForm(("a", "i", "u", "e", "o"), ("H", "L", "H", "L", "H"), (2, 4)),
            CanonicalForm(("m", "e", "N", "d", "o", "u"), ("L", "H", "H", "H"), ()),
        ]
        for cf in patterns:
            ids = tk.encode_target(cf)
            recovered = p_a_to_canonical(ids, tk.vocab)
            assert recovered == cf, f"roundtrip failed for {cf}"

    def test_encode_target_uses_custom_vocab(self) -> None:
        vocab = build_default_vocab()
        tk = PATokenizer(hf_tokenizer=_FakeHFTokenizer(), phoneme_vocab=vocab)
        assert tk.vocab is vocab

    def test_decode_output_renders_tokens(self) -> None:
        tk = PATokenizer(hf_tokenizer=_FakeHFTokenizer())
        vocab = tk.vocab
        ids = [vocab.id_of("k"), vocab.id_of("a"), vocab.high_id]
        rendered = tk.decode_output(ids)
        assert rendered == "k a H"

    def test_encode_input_real_modernbert_tokenizer(self) -> None:
        pytest.importorskip("transformers")
        try:
            tk = PATokenizer()
            out = tk.encode_input("こんにちは")
        except Exception as exc:  # network / cache miss
            pytest.skip(f"modernbert-ja tokenizer unavailable: {exc}")
        assert len(out["input_ids"]) >= 1
        assert len(out["input_ids"]) == len(out["attention_mask"])

    def test_encode_batched_shape_matches_encode_input(self) -> None:
        tk = PATokenizer(hf_tokenizer=_FakeHFTokenizer())
        texts = ["こん", "にちは"]
        out = tk.encode(texts)
        assert list(out.keys())[:2] == ["input_ids", "attention_mask"]
        assert len(out["input_ids"]) == 2
        assert len(out["attention_mask"]) == 2
        for i, t in enumerate(texts):
            per = tk.encode_input(t)
            assert out["input_ids"][i] == per["input_ids"]
            assert out["attention_mask"][i] == per["attention_mask"]

    def test_encode_batched_empty_list(self) -> None:
        tk = PATokenizer(hf_tokenizer=_FakeHFTokenizer())
        out = tk.encode([])
        assert out == {"input_ids": [], "attention_mask": []}


class TestPCTokenizer:
    def test_instantiates_with_defaults(self) -> None:
        tk = PCTokenizer(hf_tokenizer=_FakeHFTokenizer())
        assert tk.name == "p_c"
        assert tk.max_slot == 8
        assert tk.max_input_length == 512

    def test_encode_input_length_matches_text(self) -> None:
        tk = PCTokenizer(hf_tokenizer=_FakeHFTokenizer())
        text = "こんにちは"
        out = tk.encode_input(text)
        assert len(out["input_ids"]) == len(text) + 2
        assert len(out["attention_mask"]) == len(out["input_ids"])

    def test_encode_input_real_char_v2_tokenizer(self) -> None:
        pytest.importorskip("transformers")
        try:
            tk = PCTokenizer()
            out = tk.encode_input("こんにちは")
            hf = tk.get_hf_tokenizer()
            tokens = hf.convert_ids_to_tokens(out["input_ids"])
        except Exception as exc:
            pytest.skip(f"char-v2 tokenizer unavailable: {exc}")
        core = tokens[1:-1]
        continuations = [t for t in core if t.startswith("##")]
        assert len(continuations) == 0
        assert len(core) == len("こんにちは")

    def test_encode_target_shapes(self) -> None:
        tk = PCTokenizer(hf_tokenizer=_FakeHFTokenizer(), max_slot=8)
        cf = _make_canonical_watashi_wa()
        char_alignment = [[0, 1, 2, 3, 4, 5], [6, 7]]
        out = tk.encode_target(cf, char_alignment)
        assert len(out["per_char_phon_ids"]) == 2
        for row in out["per_char_phon_ids"]:
            assert len(row) == 8
        for row in out["per_char_hl_ids"]:
            assert len(row) == 8
        assert len(out["apbp_bio_ids"]) == 2

    def test_encode_target_uses_vocab_ids(self) -> None:
        tk = PCTokenizer(hf_tokenizer=_FakeHFTokenizer())
        vocab = tk.vocab
        cf = CanonicalForm(phonemes=("k", "a"), mora_accents=("L",), accent_boundaries=())
        out = tk.encode_target(cf, [[0, 1]])
        assert out["per_char_phon_ids"][0][0] == vocab.id_of("k")
        assert out["per_char_phon_ids"][0][1] == vocab.id_of("a")
        assert out["per_char_hl_ids"][0][0] == vocab.low_id
        assert out["per_char_phon_ids"][0][2] == vocab.pad_id
        assert out["per_char_hl_ids"][0][1] == vocab.pad_id

    def test_encode_target_apbp_boundary(self) -> None:
        tk = PCTokenizer(hf_tokenizer=_FakeHFTokenizer())
        cf = _make_canonical_watashi_wa()
        out = tk.encode_target(cf, [[0, 1, 2, 3, 4, 5], [6, 7]])
        assert out["apbp_bio_ids"][0] == BIO_LABEL_TO_ID["O"]
        assert out["apbp_bio_ids"][1] == BIO_LABEL_TO_ID["B"]

    def test_encode_target_truncates_slots_with_warning(self) -> None:
        tk = PCTokenizer(hf_tokenizer=_FakeHFTokenizer(), max_slot=4)
        cf = CanonicalForm(
            phonemes=("k", "a", "s", "a", "n", "a"),
            mora_accents=("H", "L", "L"),
            accent_boundaries=(),
        )
        char_alignment = [[0, 1, 2, 3, 4, 5]]
        with pytest.warns(UserWarning, match="max_slot"):
            out = tk.encode_target(cf, char_alignment)
        assert len(out["per_char_phon_ids"][0]) == 4

    def test_encode_target_roundtrip_via_p_c_to_canonical(self) -> None:
        tk = PCTokenizer(hf_tokenizer=_FakeHFTokenizer())
        vocab = tk.vocab
        cf = _make_canonical_watashi_wa()
        char_alignment = [[0, 1, 2, 3, 4, 5], [6, 7]]
        out = tk.encode_target(cf, char_alignment)
        id_to_hl = {vocab.pad_id: "", vocab.high_id: "H", vocab.low_id: "L"}
        per_char_hl_strs = [
            [id_to_hl.get(x, "") for x in row] for row in out["per_char_hl_ids"]
        ]
        apbp_tags = [ID_TO_BIO_LABEL[x] for x in out["apbp_bio_ids"]]
        recovered = p_c_to_canonical(
            out["per_char_phon_ids"], per_char_hl_strs, apbp_tags, vocab
        )
        assert recovered == cf

    def test_encode_target_first_char_bio_is_outside_default(self) -> None:
        tk = PCTokenizer(hf_tokenizer=_FakeHFTokenizer())
        cf = CanonicalForm(phonemes=("k", "a"), mora_accents=("L",), accent_boundaries=())
        out = tk.encode_target(cf, [[0, 1]])
        assert out["apbp_bio_ids"][0] == BIO_LABEL_TO_ID["O"]

    def test_encode_target_non_boundary_after_first_is_inside(self) -> None:
        tk = PCTokenizer(hf_tokenizer=_FakeHFTokenizer())
        cf = CanonicalForm(
            phonemes=("k", "a", "s", "a"),
            mora_accents=("L", "H"),
            accent_boundaries=(),
        )
        out = tk.encode_target(cf, [[0, 1], [2, 3]])
        assert out["apbp_bio_ids"][1] == BIO_LABEL_TO_ID["I"]

    def test_encode_batched_nests_char_positions(self) -> None:
        tk = PCTokenizer(hf_tokenizer=_FakeHFTokenizer())
        texts = ["こんにちは", "さよなら"]
        out = tk.encode(texts)
        assert set(out.keys()) == {"input_ids", "attention_mask", "extras"}
        assert set(out["extras"].keys()) == {"char_positions"}
        assert "char_positions" not in out
        assert len(out["input_ids"]) == 2
        assert len(out["extras"]["char_positions"]) == 2
        # _FakeHFTokenizer prepends CLS and appends SEP → char_positions start at 1
        assert out["extras"]["char_positions"][0] == list(range(1, 6))
        assert out["extras"]["char_positions"][1] == list(range(1, 5))

    def test_encode_batched_matches_encode_input(self) -> None:
        tk = PCTokenizer(hf_tokenizer=_FakeHFTokenizer())
        text = "こんにちは"
        per = tk.encode_input(text)
        out = tk.encode([text])
        assert out["input_ids"][0] == per["input_ids"]
        assert out["attention_mask"][0] == per["attention_mask"]

    def test_encode_batched_positions_within_seq_len(self) -> None:
        tk = PCTokenizer(hf_tokenizer=_FakeHFTokenizer(), max_input_length=6)
        text = "あいうえおかきくけこ"
        out = tk.encode([text])
        ids = out["input_ids"][0]
        positions = out["extras"]["char_positions"][0]
        for p in positions:
            assert 0 <= p < len(ids)
        assert len(positions) <= max(0, len(ids) - 2)

    def test_encode_batched_empty_list(self) -> None:
        tk = PCTokenizer(hf_tokenizer=_FakeHFTokenizer())
        out = tk.encode([])
        assert out["input_ids"] == []
        assert out["attention_mask"] == []
        assert out["extras"] == {"char_positions": []}


class TestGetTokenizerFactory:
    def test_returns_pa(self) -> None:
        tk = get_tokenizer("p_a", hf_tokenizer=_FakeHFTokenizer())
        assert isinstance(tk, PATokenizer)
        assert tk.name == "p_a"

    def test_returns_pc(self) -> None:
        tk = get_tokenizer("p_c", hf_tokenizer=_FakeHFTokenizer())
        assert isinstance(tk, PCTokenizer)
        assert tk.name == "p_c"

    def test_accepts_aliases(self) -> None:
        for alias in ("PA", "pa", "p-a", "P_A"):
            tk = get_tokenizer(alias, hf_tokenizer=_FakeHFTokenizer())
            assert isinstance(tk, PATokenizer)

    def test_raises_on_removed_pb(self) -> None:
        with pytest.raises(ValueError, match="unknown pilot"):
            get_tokenizer("p_b")

    def test_raises_on_unknown_pilot(self) -> None:
        with pytest.raises(ValueError, match="unknown pilot"):
            get_tokenizer("p_x")

    def test_forwards_kwargs(self) -> None:
        tk = get_tokenizer(
            "p_a",
            hf_tokenizer=_FakeHFTokenizer(),
            max_input_length=64,
        )
        assert tk.max_input_length == 64
