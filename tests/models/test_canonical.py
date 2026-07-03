"""Tests for canonical vocab and per-pilot canonicalizers (design doc §3.4)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from modernbert_g2p.models.canonical import (
    DROPPED_PHONEMES,
    JULIUS_PHONEMES,
    PAUSE_TOKEN,
    PROSODY_TOKENS,
    SPECIAL_TOKENS,
    CanonicalForm,
    Vocab,
    build_default_vocab,
    p_a_to_canonical,
    p_b_to_canonical,
    p_c_to_canonical,
)


def _ids(vocab: Vocab, tokens: str) -> list[int]:
    return [vocab.id_of(t) for t in tokens.split()]


class TestVocab:
    def test_size_matches_specials_plus_phonemes(self) -> None:
        vocab = build_default_vocab()
        assert vocab.size == len(JULIUS_PHONEMES) + len(SPECIAL_TOKENS)

    def test_reserved_ids_are_stable(self) -> None:
        vocab = build_default_vocab()
        assert vocab.pad_id == 0
        assert vocab.bos_id == 1
        assert vocab.eos_id == 2
        assert vocab.unk_id == 3
        assert vocab.token_of(0) == vocab.PAD_TOKEN
        assert vocab.token_of(1) == vocab.BOS_TOKEN
        assert vocab.token_of(2) == vocab.EOS_TOKEN
        assert vocab.token_of(3) == vocab.UNK_TOKEN

    def test_first_phoneme_id_is_four(self) -> None:
        vocab = build_default_vocab()
        assert vocab.id_of("a") == 4
        assert vocab.token_of(4) == "a"

    def test_prosody_tokens_at_tail(self) -> None:
        vocab = build_default_vocab()
        assert vocab.high_id == 4 + len(JULIUS_PHONEMES)
        assert vocab.low_id == 5 + len(JULIUS_PHONEMES)
        assert vocab.boundary_id == 6 + len(JULIUS_PHONEMES)
        for tok in PROSODY_TOKENS:
            assert vocab.token_of(vocab.id_of(tok)) == tok

    def test_id_token_roundtrip_all_phonemes(self) -> None:
        vocab = build_default_vocab()
        for ph in JULIUS_PHONEMES:
            assert vocab.token_of(vocab.id_of(ph)) == ph

    def test_unknown_token_maps_to_unk_id(self) -> None:
        vocab = build_default_vocab()
        assert vocab.id_of("ZZ_not_a_phoneme") == vocab.unk_id
        assert vocab.id_of("") == vocab.unk_id

    def test_unk_roundtrip(self) -> None:
        vocab = build_default_vocab()
        assert vocab.token_of(vocab.unk_id) == vocab.UNK_TOKEN
        assert vocab.id_of(vocab.UNK_TOKEN) == vocab.unk_id

    def test_out_of_range_token_of_falls_back_to_unk(self) -> None:
        vocab = build_default_vocab()
        assert vocab.token_of(vocab.size) == vocab.UNK_TOKEN
        assert vocab.token_of(-1) == vocab.UNK_TOKEN

    def test_custom_phoneme_set(self) -> None:
        vocab = Vocab(phoneme_tokens=("a", "b"))
        assert vocab.size == 4 + 2 + 3
        assert vocab.id_of("a") == 4
        assert vocab.id_of("b") == 5
        assert vocab.high_id == 6


class TestCanonicalForm:
    def test_is_frozen(self) -> None:
        cf = CanonicalForm(phonemes=("a",), mora_accents=("H",), accent_boundaries=())
        with pytest.raises(FrozenInstanceError):
            cf.phonemes = ("b",)  # type: ignore[misc]

    def test_equality_and_hash(self) -> None:
        a = CanonicalForm(phonemes=("a", "k"), mora_accents=("H",), accent_boundaries=(1,))
        b = CanonicalForm(phonemes=("a", "k"), mora_accents=("H",), accent_boundaries=(1,))
        assert a == b
        assert hash(a) == hash(b)
        assert {a: 1}[b] == 1


class TestPAtoCanonical:
    def setup_method(self) -> None:
        self.vocab = build_default_vocab()

    def test_watashi_hl_pattern(self) -> None:
        ids = _ids(self.vocab, "w a t a H sh i L")
        cf = p_a_to_canonical(ids, self.vocab)
        assert cf.phonemes == ("w", "a", "t", "a", "sh", "i")
        assert cf.mora_accents == ("H", "L")
        assert cf.accent_boundaries == ()

    def test_boundary_between_two_morae(self) -> None:
        ids = _ids(self.vocab, "w a L / y u H")
        cf = p_a_to_canonical(ids, self.vocab)
        assert cf.phonemes == ("w", "a", "y", "u")
        assert cf.mora_accents == ("L", "H")
        assert cf.accent_boundaries == (1,)

    def test_bos_pad_stripped_at_start(self) -> None:
        ids = [self.vocab.bos_id, self.vocab.pad_id, *_ids(self.vocab, "a H")]
        cf = p_a_to_canonical(ids, self.vocab)
        assert cf.phonemes == ("a",)
        assert cf.mora_accents == ("H",)

    def test_eos_stripped_at_end(self) -> None:
        ids = [*_ids(self.vocab, "a H k a L"), self.vocab.eos_id]
        cf = p_a_to_canonical(ids, self.vocab)
        assert cf.phonemes == ("a", "k", "a")
        assert cf.mora_accents == ("H", "L")

    def test_multiple_boundary_tokens(self) -> None:
        ids = _ids(self.vocab, "a H / k a L / m i H")
        cf = p_a_to_canonical(ids, self.vocab)
        assert cf.phonemes == ("a", "k", "a", "m", "i")
        assert cf.mora_accents == ("H", "L", "H")
        assert cf.accent_boundaries == (1, 2)

    def test_empty_input(self) -> None:
        cf = p_a_to_canonical([], self.vocab)
        assert cf == CanonicalForm((), (), ())

    def test_only_specials_input(self) -> None:
        ids = [self.vocab.bos_id, self.vocab.pad_id, self.vocab.eos_id]
        cf = p_a_to_canonical(ids, self.vocab)
        assert cf == CanonicalForm((), (), ())

    def test_trailing_phones_without_accent_are_appended(self) -> None:
        ids = _ids(self.vocab, "a H k a")
        cf = p_a_to_canonical(ids, self.vocab)
        assert cf.phonemes == ("a", "k", "a")
        assert cf.mora_accents == ("H",)

    def test_boundary_at_index_zero(self) -> None:
        ids = _ids(self.vocab, "/ a H")
        cf = p_a_to_canonical(ids, self.vocab)
        assert cf.accent_boundaries == (0,)
        assert cf.phonemes == ("a",)
        assert cf.mora_accents == ("H",)

    def test_unk_tokens_are_skipped(self) -> None:
        ids = [self.vocab.id_of("a"), self.vocab.unk_id, self.vocab.id_of("H")]
        cf = p_a_to_canonical(ids, self.vocab)
        assert cf.phonemes == ("a",)
        assert cf.mora_accents == ("H",)

    def test_polyphone_like_palatalized(self) -> None:
        ids = _ids(self.vocab, "ky o H q k a L")
        cf = p_a_to_canonical(ids, self.vocab)
        assert cf.phonemes == ("ky", "o", "q", "k", "a")
        assert cf.mora_accents == ("H", "L")


class TestPBtoCanonical:
    def setup_method(self) -> None:
        self.vocab = build_default_vocab()

    def _slot(self, tokens: str) -> list[int]:
        return [self.vocab.id_of(t) for t in tokens.split()]

    def test_two_morphs_with_b_tag_inserts_boundary(self) -> None:
        phon_slots = [self._slot("w a t a sh i"), self._slot("d e s u")]
        hl_slots = [["L", "H", "H"], ["L", "L"]]
        apbp = ["O", "B"]
        cf = p_b_to_canonical(phon_slots, hl_slots, apbp, self.vocab)
        assert cf.phonemes == ("w", "a", "t", "a", "sh", "i", "d", "e", "s", "u")
        assert cf.mora_accents == ("L", "H", "H", "L", "L")
        assert cf.accent_boundaries == (3,)

    def test_pad_slots_stripped(self) -> None:
        phon_slots = [self._slot("k a") + [self.vocab.pad_id, self.vocab.pad_id]]
        hl_slots = [["L", "H", self.vocab.PAD_TOKEN, self.vocab.PAD_TOKEN]]
        apbp = ["O"]
        cf = p_b_to_canonical(phon_slots, hl_slots, apbp, self.vocab)
        assert cf.phonemes == ("k", "a")
        assert cf.mora_accents == ("L", "H")
        assert cf.accent_boundaries == ()

    def test_all_o_produces_no_boundary(self) -> None:
        phon_slots = [self._slot("a"), self._slot("i"), self._slot("u")]
        hl_slots = [["H"], ["L"], ["H"]]
        apbp = ["O", "O", "O"]
        cf = p_b_to_canonical(phon_slots, hl_slots, apbp, self.vocab)
        assert cf.accent_boundaries == ()
        assert cf.mora_accents == ("H", "L", "H")

    def test_b_on_first_morph_ignored(self) -> None:
        phon_slots = [self._slot("a"), self._slot("i")]
        hl_slots = [["H"], ["L"]]
        apbp = ["B", "O"]
        cf = p_b_to_canonical(phon_slots, hl_slots, apbp, self.vocab)
        assert cf.accent_boundaries == ()

    def test_multi_morph_multiple_boundaries(self) -> None:
        phon_slots = [
            self._slot("a"),
            self._slot("k a"),
            self._slot("s a"),
            self._slot("t a"),
        ]
        hl_slots = [["H"], ["L"], ["H"], ["L"]]
        apbp = ["O", "B", "O", "B"]
        cf = p_b_to_canonical(phon_slots, hl_slots, apbp, self.vocab)
        assert cf.accent_boundaries == (1, 3)
        assert cf.phonemes == ("a", "k", "a", "s", "a", "t", "a")
        assert cf.mora_accents == ("H", "L", "H", "L")

    def test_empty_input(self) -> None:
        cf = p_b_to_canonical([], [], [], self.vocab)
        assert cf == CanonicalForm((), (), ())

    def test_zip_strict_mismatched_raises(self) -> None:
        with pytest.raises(ValueError):
            p_b_to_canonical([[]], [["H"], ["L"]], ["O"], self.vocab)


class TestPCtoCanonical:
    def setup_method(self) -> None:
        self.vocab = build_default_vocab()

    def _slot(self, tokens: str) -> list[int]:
        return [self.vocab.id_of(t) for t in tokens.split()]

    def test_multi_phoneme_kanji(self) -> None:
        # 漢字 → k a N j i, over 2 chars.
        phon_slots = [self._slot("k a N"), self._slot("j i")]
        hl_slots = [["L", "H", "H"], ["L", "L"]]
        apbp = ["O", "O"]
        cf = p_c_to_canonical(phon_slots, hl_slots, apbp, self.vocab)
        assert cf.phonemes == ("k", "a", "N", "j", "i")
        assert cf.mora_accents == ("L", "H", "H", "L", "L")
        assert cf.accent_boundaries == ()

    def test_pad_slots_stripped(self) -> None:
        phon_slots = [
            self._slot("k a") + [self.vocab.pad_id] * 6,
            self._slot("s a") + [self.vocab.pad_id] * 6,
        ]
        hl_slots = [
            ["L", "H", *([self.vocab.PAD_TOKEN] * 6)],
            ["L", "H", *([self.vocab.PAD_TOKEN] * 6)],
        ]
        apbp = ["O", "O"]
        cf = p_c_to_canonical(phon_slots, hl_slots, apbp, self.vocab)
        assert cf.phonemes == ("k", "a", "s", "a")
        assert cf.mora_accents == ("L", "H", "L", "H")

    def test_bio_b_inserts_boundary(self) -> None:
        phon_slots = [self._slot("k a"), self._slot("s a")]
        hl_slots = [["L", "H"], ["L", "H"]]
        apbp = ["O", "B"]
        cf = p_c_to_canonical(phon_slots, hl_slots, apbp, self.vocab)
        assert cf.accent_boundaries == (2,)

    def test_all_o_no_boundary(self) -> None:
        phon_slots = [self._slot("a"), self._slot("i"), self._slot("u")]
        hl_slots = [["H"], ["L"], ["H"]]
        apbp = ["O", "O", "O"]
        cf = p_c_to_canonical(phon_slots, hl_slots, apbp, self.vocab)
        assert cf.accent_boundaries == ()

    def test_b_on_first_char_ignored(self) -> None:
        phon_slots = [self._slot("a"), self._slot("i")]
        hl_slots = [["H"], ["L"]]
        apbp = ["B", "O"]
        cf = p_c_to_canonical(phon_slots, hl_slots, apbp, self.vocab)
        assert cf.accent_boundaries == ()

    def test_sokuon_geminate(self) -> None:
        # 促音 (っ) collapses in the phoneme stream as ``q``; carries no H/L.
        phon_slots = [self._slot("k a"), self._slot("q"), self._slot("t a")]
        hl_slots = [["L", "H"], [], ["L", "H"]]
        apbp = ["O", "O", "O"]
        cf = p_c_to_canonical(phon_slots, hl_slots, apbp, self.vocab)
        assert cf.phonemes == ("k", "a", "q", "t", "a")
        assert cf.mora_accents == ("L", "H", "L", "H")

    def test_empty_input(self) -> None:
        cf = p_c_to_canonical([], [], [], self.vocab)
        assert cf == CanonicalForm((), (), ())


class TestPauseTokenInvariants:
    """Design doc §3.4: no pilot may surface ``pau`` in the canonical form."""

    def test_pau_not_in_julius_phoneme_set(self) -> None:
        assert PAUSE_TOKEN not in JULIUS_PHONEMES

    def test_pause_token_constant(self) -> None:
        assert PAUSE_TOKEN == "pau"
        assert PAUSE_TOKEN in DROPPED_PHONEMES

    def test_default_vocab_does_not_expose_pau(self) -> None:
        vocab = build_default_vocab()
        assert vocab.id_of(PAUSE_TOKEN) == vocab.unk_id

    def test_p_a_strips_pau_from_custom_vocab(self) -> None:
        vocab = Vocab(phoneme_tokens=("a", "k", PAUSE_TOKEN))
        ids = [
            vocab.id_of("a"),
            vocab.id_of("H"),
            vocab.id_of(PAUSE_TOKEN),
            vocab.id_of("k"),
            vocab.id_of("a"),
            vocab.id_of("L"),
        ]
        cf = p_a_to_canonical(ids, vocab)
        assert cf.phonemes == ("a", "k", "a")
        assert cf.mora_accents == ("H", "L")

    def test_p_a_strips_pau_at_boundary_position(self) -> None:
        vocab = Vocab(phoneme_tokens=("a", "k", PAUSE_TOKEN))
        ids = [
            vocab.id_of("a"),
            vocab.id_of("H"),
            vocab.id_of("/"),
            vocab.id_of(PAUSE_TOKEN),
            vocab.id_of("k"),
            vocab.id_of("a"),
            vocab.id_of("L"),
        ]
        cf = p_a_to_canonical(ids, vocab)
        assert cf.phonemes == ("a", "k", "a")
        assert cf.mora_accents == ("H", "L")
        assert cf.accent_boundaries == (1,)

    def test_p_b_strips_pau_from_phon_slots(self) -> None:
        vocab = Vocab(phoneme_tokens=("k", "a", PAUSE_TOKEN))
        pau_id = vocab.id_of(PAUSE_TOKEN)
        phon_slots = [
            [vocab.id_of("k"), vocab.id_of("a"), pau_id],
            [pau_id, vocab.id_of("k"), vocab.id_of("a")],
        ]
        hl_slots = [["L", "H"], ["L", "H"]]
        apbp = ["O", "O"]
        cf = p_b_to_canonical(phon_slots, hl_slots, apbp, vocab)
        assert cf.phonemes == ("k", "a", "k", "a")
        assert cf.mora_accents == ("L", "H", "L", "H")

    def test_p_c_strips_pau_from_phon_slots(self) -> None:
        vocab = Vocab(phoneme_tokens=("k", "a", PAUSE_TOKEN))
        pau_id = vocab.id_of(PAUSE_TOKEN)
        phon_slots = [
            [vocab.id_of("k"), vocab.id_of("a")],
            [pau_id],
            [vocab.id_of("k"), vocab.id_of("a")],
        ]
        hl_slots = [["L", "H"], [], ["L", "H"]]
        apbp = ["O", "O", "O"]
        cf = p_c_to_canonical(phon_slots, hl_slots, apbp, vocab)
        assert cf.phonemes == ("k", "a", "k", "a")
        assert cf.mora_accents == ("L", "H", "L", "H")

    def test_default_vocab_unk_id_still_stripped(self) -> None:
        # If a pilot uses the default vocab (no `pau` entry), an emitted pau
        # would map to <unk> and still must not appear in canonical.
        vocab = build_default_vocab()
        ids = [
            vocab.id_of("a"),
            vocab.id_of("H"),
            vocab.id_of(PAUSE_TOKEN),
            vocab.id_of("k"),
            vocab.id_of("a"),
            vocab.id_of("L"),
        ]
        cf = p_a_to_canonical(ids, vocab)
        assert PAUSE_TOKEN not in cf.phonemes
        assert cf.phonemes == ("a", "k", "a")


class TestSpecialTokenInvariants:
    def test_special_tokens_reference_shape(self) -> None:
        # Sanity check that SPECIAL_TOKENS contains the four structural specials
        # plus the three prosody markers (see docstring for Vocab layout).
        assert "<pad>" in SPECIAL_TOKENS
        assert "<bos>" in SPECIAL_TOKENS
        assert "<eos>" in SPECIAL_TOKENS
        assert "<unk>" in SPECIAL_TOKENS
        for tok in PROSODY_TOKENS:
            assert tok in SPECIAL_TOKENS
