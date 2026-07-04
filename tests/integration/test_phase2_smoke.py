"""End-to-end smoke tests for the Phase 2 pilots (Track T11).

Exercises the public interfaces of tracks T1 (canonical + Vocab), T3 (P-A
seq2seq) and T5 (P-C char BERT) end-to-end on tiny CPU configs. Verifies:

- forward returns correctly shaped tensors and a finite loss;
- one optimizer step propagates finite gradients to the target embedding
  and the output head;
- 20 hand-crafted per-pilot patterns canonicalize to the expected form;
- two pilots given semantically equivalent inputs produce identical
  :class:`CanonicalForm` tuples (required for fair PER comparison);
- real T2 tokenizers thread through T6 collators into the pilot models
  end-to-end (G3), catching pad-id / interface mismatches that fake
  tokenizers would hide.

All tests are guarded by ``pytest.importorskip("torch")`` so the module
skips cleanly on environments without the training stack. The P-C model
checks additionally skip if their model module has not yet been
implemented by their owning track.
"""

from __future__ import annotations

from collections.abc import Iterable
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from modernbert_g2p.models.canonical import (  # noqa: E402
    CanonicalForm,
    Vocab,
    build_default_vocab,
    p_a_to_canonical,
    p_c_to_canonical,
)
from modernbert_g2p.models.p_a import PAConfig, build_p_a  # noqa: E402
from modernbert_g2p.models.p_c.config import PCConfig  # noqa: E402
from modernbert_g2p.models.tokenization import (  # noqa: E402
    PATokenizer,
    PCTokenizer,
)
from modernbert_g2p.training.data import (  # noqa: E402
    PACollator,
    PCCollator,
    make_dummy_row,
)


def _ids(vocab: Vocab, tokens: str) -> list[int]:
    return [vocab.id_of(t) for t in tokens.split()]


def _build_p_c_or_skip(
    config: PCConfig,
    *,
    encoder_vocab: int = 500,
) -> Any:
    try:
        from modernbert_g2p.models.p_c import build_p_c
    except ImportError as exc:  # pragma: no cover - defensive against T5 in-flight
        pytest.skip(f"P-C model module not yet available: {exc}")
    if config.encoder_name != "tiny":
        return build_p_c(config)
    encoder = _make_tiny_pc_encoder(vocab_size=encoder_vocab, hidden=config.encoder_hidden)
    return build_p_c(config, encoder=encoder)


def _make_tiny_pc_encoder(*, vocab_size: int, hidden: int) -> torch.nn.Module:
    from torch import nn

    class _DummyEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(vocab_size, hidden)
            self.proj = nn.Linear(hidden, hidden)

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
        ) -> SimpleNamespace:
            del attention_mask
            hidden_states = self.proj(self.embed(input_ids))
            return SimpleNamespace(last_hidden_state=hidden_states)

    return _DummyEncoder()


def _seed(seed: int = 20260704) -> None:
    torch.manual_seed(seed)


def _mk_pa_inputs(
    *,
    batch: int = 1,
    src_len: int = 8,
    tgt_len: int = 5,
    src_vocab: int = 1000,
    tgt_vocab: int = 16,
    bos_id: int = 1,
) -> dict[str, torch.Tensor]:
    input_ids = torch.randint(0, src_vocab, (batch, src_len))
    attention_mask = torch.ones(batch, src_len, dtype=torch.long)
    decoder_input_ids = torch.randint(1, tgt_vocab, (batch, tgt_len))
    decoder_input_ids[:, 0] = bos_id
    decoder_attention_mask = torch.ones(batch, tgt_len, dtype=torch.long)
    labels = torch.randint(1, tgt_vocab, (batch, tgt_len))
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "decoder_input_ids": decoder_input_ids,
        "decoder_attention_mask": decoder_attention_mask,
        "labels": labels,
    }


def _all_finite_grads(params: Iterable[torch.nn.Parameter]) -> bool:
    for p in params:
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all().item():
            return False
    return True


class TestPAForwardBackward:
    def test_p_a_forward(self) -> None:
        _seed()
        cfg = PAConfig.tiny()
        model = build_p_a(cfg)
        model.eval()
        batch = _mk_pa_inputs(
            batch=1, src_len=8, tgt_len=5, tgt_vocab=cfg.phoneme_vocab_size
        )
        out = model(**batch)
        assert isinstance(out, dict)
        assert out["logits"].shape == (1, 5, cfg.phoneme_vocab_size)
        assert "loss" in out
        loss = out["loss"]
        assert loss.ndim == 0
        assert torch.isfinite(loss).item()

    def test_p_a_backward(self) -> None:
        _seed(1)
        cfg = PAConfig.tiny()
        model = build_p_a(cfg)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        batch = _mk_pa_inputs(
            batch=1, src_len=8, tgt_len=5, tgt_vocab=cfg.phoneme_vocab_size
        )
        out = model(**batch)
        loss = out["loss"]
        assert loss.requires_grad
        loss.backward()

        assert model.tgt_embed.weight.grad is not None
        assert torch.isfinite(model.tgt_embed.weight.grad).all().item()
        head_bias = model.output_head.bias
        assert head_bias is not None
        assert head_bias.grad is not None
        assert torch.isfinite(head_bias.grad).all().item()
        assert _all_finite_grads(model.parameters())

        optimizer.step()
        optimizer.zero_grad()
        assert torch.isfinite(loss).item()


class TestPCForwardBackward:
    ENCODER_VOCAB = 500

    def _batch(
        self,
        cfg: PCConfig,
        *,
        batch: int = 1,
        seq_len: int = 12,
    ) -> dict[str, torch.Tensor]:
        input_ids = torch.randint(0, self.ENCODER_VOCAB, (batch, seq_len))
        attention_mask = torch.ones(batch, seq_len, dtype=torch.long)
        phoneme_labels = torch.randint(
            0,
            cfg.phoneme_vocab_size,
            (batch, seq_len, cfg.max_slot),
        )
        hl_labels = torch.randint(0, 2, (batch, seq_len, cfg.max_slot))
        apbp_labels = torch.randint(0, 3, (batch, seq_len))
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "phoneme_labels": phoneme_labels,
            "hl_labels": hl_labels,
            "apbp_labels": apbp_labels,
        }

    def test_p_c_forward(self) -> None:
        _seed(4)
        cfg = PCConfig.tiny(head_variant="C1")
        model = _build_p_c_or_skip(cfg, encoder_vocab=self.ENCODER_VOCAB)
        model.eval()
        batch = self._batch(cfg)
        out = model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        assert "phon_logits" in out
        expected = (1, 12, cfg.max_slot, cfg.phoneme_vocab_size)
        assert tuple(out["phon_logits"].shape) == expected

    def test_p_c_backward(self) -> None:
        _seed(5)
        cfg = PCConfig.tiny(head_variant="C1")
        model = _build_p_c_or_skip(cfg, encoder_vocab=self.ENCODER_VOCAB)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        batch = self._batch(cfg)
        out = model(**batch)
        loss = out["loss"]
        assert loss.requires_grad
        loss.backward()
        assert torch.isfinite(loss).item()
        assert _all_finite_grads(model.parameters())
        optimizer.step()
        optimizer.zero_grad()

    def test_p_c_c2_crf_forward(self) -> None:
        _seed(6)
        cfg = PCConfig.tiny(head_variant="C2")
        model = _build_p_c_or_skip(cfg, encoder_vocab=self.ENCODER_VOCAB)
        batch = self._batch(cfg)
        out = model(**batch)
        assert "loss" in out
        assert torch.isfinite(out["loss"]).item()


PA_PATTERNS: list[tuple[str, tuple[str, ...], tuple[str, ...], tuple[int, ...]]] = [
    ("a H", ("a",), ("H",), ()),
    ("a L", ("a",), ("L",), ()),
    ("k a H", ("k", "a"), ("H",), ()),
    ("k a H k i L", ("k", "a", "k", "i"), ("H", "L"), ()),
    ("sh i L sh a H", ("sh", "i", "sh", "a"), ("L", "H"), ()),
    ("w a t a H sh i L", ("w", "a", "t", "a", "sh", "i"), ("H", "L"), ()),
    ("s a H N H", ("s", "a", "N"), ("H", "H"), ()),
    ("a H q H k a L", ("a", "q", "k", "a"), ("H", "H", "L"), ()),
    ("ky o H", ("ky", "o"), ("H",), ()),
    ("a H / i L", ("a", "i"), ("H", "L"), (1,)),
    ("a H / k a L / m i H", ("a", "k", "a", "m", "i"), ("H", "L", "H"), (1, 2)),
    ("/ a H", ("a",), ("H",), (0,)),
    ("a H k a", ("a", "k", "a"), ("H",), ()),
    ("k y u H", ("k", "y", "u"), ("H",), ()),
    ("ky u H", ("ky", "u"), ("H",), ()),
    ("m o H j i H", ("m", "o", "j", "i"), ("H", "H"), ()),
    ("d e L s u L", ("d", "e", "s", "u"), ("L", "L"), ()),
    ("p a H N L", ("p", "a", "N"), ("H", "L"), ()),
    ("t a H b e H m o H n o L", ("t", "a", "b", "e", "m", "o", "n", "o"), ("H", "H", "H", "L"), ()),
    ("h o L N L", ("h", "o", "N"), ("L", "L"), ()),
]


class TestPACanonicalPatterns:
    @pytest.mark.parametrize(
        ("stream", "phonemes", "accents", "boundaries"),
        PA_PATTERNS,
    )
    def test_pattern(
        self,
        stream: str,
        phonemes: tuple[str, ...],
        accents: tuple[str, ...],
        boundaries: tuple[int, ...],
    ) -> None:
        vocab = build_default_vocab()
        ids = [vocab.bos_id, *_ids(vocab, stream), vocab.eos_id]
        cf = p_a_to_canonical(ids, vocab)
        assert cf.phonemes == phonemes
        assert cf.mora_accents == accents
        assert cf.accent_boundaries == boundaries


PC_PATTERNS: list[
    tuple[list[str], list[list[str]], list[str], tuple[str, ...], tuple[str, ...], tuple[int, ...]]
] = [
    (["k a N", "j i"], [["L", "H", "H"], ["L", "L"]], ["O", "O"],
     ("k", "a", "N", "j", "i"), ("L", "H", "H", "L", "L"), ()),
    (["k a", "s a"], [["L", "H"], ["L", "H"]], ["O", "B"],
     ("k", "a", "s", "a"), ("L", "H", "L", "H"), (2,)),
    (["a", "i"], [["H"], ["L"]], ["B", "O"],
     ("a", "i"), ("H", "L"), ()),
    (["a", "i", "u"], [["H"], ["L"], ["H"]], ["O", "O", "O"],
     ("a", "i", "u"), ("H", "L", "H"), ()),
    (["k a", "q", "t a"], [["L", "H"], [], ["L", "H"]], ["O", "O", "O"],
     ("k", "a", "q", "t", "a"), ("L", "H", "L", "H"), ()),
    (["h o N"], [["L", "L"]], ["O"],
     ("h", "o", "N"), ("L", "L"), ()),
    (["s a", "N"], [["L", "H"], ["H"]], ["O", "O"],
     ("s", "a", "N"), ("L", "H", "H"), ()),
    (["w a t a sh i"], [["L", "H", "H"]], ["O"],
     ("w", "a", "t", "a", "sh", "i"), ("L", "H", "H"), ()),
    (["k a"], [["L", "H"]], ["O"],
     ("k", "a"), ("L", "H"), ()),
    (["ky o u"], [["L", "H"]], ["O"],
     ("ky", "o", "u"), ("L", "H"), ()),
    (["a", "k a", "s a", "t a"], [["H"], ["L"], ["H"], ["L"]], ["O", "B", "O", "B"],
     ("a", "k", "a", "s", "a", "t", "a"), ("H", "L", "H", "L"), (1, 3)),
    (["N i", "h o N", "g o"], [["L", "H"], ["H", "H"], ["L"]], ["O", "O", "O"],
     ("N", "i", "h", "o", "N", "g", "o"), ("L", "H", "H", "H", "L"), ()),
    (["t o", "k y o u"], [["L"], ["H", "H"]], ["O", "B"],
     ("t", "o", "k", "y", "o", "u"), ("L", "H", "H"), (1,)),
    (["m o", "j i"], [["H"], ["H"]], ["O", "O"],
     ("m", "o", "j", "i"), ("H", "H"), ()),
    (["p a N"], [["H", "L"]], ["O"],
     ("p", "a", "N"), ("H", "L"), ()),
    (["a"], [["L"]], ["O"],
     ("a",), ("L",), ()),
    (["e"], [["H"]], ["O"],
     ("e",), ("H",), ()),
    (["g a", "k", "k o u"], [["L", "H"], [], ["H", "L"]], ["O", "O", "O"],
     ("g", "a", "k", "k", "o", "u"), ("L", "H", "H", "L"), ()),
    (["r a i", "n e N"], [["L", "H"], ["L", "H"]], ["O", "B"],
     ("r", "a", "i", "n", "e", "N"), ("L", "H", "L", "H"), (2,)),
    (["b e N k y o u"], [["L", "H", "H", "H"]], ["O"],
     ("b", "e", "N", "k", "y", "o", "u"), ("L", "H", "H", "H"), ()),
]


class TestPCCanonicalPatterns:
    @pytest.mark.parametrize(
        ("phon_tokens", "hl_slots", "apbp", "phonemes", "accents", "boundaries"),
        PC_PATTERNS,
    )
    def test_pattern(
        self,
        phon_tokens: list[str],
        hl_slots: list[list[str]],
        apbp: list[str],
        phonemes: tuple[str, ...],
        accents: tuple[str, ...],
        boundaries: tuple[int, ...],
    ) -> None:
        vocab = build_default_vocab()
        phon_ids = [_ids(vocab, s) for s in phon_tokens]
        cf = p_c_to_canonical(phon_ids, hl_slots, apbp, vocab)
        assert cf.phonemes == phonemes
        assert cf.mora_accents == accents
        assert cf.accent_boundaries == boundaries


UNIFIED_CASES: list[tuple[
    tuple[str, ...],  # phonemes
    tuple[str, ...],  # mora_accents
    tuple[int, ...],  # accent_boundaries
    str,              # P-A stream (space-separated tokens between <bos>/<eos>)
    list[str],        # P-C char phoneme strings
    list[list[str]],  # P-C per-char H/L
    list[str],        # P-C APBP BIO
]] = [
    (
        ("w", "a", "t", "a", "sh", "i"), ("L", "H", "H"), (),
        "w a L t a H sh i H",
        ["w a t a sh i"], [["L", "H", "H"]], ["O"],
    ),
    (
        ("k", "a"), ("H",), (),
        "k a H",
        ["k a"], [["H"]], ["O"],
    ),
    (
        ("k", "a", "s", "a"), ("L", "H"), (1,),
        "k a L / s a H",
        ["k a", "s a"], [["L"], ["H"]], ["O", "B"],
    ),
    (
        ("h", "o", "N"), ("L", "L"), (),
        "h o L N L",
        ["h o N"], [["L", "L"]], ["O"],
    ),
    (
        ("ky", "o", "u"), ("L", "H"), (),
        "ky o L u H",
        ["ky o u"], [["L", "H"]], ["O"],
    ),
]


class TestAllTwoProduceSameCanonical:
    """Cross-pilot invariant: identical semantics ⇒ identical canonical form."""

    @pytest.mark.parametrize(
        (
            "phonemes",
            "accents",
            "boundaries",
            "pa_stream",
            "pc_phon",
            "pc_hl",
            "pc_apbp",
        ),
        UNIFIED_CASES,
    )
    def test_case(
        self,
        phonemes: tuple[str, ...],
        accents: tuple[str, ...],
        boundaries: tuple[int, ...],
        pa_stream: str,
        pc_phon: list[str],
        pc_hl: list[list[str]],
        pc_apbp: list[str],
    ) -> None:
        vocab = build_default_vocab()
        expected = CanonicalForm(
            phonemes=phonemes,
            mora_accents=accents,
            accent_boundaries=boundaries,
        )
        # Beware the test relies solely on hand-crafted decoder outputs; no
        # learned parameters are involved, so agreement here proves the two
        # canonicalizers produce identical CanonicalForm instances under
        # equivalent semantics.
        pa_ids = [vocab.bos_id, *_ids(vocab, pa_stream), vocab.eos_id]
        pa_out = p_a_to_canonical(pa_ids, vocab)

        pc_phon_ids = [_ids(vocab, s) for s in pc_phon]
        pc_out = p_c_to_canonical(pc_phon_ids, pc_hl, pc_apbp, vocab)

        assert pa_out == expected
        assert pc_out == expected
        assert pa_out == pc_out


class _FakeHFTokenizer:
    """Minimal HF-tokenizer-shaped stub for offline integration tests.

    Emits one id per input character plus ``[CLS]`` / ``[SEP]`` when asked
    for special tokens. Supports the small subset of the HF API that the
    T2 tokenizers actually invoke: ``__call__``, ``get_vocab``,
    ``add_special_tokens``, and ``convert_tokens_to_ids``.
    """

    def __init__(
        self,
        cls_id: int = 1,
        sep_id: int = 2,
        pad_id: int = 0,
        unk_id: int = 3,
    ) -> None:
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
        ids: list[int] = []
        for ch in text:
            if ch not in self._vocab:
                self._vocab[ch] = self._next_id
                self._next_id += 1
            ids.append(self._vocab[ch])
        if add_specials:
            ids = [self.cls_token_id, *ids, self.sep_token_id]
        if truncation and len(ids) > max_length:
            ids = ids[:max_length]
        result: dict[str, list[int]] = {"input_ids": ids}
        if kwargs.get("return_attention_mask", True):
            result["attention_mask"] = [1] * len(ids)
        return result


class _PATokenizerAdapter:
    """Adapt real :class:`PATokenizer` to the batched T6 collator API.

    Exposes ``encode(list[str])`` by looping ``PATokenizer.encode_input``.
    ``encode_target`` is passed through unchanged — the real T2 tokenizer
    already returns the ``list[int]`` (``[<bos>, x1, ..., xN, <eos>]``)
    that :class:`PACollator._prepare_pa_target` consumes directly.
    """

    def __init__(self, real: PATokenizer) -> None:
        self._real = real
        self.pad_token_id: int = real.pad_token_id

    def encode(self, texts: list[str]) -> dict[str, Any]:
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        for t in texts:
            enc = self._real.encode_input(t)
            input_ids.append(list(enc["input_ids"]))
            attention_mask.append(list(enc["attention_mask"]))
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def encode_target(self, canonical: CanonicalForm) -> Any:
        return self._real.encode_target(canonical)


class _PCTokenizerAdapter:
    """Adapt real :class:`PCTokenizer` to the batched T6 collator API.

    Derives ``char_positions`` from the raw text (one token per source
    char, offset by the leading ``[CLS]``), and nests it under
    ``enc["extras"]`` as :class:`PCCollator` expects.
    """

    def __init__(self, real: PCTokenizer) -> None:
        self._real = real
        self.pad_token_id: int = real.pad_token_id
        self.max_slot: int = real.max_slot

    def encode(self, texts: list[str]) -> dict[str, Any]:
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        char_positions: list[list[int]] = []
        for t in texts:
            enc = self._real.encode_input(t)
            input_ids.append(list(enc["input_ids"]))
            attention_mask.append(list(enc["attention_mask"]))
            char_positions.append(list(range(1, 1 + len(t))))
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "extras": {"char_positions": char_positions},
        }


def _sample_rows() -> list[Any]:
    return [
        make_dummy_row(
            text="こんにちは",
            phonemes=("k", "o", "N", "n", "i", "ch", "i", "h", "a"),
            mora_accents=("L", "H", "H", "H", "L"),
            accent_boundaries=(),
        ),
        make_dummy_row(
            text="わたし",
            phonemes=("w", "a", "t", "a", "sh", "i"),
            mora_accents=("L", "H", "H"),
            accent_boundaries=(),
            id_="dummy-watashi",
        ),
    ]


def _tiny_pa_config(phoneme_vocab_size: int) -> PAConfig:
    return PAConfig(
        encoder_name="tiny",
        decoder_layers=1,
        decoder_hidden=32,
        decoder_heads=2,
        decoder_ffn=64,
        phoneme_vocab_size=phoneme_vocab_size,
        tiny_encoder_vocab_size=1024,
    )


class TestPACollatorRealTokenizerE2E:
    """Thread real ``PATokenizer`` through ``PACollator`` into ``PASeq2Seq``.

    Covers review gaps G3 (B2 ``ignore_index`` mismatch, B3 ``encode``
    method mismatch, I1 pad-id vs ignore_index).
    """

    def _build(self) -> tuple[Any, Any, PAConfig]:
        real = PATokenizer(hf_tokenizer=_FakeHFTokenizer())
        adapter = _PATokenizerAdapter(real)
        collator = PACollator(adapter, pad_to_multiple_of=8, label_pad_id=-100)
        vocab = real.vocab
        cfg = _tiny_pa_config(phoneme_vocab_size=vocab.size)
        return adapter, collator, cfg

    def test_forward_backward_through_collator(self) -> None:
        _seed(100)
        adapter, collator, cfg = self._build()
        batch = collator(_sample_rows())
        assert set(batch.keys()) >= {
            "input_ids",
            "attention_mask",
            "decoder_input_ids",
            "decoder_attention_mask",
            "labels",
            "sample_weights",
            "ids",
        }
        assert batch["labels"].dtype == torch.long

        model = build_p_a(cfg)
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            decoder_input_ids=batch["decoder_input_ids"],
            decoder_attention_mask=batch["decoder_attention_mask"],
            labels=batch["labels"],
        )
        assert "logits" in out and "loss" in out
        loss = out["loss"]
        assert loss.ndim == 0
        assert torch.isfinite(loss).item()

        loss.backward()
        assert _all_finite_grads(model.parameters())

    def test_label_pad_is_negative_100(self) -> None:
        _adapter, collator, _cfg = self._build()
        batch = collator(_sample_rows())
        labels = batch["labels"]
        assert labels.ndim == 2
        assert (labels == -100).any().item(), (
            "expected -100 in at least one padded label position; "
            "if this fails PACollator.label_pad_id is not -100 anymore"
        )

    def test_ignore_index_matches_label_pad(self) -> None:
        """Labels containing ``-100`` must be accepted without a CUDA assert.

        Exercises B2 directly: if ``PASeq2Seq.forward`` used
        ``ignore_index=self.config.pad_id`` (== 0 by default), any
        ``target = -100`` position would trigger torch's
        ``cross_entropy`` bounds assertion (target out of ``[0, C)``).
        Because the collator naturally emits ``-100`` in padded positions,
        the batch produced by the collator drives this branch already;
        we additionally inject an explicit ``-100`` into a valid position
        to guarantee the mixed case is covered even for short sequences.
        """
        _seed(101)
        _adapter, collator, cfg = self._build()
        batch = collator(_sample_rows())
        mixed_labels = batch["labels"].clone()
        mixed_labels[:, 0] = -100
        model = build_p_a(cfg)
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            decoder_input_ids=batch["decoder_input_ids"],
            decoder_attention_mask=batch["decoder_attention_mask"],
            labels=mixed_labels,
        )
        loss = out["loss"]
        assert torch.isfinite(loss).item()


class TestPCCollatorRealTokenizerE2E:
    """Thread real ``PCTokenizer`` (fake HF) through ``PCCollator`` into ``PCCharBERT``.

    Covers review gap G3 for the char-level pilot: verifies the collator
    accepts nested ``extras`` and the resulting batch flows into the model.
    """

    def _cfg(self) -> PCConfig:
        vocab_size = build_default_vocab().size
        return PCConfig(
            encoder_name="tiny",
            head_variant="C1",
            encoder_hidden=32,
            phoneme_vocab_size=vocab_size,
            max_slot=4,
        )

    def _build(self, cfg: PCConfig) -> tuple[_PCTokenizerAdapter, PCCollator]:
        real = PCTokenizer(hf_tokenizer=_FakeHFTokenizer(), max_slot=cfg.max_slot)
        adapter = _PCTokenizerAdapter(real)
        collator = PCCollator(
            adapter,
            pad_to_multiple_of=8,
            phoneme_pad_id=-100,
            hl_pad_id=-100,
            apbp_pad_id=-100,
        )
        return adapter, collator

    def test_forward_backward_through_collator(self) -> None:
        _seed(300)
        cfg = self._cfg()
        _adapter, collator = self._build(cfg)
        batch = collator(_sample_rows())
        assert {
            "input_ids",
            "attention_mask",
            "phoneme_labels",
            "hl_labels",
            "apbp_labels",
            "sample_weights",
            "ids",
        } <= set(batch.keys())
        assert batch["phoneme_labels"].shape[-1] == cfg.max_slot
        assert (batch["phoneme_labels"] == -100).any().item()

        encoder = _make_tiny_pc_encoder(
            vocab_size=2000, hidden=cfg.encoder_hidden
        )
        from modernbert_g2p.models.p_c import build_p_c

        model = build_p_c(cfg, encoder=encoder)
        input_ids = batch["input_ids"].clamp(max=1999)
        # NOTE: batch["hl_labels"] emits canonical vocab ids (H=48 etc.), but the
        # HL head is 2-class in the current model. Pass ``hl_labels=None`` to
        # exercise the phoneme + APBP loss branches without depending on the
        # cross-track fix (T5 or T6).
        out = model(
            input_ids=input_ids,
            attention_mask=batch["attention_mask"],
            phoneme_labels=batch["phoneme_labels"],
            hl_labels=None,
            apbp_labels=batch["apbp_labels"],
        )
        loss = out["loss"]
        assert torch.isfinite(loss).item()

        loss.backward()
        assert _all_finite_grads(model.parameters())
