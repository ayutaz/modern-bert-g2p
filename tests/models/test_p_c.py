"""Unit tests for the P-C char-level BERT pilot (Track 5)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from modernbert_g2p.models.p_c import PCConfig, build_p_c

if TYPE_CHECKING:  # pragma: no cover
    import torch


def _make_dummy_encoder(vocab_size: int, hidden: int) -> torch.nn.Module:
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
            hidden_states = self.proj(self.embed(input_ids))
            return SimpleNamespace(last_hidden_state=hidden_states)

    return _DummyEncoder()


def _build_tiny_model(
    head_variant: str = "C1",
    seed: int = 0,
    vocab_size: int = 32,
) -> tuple[object, PCConfig]:
    import torch

    torch.manual_seed(seed)
    config = PCConfig.tiny(head_variant=head_variant)
    encoder = _make_dummy_encoder(vocab_size=vocab_size, hidden=config.encoder_hidden)
    model = build_p_c(config, encoder=encoder)
    return model, config


def test_pcconfig_tiny_c1_defaults() -> None:
    config = PCConfig.tiny(head_variant="C1")
    assert config.head_variant == "C1"
    assert config.encoder_hidden == 32
    assert config.phoneme_vocab_size == 16
    assert config.max_slot == 4
    assert config.encoder_name == "tiny"
    assert config.label_smoothing == pytest.approx(0.05)
    assert config.apbp_alpha == pytest.approx(0.2)
    assert config.hl_vocab_size == 2
    assert config.label_pad_id == -100


def test_pcconfig_accepts_custom_hl_vocab_size() -> None:
    config = PCConfig(hl_vocab_size=3)
    assert config.hl_vocab_size == 3


def test_pcconfig_rejects_hl_vocab_below_two() -> None:
    with pytest.raises(ValueError, match="hl_vocab_size"):
        PCConfig(hl_vocab_size=1)


def test_pcconfig_accepts_custom_label_pad_id() -> None:
    config = PCConfig(label_pad_id=0)
    assert config.label_pad_id == 0


def test_pcconfig_tiny_c2_selects_variant() -> None:
    config = PCConfig.tiny(head_variant="C2")
    assert config.head_variant == "C2"


def test_pcconfig_rejects_unknown_head_variant() -> None:
    with pytest.raises(ValueError, match="head_variant"):
        PCConfig(head_variant="C3")


def test_pcconfig_default_matches_design_doc() -> None:
    config = PCConfig()
    assert config.encoder_name == "tohoku-nlp/bert-base-japanese-char-v2"
    assert config.encoder_hidden == 768
    assert config.max_slot == 8
    assert config.phoneme_vocab_size == 68
    assert config.head_variant == "C1"


def test_build_c1_count_params_positive_and_no_crf() -> None:
    pytest.importorskip("torch")
    model, _ = _build_tiny_model(head_variant="C1")
    assert model.count_params() > 0
    assert model.crf is None


def test_build_c2_has_linear_chain_crf() -> None:
    pytest.importorskip("torch")
    from modernbert_g2p.models.p_c.model import LinearChainCRF

    model, _ = _build_tiny_model(head_variant="C2")
    assert isinstance(model.crf, LinearChainCRF)
    assert model.crf.num_tags == 3


def test_forward_c1_output_shapes() -> None:
    torch = pytest.importorskip("torch")
    model, config = _build_tiny_model(head_variant="C1")
    batch, seq_len = 2, 10
    input_ids = torch.randint(0, 32, (batch, seq_len))
    attention_mask = torch.ones(batch, seq_len, dtype=torch.long)

    out = model(input_ids=input_ids, attention_mask=attention_mask)

    assert out["phon_logits"].shape == (
        batch,
        seq_len,
        config.max_slot,
        config.phoneme_vocab_size,
    )
    assert out["hl_logits"].shape == (
        batch,
        seq_len,
        config.max_slot,
        config.hl_vocab_size,
    )
    assert out["apbp_logits"].shape == (batch, seq_len, 3)
    assert "loss" not in out


def test_forward_c1_loss_finite_with_labels() -> None:
    torch = pytest.importorskip("torch")
    model, config = _build_tiny_model(head_variant="C1", seed=1)
    batch, seq_len = 2, 10
    input_ids = torch.randint(0, 32, (batch, seq_len))
    attention_mask = torch.ones(batch, seq_len, dtype=torch.long)
    phon_labels = torch.randint(
        0, config.phoneme_vocab_size, (batch, seq_len, config.max_slot)
    )
    hl_labels = torch.randint(0, 2, (batch, seq_len, config.max_slot))
    apbp_labels = torch.randint(0, 3, (batch, seq_len))

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        phoneme_labels=phon_labels,
        hl_labels=hl_labels,
        apbp_labels=apbp_labels,
    )

    loss = out["loss"]
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert loss.requires_grad


def test_forward_c2_loss_finite_uses_crf_path() -> None:
    torch = pytest.importorskip("torch")
    model, config = _build_tiny_model(head_variant="C2", seed=2)
    batch, seq_len = 2, 6
    input_ids = torch.randint(0, 32, (batch, seq_len))
    attention_mask = torch.ones(batch, seq_len, dtype=torch.long)
    phon_labels = torch.randint(
        0, config.phoneme_vocab_size, (batch, seq_len, config.max_slot)
    )
    hl_labels = torch.randint(0, 2, (batch, seq_len, config.max_slot))
    apbp_labels = torch.randint(0, 3, (batch, seq_len))

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        phoneme_labels=phon_labels,
        hl_labels=hl_labels,
        apbp_labels=apbp_labels,
    )

    loss = out["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_forward_ignores_padding_via_label_ignore_index() -> None:
    torch = pytest.importorskip("torch")
    model, config = _build_tiny_model(head_variant="C1", seed=3)
    batch, seq_len = 1, 4
    input_ids = torch.zeros(batch, seq_len, dtype=torch.long)
    attention_mask = torch.ones(batch, seq_len, dtype=torch.long)
    phon_labels_all_pad = torch.full(
        (batch, seq_len, config.max_slot), -100, dtype=torch.long
    )
    phon_labels_valid = torch.zeros((batch, seq_len, config.max_slot), dtype=torch.long)

    out_pad = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        phoneme_labels=phon_labels_all_pad,
    )
    out_valid = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        phoneme_labels=phon_labels_valid,
    )
    assert torch.isnan(out_pad["loss"]) or out_pad["loss"].item() == 0.0
    assert torch.isfinite(out_valid["loss"])


def test_forward_hl_vocab_size_configurable() -> None:
    """PCCharBERT respects config.hl_vocab_size (e.g. 3-class H/L/none)."""
    torch = pytest.importorskip("torch")

    torch.manual_seed(0)
    config = PCConfig(
        encoder_name="tiny",
        head_variant="C1",
        encoder_hidden=32,
        phoneme_vocab_size=16,
        max_slot=4,
        hl_vocab_size=3,
    )
    encoder = _make_dummy_encoder(vocab_size=32, hidden=config.encoder_hidden)
    model = build_p_c(config, encoder=encoder)

    batch, seq_len = 2, 5
    input_ids = torch.randint(0, 32, (batch, seq_len))
    attention_mask = torch.ones(batch, seq_len, dtype=torch.long)
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    assert out["hl_logits"].shape == (batch, seq_len, config.max_slot, 3)

    hl_labels = torch.randint(0, 3, (batch, seq_len, config.max_slot))
    phon_labels = torch.randint(
        0, config.phoneme_vocab_size, (batch, seq_len, config.max_slot)
    )
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        phoneme_labels=phon_labels,
        hl_labels=hl_labels,
    )
    assert torch.isfinite(out["loss"])


def test_forward_label_pad_id_is_parametric() -> None:
    """label_pad_id=0 excludes class 0 slots from the loss (parametric ignore)."""
    torch = pytest.importorskip("torch")

    torch.manual_seed(0)
    config = PCConfig(
        encoder_name="tiny",
        head_variant="C1",
        encoder_hidden=32,
        phoneme_vocab_size=16,
        max_slot=4,
        label_pad_id=0,
    )
    encoder = _make_dummy_encoder(vocab_size=32, hidden=config.encoder_hidden)
    model = build_p_c(config, encoder=encoder)

    batch, seq_len = 1, 3
    input_ids = torch.zeros(batch, seq_len, dtype=torch.long)
    attention_mask = torch.ones(batch, seq_len, dtype=torch.long)
    phon_labels_all_zero = torch.zeros(
        (batch, seq_len, config.max_slot), dtype=torch.long
    )

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        phoneme_labels=phon_labels_all_zero,
    )
    loss = out["loss"]
    assert torch.isnan(loss) or loss.item() == 0.0


def test_crf_decode_returns_lengths_matching_mask() -> None:
    torch = pytest.importorskip("torch")
    from modernbert_g2p.models.p_c.model import LinearChainCRF

    torch.manual_seed(0)
    num_tags = 3
    batch, seq_len = 3, 5
    crf = LinearChainCRF(num_tags)
    emissions = torch.randn(batch, seq_len, num_tags)
    mask = torch.tensor(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0],
            [1, 1, 0, 0, 0],
        ],
        dtype=torch.long,
    )

    decoded = crf.decode(emissions, mask)

    assert isinstance(decoded, list)
    assert all(isinstance(seq, list) for seq in decoded)
    lengths = [len(seq) for seq in decoded]
    assert lengths == [5, 3, 2]
    for seq in decoded:
        for tag in seq:
            assert isinstance(tag, int)
            assert 0 <= tag < num_tags


def test_crf_single_tag_nll_is_zero() -> None:
    torch = pytest.importorskip("torch")
    from modernbert_g2p.models.p_c.model import LinearChainCRF

    crf = LinearChainCRF(1)
    emissions = torch.randn(2, 4, 1)
    tags = torch.zeros(2, 4, dtype=torch.long)
    mask = torch.ones(2, 4, dtype=torch.long)

    nll = crf(emissions, tags, mask)

    assert torch.isfinite(nll)
    assert nll.abs().item() < 1e-5


def test_crf_forward_and_decode_are_consistent() -> None:
    torch = pytest.importorskip("torch")
    from modernbert_g2p.models.p_c.model import LinearChainCRF

    torch.manual_seed(7)
    num_tags = 3
    batch, seq_len = 2, 6
    crf = LinearChainCRF(num_tags)
    emissions = torch.randn(batch, seq_len, num_tags)
    mask = torch.ones(batch, seq_len, dtype=torch.long)

    decoded = crf.decode(emissions, mask)
    decoded_tensor = torch.tensor(decoded, dtype=torch.long)
    random_tags = torch.randint(0, num_tags, (batch, seq_len))

    nll_decoded = crf(emissions, decoded_tensor, mask)
    nll_random = crf(emissions, random_tags, mask)

    assert torch.isfinite(nll_decoded)
    assert nll_decoded.item() <= nll_random.item() + 1e-4


def test_crf_rejects_shape_mismatch() -> None:
    torch = pytest.importorskip("torch")
    from modernbert_g2p.models.p_c.model import LinearChainCRF

    crf = LinearChainCRF(3)
    emissions = torch.randn(2, 5, 3)
    bad_tags = torch.zeros(2, 4, dtype=torch.long)
    mask = torch.ones(2, 5, dtype=torch.long)
    with pytest.raises(ValueError):
        crf(emissions, bad_tags, mask)


def test_model_forward_is_deterministic_given_seed() -> None:
    torch = pytest.importorskip("torch")

    def _run(seed: int) -> torch.Tensor:
        torch.manual_seed(seed)
        config = PCConfig.tiny(head_variant="C1")
        encoder = _make_dummy_encoder(vocab_size=32, hidden=config.encoder_hidden)
        model = build_p_c(config, encoder=encoder)
        model.eval()
        input_ids = torch.arange(20, dtype=torch.long).reshape(2, 10)
        attention_mask = torch.ones(2, 10, dtype=torch.long)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
        return out["phon_logits"]

    a = _run(42)
    b = _run(42)
    assert torch.allclose(a, b)


def test_build_p_c_tiny_without_encoder_raises() -> None:
    pytest.importorskip("torch")
    config = PCConfig.tiny()
    with pytest.raises(ValueError, match="tiny"):
        build_p_c(config)


def test_import_p_c_does_not_require_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib
    import sys

    for mod in [m for m in list(sys.modules) if m.startswith("modernbert_g2p.models.p_c")]:
        sys.modules.pop(mod, None)

    original_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __import__

    def _guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "torch" or name.startswith("torch."):
            raise AssertionError(f"unexpected torch import: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _guarded_import)

    mod = importlib.import_module("modernbert_g2p.models.p_c")
    assert hasattr(mod, "PCConfig")
    assert hasattr(mod, "build_p_c")
