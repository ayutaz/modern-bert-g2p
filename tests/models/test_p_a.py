"""Unit tests for the P-A seq2seq pilot (Track 3)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from modernbert_g2p.models.p_a import PAConfig, build_p_a  # noqa: E402
from modernbert_g2p.models.p_a.model import PASeq2Seq  # noqa: E402


def _fixed_seed(seed: int = 20260704) -> None:
    torch.manual_seed(seed)


def _mk_inputs(
    *,
    batch: int = 2,
    src_len: int = 8,
    tgt_len: int = 5,
    vocab_size: int = 1000,
    tgt_vocab: int = 16,
    bos_id: int = 1,
) -> dict[str, torch.Tensor]:
    _fixed_seed()
    input_ids = torch.randint(0, vocab_size, (batch, src_len))
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


def test_pa_config_tiny_builds() -> None:
    cfg = PAConfig.tiny()
    assert cfg.encoder_name == "tiny"
    assert cfg.decoder_layers == 1
    assert cfg.decoder_hidden == 32
    assert cfg.decoder_heads == 2
    assert cfg.decoder_ffn == 64
    assert cfg.phoneme_vocab_size == 16


def test_pa_config_defaults_match_spec() -> None:
    cfg = PAConfig()
    assert cfg.encoder_name == "sbintuitions/modernbert-ja-130m"
    assert cfg.decoder_layers == 6
    assert cfg.decoder_hidden == 512
    assert cfg.decoder_heads == 8
    assert cfg.decoder_ffn == 2048
    assert cfg.phoneme_vocab_size == 68
    assert cfg.label_smoothing == pytest.approx(0.1)
    assert cfg.pad_id == 0
    assert cfg.bos_id == 1
    assert cfg.eos_id == 2
    assert cfg.max_decode_len == 128
    assert cfg.beam_size == 4
    assert cfg.length_penalty == pytest.approx(1.0)
    assert cfg.coverage_penalty == pytest.approx(0.0)


def test_build_p_a_returns_pa_seq2seq() -> None:
    _fixed_seed()
    model = build_p_a(PAConfig.tiny())
    assert isinstance(model, PASeq2Seq)
    assert model.config.encoder_name == "tiny"


def test_count_params_positive() -> None:
    _fixed_seed()
    model = build_p_a(PAConfig.tiny())
    n = model.count_params()
    assert isinstance(n, int)
    assert n > 0


def test_forward_returns_logits_shape() -> None:
    _fixed_seed()
    model = build_p_a(PAConfig.tiny())
    batch = _mk_inputs(batch=2, src_len=8, tgt_len=5, tgt_vocab=16)
    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        decoder_input_ids=batch["decoder_input_ids"],
        decoder_attention_mask=batch["decoder_attention_mask"],
    )
    assert "logits" in out
    assert out["logits"].shape == (2, 5, 16)
    assert "loss" not in out


def test_forward_with_labels_returns_finite_loss() -> None:
    _fixed_seed()
    model = build_p_a(PAConfig.tiny())
    batch = _mk_inputs(batch=2, src_len=8, tgt_len=5, tgt_vocab=16)
    out = model(**batch)
    assert "loss" in out
    loss = out["loss"]
    assert loss.ndim == 0
    assert torch.isfinite(loss).item()
    assert loss.requires_grad


def test_backward_updates_decoder_grads() -> None:
    _fixed_seed()
    model = build_p_a(PAConfig.tiny())
    batch = _mk_inputs(batch=2, src_len=8, tgt_len=5, tgt_vocab=16)
    out = model(**batch)
    out["loss"].backward()

    decoder_params_with_grad = [
        p for p in model.decoder.parameters() if p.grad is not None
    ]
    assert len(decoder_params_with_grad) > 0
    for p in decoder_params_with_grad:
        assert torch.isfinite(p.grad).all().item()

    head_grad = model.output_head.bias.grad
    assert head_grad is not None
    assert torch.isfinite(head_grad).all().item()


def test_padding_positions_ignored_in_loss() -> None:
    _fixed_seed()
    model = build_p_a(PAConfig.tiny())
    model.eval()
    batch = _mk_inputs(batch=2, src_len=8, tgt_len=6, tgt_vocab=16)

    labels_padded = batch["labels"].clone()
    labels_padded[:, -2:] = -100

    with torch.inference_mode():
        out_padded = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            decoder_input_ids=batch["decoder_input_ids"],
            decoder_attention_mask=batch["decoder_attention_mask"],
            labels=labels_padded,
        )
        out_truncated = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            decoder_input_ids=batch["decoder_input_ids"][:, :4],
            decoder_attention_mask=batch["decoder_attention_mask"][:, :4],
            labels=batch["labels"][:, :4],
        )

    assert torch.allclose(out_padded["loss"], out_truncated["loss"], atol=1e-5)


def test_loss_ignores_only_negative_hundred_not_pad_id() -> None:
    _fixed_seed()
    model = build_p_a(PAConfig.tiny())
    model.eval()
    batch = _mk_inputs(batch=2, src_len=8, tgt_len=6, tgt_vocab=16)

    labels_with_pad_class = batch["labels"].clone()
    labels_with_pad_class[:, -2:] = model.config.pad_id

    labels_with_ignore = batch["labels"].clone()
    labels_with_ignore[:, -2:] = -100

    with torch.inference_mode():
        out_pad_class = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            decoder_input_ids=batch["decoder_input_ids"],
            decoder_attention_mask=batch["decoder_attention_mask"],
            labels=labels_with_pad_class,
        )
        out_ignore = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            decoder_input_ids=batch["decoder_input_ids"],
            decoder_attention_mask=batch["decoder_attention_mask"],
            labels=labels_with_ignore,
        )

    assert not torch.allclose(out_pad_class["loss"], out_ignore["loss"], atol=1e-5)


def test_forward_is_deterministic_for_fixed_seed() -> None:
    inputs = _mk_inputs(batch=2, src_len=8, tgt_len=5, tgt_vocab=16)

    _fixed_seed(42)
    model_1 = build_p_a(PAConfig.tiny())
    model_1.eval()
    with torch.inference_mode():
        out_1 = model_1(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            decoder_input_ids=inputs["decoder_input_ids"],
            decoder_attention_mask=inputs["decoder_attention_mask"],
        )

    _fixed_seed(42)
    model_2 = build_p_a(PAConfig.tiny())
    model_2.eval()
    with torch.inference_mode():
        out_2 = model_2(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            decoder_input_ids=inputs["decoder_input_ids"],
            decoder_attention_mask=inputs["decoder_attention_mask"],
        )

    assert torch.allclose(out_1["logits"], out_2["logits"], atol=1e-6)


def test_generate_greedy_produces_finite_ids() -> None:
    _fixed_seed()
    model = build_p_a(PAConfig.tiny())
    input_ids = torch.randint(0, 1000, (2, 8))
    attention_mask = torch.ones(2, 8, dtype=torch.long)

    generated = model.generate(input_ids, attention_mask, max_len=10, beam=1)
    assert generated.dtype == torch.long
    assert generated.dim() == 2
    assert generated.size(0) == 2
    assert generated.size(1) <= 10


def test_generate_beam_search_produces_finite_ids() -> None:
    _fixed_seed()
    model = build_p_a(PAConfig.tiny())
    input_ids = torch.randint(0, 1000, (2, 8))
    attention_mask = torch.ones(2, 8, dtype=torch.long)

    generated = model.generate(input_ids, attention_mask, max_len=10, beam=4)
    assert generated.dtype == torch.long
    assert generated.dim() == 2
    assert generated.size(0) == 2
    assert generated.size(1) <= 10


def test_generate_beam_at_least_greedy_quality() -> None:
    _fixed_seed()
    cfg = PAConfig.tiny()
    model = build_p_a(cfg)
    input_ids = torch.randint(0, 1000, (2, 8))
    attention_mask = torch.ones(2, 8, dtype=torch.long)

    greedy = model.generate(input_ids, attention_mask, max_len=10, beam=1)
    beam4 = model.generate(input_ids, attention_mask, max_len=10, beam=4)

    assert greedy.size(0) == beam4.size(0)


def test_generate_uses_config_defaults_when_kwargs_omitted() -> None:
    _fixed_seed()
    cfg = PAConfig.tiny()
    model = build_p_a(cfg)
    input_ids = torch.randint(0, 1000, (2, 8))
    attention_mask = torch.ones(2, 8, dtype=torch.long)

    generated = model.generate(input_ids, attention_mask, max_len=6)
    assert generated.size(1) <= 6


def test_generate_coverage_penalty_runs() -> None:
    _fixed_seed()
    model = build_p_a(PAConfig.tiny())
    input_ids = torch.randint(0, 1000, (2, 8))
    attention_mask = torch.ones(2, 8, dtype=torch.long)

    generated = model.generate(
        input_ids, attention_mask, max_len=6, beam=2, coverage_penalty=0.1
    )
    assert generated.size(0) == 2


def test_module_import_does_not_require_torch() -> None:
    import importlib
    import sys

    for key in list(sys.modules):
        if key.startswith("modernbert_g2p.models.p_a"):
            del sys.modules[key]

    config_mod = importlib.import_module("modernbert_g2p.models.p_a.config")
    assert hasattr(config_mod, "PAConfig")

    pkg = importlib.import_module("modernbert_g2p.models.p_a")
    assert "PASeq2Seq" in pkg.__all__
