"""Unit tests for the P-B MeCab + [MORPH] + ModernBERT pilot (Track 4)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from modernbert_g2p.models.p_b import MeCabPretokenizer, MeCabToken, PBConfig

if TYPE_CHECKING:  # pragma: no cover
    pass


def _build_model(head_variant: str = "B1", seed: int = 0):
    import torch

    from modernbert_g2p.models.p_b import build_p_b

    torch.manual_seed(seed)
    config = PBConfig.tiny(head_variant=head_variant)
    model = build_p_b(config, morph_token_id=config.morph_token_id)
    return model, config


def _make_batch(config: PBConfig, batch_size: int = 2, seq_len: int = 16, morphs: int = 4):
    import torch

    input_ids = torch.randint(0, config.encoder_vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    stride = seq_len // morphs
    morph_positions = torch.tensor(
        [[j * stride for j in range(morphs)] for _ in range(batch_size)]
    )
    morph_lens = torch.tensor(
        [[max(1, stride - 1) for _ in range(morphs)] for _ in range(batch_size)]
    )
    return input_ids, attention_mask, morph_positions, morph_lens


def _make_labels(config: PBConfig, batch_size: int, morphs: int):
    import torch

    max_mora = config.max_mora_per_morph
    labels_phon = torch.randint(
        0, config.phoneme_vocab_size, (batch_size, morphs, max_mora)
    )
    labels_hl = torch.randint(0, 2, (batch_size, morphs, max_mora))
    labels_apbp = torch.randint(0, 3, (batch_size, morphs))
    return labels_phon, labels_hl, labels_apbp


def test_pbconfig_tiny_b1_defaults() -> None:
    config = PBConfig.tiny("B1")
    assert config.head_variant == "B1"
    assert config.encoder_hidden == 32
    assert config.phoneme_vocab_size == 16
    assert config.max_mora_per_morph == 4
    assert config.encoder_name == "tiny"
    assert config.morph_token_id == 15
    assert config.dict_hit_weight == pytest.approx(0.3)
    assert config.apbp_alpha == pytest.approx(0.2)
    assert config.label_smoothing == pytest.approx(0.05)


def test_pbconfig_tiny_b2_defaults() -> None:
    config = PBConfig.tiny("B2")
    assert config.head_variant == "B2"
    assert config.lstm_hidden == 16
    assert config.encoder_hidden == 32


def test_pbconfig_defaults_match_design() -> None:
    config = PBConfig()
    assert config.encoder_name == "sbintuitions/modernbert-ja-130m"
    assert config.max_mora_per_morph == 8
    assert config.phoneme_vocab_size == 68
    assert config.apbp_alpha == pytest.approx(0.2)
    assert config.dict_hit_weight == pytest.approx(0.3)
    assert config.label_smoothing == pytest.approx(0.05)


def test_pbconfig_morph_token_id_default_is_non_negative() -> None:
    """Regression: default must not be -100 (silent CE-ignore trap in build_p_b)."""
    config = PBConfig()
    assert config.morph_token_id >= 0
    assert config.morph_token_id == 0


def test_pbconfig_unknown_head_rejected() -> None:
    pytest.importorskip("torch")
    from modernbert_g2p.models.p_b import PBMorphBERT

    bad = PBConfig.tiny("B1")
    from dataclasses import replace

    bad = replace(bad, head_variant="B9")
    with pytest.raises(ValueError, match="Unknown head_variant"):
        PBMorphBERT(bad)


def test_build_b1_positive_params() -> None:
    pytest.importorskip("torch")
    model, _ = _build_model("B1")
    assert model.count_params() > 0


def test_build_b2_positive_params_and_larger_than_b1() -> None:
    pytest.importorskip("torch")
    model_b1, _ = _build_model("B1")
    model_b2, _ = _build_model("B2")
    assert model_b2.count_params() > 0
    assert model_b2.count_params() > model_b1.count_params()


def test_forward_b1_shapes() -> None:
    pytest.importorskip("torch")
    model, config = _build_model("B1")
    input_ids, attention_mask, morph_positions, morph_lens = _make_batch(config)
    out = model(input_ids, attention_mask, morph_positions, morph_lens)
    b, m = morph_positions.shape
    assert out["phon_logits"].shape == (b, m, config.max_mora_per_morph, config.phoneme_vocab_size)
    assert out["hl_logits"].shape == (b, m, config.max_mora_per_morph, 2)
    assert out["apbp_logits"].shape == (b, m, 3)
    assert out["loss"] is None


def test_forward_b1_loss_finite_and_backward() -> None:
    import torch

    pytest.importorskip("torch")
    model, config = _build_model("B1")
    input_ids, attention_mask, morph_positions, morph_lens = _make_batch(config)
    b, m = morph_positions.shape
    labels_phon, labels_hl, labels_apbp = _make_labels(config, b, m)
    out = model(
        input_ids,
        attention_mask,
        morph_positions,
        morph_lens,
        morph_labels_phon=labels_phon,
        morph_labels_hl=labels_hl,
        morph_labels_apbp=labels_apbp,
    )
    loss = out["loss"]
    assert loss is not None
    assert torch.isfinite(loss)
    assert loss.requires_grad
    loss.backward()


def test_forward_b2_shapes_match_b1() -> None:
    pytest.importorskip("torch")
    model, config = _build_model("B2")
    input_ids, attention_mask, morph_positions, morph_lens = _make_batch(config)
    b, m = morph_positions.shape
    labels_phon, _, _ = _make_labels(config, b, m)
    out = model(
        input_ids,
        attention_mask,
        morph_positions,
        morph_lens,
        morph_labels_phon=labels_phon,
    )
    assert out["phon_logits"].shape == (
        b,
        m,
        config.max_mora_per_morph,
        config.phoneme_vocab_size,
    )
    import torch

    assert torch.isfinite(out["loss"])


def test_dict_hit_mask_downweights_loss() -> None:
    import torch

    pytest.importorskip("torch")
    model, config = _build_model("B1")
    input_ids, attention_mask, morph_positions, morph_lens = _make_batch(config)
    b, m = morph_positions.shape
    labels_phon, labels_hl, labels_apbp = _make_labels(config, b, m)

    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "morph_positions": morph_positions,
        "morph_lens": morph_lens,
        "morph_labels_phon": labels_phon,
        "morph_labels_hl": labels_hl,
        "morph_labels_apbp": labels_apbp,
    }
    mask_nohit = torch.zeros((b, m), dtype=torch.bool)
    mask_hit = torch.ones((b, m), dtype=torch.bool)

    out_nohit = model(**kwargs, dict_hit_mask=mask_nohit)
    out_hit = model(**kwargs, dict_hit_mask=mask_hit)
    assert out_hit["loss"].item() < out_nohit["loss"].item()
    ratio = out_hit["loss"].item() / out_nohit["loss"].item()
    assert ratio == pytest.approx(config.dict_hit_weight, rel=1e-3)


def test_sample_weights_scale_loss() -> None:
    import torch

    pytest.importorskip("torch")
    model, config = _build_model("B1")
    input_ids, attention_mask, morph_positions, morph_lens = _make_batch(config)
    b, m = morph_positions.shape
    labels_phon, _, _ = _make_labels(config, b, m)

    out_uniform = model(
        input_ids,
        attention_mask,
        morph_positions,
        morph_lens,
        morph_labels_phon=labels_phon,
        sample_weights=torch.ones(b),
    )
    out_scaled = model(
        input_ids,
        attention_mask,
        morph_positions,
        morph_lens,
        morph_labels_phon=labels_phon,
        sample_weights=torch.full((b,), 2.0),
    )
    assert out_scaled["loss"].item() == pytest.approx(2.0 * out_uniform["loss"].item(), rel=1e-4)


def test_padded_morphs_do_not_contribute() -> None:

    pytest.importorskip("torch")
    model, config = _build_model("B1")
    input_ids, attention_mask, morph_positions, morph_lens = _make_batch(config)
    b, m = morph_positions.shape
    labels_phon, _, _ = _make_labels(config, b, m)

    labels_padded = labels_phon.clone()
    labels_padded[..., :] = -100
    out = model(
        input_ids,
        attention_mask,
        morph_positions,
        morph_lens,
        morph_labels_phon=labels_padded,
    )
    assert out["loss"].item() == pytest.approx(0.0, abs=1e-6)


def test_apbp_alpha_scales_loss() -> None:
    from dataclasses import replace

    import torch

    pytest.importorskip("torch")
    torch.manual_seed(0)
    config = PBConfig.tiny("B1")
    from modernbert_g2p.models.p_b import build_p_b

    model = build_p_b(config, morph_token_id=config.morph_token_id)

    input_ids, attention_mask, morph_positions, morph_lens = _make_batch(config)
    b, m = morph_positions.shape
    _, _, labels_apbp = _make_labels(config, b, m)

    out_default = model(
        input_ids,
        attention_mask,
        morph_positions,
        morph_lens,
        morph_labels_apbp=labels_apbp,
    )
    torch.manual_seed(0)
    scaled_config = replace(config, apbp_alpha=1.0)
    model2 = build_p_b(scaled_config, morph_token_id=scaled_config.morph_token_id)
    out_alpha1 = model2(
        input_ids,
        attention_mask,
        morph_positions,
        morph_lens,
        morph_labels_apbp=labels_apbp,
    )
    # apbp_alpha 0.2 vs 1.0 → 5x factor on the same APBP CE
    assert out_alpha1["loss"].item() == pytest.approx(5.0 * out_default["loss"].item(), rel=1e-4)


def test_deterministic_given_seed() -> None:
    pytest.importorskip("torch")
    import torch

    torch.manual_seed(42)
    m1, config = _build_model("B1", seed=42)
    input_ids, attention_mask, morph_positions, morph_lens = _make_batch(config)
    out1 = m1(input_ids, attention_mask, morph_positions, morph_lens)

    m2, _ = _build_model("B1", seed=42)
    out2 = m2(input_ids, attention_mask, morph_positions, morph_lens)
    assert torch.allclose(out1["phon_logits"], out2["phon_logits"])
    assert torch.allclose(out1["hl_logits"], out2["hl_logits"])
    assert torch.allclose(out1["apbp_logits"], out2["apbp_logits"])


def test_build_p_b_rejects_negative_morph_token_id() -> None:
    """Regression: negative morph_token_id makes the resize/init path a silent no-op."""
    pytest.importorskip("torch")
    from dataclasses import replace

    from modernbert_g2p.models.p_b import build_p_b

    bad = replace(PBConfig.tiny("B1"), morph_token_id=-100)
    with pytest.raises(ValueError, match="morph_token_id must be a non-negative"):
        build_p_b(bad)

    good = PBConfig.tiny("B1")
    with pytest.raises(ValueError, match="morph_token_id must be a non-negative"):
        build_p_b(good, morph_token_id=-1)


def test_build_p_b_uses_default_placeholder_when_no_override() -> None:
    """Confirms the safe placeholder default lets build succeed without an explicit id."""
    pytest.importorskip("torch")
    import torch

    from modernbert_g2p.models.p_b import build_p_b

    torch.manual_seed(0)
    model = build_p_b(PBConfig.tiny("B1"))
    assert model.count_params() > 0


def test_build_p_b_resizes_embedding_when_morph_id_out_of_range() -> None:
    pytest.importorskip("torch")
    from dataclasses import replace

    import torch

    from modernbert_g2p.models.p_b import build_p_b

    base = PBConfig.tiny("B1")
    beyond = replace(base, morph_token_id=base.encoder_vocab_size + 5)
    torch.manual_seed(0)
    model = build_p_b(beyond)
    embed = model.encoder.embed
    assert embed.num_embeddings >= beyond.morph_token_id + 1
    morph_row = embed.weight[beyond.morph_token_id]
    assert torch.linalg.norm(morph_row) > 0


def test_mecab_pretokenizer_requires_fugashi_or_skipped() -> None:
    pytest.importorskip("fugashi")
    pre = MeCabPretokenizer()
    tokens = pre.pretokenize("東京都に住む")
    assert isinstance(tokens, list)
    for tok in tokens:
        assert isinstance(tok, MeCabToken)
        assert tok.end > tok.start
    surfaces = "".join(t.surface for t in tokens)
    assert surfaces == "東京都に住む"


def test_mecab_pretokenizer_empty_string() -> None:
    pre = MeCabPretokenizer()
    assert pre.pretokenize("") == []


def test_mock_pretokenizer_dispatch() -> None:
    from modernbert_g2p.models.p_b import build_mock_pretokenizer

    mapping = {
        "私は東京で本を読みました": [
            MeCabToken(surface="私", start=0, end=1, pos="名詞"),
            MeCabToken(surface="は", start=1, end=2, pos="助詞"),
            MeCabToken(surface="東京", start=2, end=4, pos="名詞"),
        ],
    }
    pre = build_mock_pretokenizer(mapping)
    tokens = pre.pretokenize("私は東京で本を読みました")
    assert len(tokens) == 3
    assert tokens[0].surface == "私"
    assert tokens[2].surface == "東京"
    assert pre.pretokenize("未知") == []
