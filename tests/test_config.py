"""Tests for Phase 2 config loader (Track 9)."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

from modernbert_g2p.config import (
    DataConfig,
    Phase2Config,
    TrainingConfig,
    config_to_dict,
    dump_config,
    load_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"


@pytest.mark.parametrize(
    ("yaml_name", "expected_pilot", "expected_encoder_lr", "expected_total_steps"),
    [
        ("p_a.yaml", "P-A", 3.0e-5, 60_000),
        ("p_b.yaml", "P-B", 5.0e-5, 45_000),
        ("p_c.yaml", "P-C", 5.0e-5, 40_000),
    ],
)
def test_load_config_for_each_pilot_returns_valid_phase2config(
    yaml_name: str,
    expected_pilot: str,
    expected_encoder_lr: float,
    expected_total_steps: int,
) -> None:
    cfg = load_config(CONFIG_DIR / yaml_name)

    assert isinstance(cfg, Phase2Config)
    assert cfg.pilot == expected_pilot
    assert isinstance(cfg.data, DataConfig)
    assert isinstance(cfg.training, TrainingConfig)
    assert isinstance(cfg.model, dict)
    assert cfg.training.encoder_lr == pytest.approx(expected_encoder_lr)
    assert cfg.training.total_steps == expected_total_steps
    assert cfg.training.precision == "bf16"


def test_p_a_yaml_hyperparameters_match_design_doc() -> None:
    cfg = load_config(CONFIG_DIR / "p_a.yaml")

    assert cfg.model["encoder_name"] == "sbintuitions/modernbert-ja-130m"
    assert cfg.model["decoder_layers"] == 6
    assert cfg.model["decoder_hidden"] == 512
    assert cfg.model["decoder_heads"] == 8
    assert cfg.model["decoder_ffn"] == 2048
    assert cfg.model["phoneme_vocab_size"] == 68
    assert cfg.model["label_smoothing"] == pytest.approx(0.1)
    assert cfg.training.encoder_lr == pytest.approx(3.0e-5)
    assert cfg.training.head_lr == pytest.approx(1.0e-4)
    assert cfg.training.warmup_steps == 2000
    assert cfg.training.total_steps == 60_000


def test_p_b_yaml_hyperparameters_match_design_doc() -> None:
    cfg = load_config(CONFIG_DIR / "p_b.yaml")

    assert cfg.model["encoder_name"] == "sbintuitions/modernbert-ja-130m"
    assert cfg.model["head_variant"] == "B1"
    assert cfg.model["max_mora_per_morph"] == 8
    assert cfg.model["phoneme_vocab_size"] == 68
    assert cfg.model["apbp_alpha"] == pytest.approx(0.2)
    assert cfg.model["dict_hit_weight"] == pytest.approx(0.3)
    assert cfg.model["label_smoothing"] == pytest.approx(0.05)
    assert cfg.training.encoder_lr == pytest.approx(5.0e-5)
    assert cfg.training.head_lr == pytest.approx(3.0e-4)
    assert cfg.training.warmup_steps == 1500
    assert cfg.training.total_steps == 45_000


def test_p_c_yaml_hyperparameters_match_design_doc() -> None:
    cfg = load_config(CONFIG_DIR / "p_c.yaml")

    assert cfg.model["encoder_name"] == "tohoku-nlp/bert-base-japanese-char-v2"
    assert cfg.model["head_variant"] == "C1"
    assert cfg.model["max_slot"] == 8
    assert cfg.model["phoneme_vocab_size"] == 68
    assert cfg.model["apbp_alpha"] == pytest.approx(0.2)
    assert cfg.training.encoder_lr == pytest.approx(5.0e-5)
    assert cfg.training.head_lr == pytest.approx(5.0e-5)
    assert cfg.training.warmup_steps == 1000
    assert cfg.training.total_steps == 40_000


def test_unknown_pilot_raises_value_error(tmp_path: Path) -> None:
    payload = {
        "pilot": "P-X",
        "model": {},
        "data": {"train_path": "a", "val_path": "b"},
        "training": {
            "encoder_lr": 1e-4,
            "head_lr": 1e-4,
            "warmup_steps": 10,
            "total_steps": 100,
        },
    }
    path = tmp_path / "bad_pilot.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown pilot"):
        load_config(path)


def test_unknown_top_level_key_raises_value_error(tmp_path: Path) -> None:
    payload = {
        "pilot": "P-A",
        "model": {},
        "data": {"train_path": "a", "val_path": "b"},
        "training": {
            "encoder_lr": 1e-4,
            "head_lr": 1e-4,
            "warmup_steps": 10,
            "total_steps": 100,
        },
        "made_up_field": 42,
    }
    path = tmp_path / "unknown_top.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown keys in Phase2Config"):
        load_config(path)


def test_unknown_training_key_raises_value_error(tmp_path: Path) -> None:
    payload = {
        "pilot": "P-A",
        "model": {},
        "data": {"train_path": "a", "val_path": "b"},
        "training": {
            "encoder_lr": 1e-4,
            "head_lr": 1e-4,
            "warmup_steps": 10,
            "total_steps": 100,
            "typo_field": 1,
        },
    }
    path = tmp_path / "unknown_training.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown keys in training"):
        load_config(path)


def test_missing_required_training_field_raises(tmp_path: Path) -> None:
    payload = {
        "pilot": "P-A",
        "model": {},
        "data": {"train_path": "a", "val_path": "b"},
        "training": {"encoder_lr": 1e-4, "warmup_steps": 10, "total_steps": 100},
    }
    path = tmp_path / "missing.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Missing required training keys"):
        load_config(path)


def test_missing_required_data_field_raises(tmp_path: Path) -> None:
    payload = {
        "pilot": "P-A",
        "model": {},
        "data": {"train_path": "a"},
        "training": {
            "encoder_lr": 1e-4,
            "head_lr": 1e-4,
            "warmup_steps": 10,
            "total_steps": 100,
        },
    }
    path = tmp_path / "missing_data.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Missing required data keys"):
        load_config(path)


def test_missing_top_level_key_raises(tmp_path: Path) -> None:
    payload = {"pilot": "P-A", "model": {}, "data": {"train_path": "a", "val_path": "b"}}
    path = tmp_path / "missing_top.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Missing required top-level keys"):
        load_config(path)


def test_invalid_precision_raises(tmp_path: Path) -> None:
    payload = {
        "pilot": "P-A",
        "model": {},
        "data": {"train_path": "a", "val_path": "b"},
        "training": {
            "encoder_lr": 1e-4,
            "head_lr": 1e-4,
            "warmup_steps": 10,
            "total_steps": 100,
            "precision": "int4",
        },
    }
    path = tmp_path / "bad_precision.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown precision"):
        load_config(path)


def test_nested_dataclass_field_types() -> None:
    cfg = load_config(CONFIG_DIR / "p_a.yaml")

    assert isinstance(cfg.data.train_path, str)
    assert isinstance(cfg.data.val_path, str)
    assert isinstance(cfg.data.batch_size, int)
    assert isinstance(cfg.training.encoder_lr, float)
    assert isinstance(cfg.training.head_lr, float)
    assert isinstance(cfg.training.warmup_steps, int)
    assert isinstance(cfg.training.total_steps, int)
    assert isinstance(cfg.training.seed, int)


def test_determinism_load_twice_returns_equal_configs() -> None:
    a = load_config(CONFIG_DIR / "p_a.yaml")
    b = load_config(CONFIG_DIR / "p_a.yaml")

    assert a == b
    assert a.data == b.data
    assert a.training == b.training
    assert a.model == b.model


@pytest.mark.parametrize("yaml_name", ["p_a.yaml", "p_b.yaml", "p_c.yaml"])
def test_yaml_roundtrip_via_dict_is_consistent(yaml_name: str, tmp_path: Path) -> None:
    cfg = load_config(CONFIG_DIR / yaml_name)
    dumped = tmp_path / yaml_name
    dump_config(cfg, dumped)

    reloaded = load_config(dumped)

    assert reloaded == cfg
    assert config_to_dict(cfg) == config_to_dict(reloaded)
    assert config_to_dict(cfg) == asdict(cfg)


def test_output_dir_default_and_override() -> None:
    cfg = load_config(CONFIG_DIR / "p_a.yaml")
    assert cfg.output_dir == "reports/phase2/p_a/{seed}"


def test_config_to_dict_produces_yaml_safe_structure() -> None:
    cfg = load_config(CONFIG_DIR / "p_b.yaml")
    d = config_to_dict(cfg)

    assert d["pilot"] == "P-B"
    assert isinstance(d["data"], dict)
    assert isinstance(d["training"], dict)
    assert d["training"]["encoder_lr"] == pytest.approx(5.0e-5)
    rendered = yaml.safe_dump(d)
    assert "P-B" in rendered


def test_frozen_dataclasses_reject_mutation() -> None:
    cfg = load_config(CONFIG_DIR / "p_a.yaml")

    with pytest.raises((AttributeError, TypeError)):
        cfg.pilot = "P-B"  # type: ignore[misc]
    with pytest.raises((AttributeError, TypeError)):
        cfg.training.encoder_lr = 1.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("yaml_name", "pilot_config_path", "pilot_config_name"),
    [
        ("p_a.yaml", "modernbert_g2p.models.p_a.config", "PAConfig"),
        ("p_b.yaml", "modernbert_g2p.models.p_b.config", "PBConfig"),
        ("p_c.yaml", "modernbert_g2p.models.p_c.config", "PCConfig"),
    ],
)
def test_yaml_model_keys_are_valid_pilot_config_fields(
    yaml_name: str, pilot_config_path: str, pilot_config_name: str
) -> None:
    from dataclasses import fields as dc_fields
    from importlib import import_module

    cfg = load_config(CONFIG_DIR / yaml_name)
    module = import_module(pilot_config_path)
    pilot_config_cls = getattr(module, pilot_config_name)
    valid_fields = {f.name for f in dc_fields(pilot_config_cls)}
    yaml_keys = set(cfg.model.keys())
    unknown = yaml_keys - valid_fields
    assert not unknown, (
        f"{yaml_name} has model.* keys not present in {pilot_config_name}: "
        f"{sorted(unknown)}. Either add the field to the dataclass or drop the key."
    )


def test_p_a_yaml_does_not_contain_dead_keys() -> None:
    cfg = load_config(CONFIG_DIR / "p_a.yaml")
    assert "beam_size" not in cfg.model
    assert "max_decode_len" not in cfg.model


def test_p_c_yaml_does_not_contain_dead_keys() -> None:
    cfg = load_config(CONFIG_DIR / "p_c.yaml")
    assert "hl_vocab_size" not in cfg.model


def test_training_label_smoothing_field_removed() -> None:
    from dataclasses import fields as dc_fields

    cfg = load_config(CONFIG_DIR / "p_a.yaml")
    training_field_names = {f.name for f in dc_fields(cfg.training)}
    assert "label_smoothing" not in training_field_names, (
        "training.label_smoothing was dead code (models pull label_smoothing from "
        "their own PAConfig/PBConfig/PCConfig). It must not be re-introduced."
    )


def test_yaml_training_block_rejects_label_smoothing(tmp_path: Path) -> None:
    payload = {
        "pilot": "P-A",
        "model": {},
        "data": {"train_path": "a", "val_path": "b"},
        "training": {
            "encoder_lr": 1e-4,
            "head_lr": 1e-4,
            "warmup_steps": 10,
            "total_steps": 100,
            "label_smoothing": 0.1,
        },
    }
    path = tmp_path / "reintroduced_label_smoothing.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown keys in training"):
        load_config(path)
