"""Phase 2 configuration dataclasses and YAML loader.

Loads pilot-specific YAML files (``configs/p_a.yaml`` / ``p_b.yaml`` / ``p_c.yaml``)
into strongly-typed ``Phase2Config`` objects. ``model`` remains an untyped mapping
here so this module does not depend on the pilot-specific ``PAConfig`` / ``PBConfig``
/ ``PCConfig`` dataclasses (validated downstream by each pilot's factory).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml

VALID_PILOTS: frozenset[str] = frozenset({"P-A", "P-B", "P-C"})
VALID_PRECISIONS: frozenset[str] = frozenset({"bf16", "fp16", "fp32"})


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Paths + loader knobs for a Phase 2 training run."""

    train_path: str
    val_path: str
    test_path: str | None = None
    hard_set_path: str | None = None
    jsut_yaml: str | None = None
    jvs_dir: str | None = None
    rohan_txt: str | None = None
    batch_size: int = 8
    num_workers: int = 0


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Optimizer / scheduler / precision knobs shared by all three pilots."""

    encoder_lr: float
    head_lr: float
    warmup_steps: int
    total_steps: int
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    precision: str = "bf16"
    seed: int = 20260704
    accumulate_grad_batches: int = 1


@dataclass(frozen=True, slots=True)
class Phase2Config:
    """Root Phase 2 config. ``model`` is pilot-specific and validated downstream."""

    pilot: str
    model: dict[str, Any]
    data: DataConfig
    training: TrainingConfig
    output_dir: str = "reports/phase2/{pilot}/{seed}"


def _reject_unknown_keys(section: str, expected: set[str], actual: set[str]) -> None:
    unknown = actual - expected
    if unknown:
        raise ValueError(
            f"Unknown keys in {section}: {sorted(unknown)}. Expected subset of {sorted(expected)}"
        )


def _require_dict(section: str, value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{section} must be a mapping, got {type(value).__name__}")
    return value


def _build_data_config(raw: dict[str, Any]) -> DataConfig:
    valid = {f.name for f in fields(DataConfig)}
    _reject_unknown_keys("data", valid, set(raw))
    required = {"train_path", "val_path"}
    missing = required - set(raw)
    if missing:
        raise ValueError(f"Missing required data keys: {sorted(missing)}")
    return DataConfig(**raw)


def _build_training_config(raw: dict[str, Any]) -> TrainingConfig:
    valid = {f.name for f in fields(TrainingConfig)}
    _reject_unknown_keys("training", valid, set(raw))
    required = {"encoder_lr", "head_lr", "warmup_steps", "total_steps"}
    missing = required - set(raw)
    if missing:
        raise ValueError(f"Missing required training keys: {sorted(missing)}")
    return TrainingConfig(**raw)


def _build_phase2_config(raw: dict[str, Any]) -> Phase2Config:
    top_valid = {f.name for f in fields(Phase2Config)}
    _reject_unknown_keys("Phase2Config", top_valid, set(raw))

    required = {"pilot", "model", "data", "training"}
    missing = required - set(raw)
    if missing:
        raise ValueError(f"Missing required top-level keys: {sorted(missing)}")

    pilot = raw["pilot"]
    if pilot not in VALID_PILOTS:
        raise ValueError(
            f"Unknown pilot value: {pilot!r}. Must be one of {sorted(VALID_PILOTS)}"
        )

    model = _require_dict("model", raw["model"])
    data = _build_data_config(_require_dict("data", raw["data"]))
    training = _build_training_config(_require_dict("training", raw["training"]))

    if training.precision not in VALID_PRECISIONS:
        raise ValueError(
            f"Unknown precision: {training.precision!r}. Must be one of {sorted(VALID_PRECISIONS)}"
        )

    output_dir = raw.get("output_dir", Phase2Config.__dataclass_fields__["output_dir"].default)
    return Phase2Config(
        pilot=pilot,
        model=dict(model),
        data=data,
        training=training,
        output_dir=output_dir,
    )


def load_config(yaml_path: Path | str) -> Phase2Config:
    """Load a Phase 2 YAML config and return a validated ``Phase2Config``."""
    path = Path(yaml_path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Top-level YAML must be a mapping, got {type(raw).__name__}")
    return _build_phase2_config(raw)


def config_to_dict(cfg: Phase2Config) -> dict[str, Any]:
    """Convert a ``Phase2Config`` to a plain dict suitable for ``yaml.safe_dump``."""
    return asdict(cfg)


def dump_config(cfg: Phase2Config, path: Path | str) -> None:
    """Serialize a ``Phase2Config`` back to YAML."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config_to_dict(cfg), f, sort_keys=False, allow_unicode=True)


__all__ = [
    "VALID_PILOTS",
    "VALID_PRECISIONS",
    "DataConfig",
    "TrainingConfig",
    "Phase2Config",
    "load_config",
    "dump_config",
    "config_to_dict",
]
