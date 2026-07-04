"""Phase 2 training utilities: loss functions, optimizer/scheduler builders, trainer, data."""

from __future__ import annotations

__all__ = [
    "APBPBIOLoss",
    "G2PDataset",
    "LabelSmoothingCELoss",
    "PACollator",
    "PCCollator",
    "Trainer",
    "TrainingArgs",
    "build_optimizer",
    "build_scheduler",
    "build_smoke_pipeline",
    "make_dummy_row",
    "run_training",
]


def __getattr__(name: str) -> object:
    if name in {"LabelSmoothingCELoss", "APBPBIOLoss"}:
        from modernbert_g2p.training.loss import APBPBIOLoss, LabelSmoothingCELoss

        return {"LabelSmoothingCELoss": LabelSmoothingCELoss, "APBPBIOLoss": APBPBIOLoss}[name]
    if name in {"build_optimizer", "build_scheduler"}:
        from modernbert_g2p.training.optim import build_optimizer, build_scheduler

        return {"build_optimizer": build_optimizer, "build_scheduler": build_scheduler}[name]
    if name in {"Trainer", "TrainingArgs"}:
        from modernbert_g2p.training.trainer import Trainer, TrainingArgs

        return {"Trainer": Trainer, "TrainingArgs": TrainingArgs}[name]
    if name in {"G2PDataset", "PACollator", "PCCollator", "make_dummy_row"}:
        from modernbert_g2p.training import data as _data

        return getattr(_data, name)
    if name in {"build_smoke_pipeline", "run_training"}:
        from modernbert_g2p.training.factory import build_smoke_pipeline, run_training

        return {"build_smoke_pipeline": build_smoke_pipeline, "run_training": run_training}[name]
    raise AttributeError(name)
