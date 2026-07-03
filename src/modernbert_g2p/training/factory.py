"""End-to-end factories that assemble the Phase 2 training pipeline from a config.

Two public entry points, both consumed by ``modernbert_g2p.cli``:

- :func:`build_smoke_pipeline`: constructs a :class:`Trainer` wired to a
  minimal one-step configuration; the CLI invokes ``trainer.run(steps=1)``
  to verify that forward + backward + optimizer step all run without error.
- :func:`run_training`: constructs the same pipeline with full-scale
  hyperparameters, calls :meth:`Trainer.run`, and returns the resulting
  scalar metrics dict.

Pilot dispatch (``cfg.pilot in {"P-A", "P-B", "P-C"}``) selects the
per-pilot ``PXConfig`` dataclass, tokenizer, collator, and model builder.
Heavy imports (``torch``, ``transformers``, ``modernbert_g2p.models.*``)
stay lazy so ``import modernbert_g2p.training`` still costs no more than
its Phase-1 dependencies.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from modernbert_g2p.config import Phase2Config
    from modernbert_g2p.training.trainer import Trainer, TrainingArgs


TINY_MODE_ENV_VAR: str = "MODERNBERT_G2P_TEST_TINY"


def _is_tiny_mode() -> bool:
    """Return ``True`` when ``$MODERNBERT_G2P_TEST_TINY == "1"``.

    Tiny mode swaps the real HF-encoder + Parquet-dataset pipeline for a
    fully-synthetic random-init model + in-memory dataset so CI can exercise
    the Trainer loop end-to-end without hitting the internet or GPU.
    """
    return os.environ.get(TINY_MODE_ENV_VAR) == "1"


_PILOT_ALIASES: dict[str, str] = {
    "P-A": "P-A",
    "P-B": "P-B",
    "P-C": "P-C",
    "p_a": "P-A",
    "p_b": "P-B",
    "p_c": "P-C",
}


def _normalize_pilot(pilot: str) -> str:
    key = _PILOT_ALIASES.get(str(pilot))
    if key is None:
        raise ValueError(
            f"Unknown pilot {pilot!r}; must be one of {sorted(set(_PILOT_ALIASES))}"
        )
    return key


def _training_args_from_cfg(
    cfg: Phase2Config,
    *,
    seed: int,
    output_dir: Path,
    total_steps: int | None = None,
    warmup_steps: int | None = None,
    batch_size: int | None = None,
    eval_every: int = 0,
    save_every: int = 0,
) -> TrainingArgs:
    from modernbert_g2p.training.trainer import TrainingArgs

    training = cfg.training
    data = cfg.data
    return TrainingArgs(
        output_dir=Path(output_dir),
        total_steps=total_steps if total_steps is not None else training.total_steps,
        warmup_steps=warmup_steps if warmup_steps is not None else training.warmup_steps,
        encoder_lr=training.encoder_lr,
        head_lr=training.head_lr,
        weight_decay=training.weight_decay,
        grad_clip=training.grad_clip,
        eval_every=eval_every,
        save_every=save_every,
        batch_size=batch_size if batch_size is not None else data.batch_size,
        accumulate_grad_batches=training.accumulate_grad_batches,
        precision=training.precision,
        seed=seed,
        num_workers=data.num_workers,
    )


def _build_dataset(path: str | Path) -> Any:
    from modernbert_g2p.training.data import G2PDataset

    return G2PDataset(Path(path))


def _build_p_a_pipeline(cfg: Phase2Config) -> tuple[Any, Any, Any, Any]:
    from modernbert_g2p.models.p_a import PAConfig, build_p_a
    from modernbert_g2p.models.tokenization import PATokenizer
    from modernbert_g2p.training.data import PACollator

    pa_cfg = _instantiate_config(PAConfig, cfg.model)
    tokenizer = PATokenizer(tokenizer_name=pa_cfg.encoder_name)
    model = build_p_a(pa_cfg)
    collator = PACollator(tokenizer)
    return pa_cfg, tokenizer, collator, model


def _build_p_b_pipeline(cfg: Phase2Config) -> tuple[Any, Any, Any, Any]:
    from modernbert_g2p.models.p_b import PBConfig, build_p_b
    from modernbert_g2p.models.tokenization import PBTokenizer
    from modernbert_g2p.training.data import PBCollator

    pb_cfg = _instantiate_config(PBConfig, cfg.model)
    tokenizer = PBTokenizer(tokenizer_name=pb_cfg.encoder_name)
    model = build_p_b(pb_cfg, morph_token_id=getattr(tokenizer, "morph_id", 0))
    collator = PBCollator(tokenizer, max_slot=pb_cfg.max_mora_per_morph)
    return pb_cfg, tokenizer, collator, model


def _build_p_c_pipeline(cfg: Phase2Config) -> tuple[Any, Any, Any, Any]:
    from modernbert_g2p.models.p_c import PCConfig, build_p_c
    from modernbert_g2p.models.tokenization import PCTokenizer
    from modernbert_g2p.training.data import PCCollator

    pc_cfg = _instantiate_config(PCConfig, cfg.model)
    tokenizer = PCTokenizer(tokenizer_name=pc_cfg.encoder_name)
    model = build_p_c(pc_cfg)
    collator = PCCollator(tokenizer)
    return pc_cfg, tokenizer, collator, model


def _instantiate_config(cls: type, raw: dict[str, Any]) -> Any:
    """Instantiate a pilot config dataclass, dropping unknown keys.

    YAML configs may carry keys that a pilot's config does not consume yet
    (e.g. ``hl_vocab_size`` for P-C). We drop them here so a partially
    landed set of Track T3/T4/T5 config fields does not block the trainer
    factory. Unknown keys are logged to stderr but not fatal.
    """
    import dataclasses
    import sys

    valid = {f.name for f in dataclasses.fields(cls)}
    known = {k: v for k, v in raw.items() if k in valid}
    unknown = set(raw) - valid
    if unknown:
        print(
            f"[factory] warning: dropping unknown {cls.__name__} keys: {sorted(unknown)}",
            file=sys.stderr,
        )
    return cls(**known)


def _dispatch_pipeline(cfg: Phase2Config) -> tuple[Any, Any, Any, Any]:
    pilot = _normalize_pilot(cfg.pilot)
    builders: dict[str, Callable[[Phase2Config], tuple[Any, Any, Any, Any]]] = {
        "P-A": _build_p_a_pipeline,
        "P-B": _build_p_b_pipeline,
        "P-C": _build_p_c_pipeline,
    }
    return builders[pilot](cfg)


def _build_tiny_smoke_trainer(
    cfg: Phase2Config,
    *,
    seed: int,
    output_dir: Path,
) -> Trainer:
    """Assemble a fully-synthetic Trainer that runs one step without HF Hub.

    Used when :data:`TINY_MODE_ENV_VAR` == ``"1"``. Bypasses the real
    tokenizer / dataset / encoder to sidestep the T2/T6 interface
    misalignment listed in the cross-track review and the offline sandbox
    that cannot reach ``huggingface.co``. Verifies only that the Trainer
    loop (forward, backward, optimizer.step, scheduler.step) is wired.
    """
    import torch
    from torch import nn

    from modernbert_g2p.training.trainer import Trainer

    class _TinyStubModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Linear(4, 8)
            self.head = nn.Linear(8, 5)

        def forward(
            self,
            input_ids: torch.Tensor,
            labels: torch.Tensor,
            **_: object,
        ) -> dict[str, torch.Tensor]:
            h = self.encoder(input_ids)
            logits = self.head(h)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
            return {"loss": loss}

    class _TinyStubDataset:
        def __init__(self, n: int) -> None:
            torch.manual_seed(seed)
            self._items = [
                {
                    "input_ids": torch.randn(3, 4),
                    "labels": torch.randint(0, 5, (3,)),
                }
                for _ in range(n)
            ]

        def __len__(self) -> int:
            return len(self._items)

        def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
            return self._items[idx]

    def _stub_collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "labels": torch.stack([b["labels"] for b in batch]),
        }

    args = _training_args_from_cfg(
        cfg,
        seed=seed,
        output_dir=Path(output_dir),
        total_steps=1,
        warmup_steps=0,
        batch_size=2,
        eval_every=0,
        save_every=0,
    )
    return Trainer(
        _TinyStubModel(),
        _TinyStubDataset(4),
        _TinyStubDataset(2),
        _stub_collate,
        args,
        device="cpu",
    )


def build_smoke_pipeline(
    cfg: Phase2Config,
    *,
    seed: int,
    output_dir: Path,
) -> Trainer:
    """Assemble a :class:`Trainer` sized for a one-step smoke run.

    ``total_steps`` is forced to 1 and ``warmup_steps`` to 0 so that
    ``trainer.run(steps=1)`` completes a full forward + backward +
    optimizer step irrespective of the config's production settings.
    Batch size is capped at ``min(cfg.data.batch_size, 2)`` to keep CPU
    memory bounded.

    When ``$MODERNBERT_G2P_TEST_TINY == "1"`` the real HF-encoder + Parquet
    pipeline is replaced by a random-init stub so CI can verify the
    Trainer loop without hitting the HuggingFace Hub or requiring a real
    training corpus on disk.
    """
    from modernbert_g2p.training.trainer import Trainer

    if _is_tiny_mode():
        return _build_tiny_smoke_trainer(cfg, seed=seed, output_dir=Path(output_dir))

    _pilot_cfg, _tokenizer, collator, model = _dispatch_pipeline(cfg)

    train_dataset = _build_dataset(cfg.data.train_path)
    val_dataset = _build_dataset(cfg.data.val_path)

    args = _training_args_from_cfg(
        cfg,
        seed=seed,
        output_dir=Path(output_dir),
        total_steps=1,
        warmup_steps=0,
        batch_size=min(cfg.data.batch_size, 2),
        eval_every=0,
        save_every=0,
    )
    return Trainer(model, train_dataset, val_dataset, collator, args, device="cpu")


def run_training(
    cfg: Phase2Config,
    *,
    seed: int,
    output_dir: Path,
) -> dict[str, float]:
    """Assemble the full pipeline, run :meth:`Trainer.run`, return scalar metrics.

    The returned dict is derived from :meth:`Trainer.run`'s history: it
    always contains ``final_train_loss`` (the last observed train loss)
    and, when the eval loop has run at least once, ``final_val_loss``.
    """
    from modernbert_g2p.training.trainer import Trainer

    _pilot_cfg, _tokenizer, collator, model = _dispatch_pipeline(cfg)

    train_dataset = _build_dataset(cfg.data.train_path)
    val_dataset = _build_dataset(cfg.data.val_path)

    args = _training_args_from_cfg(
        cfg,
        seed=seed,
        output_dir=Path(output_dir),
        eval_every=cfg.training.total_steps // 4 or 1,
        save_every=cfg.training.total_steps // 2 or 1,
    )
    trainer = Trainer(model, train_dataset, val_dataset, collator, args, device="cpu")
    history = trainer.run()

    metrics: dict[str, float] = {}
    train_losses = history.get("train_loss") or []
    if train_losses:
        metrics["final_train_loss"] = float(train_losses[-1])
    val_losses = history.get("val_loss") or []
    if val_losses:
        metrics["final_val_loss"] = float(val_losses[-1])
    return metrics


__all__ = [
    "build_smoke_pipeline",
    "run_training",
]
