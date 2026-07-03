"""Optimizer + scheduler builders for Phase 2 pilots.

Two param groups by name prefix:
- Encoder (``encoder.*``): LR ``encoder_lr``, weight decay applied to
  linear/embedding weights, skipped for norm/bias/embedding-1D tensors.
- Head / decoder / other: LR ``head_lr``, same WD filtering.

Scheduler: linear warmup to peak LR followed by linear decay to 0 at
``total_steps``. Returns ``torch.optim.lr_scheduler.LambdaLR``.

`torch` is imported lazily inside each builder so importing this module has
no top-level torch dependency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


_NO_DECAY_SUFFIXES: tuple[str, ...] = ("bias", "LayerNorm.weight", "layer_norm.weight")
_NO_DECAY_KEYWORDS: tuple[str, ...] = ("norm.weight", "embeddings.weight")


def _is_no_decay(name: str, param_dim: int) -> bool:
    if param_dim <= 1:
        return True
    if any(name.endswith(suffix) for suffix in _NO_DECAY_SUFFIXES):
        return True
    return any(kw in name for kw in _NO_DECAY_KEYWORDS)


def build_optimizer(
    model: Any,
    *,
    encoder_lr: float,
    head_lr: float,
    weight_decay: float = 0.01,
) -> Any:
    """Build an ``AdamW`` optimizer with encoder/head param groups.

    Params whose name begins with ``"encoder."`` are placed in the encoder
    group at ``encoder_lr``; everything else goes into the head group at
    ``head_lr``. Bias / norm / embedding weights have weight-decay disabled.
    """
    import torch

    encoder_decay: list[torch.nn.Parameter] = []
    encoder_no_decay: list[torch.nn.Parameter] = []
    head_decay: list[torch.nn.Parameter] = []
    head_no_decay: list[torch.nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_encoder = name.startswith("encoder.")
        if _is_no_decay(name, param.dim()):
            (encoder_no_decay if is_encoder else head_no_decay).append(param)
        else:
            (encoder_decay if is_encoder else head_decay).append(param)

    param_groups: list[dict[str, object]] = []
    if encoder_decay:
        param_groups.append(
            {"params": encoder_decay, "lr": encoder_lr, "weight_decay": weight_decay}
        )
    if encoder_no_decay:
        param_groups.append(
            {"params": encoder_no_decay, "lr": encoder_lr, "weight_decay": 0.0}
        )
    if head_decay:
        param_groups.append(
            {"params": head_decay, "lr": head_lr, "weight_decay": weight_decay}
        )
    if head_no_decay:
        param_groups.append(
            {"params": head_no_decay, "lr": head_lr, "weight_decay": 0.0}
        )

    if not param_groups:
        raise ValueError("build_optimizer: model has no trainable parameters")

    return torch.optim.AdamW(param_groups)


def build_scheduler(
    optimizer: Any,
    *,
    warmup_steps: int,
    total_steps: int,
) -> Any:
    """Linear warmup to peak LR then linear decay to 0 at ``total_steps``."""
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
    if total_steps <= 0:
        raise ValueError(f"total_steps must be > 0, got {total_steps}")
    if warmup_steps >= total_steps:
        raise ValueError(
            f"warmup_steps ({warmup_steps}) must be < total_steps ({total_steps})"
        )

    import torch

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            if warmup_steps == 0:
                return 1.0
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 1.0 - progress)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


__all__ = ["build_optimizer", "build_scheduler"]
