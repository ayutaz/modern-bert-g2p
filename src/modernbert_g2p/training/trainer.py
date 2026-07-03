"""Phase 2 trainer: single-device loop with grad accumulation + checkpointing.

Implements a minimal ``Trainer`` per `docs/design/phase2_tokenizer_pilots.md`.
Precision defaults to fp32 (CPU / tests); ``bf16`` and ``fp16`` are honoured
only when CUDA is available. All ``torch`` imports live inside functions.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch  # noqa: F401


@dataclass
class TrainingArgs:
    """Training hyperparameters + IO paths for :class:`Trainer`.

    Field defaults match a CPU smoke run. Production configs override via
    :mod:`modernbert_g2p.config` YAML loading in Track T9.
    """

    output_dir: Path
    total_steps: int
    warmup_steps: int
    encoder_lr: float
    head_lr: float
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    eval_every: int = 2000
    save_every: int = 2000
    batch_size: int = 8
    accumulate_grad_batches: int = 1
    precision: str = "fp32"
    seed: int = 42
    num_workers: int = 0
    log_every: int = 50


BatchLike = dict[str, Any]
Collator = Callable[[list[Any]], BatchLike]


class Trainer:
    """Minimal single-device trainer.

    Responsibilities:
    - Set ``torch.manual_seed`` from ``args.seed``.
    - Build optimizer + linear-warmup / linear-decay scheduler.
    - Iterate up to ``args.total_steps`` with gradient accumulation.
    - Periodically run ``eval_step`` on ``val_dataset``.
    - Save / load full training state as a single ``torch.save`` dict.
    """

    def __init__(
        self,
        model: Any,
        train_dataset: Any,
        val_dataset: Any,
        collator: Collator,
        args: TrainingArgs,
        *,
        device: str = "cpu",
    ) -> None:
        import torch

        from modernbert_g2p.training.optim import build_optimizer, build_scheduler

        self.args = args
        self.device = torch.device(device)
        torch.manual_seed(args.seed)

        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.collator = collator

        self.optimizer = build_optimizer(
            self.model,
            encoder_lr=args.encoder_lr,
            head_lr=args.head_lr,
            weight_decay=args.weight_decay,
        )
        self.scheduler = build_scheduler(
            self.optimizer,
            warmup_steps=args.warmup_steps,
            total_steps=args.total_steps,
        )

        self._train_loader = self._build_loader(train_dataset, shuffle=True)
        self._val_loader = self._build_loader(val_dataset, shuffle=False)

        self.step: int = 0
        self._precision_dtype = self._resolve_precision(args.precision)
        self._accum_batches = 0

    def _resolve_precision(self, precision: str) -> Any:
        import torch

        precision = precision.lower()
        if precision == "fp32":
            return torch.float32
        if precision == "bf16":
            return torch.bfloat16
        if precision == "fp16":
            return torch.float16
        raise ValueError(f"Unsupported precision {precision!r}")

    def _build_loader(self, dataset: Any, *, shuffle: bool) -> Any:
        import torch
        from torch.utils.data import DataLoader

        return DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=shuffle,
            num_workers=self.args.num_workers,
            collate_fn=self.collator,
            generator=torch.Generator().manual_seed(self.args.seed) if shuffle else None,
        )

    def _to_device(self, batch: BatchLike) -> BatchLike:
        import torch

        moved: BatchLike = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(self.device, non_blocking=False)
            else:
                moved[key] = value
        return moved

    def _autocast_context(self) -> Any:
        import contextlib

        import torch

        if self._precision_dtype == torch.float32:
            return contextlib.nullcontext()
        if self.device.type != "cuda":
            return contextlib.nullcontext()
        return torch.autocast(device_type="cuda", dtype=self._precision_dtype)

    def _run_forward(self, batch: BatchLike) -> Any:
        import torch

        kwargs = {k: v for k, v in batch.items() if k != "ids"}
        with self._autocast_context():
            outputs = self.model(**kwargs)
        if isinstance(outputs, dict):
            return outputs
        if isinstance(outputs, torch.Tensor):
            return {"loss": outputs, "logits": None}
        raise TypeError(f"model forward returned unsupported type {type(outputs).__name__}")

    def train_step(self, batch: BatchLike) -> dict[str, float]:
        """Perform one optimizer update from ``batch`` (respecting grad accumulation)."""
        import torch

        self.model.train()
        batch = self._to_device(batch)
        outputs = self._run_forward(batch)
        loss = outputs.get("loss")
        if loss is None:
            raise ValueError("model forward did not return 'loss'")

        accum = max(1, self.args.accumulate_grad_batches)
        (loss / accum).backward()
        self._accum_batches += 1
        did_step = False

        if self._accum_batches >= accum:
            if self.args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self._accum_batches = 0
            self.step += 1
            did_step = True

        return {
            "loss": float(loss.detach().cpu().item()),
            "lr": float(self.optimizer.param_groups[0]["lr"]),
            "step": float(self.step),
            "did_step": 1.0 if did_step else 0.0,
        }

    def eval_step(self, batch: BatchLike) -> dict[str, float]:
        """Forward-only evaluation of a single batch."""
        import torch

        self.model.eval()
        batch = self._to_device(batch)
        with torch.no_grad():
            outputs = self._run_forward(batch)
        loss = outputs.get("loss")
        if loss is None:
            return {"loss": float("nan")}
        return {"loss": float(loss.detach().cpu().item())}

    def eval_loop(self) -> dict[str, float]:
        """Evaluate over the full validation loader; return mean loss."""
        losses: list[float] = []
        for batch in self._val_loader:
            metrics = self.eval_step(batch)
            losses.append(metrics["loss"])
        if not losses:
            return {"val_loss": float("nan")}
        return {"val_loss": sum(losses) / len(losses)}

    def run(self, steps: int | None = None) -> dict[str, Any]:
        """Iterate up to ``steps`` (or ``args.total_steps`` when omitted).

        Returns a dict with per-step ``train_loss`` and ``val_loss`` traces
        under those keys, plus ``loss`` — the final observed train loss —
        so callers wanting a single scalar (e.g. the CLI smoke path) can
        read ``result["loss"]`` without inspecting the trace lengths.
        """
        target_step = self.args.total_steps if steps is None else self.step + int(steps)
        history: dict[str, Any] = {"train_loss": [], "val_loss": []}
        if target_step <= self.step:
            history["loss"] = float("nan")
            return history

        train_iter = _cycle(self._train_loader)
        last_loss = float("nan")
        while self.step < target_step:
            batch = next(train_iter)
            metrics = self.train_step(batch)
            history["train_loss"].append(metrics["loss"])
            last_loss = metrics["loss"]

            if metrics["did_step"] and self.step > 0:
                if self.args.eval_every > 0 and self.step % self.args.eval_every == 0:
                    eval_metrics = self.eval_loop()
                    history["val_loss"].append(eval_metrics["val_loss"])
                if self.args.save_every > 0 and self.step % self.args.save_every == 0:
                    self.save_checkpoint(
                        self.step,
                        Path(self.args.output_dir) / f"checkpoint_step_{self.step}.pt",
                    )
        history["loss"] = last_loss
        return history

    def save_checkpoint(self, step: int, path: Path) -> None:
        """Persist model + optimizer + scheduler + step to ``path``."""
        import torch

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "step": int(step),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "args": _training_args_to_dict(self.args),
        }
        torch.save(payload, str(path))

    def load_checkpoint(self, path: Path) -> None:
        """Restore model + optimizer + scheduler + step from ``path``."""
        import torch

        payload = torch.load(str(path), map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model_state_dict"])
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.scheduler.load_state_dict(payload["scheduler_state_dict"])
        self.step = int(payload.get("step", 0))


def _cycle(iterable: Iterable[Any]) -> Iterator[Any]:
    while True:
        yield from iterable


def _training_args_to_dict(args: TrainingArgs) -> dict[str, object]:
    return {
        "output_dir": str(args.output_dir),
        "total_steps": args.total_steps,
        "warmup_steps": args.warmup_steps,
        "encoder_lr": args.encoder_lr,
        "head_lr": args.head_lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "eval_every": args.eval_every,
        "save_every": args.save_every,
        "batch_size": args.batch_size,
        "accumulate_grad_batches": args.accumulate_grad_batches,
        "precision": args.precision,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "log_every": args.log_every,
    }


__all__ = ["Trainer", "TrainingArgs"]
