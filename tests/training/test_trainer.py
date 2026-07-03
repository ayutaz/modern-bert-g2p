"""Tests for Phase 2 trainer, loss, and optimizer builders (Track T7)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from modernbert_g2p.training.loss import APBPBIOLoss, LabelSmoothingCELoss  # noqa: E402
from modernbert_g2p.training.optim import build_optimizer, build_scheduler  # noqa: E402
from modernbert_g2p.training.trainer import Trainer, TrainingArgs  # noqa: E402

# ---------- LabelSmoothingCELoss ----------


def test_label_smoothing_ce_finite_scalar():
    loss_fn = LabelSmoothingCELoss(smoothing=0.1, ignore_index=-100)
    logits = torch.randn(2, 4, 8, requires_grad=True)
    labels = torch.randint(0, 8, (2, 4))
    out = loss_fn(logits, labels)
    assert out.dim() == 0
    assert torch.isfinite(out).item()
    out.backward()
    assert logits.grad is not None


def test_label_smoothing_ce_ignores_padding():
    torch.manual_seed(0)
    loss_fn = LabelSmoothingCELoss(smoothing=0.0, ignore_index=-100)
    logits = torch.randn(2, 4, 8)
    labels = torch.tensor([[1, 2, -100, -100], [0, -100, -100, -100]])

    out_masked = loss_fn(logits, labels).item()

    log_probs = torch.log_softmax(logits, dim=-1)
    manual = -log_probs[0, 0, 1].item() - log_probs[0, 1, 2].item() - log_probs[1, 0, 0].item()
    manual /= 3.0
    assert abs(out_masked - manual) < 1e-5


def test_label_smoothing_ce_sample_weight_reweights_rows():
    torch.manual_seed(42)
    loss_fn = LabelSmoothingCELoss(smoothing=0.0, ignore_index=-100)
    logits = torch.randn(2, 4, 8)
    labels = torch.randint(0, 8, (2, 4))

    log_probs = torch.log_softmax(logits, dim=-1)
    row_losses = []
    for b in range(2):
        row = 0.0
        for t in range(4):
            row -= log_probs[b, t, labels[b, t]].item()
        row_losses.append(row / 4.0)

    weights_uniform = torch.ones(2)
    out_uniform = loss_fn(logits, labels, sample_weight=weights_uniform).item()
    expected_uniform = (row_losses[0] + row_losses[1]) / 2.0
    assert abs(out_uniform - expected_uniform) < 1e-5

    weights_biased = torch.tensor([2.0, 0.5])
    out_biased = loss_fn(logits, labels, sample_weight=weights_biased).item()
    expected_biased = (2.0 * row_losses[0] + 0.5 * row_losses[1]) / 2.0
    assert abs(out_biased - expected_biased) < 1e-5


def test_label_smoothing_ce_matches_formula_for_confident_logits():
    logits = torch.zeros(1, 1, 4)
    logits[0, 0, 0] = 10.0
    labels = torch.zeros(1, 1, dtype=torch.long)
    ce = LabelSmoothingCELoss(smoothing=0.0)(logits, labels).item()
    smoothed = LabelSmoothingCELoss(smoothing=0.2)(logits, labels).item()
    log_probs = torch.log_softmax(logits, dim=-1)
    nll = -log_probs[0, 0, 0].item()
    smooth = -log_probs[0, 0].mean().item()
    expected = 0.8 * nll + 0.2 * smooth
    assert ce == pytest.approx(nll, rel=1e-5)
    assert smoothed == pytest.approx(expected, rel=1e-5)
    assert smoothed > ce


def test_label_smoothing_ce_invalid_smoothing_raises():
    with pytest.raises(ValueError, match="smoothing"):
        LabelSmoothingCELoss(smoothing=1.0)


# ---------- APBPBIOLoss ----------


def test_apbp_bio_loss_finite_and_scales_with_alpha():
    torch.manual_seed(0)
    logits = torch.randn(2, 6, 3, requires_grad=True)
    labels = torch.randint(0, 3, (2, 6))

    base = APBPBIOLoss(alpha=1.0)(logits, labels).item()
    scaled = APBPBIOLoss(alpha=0.2)(logits, labels).item()
    assert torch.isfinite(torch.tensor(base)).item()
    assert abs(scaled - 0.2 * base) < 1e-6

    loss_fn = APBPBIOLoss(alpha=0.5)
    out = loss_fn(logits, labels)
    out.backward()
    assert logits.grad is not None


def test_apbp_bio_loss_rejects_wrong_class_dim():
    logits = torch.randn(2, 4, 5)
    labels = torch.zeros(2, 4, dtype=torch.long)
    with pytest.raises(ValueError, match="last dim=3"):
        APBPBIOLoss(alpha=1.0)(logits, labels)


# ---------- build_optimizer / build_scheduler ----------


class _TwoGroupModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(4, 8),
            torch.nn.LayerNorm(8),
        )
        self.head = torch.nn.Linear(8, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


def test_build_optimizer_creates_encoder_and_head_groups():
    model = _TwoGroupModel()
    opt = build_optimizer(model, encoder_lr=3e-5, head_lr=1e-4, weight_decay=0.01)
    assert isinstance(opt, torch.optim.AdamW)

    lrs = {round(g["lr"], 8) for g in opt.param_groups}
    assert 3e-5 in lrs
    assert 1e-4 in lrs

    for group in opt.param_groups:
        for param in group["params"]:
            if param.dim() <= 1 and group["weight_decay"] != 0.0:
                raise AssertionError("bias/norm param should have wd=0")


def test_build_optimizer_rejects_empty_model():
    model = torch.nn.Linear(1, 1)
    for p in model.parameters():
        p.requires_grad_(False)
    with pytest.raises(ValueError, match="no trainable parameters"):
        build_optimizer(model, encoder_lr=1e-4, head_lr=1e-4)


def test_build_scheduler_warmup_then_decay():
    model = _TwoGroupModel()
    opt = build_optimizer(model, encoder_lr=1e-3, head_lr=1e-3)
    sched = build_scheduler(opt, warmup_steps=10, total_steps=100)

    initial_lr = opt.param_groups[0]["lr"]
    assert initial_lr == pytest.approx(0.0, abs=1e-9)

    peak_lr = None
    for _ in range(10):
        opt.step()
        sched.step()
    peak_lr = opt.param_groups[0]["lr"]
    assert peak_lr == pytest.approx(1e-3, rel=1e-5)

    for _ in range(90):
        opt.step()
        sched.step()
    tail_lr = opt.param_groups[0]["lr"]
    assert tail_lr == pytest.approx(0.0, abs=1e-9)


def test_build_scheduler_rejects_bad_args():
    model = _TwoGroupModel()
    opt = build_optimizer(model, encoder_lr=1e-3, head_lr=1e-3)
    with pytest.raises(ValueError):
        build_scheduler(opt, warmup_steps=100, total_steps=50)


# ---------- Trainer ----------


class _TinyG2PModel(torch.nn.Module):
    """Two-linear stack that mimics the encoder/head split for optim tests."""

    def __init__(self, in_dim: int = 4, hidden: int = 8, vocab: int = 5) -> None:
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.GELU(),
            torch.nn.LayerNorm(hidden),
        )
        self.head = torch.nn.Linear(hidden, vocab)
        self._loss = LabelSmoothingCELoss(smoothing=0.05, ignore_index=-100)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        h = self.encoder(input_ids)
        logits = self.head(h)
        loss = self._loss(logits, labels, sample_weight=sample_weights)
        return {"loss": loss, "logits": logits}


class _TinyDataset:
    def __init__(self, n: int = 4, in_dim: int = 4, seq_len: int = 6, vocab: int = 5) -> None:
        torch.manual_seed(123)
        self._items = [
            {
                "input_ids": torch.randn(seq_len, in_dim),
                "labels": torch.randint(0, vocab, (seq_len,)),
                "sample_weights": torch.tensor(1.0),
            }
            for _ in range(n)
        ]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self._items[idx]


def _collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "sample_weights": torch.stack([b["sample_weights"] for b in batch]),
    }


def _make_trainer(tmp_path: Path, *, model_seed: int = 0, **overrides: Any) -> Trainer:
    torch.manual_seed(model_seed)
    model = _TinyG2PModel()
    train_ds = _TinyDataset(n=4)
    val_ds = _TinyDataset(n=2)
    args_kwargs: dict[str, Any] = {
        "output_dir": tmp_path,
        "total_steps": 4,
        "warmup_steps": 1,
        "encoder_lr": 5e-4,
        "head_lr": 1e-3,
        "batch_size": 2,
        "eval_every": 0,
        "save_every": 0,
        "grad_clip": 1.0,
        "seed": 0,
    }
    args_kwargs.update(overrides)
    args = TrainingArgs(**args_kwargs)
    return Trainer(model, train_ds, val_ds, _collate, args, device="cpu")


def test_trainer_smoke_one_step_produces_finite_loss(tmp_path: Path):
    trainer = _make_trainer(tmp_path, total_steps=2, warmup_steps=1)
    loader = iter(trainer._train_loader)
    batch = next(loader)
    metrics = trainer.train_step(batch)
    assert torch.isfinite(torch.tensor(metrics["loss"])).item()
    assert metrics["did_step"] == 1.0
    assert trainer.step == 1


def test_trainer_run_decreases_or_finite_loss(tmp_path: Path):
    trainer = _make_trainer(tmp_path, total_steps=6, warmup_steps=1)
    history = trainer.run()
    losses = history["train_loss"]
    assert len(losses) >= 6
    assert all(torch.isfinite(torch.tensor(v)).item() for v in losses)
    assert trainer.step == 6


def test_trainer_grad_accumulation_delays_step(tmp_path: Path):
    trainer = _make_trainer(tmp_path, total_steps=2, accumulate_grad_batches=2)
    loader = iter(trainer._train_loader)
    m1 = trainer.train_step(next(loader))
    assert m1["did_step"] == 0.0
    assert trainer.step == 0
    m2 = trainer.train_step(next(loader))
    assert m2["did_step"] == 1.0
    assert trainer.step == 1


def test_trainer_eval_loop_returns_finite(tmp_path: Path):
    trainer = _make_trainer(tmp_path)
    metrics = trainer.eval_loop()
    assert "val_loss" in metrics
    assert torch.isfinite(torch.tensor(metrics["val_loss"])).item()


def test_trainer_checkpoint_roundtrip(tmp_path: Path):
    trainer = _make_trainer(tmp_path, total_steps=3, warmup_steps=1)
    loader = iter(trainer._train_loader)
    for _ in range(2):
        trainer.train_step(next(loader))

    ckpt = tmp_path / "ckpt.pt"
    trainer.save_checkpoint(trainer.step, ckpt)
    assert ckpt.exists()

    fresh = _make_trainer(tmp_path, total_steps=3, warmup_steps=1, seed=99)
    fresh.load_checkpoint(ckpt)
    assert fresh.step == trainer.step

    for (n1, p1), (n2, p2) in zip(
        trainer.model.state_dict().items(),
        fresh.model.state_dict().items(),
        strict=True,
    ):
        assert n1 == n2
        assert torch.allclose(p1, p2, atol=1e-6)

    for g1, g2 in zip(
        trainer.optimizer.state_dict()["param_groups"],
        fresh.optimizer.state_dict()["param_groups"],
        strict=True,
    ):
        assert g1["lr"] == pytest.approx(g2["lr"])
        assert g1["weight_decay"] == pytest.approx(g2["weight_decay"])


def test_trainer_seed_determinism(tmp_path: Path):
    trainer_a = _make_trainer(
        tmp_path / "a", model_seed=13, total_steps=3, warmup_steps=1, seed=7
    )
    loader_a = iter(trainer_a._train_loader)
    m1 = trainer_a.train_step(next(loader_a))

    trainer_b = _make_trainer(
        tmp_path / "b", model_seed=13, total_steps=3, warmup_steps=1, seed=7
    )
    loader_b = iter(trainer_b._train_loader)
    m2 = trainer_b.train_step(next(loader_b))

    assert m1["loss"] == pytest.approx(m2["loss"], rel=1e-6)


def test_trainer_lazy_imports_at_module_load():
    import importlib

    loss_mod = importlib.import_module("modernbert_g2p.training.loss")
    optim_mod = importlib.import_module("modernbert_g2p.training.optim")
    trainer_mod = importlib.import_module("modernbert_g2p.training.trainer")
    assert loss_mod is not None
    assert optim_mod is not None
    assert trainer_mod is not None
    assert hasattr(loss_mod, "LabelSmoothingCELoss")
    assert hasattr(loss_mod, "APBPBIOLoss")


def test_trainer_run_steps_override_stops_after_n_steps(tmp_path: Path):
    trainer = _make_trainer(tmp_path, total_steps=100, warmup_steps=1)
    history = trainer.run(steps=2)
    assert trainer.step == 2
    assert len(history["train_loss"]) == 2
    assert torch.isfinite(torch.tensor(history["loss"])).item()


def test_trainer_run_returns_loss_scalar_for_cli_smoke(tmp_path: Path):
    trainer = _make_trainer(tmp_path, total_steps=1, warmup_steps=0)
    history = trainer.run(steps=1)
    assert "loss" in history
    assert isinstance(history["loss"], float)
    assert torch.isfinite(torch.tensor(history["loss"])).item()


def test_trainer_run_forward_ignores_ids_key(tmp_path: Path):
    """The 'ids' key emitted by real collators is stripped before model forward."""

    class _NoIdsModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Linear(4, 8)
            self.head = torch.nn.Linear(8, 5)
            self._loss = LabelSmoothingCELoss(smoothing=0.0, ignore_index=-100)

        def forward(
            self,
            input_ids: torch.Tensor,
            labels: torch.Tensor,
            sample_weights: torch.Tensor | None = None,
        ) -> dict[str, torch.Tensor]:
            logits = self.head(self.encoder(input_ids))
            loss = self._loss(logits, labels, sample_weight=sample_weights)
            return {"loss": loss}

    def _collate_with_ids(batch: list[dict[str, torch.Tensor]]) -> dict[str, Any]:
        return {
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "labels": torch.stack([b["labels"] for b in batch]),
            "sample_weights": torch.stack([b["sample_weights"] for b in batch]),
            "ids": ["row-a", "row-b"],
        }

    torch.manual_seed(0)
    model = _NoIdsModel()
    train_ds = _TinyDataset(n=4)
    val_ds = _TinyDataset(n=2)
    args = TrainingArgs(
        output_dir=tmp_path,
        total_steps=2,
        warmup_steps=1,
        encoder_lr=1e-3,
        head_lr=1e-3,
        batch_size=2,
        eval_every=0,
        save_every=0,
    )
    trainer = Trainer(model, train_ds, val_ds, _collate_with_ids, args, device="cpu")
    loader = iter(trainer._train_loader)
    metrics = trainer.train_step(next(loader))
    assert torch.isfinite(torch.tensor(metrics["loss"])).item()


# ---------- build_smoke_pipeline / run_training factories ----------


def _fake_phase2_config(pilot: str, tmp_path: Path) -> Any:
    """Minimal Phase2Config-like object; dispatch tests only care about pilot + paths."""
    from types import SimpleNamespace

    data = SimpleNamespace(
        train_path=str(tmp_path / "train.parquet"),
        val_path=str(tmp_path / "val.parquet"),
        batch_size=2,
        num_workers=0,
    )
    training = SimpleNamespace(
        encoder_lr=1e-4,
        head_lr=1e-3,
        warmup_steps=0,
        total_steps=1,
        weight_decay=0.01,
        grad_clip=1.0,
        precision="fp32",
        accumulate_grad_batches=1,
    )
    return SimpleNamespace(
        pilot=pilot,
        model={},
        data=data,
        training=training,
    )


def test_factory_normalizes_pilot_alias() -> None:
    from modernbert_g2p.training.factory import _normalize_pilot

    assert _normalize_pilot("P-A") == "P-A"
    assert _normalize_pilot("p_a") == "P-A"
    assert _normalize_pilot("p_b") == "P-B"
    assert _normalize_pilot("p_c") == "P-C"
    with pytest.raises(ValueError, match="Unknown pilot"):
        _normalize_pilot("P-D")


def test_factory_training_args_from_cfg(tmp_path: Path) -> None:
    from modernbert_g2p.training.factory import _training_args_from_cfg

    cfg = _fake_phase2_config("P-A", tmp_path)
    cfg.training.total_steps = 100
    cfg.training.warmup_steps = 10
    args = _training_args_from_cfg(cfg, seed=42, output_dir=tmp_path)
    assert args.total_steps == 100
    assert args.warmup_steps == 10
    assert args.seed == 42
    assert args.output_dir == tmp_path

    smoke_args = _training_args_from_cfg(
        cfg, seed=1, output_dir=tmp_path, total_steps=1, warmup_steps=0
    )
    assert smoke_args.total_steps == 1
    assert smoke_args.warmup_steps == 0


def test_factory_instantiate_config_drops_unknown_keys(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from modernbert_g2p.models.p_a import PAConfig
    from modernbert_g2p.training.factory import _instantiate_config

    raw = {
        "encoder_name": "tiny",
        "phoneme_vocab_size": 16,
        "hl_vocab_size": 3,
        "some_unknown_key": "ignored",
    }
    pa_cfg = _instantiate_config(PAConfig, raw)
    assert pa_cfg.encoder_name == "tiny"
    assert pa_cfg.phoneme_vocab_size == 16
    err = capsys.readouterr().err
    assert "hl_vocab_size" in err or "some_unknown_key" in err


def test_factory_build_smoke_pipeline_dispatches_per_pilot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from modernbert_g2p.training import factory as _factory
    from modernbert_g2p.training.trainer import Trainer, TrainingArgs

    class _StubModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Linear(1, 1)

        def forward(self, **kwargs: Any) -> dict[str, torch.Tensor]:
            return {"loss": torch.tensor(0.0, requires_grad=True) + self.encoder.weight.sum()}

    def _fake_dispatch(cfg: Any) -> tuple[Any, Any, Any, Any]:
        return (object(), object(), lambda batch: batch, _StubModel())

    def _fake_dataset(path: str | Path) -> Any:
        class _DS:
            def __len__(self) -> int:
                return 1

            def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
                return {"input_ids": torch.zeros(1, 1), "labels": torch.zeros(1, dtype=torch.long)}

        return _DS()

    monkeypatch.setattr(_factory, "_dispatch_pipeline", _fake_dispatch)
    monkeypatch.setattr(_factory, "_build_dataset", _fake_dataset)

    cfg = _fake_phase2_config("P-A", tmp_path)
    trainer = _factory.build_smoke_pipeline(cfg, seed=1, output_dir=tmp_path)
    assert isinstance(trainer, Trainer)
    assert isinstance(trainer.args, TrainingArgs)
    assert trainer.args.total_steps == 1
    assert trainer.args.warmup_steps == 0
    assert trainer.args.batch_size <= 2
    assert trainer.args.seed == 1


def test_factory_build_smoke_pipeline_step_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from modernbert_g2p.training import factory as _factory

    class _StubModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))

        def forward(self, **kwargs: Any) -> dict[str, torch.Tensor]:
            input_ids = kwargs.get("input_ids", torch.zeros(1))
            loss = (self.w * input_ids.float().sum()) ** 2 + 1.0
            return {"loss": loss}

    def _batch_collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        return {"input_ids": torch.stack([b["input_ids"] for b in batch])}

    class _DS:
        def __init__(self, n: int) -> None:
            self._n = n

        def __len__(self) -> int:
            return self._n

        def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
            return {"input_ids": torch.tensor([1.0, 2.0])}

    monkeypatch.setattr(
        _factory,
        "_dispatch_pipeline",
        lambda cfg: (object(), object(), _batch_collate, _StubModel()),
    )
    monkeypatch.setattr(_factory, "_build_dataset", lambda p: _DS(4))

    cfg = _fake_phase2_config("P-A", tmp_path)
    trainer = _factory.build_smoke_pipeline(cfg, seed=0, output_dir=tmp_path)
    history = trainer.run(steps=1)
    assert history["loss"] == pytest.approx(1.0, rel=1e-5)
    assert trainer.step == 1


def test_factory_run_training_returns_final_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from modernbert_g2p.training import factory as _factory

    class _StubModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))

        def forward(self, **kwargs: Any) -> dict[str, torch.Tensor]:
            input_ids = kwargs.get("input_ids", torch.zeros(1))
            loss = (self.w * input_ids.float().sum()) ** 2 + 2.5
            return {"loss": loss}

    def _batch_collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        return {"input_ids": torch.stack([b["input_ids"] for b in batch])}

    class _DS:
        def __len__(self) -> int:
            return 2

        def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
            return {"input_ids": torch.tensor([1.0])}

    monkeypatch.setattr(
        _factory,
        "_dispatch_pipeline",
        lambda cfg: (object(), object(), _batch_collate, _StubModel()),
    )
    monkeypatch.setattr(_factory, "_build_dataset", lambda p: _DS())

    cfg = _fake_phase2_config("P-A", tmp_path)
    cfg.training.total_steps = 3
    cfg.training.warmup_steps = 1
    metrics = _factory.run_training(cfg, seed=7, output_dir=tmp_path)
    assert "final_train_loss" in metrics
    assert torch.isfinite(torch.tensor(metrics["final_train_loss"])).item()


def test_factory_run_training_rejects_unknown_pilot(tmp_path: Path) -> None:
    from modernbert_g2p.training import factory as _factory

    cfg = _fake_phase2_config("P-X", tmp_path)
    with pytest.raises(ValueError, match="Unknown pilot"):
        _factory.run_training(cfg, seed=1, output_dir=tmp_path)


def test_training_public_api_exports_factories() -> None:
    from modernbert_g2p import training as training_mod

    assert "build_smoke_pipeline" in training_mod.__all__
    assert "run_training" in training_mod.__all__
    assert callable(training_mod.build_smoke_pipeline)
    assert callable(training_mod.run_training)


def test_factory_tiny_mode_bypasses_real_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MODERNBERT_G2P_TEST_TINY=1 → synthetic Trainer runs 1 step, no HF/dataset."""
    from modernbert_g2p.training import factory as _factory
    from modernbert_g2p.training.trainer import Trainer

    monkeypatch.setenv(_factory.TINY_MODE_ENV_VAR, "1")

    called: list[str] = []

    def _guard_dispatch(cfg: Any) -> Any:
        called.append("dispatch")
        raise AssertionError("_dispatch_pipeline must not be called in tiny mode")

    def _guard_dataset(path: Any) -> Any:
        called.append("dataset")
        raise AssertionError("_build_dataset must not be called in tiny mode")

    monkeypatch.setattr(_factory, "_dispatch_pipeline", _guard_dispatch)
    monkeypatch.setattr(_factory, "_build_dataset", _guard_dataset)

    cfg = _fake_phase2_config("P-A", tmp_path)
    trainer = _factory.build_smoke_pipeline(cfg, seed=0, output_dir=tmp_path)
    assert isinstance(trainer, Trainer)
    assert called == []
    history = trainer.run(steps=1)
    assert trainer.step == 1
    assert torch.isfinite(torch.tensor(history["loss"])).item()
