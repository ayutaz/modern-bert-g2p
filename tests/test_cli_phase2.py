"""CLI-level smoke tests for the Phase 2 top-level ``python -m modernbert_g2p``."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from modernbert_g2p.cli import main, make_parser

REPO_ROOT = Path(__file__).resolve().parent.parent
SMOKE_SCRIPT = REPO_ROOT / "scripts" / "phase2_smoke.sh"


TINY_P_A_YAML = """\
pilot: P-A
model:
  encoder_name: sbintuitions/modernbert-ja-130m
  encoder_hidden_size: 64
  encoder_num_hidden_layers: 2
  encoder_num_attention_heads: 2
  encoder_intermediate_size: 128
  decoder_layers: 1
  decoder_hidden: 32
  decoder_heads: 2
  decoder_ffn: 64
  decoder_dropout: 0.1
  phoneme_vocab_size: 68
  label_smoothing: 0.1
  max_decode_len: 32
  beam_size: 1
data:
  train_path: {train_path}
  val_path: {val_path}
  batch_size: 2
  num_workers: 0
training:
  encoder_lr: 1.0e-4
  head_lr: 1.0e-3
  warmup_steps: 0
  total_steps: 1
  weight_decay: 0.0
  grad_clip: 1.0
  precision: fp32
  seed: 20260704
  accumulate_grad_batches: 1
output_dir: {output_dir}
"""


def _has_training_hook(name: str) -> bool:
    try:
        mod = importlib.import_module("modernbert_g2p.training")
    except ImportError:
        return False
    return hasattr(mod, name)


def _has_evaluation_hook(name: str) -> bool:
    try:
        mod = importlib.import_module("modernbert_g2p.evaluation")
    except ImportError:
        return False
    return hasattr(mod, name)


def test_make_parser_has_three_subcommands() -> None:
    parser = make_parser()
    subactions = [a for a in parser._actions if a.dest == "command"]
    assert len(subactions) == 1
    choices = set(subactions[0].choices or {})
    assert {"train", "eval", "compare"} <= choices


def test_main_top_level_help_exit_0(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["--help"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "train" in out
    assert "eval" in out
    assert "compare" in out


def test_main_train_help_exit_0(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["train", "--help"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "--config" in out
    assert "--seed" in out
    assert "--smoke" in out
    assert "--output-dir" in out


def test_main_eval_help_exit_0(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["eval", "--help"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "--checkpoint" in out
    assert "--config" in out
    assert "--pilot" in out
    assert "--output" in out


def test_main_compare_help_exit_0(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["compare", "--help"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "--checkpoint" in out
    assert "--output" in out


def test_main_unknown_subcommand_nonzero() -> None:
    rc = main(["nonexistent-cmd"])
    assert rc != 0


def test_main_no_args_nonzero() -> None:
    rc = main([])
    assert rc != 0


def test_main_train_missing_config_returns_2(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.yaml"
    rc = main(["train", "--config", str(missing)])
    assert rc == 2


def test_main_eval_missing_checkpoint_returns_2(tmp_path: Path) -> None:
    missing_ckpt = tmp_path / "no.ckpt"
    dummy_cfg = tmp_path / "cfg.yaml"
    dummy_cfg.write_text("pilot: p_a\n", encoding="utf-8")
    rc = main(
        [
            "eval",
            "--checkpoint",
            str(missing_ckpt),
            "--config",
            str(dummy_cfg),
            "--pilot",
            "P-A",
            "--output",
            str(tmp_path / "out.json"),
        ]
    )
    assert rc == 2


def test_main_compare_missing_checkpoint_returns_2(tmp_path: Path) -> None:
    missing = tmp_path / "missing_dir"
    rc = main(
        [
            "compare",
            "--checkpoint",
            str(missing),
            "--output",
            str(tmp_path / "table.md"),
        ]
    )
    assert rc == 2


def test_main_train_smoke_uses_build_smoke_pipeline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """--smoke calls build_smoke_pipeline and Trainer.run(steps=1)."""
    pytest.importorskip("torch")
    from modernbert_g2p import cli as cli_mod

    cfg_yaml = tmp_path / "fake.yaml"
    cfg_yaml.write_text("pilot: p_a\n", encoding="utf-8")

    class _FakeCfg:
        pilot = "p_a"
        output_dir = tmp_path / "out"

    class _FakeTrainer:
        def __init__(self) -> None:
            self.calls: list[int] = []

        def run(self, steps: int) -> dict[str, float]:
            self.calls.append(steps)
            return {"loss": 0.1234}

    fake_trainer = _FakeTrainer()

    fake_config_mod = type(sys)("modernbert_g2p.config")
    fake_config_mod.load_config = lambda path: _FakeCfg()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "modernbert_g2p.config", fake_config_mod)

    fake_training_mod = type(sys)("modernbert_g2p.training")
    fake_training_mod.build_smoke_pipeline = (  # type: ignore[attr-defined]
        lambda cfg, *, seed, output_dir: fake_trainer
    )
    monkeypatch.setitem(sys.modules, "modernbert_g2p.training", fake_training_mod)

    fake_trainer_mod = type(sys)("modernbert_g2p.training.trainer")
    fake_trainer_mod.Trainer = _FakeTrainer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "modernbert_g2p.training.trainer", fake_trainer_mod)

    rc = cli_mod.main(
        [
            "train",
            "--config",
            str(cfg_yaml),
            "--seed",
            "20260704",
            "--output-dir",
            str(tmp_path / "out"),
            "--smoke",
        ]
    )

    assert rc == 0
    assert fake_trainer.calls == [1]


def test_smoke_shell_script_exists_and_executable() -> None:
    assert SMOKE_SCRIPT.is_file()
    assert os.access(SMOKE_SCRIPT, os.X_OK)
    contents = SMOKE_SCRIPT.read_text(encoding="utf-8")
    assert "-m modernbert_g2p train" in contents
    assert "--smoke" in contents
    for pilot in ("p_a", "p_b", "p_c"):
        assert pilot in contents


def test_dash_m_help_via_subprocess() -> None:
    """``python -m modernbert_g2p --help`` exits 0 and never imports torch."""
    env = dict(os.environ)
    result = subprocess.run(
        [sys.executable, "-m", "modernbert_g2p", "--help"],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "train" in result.stdout
    assert "eval" in result.stdout
    assert "compare" in result.stdout


def test_data_subcommand_still_reachable_via_dash_m() -> None:
    """Phase 1 ``python -m modernbert_g2p.data`` remains functional."""
    result = subprocess.run(
        [sys.executable, "-m", "modernbert_g2p.data", "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "build" in result.stdout


def test_cli_module_does_not_import_torch_at_load() -> None:
    """Importing modernbert_g2p.cli must not import torch."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, modernbert_g2p.cli;"
                "print('torch' in sys.modules)"
            ),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_main_train_smoke_missing_t7_hook_returns_3(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Absent T7 build_smoke_pipeline, the CLI must exit 3 with a clear error."""
    pytest.importorskip("torch")
    from modernbert_g2p import cli as cli_mod

    cfg_yaml = tmp_path / "fake.yaml"
    cfg_yaml.write_text("pilot: p_a\n", encoding="utf-8")

    class _FakeCfg:
        pilot = "p_a"
        output_dir = tmp_path / "out"

    fake_config_mod = type(sys)("modernbert_g2p.config")
    fake_config_mod.load_config = lambda path: _FakeCfg()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "modernbert_g2p.config", fake_config_mod)

    fake_training_mod = type(sys)("modernbert_g2p.training")
    monkeypatch.setitem(sys.modules, "modernbert_g2p.training", fake_training_mod)

    fake_trainer_mod = type(sys)("modernbert_g2p.training.trainer")

    class _FakeTrainer:
        pass

    fake_trainer_mod.Trainer = _FakeTrainer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "modernbert_g2p.training.trainer", fake_trainer_mod)

    rc = cli_mod.main(
        [
            "train",
            "--config",
            str(cfg_yaml),
            "--output-dir",
            str(tmp_path / "out"),
            "--smoke",
        ]
    )
    assert rc == 3


def test_main_train_full_missing_t7_run_training_returns_3(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Absent T7 run_training hook, the full train path must exit 3."""
    from modernbert_g2p import cli as cli_mod

    cfg_yaml = tmp_path / "fake.yaml"
    cfg_yaml.write_text("pilot: p_a\n", encoding="utf-8")

    class _FakeCfg:
        pilot = "p_a"
        output_dir = tmp_path / "out"

    fake_config_mod = type(sys)("modernbert_g2p.config")
    fake_config_mod.load_config = lambda path: _FakeCfg()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "modernbert_g2p.config", fake_config_mod)

    fake_training_mod = type(sys)("modernbert_g2p.training")
    monkeypatch.setitem(sys.modules, "modernbert_g2p.training", fake_training_mod)

    rc = cli_mod.main(
        [
            "train",
            "--config",
            str(cfg_yaml),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 3


def test_main_eval_missing_t8_hook_returns_3(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Absent T8 evaluate_checkpoint, the CLI must exit 3."""
    from modernbert_g2p import cli as cli_mod

    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_bytes(b"stub")
    cfg_yaml = tmp_path / "fake.yaml"
    cfg_yaml.write_text("pilot: p_a\n", encoding="utf-8")

    fake_config_mod = type(sys)("modernbert_g2p.config")
    fake_config_mod.load_config = lambda path: object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "modernbert_g2p.config", fake_config_mod)

    fake_eval_mod = type(sys)("modernbert_g2p.evaluation")
    monkeypatch.setitem(sys.modules, "modernbert_g2p.evaluation", fake_eval_mod)

    rc = cli_mod.main(
        [
            "eval",
            "--checkpoint",
            str(ckpt),
            "--config",
            str(cfg_yaml),
            "--pilot",
            "P-A",
            "--output",
            str(tmp_path / "out.json"),
        ]
    )
    assert rc == 3


def test_main_compare_missing_t8_hook_returns_3(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Absent T8 build_comparison_table, the CLI must exit 3."""
    from modernbert_g2p import cli as cli_mod

    ckpt_dir = tmp_path / "ckpt_dir"
    ckpt_dir.mkdir()

    fake_eval_mod = type(sys)("modernbert_g2p.evaluation")
    monkeypatch.setitem(sys.modules, "modernbert_g2p.evaluation", fake_eval_mod)

    rc = cli_mod.main(
        [
            "compare",
            "--checkpoint",
            str(ckpt_dir),
            "--output",
            str(tmp_path / "table.md"),
        ]
    )
    assert rc == 3


@pytest.mark.skipif(
    os.environ.get("MODERNBERT_G2P_E2E_TESTS") != "1",
    reason=(
        "Full-stack E2E: requires network + real HF weights + real jsut-label data. "
        "Set MODERNBERT_G2P_E2E_TESTS=1 to run."
    ),
)
def test_main_train_smoke_end_to_end_subprocess(tmp_path: Path) -> None:
    """End-to-end subprocess: ``python -m modernbert_g2p train --smoke`` exits 0.

    Requested by review G2. Drives the real CLI -> config -> tokenizer ->
    collator -> model -> Trainer pipeline for P-A on CPU using a tiny-model
    YAML fixture. Gated behind ``MODERNBERT_G2P_E2E_TESTS=1`` because it
    requires HF weight availability which the sandbox CI cannot provide.
    """
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    if not _has_training_hook("build_smoke_pipeline"):
        pytest.skip(
            "Track 7 build_smoke_pipeline hook not yet exported from "
            "modernbert_g2p.training — test comes online once T7 lands B1."
        )

    train_stub = tmp_path / "train.parquet"
    val_stub = tmp_path / "val.parquet"
    tiny_cfg = tmp_path / "tiny_p_a.yaml"
    output_dir = tmp_path / "out"
    tiny_cfg.write_text(
        TINY_P_A_YAML.format(
            train_path=str(train_stub),
            val_path=str(val_stub),
            output_dir=str(output_dir),
        ),
        encoding="utf-8",
    )

    env = dict(os.environ)
    env.setdefault("MODERNBERT_G2P_TEST_TINY", "1")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "modernbert_g2p",
            "train",
            "--config",
            str(tiny_cfg),
            "--seed",
            "20260704",
            "--output-dir",
            str(output_dir),
            "--smoke",
        ],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"train --smoke subprocess exit={result.returncode}\n"
        f"stdout=\n{result.stdout}\nstderr=\n{result.stderr}"
    )
    assert "[smoke]" in result.stdout


@pytest.mark.skipif(
    os.environ.get("MODERNBERT_G2P_E2E_TESTS") != "1",
    reason=(
        "Full-stack E2E: requires network + real HF weights + real jsut-label data. "
        "Set MODERNBERT_G2P_E2E_TESTS=1 to run."
    ),
)
def test_main_eval_end_to_end_subprocess(tmp_path: Path) -> None:
    """End-to-end subprocess: ``python -m modernbert_g2p eval`` reaches T8 evaluator.

    Gated behind ``MODERNBERT_G2P_E2E_TESTS=1`` because the T8 evaluator
    demands real dataset paths (``cfg.data.jsut_yaml``) and a real checkpoint,
    neither of which the sandbox CI can provide.
    """
    pytest.importorskip("torch")
    if not _has_evaluation_hook("evaluate_checkpoint"):
        pytest.skip(
            "Track 8 evaluate_checkpoint hook not yet exported from "
            "modernbert_g2p.evaluation — test comes online once T8 lands B1."
        )

    train_stub = tmp_path / "train.parquet"
    val_stub = tmp_path / "val.parquet"
    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_bytes(b"stub")
    out_json = tmp_path / "eval_out.json"
    tiny_cfg = tmp_path / "tiny_p_a.yaml"
    tiny_cfg.write_text(
        TINY_P_A_YAML.format(
            train_path=str(train_stub),
            val_path=str(val_stub),
            output_dir=str(tmp_path / "out"),
        ),
        encoding="utf-8",
    )

    env = dict(os.environ)
    env.setdefault("MODERNBERT_G2P_TEST_TINY", "1")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "modernbert_g2p",
            "eval",
            "--checkpoint",
            str(ckpt),
            "--config",
            str(tiny_cfg),
            "--pilot",
            "P-A",
            "--dataset",
            "jsut",
            "--output",
            str(out_json),
        ],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"eval subprocess exit={result.returncode}\n"
        f"stdout=\n{result.stdout}\nstderr=\n{result.stderr}"
    )


def test_main_compare_end_to_end_subprocess(tmp_path: Path) -> None:
    """End-to-end subprocess: ``python -m modernbert_g2p compare`` writes Markdown.

    Skips until T8 lands ``build_comparison_table``. Once T8 lands, this test
    verifies that the CLI reads N checkpoint dirs and produces a valid MD table.
    """
    if not _has_evaluation_hook("build_comparison_table"):
        pytest.skip(
            "Track 8 build_comparison_table hook not yet exported from "
            "modernbert_g2p.evaluation — test comes online once T8 lands B1."
        )

    ckpt_a = tmp_path / "p_a"
    ckpt_b = tmp_path / "p_b"
    ckpt_c = tmp_path / "p_c"
    for d in (ckpt_a, ckpt_b, ckpt_c):
        d.mkdir()
    out_md = tmp_path / "table.md"

    env = dict(os.environ)
    env.setdefault("MODERNBERT_G2P_TEST_TINY", "1")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "modernbert_g2p",
            "compare",
            "--checkpoint",
            str(ckpt_a),
            "--checkpoint",
            str(ckpt_b),
            "--checkpoint",
            str(ckpt_c),
            "--output",
            str(out_md),
        ],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"compare subprocess exit={result.returncode}\n"
        f"stdout=\n{result.stdout}\nstderr=\n{result.stderr}"
    )
    assert out_md.is_file()
