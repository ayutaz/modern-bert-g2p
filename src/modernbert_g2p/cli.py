"""Top-level Phase 2 CLI.

Provides ``train``/``eval``/``compare`` subcommands invoked as
``python -m modernbert_g2p <subcommand> ...``.  The heavy imports
(``torch``, ``transformers``, model modules, training loop) are
performed inside the subcommand handlers so that ``--help``, argument
parsing, and unit tests remain fast and do not require the full
training stack.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

_PILOT_CHOICES: tuple[str, ...] = ("P-A", "P-C")

_PILOT_TO_CONFIG_KEY: dict[str, str] = {
    "P-A": "p_a",
    "P-C": "p_c",
}


def make_parser() -> argparse.ArgumentParser:
    """Build the Phase 2 top-level argparse parser."""
    parser = argparse.ArgumentParser(
        prog="python -m modernbert_g2p",
        description=(
            "ModernBERT G2P Phase 2 CLI — train / eval / compare the two "
            "pure-NN tokenizer pilots (P-A seq2seq, P-C char BERT)."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    _add_train_subparser(subparsers)
    _add_eval_subparser(subparsers)
    _add_compare_subparser(subparsers)

    return parser


def _add_train_subparser(subparsers: argparse._SubParsersAction) -> None:
    p_train = subparsers.add_parser(
        "train",
        help="Train a single pilot end-to-end from a YAML config.",
        description=(
            "Train a Phase 2 pilot (P-A / P-C) from a YAML config. "
            "Use --smoke for a single-batch single-step dry run."
        ),
    )
    p_train.add_argument(
        "--config",
        type=Path,
        required=True,
        metavar="PATH",
        help="Phase 2 pilot config YAML (e.g. configs/p_a.yaml).",
    )
    p_train.add_argument(
        "--seed",
        type=int,
        default=20260704,
        metavar="INT",
        help="RNG seed (default: 20260704).",
    )
    p_train.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="PATH",
        help="Checkpoint / log output directory (overrides config.output_dir).",
    )
    p_train.add_argument(
        "--smoke",
        action="store_true",
        help="Run one batch / one optimizer step then exit 0 (CI dry-run).",
    )


def _add_eval_subparser(subparsers: argparse._SubParsersAction) -> None:
    p_eval = subparsers.add_parser(
        "eval",
        help="Evaluate a trained pilot checkpoint on JSUT / JVS / ROHAN / hard-set.",
        description=(
            "Run evaluation on a saved checkpoint. Writes a per-row JSON report "
            "and prints aggregate PER/CER/KER with bootstrap 95% CI."
        ),
    )
    p_eval.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        metavar="PATH",
        help="Path to a checkpoint file saved by Trainer.save_checkpoint.",
    )
    p_eval.add_argument(
        "--config",
        type=Path,
        required=True,
        metavar="PATH",
        help="Pilot config YAML used at training time (rebuilds model architecture).",
    )
    p_eval.add_argument(
        "--pilot",
        choices=list(_PILOT_CHOICES),
        required=True,
        help="Pilot family: P-A (seq2seq) / P-C (char BERT).",
    )
    p_eval.add_argument(
        "--dataset",
        choices=("jsut", "jvs", "rohan", "hardset"),
        default="jsut",
        help="Evaluation dataset (default: jsut).",
    )
    p_eval.add_argument(
        "--output",
        type=Path,
        required=True,
        metavar="PATH",
        help="Output JSON path for per-row scores and aggregates.",
    )
    p_eval.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="INT",
        help="Eval batch size (default: 32).",
    )


def _add_compare_subparser(subparsers: argparse._SubParsersAction) -> None:
    p_compare = subparsers.add_parser(
        "compare",
        help="Aggregate PER/CER/KER + hard-set + throughput across pilots into Markdown.",
        description=(
            "Read per-pilot checkpoint directories and emit a comparison table "
            "(reports/phase2_pilot_table.md) with bootstrap 95% CIs."
        ),
    )
    p_compare.add_argument(
        "--checkpoint",
        dest="checkpoints",
        action="append",
        required=True,
        metavar="DIR",
        help="Checkpoint directory for one pilot (repeatable).",
    )
    p_compare.add_argument(
        "--output",
        type=Path,
        default=Path("reports/phase2_pilot_table.md"),
        metavar="PATH",
        help="Output Markdown path (default: reports/phase2_pilot_table.md).",
    )


def cmd_train(args: argparse.Namespace) -> int:
    """Handle ``train`` subcommand — dispatches to Trainer or smoke path."""
    config_path: Path = args.config
    if not config_path.is_file():
        print(f"error: --config file not found: {config_path}", file=sys.stderr)
        return 2

    try:
        from modernbert_g2p.config import load_config
    except ImportError as exc:
        print(
            "error: modernbert_g2p.config not available "
            f"(Track 9 not yet implemented): {exc}",
            file=sys.stderr,
        )
        return 3

    try:
        cfg = load_config(config_path)
    except (ValueError, OSError) as exc:
        print(f"error: failed to load config {config_path}: {exc}", file=sys.stderr)
        return 2

    seed = args.seed
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        # Config output_dir is a template (e.g. "reports/phase2/p_a_30k/{seed}",
        # default "reports/phase2/{pilot}/{seed}"); expand the placeholders so
        # checkpoints do not land in a literal "{seed}" directory.
        template = str(getattr(cfg, "output_dir", "reports/phase2/run"))
        pilot = str(getattr(cfg, "pilot", ""))
        output_dir = Path(template.replace("{seed}", str(seed)).replace("{pilot}", pilot))
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        return _run_train_smoke(cfg, seed=seed, output_dir=output_dir)
    return _run_train_full(cfg, seed=seed, output_dir=output_dir)


def _run_train_smoke(cfg: object, *, seed: int, output_dir: Path) -> int:
    """Single-batch single-step smoke path — exits 0 if a step is executed."""
    try:
        import torch
    except ImportError:
        print("error: torch not installed; install extras [training]", file=sys.stderr)
        return 3
    torch.manual_seed(seed)

    try:
        from modernbert_g2p.training.trainer import Trainer  # noqa: F401
    except ImportError as exc:
        print(
            "error: modernbert_g2p.training.trainer not available "
            f"(Track 7 not yet implemented): {exc}",
            file=sys.stderr,
        )
        return 3

    try:
        from modernbert_g2p.training import build_smoke_pipeline
    except ImportError as exc:
        print(
            "error: modernbert_g2p.training.build_smoke_pipeline not available "
            f"(Track 7 not yet implemented): {exc}",
            file=sys.stderr,
        )
        return 3

    trainer = build_smoke_pipeline(cfg, seed=seed, output_dir=output_dir)
    metrics = trainer.run(steps=1)
    _emit_smoke_line(cfg, metrics=metrics, output_dir=output_dir)
    return 0


def _run_train_full(cfg: object, *, seed: int, output_dir: Path) -> int:
    """Full training run — delegates to modernbert_g2p.training.run_training."""
    try:
        from modernbert_g2p.training import run_training
    except ImportError as exc:
        print(
            "error: modernbert_g2p.training.run_training not available "
            f"(Track 7 not yet implemented): {exc}",
            file=sys.stderr,
        )
        return 3

    metrics = run_training(cfg, seed=seed, output_dir=output_dir)
    print(f"[train] final metrics: {metrics}")
    return 0


def _emit_smoke_line(cfg: object, *, metrics: dict[str, float], output_dir: Path) -> None:
    pilot = getattr(cfg, "pilot", "unknown")
    loss = metrics.get("loss", float("nan"))
    print(f"[smoke] pilot={pilot} steps=1 loss={loss:.4f} out={output_dir}")


def cmd_eval(args: argparse.Namespace) -> int:
    """Handle ``eval`` subcommand."""
    checkpoint: Path = args.checkpoint
    if not checkpoint.exists():
        print(f"error: --checkpoint not found: {checkpoint}", file=sys.stderr)
        return 2
    config_path: Path = args.config
    if not config_path.is_file():
        print(f"error: --config file not found: {config_path}", file=sys.stderr)
        return 2

    try:
        from modernbert_g2p.config import load_config
        from modernbert_g2p.evaluation import evaluate_checkpoint
    except ImportError as exc:
        print(
            "error: evaluation stack not available "
            f"(Track 8/9 not yet implemented): {exc}",
            file=sys.stderr,
        )
        return 3

    cfg = load_config(config_path)
    pilot_key = _PILOT_TO_CONFIG_KEY[args.pilot]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = evaluate_checkpoint(
        cfg=cfg,
        pilot=pilot_key,
        checkpoint=checkpoint,
        dataset=args.dataset,
        batch_size=args.batch_size,
        output=args.output,
    )
    aggregate = result.get("aggregate", {}) if isinstance(result, dict) else {}
    print(f"[eval] pilot={args.pilot} dataset={args.dataset} aggregate={aggregate}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Handle ``compare`` subcommand — writes a per-pilot Markdown table."""
    checkpoint_dirs = [Path(p) for p in args.checkpoints]
    missing = [p for p in checkpoint_dirs if not p.exists()]
    if missing:
        for p in missing:
            print(f"error: checkpoint dir not found: {p}", file=sys.stderr)
        return 2

    try:
        from modernbert_g2p.evaluation import build_comparison_table
    except ImportError as exc:
        print(
            "error: modernbert_g2p.evaluation.build_comparison_table not available "
            f"(Track 8 not yet implemented): {exc}",
            file=sys.stderr,
        )
        return 3

    args.output.parent.mkdir(parents=True, exist_ok=True)
    table_md = build_comparison_table(checkpoint_dirs)
    args.output.write_text(table_md, encoding="utf-8")
    print(f"[compare] wrote {args.output} ({len(checkpoint_dirs)} pilots)")
    return 0


_COMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {
    "train": cmd_train,
    "eval": cmd_eval,
    "compare": cmd_compare,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Program entry point. Returns process exit code."""
    parser = make_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 2
        return code
    handler = _COMMANDS.get(args.command)
    if handler is None:
        parser.print_help(sys.stderr)
        return 2
    return handler(args)


__all__ = [
    "cmd_compare",
    "cmd_eval",
    "cmd_train",
    "main",
    "make_parser",
]
