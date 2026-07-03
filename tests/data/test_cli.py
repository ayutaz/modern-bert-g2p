"""Unit tests for the Phase 1 CLI (`python -m modernbert_g2p.data`)."""

from __future__ import annotations

import io
import os
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from modernbert_g2p.data.cli import (
    _DEFAULT_RATIOS,
    _SOURCE_CHOICES,
    main,
    make_parser,
    parse_source_root_pairs,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"


def _naist_jdic_fixture_rows() -> list[str]:
    return [
        "今日,-1,-1,5000,名詞,一般,*,*,*,*,今日,キョウ,キョー,今日,1",
        "友達,-1,-1,5000,名詞,一般,*,*,*,*,友達,トモダチ,トモダチ,友達,0",
        "学校,-1,-1,5000,名詞,一般,*,*,*,*,学校,ガッコウ,ガッコー,学校,0",
    ]


def _write_naist_jdic_fixture(directory: Path) -> Path:
    csv_path = directory / "mini.csv"
    csv_path.write_text("\n".join(_naist_jdic_fixture_rows()) + "\n", encoding="utf-8")
    return csv_path


def _pipeline_modules_available() -> bool:
    try:
        import modernbert_g2p.data.contamination  # noqa: F401
        import modernbert_g2p.data.dedup  # noqa: F401
        import modernbert_g2p.data.ingest.pyopenjtalk_plus  # noqa: F401  # type: ignore[import-not-found]
        import modernbert_g2p.data.output  # noqa: F401
        import modernbert_g2p.data.split  # noqa: F401
    except ImportError:
        return False
    return True


# ---------------------------------------------------------------------------
# parser + helpers
# ---------------------------------------------------------------------------
def test_make_parser_returns_argparse_parser() -> None:
    parser = make_parser()
    assert parser.prog == "python -m modernbert_g2p.data"


def test_top_level_help_exits_zero() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


def test_build_help_exits_zero() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["build", "--help"])
    assert exc.value.code == 0


def test_build_parser_defaults() -> None:
    parser = make_parser()
    args = parser.parse_args(["build", "--output", "outdir"])
    assert args.command == "build"
    assert args.source is None or args.source == []
    assert args.output == Path("outdir")
    assert args.seed == 42
    assert args.limit is None
    assert list(args.ratios) == list(_DEFAULT_RATIOS)
    assert args.format is None
    assert args.jsut_yaml is None
    assert args.rohan_txt is None
    assert args.jvs_txt is None
    assert args.hard_set_jsonl is None


def test_parse_source_root_pairs_happy(tmp_path: Path) -> None:
    result = parse_source_root_pairs(
        [f"pyopenjtalk_plus={tmp_path}/a", f"unidic={tmp_path}/b"]
    )
    assert result == {
        "pyopenjtalk_plus": Path(f"{tmp_path}/a"),
        "unidic": Path(f"{tmp_path}/b"),
    }


def test_parse_source_root_pairs_malformed_raises() -> None:
    with pytest.raises(ValueError, match="NAME=PATH"):
        parse_source_root_pairs(["nomeansnobody"])


def test_parse_source_root_pairs_empty_name_raises() -> None:
    with pytest.raises(ValueError):
        parse_source_root_pairs(["=/tmp/x"])


# ---------------------------------------------------------------------------
# info command — must work even without other tracks fully implemented
# ---------------------------------------------------------------------------
def test_info_lists_all_five_sources(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["info"])
    assert rc == 0
    out = capsys.readouterr().out
    for name in _SOURCE_CHOICES:
        assert name in out, f"expected source {name!r} in info output"


# ---------------------------------------------------------------------------
# build command — validation of arg surface (no heavy imports triggered)
# ---------------------------------------------------------------------------
def test_build_rejects_no_source(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    rc = main(["build", "--output", str(tmp_path)])
    assert rc == 2
    assert "--source" in capsys.readouterr().err


def test_build_rejects_missing_root(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    rc = main(
        [
            "build",
            "--source",
            "pyopenjtalk_plus",
            "--output",
            str(tmp_path),
        ]
    )
    assert rc == 2
    assert "--root" in capsys.readouterr().err


def test_build_rejects_malformed_root_pair(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    rc = main(
        [
            "build",
            "--source",
            "pyopenjtalk_plus",
            "--root",
            "no-equals-sign",
            "--output",
            str(tmp_path),
        ]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "NAME=PATH" in err or "Malformed" in err


def test_build_rejects_bad_ratios(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    rc = main(
        [
            "build",
            "--source",
            "pyopenjtalk_plus",
            "--root",
            f"pyopenjtalk_plus={tmp_path}",
            "--output",
            str(tmp_path / "out"),
            "--ratios",
            "0.5",
            "0.2",
            "0.2",
        ]
    )
    assert rc == 2
    assert "sum to 1.0" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# `python -m modernbert_g2p.data --help` via subprocess
# ---------------------------------------------------------------------------
def test_dunder_main_help_via_subprocess() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-m", "modernbert_g2p.data", "--help"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"
    assert "build" in result.stdout
    assert "info" in result.stdout
    assert "verify-manifest" in result.stdout


# ---------------------------------------------------------------------------
# End-to-end build — depends on all other tracks being present. Guarded.
# ---------------------------------------------------------------------------
def test_cmd_build_end_to_end_writes_manifest(tmp_path: Path) -> None:
    if not _pipeline_modules_available():
        pytest.skip("required pipeline tracks not yet implemented")
    import yaml

    src_root = tmp_path / "src_root"
    src_root.mkdir()
    _write_naist_jdic_fixture(src_root)

    out_dir = tmp_path / "out"

    rc = main(
        [
            "build",
            "--source",
            "pyopenjtalk_plus",
            "--root",
            f"pyopenjtalk_plus={src_root}",
            "--output",
            str(out_dir),
            "--jsonl",
        ]
    )
    assert rc == 0, "cmd_build should succeed for a well-formed fixture"

    manifest_path = out_dir / "manifest.yaml"
    assert manifest_path.exists(), "manifest.yaml must be written"

    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert manifest.get("schema_version"), "manifest missing schema_version"
    assert set(manifest["splits"].keys()) == {"train", "val", "test"}
    for split_name in ("train", "val", "test"):
        entry = manifest["splits"][split_name]
        assert "path" in entry and "sha256" in entry
        assert (out_dir / entry["path"]).exists()


# ---------------------------------------------------------------------------
# verify-manifest — happy path + tamper detection
# ---------------------------------------------------------------------------
def _build_from_fixture(tmp_path: Path) -> Path:
    src_root = tmp_path / "src_root"
    src_root.mkdir()
    _write_naist_jdic_fixture(src_root)
    out_dir = tmp_path / "out"
    rc = main(
        [
            "build",
            "--source",
            "pyopenjtalk_plus",
            "--root",
            f"pyopenjtalk_plus={src_root}",
            "--output",
            str(out_dir),
            "--jsonl",
        ]
    )
    assert rc == 0
    return out_dir


def test_verify_manifest_passes_on_fresh_build(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    if not _pipeline_modules_available():
        pytest.skip("required pipeline tracks not yet implemented")
    out_dir = _build_from_fixture(tmp_path)
    capsys.readouterr()
    rc = main(["verify-manifest", str(out_dir)])
    assert rc == 0
    out = capsys.readouterr().out
    for split_name in ("train", "val", "test"):
        assert split_name in out


def test_verify_manifest_fails_when_file_modified(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    if not _pipeline_modules_available():
        pytest.skip("required pipeline tracks not yet implemented")
    out_dir = _build_from_fixture(tmp_path)
    tampered = None
    for candidate in out_dir.glob("*.jsonl"):
        candidate.write_text(
            candidate.read_text(encoding="utf-8") + "extra\n", encoding="utf-8"
        )
        tampered = candidate
        break
    assert tampered is not None, "no split file to tamper with"
    capsys.readouterr()
    rc = main(["verify-manifest", str(out_dir)])
    assert rc == 1
    captured = capsys.readouterr()
    combined = captured.err + captured.out
    assert "FAIL" in combined or "expected" in combined or "actual" in combined


def test_verify_manifest_missing_yaml(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = main(["verify-manifest", str(tmp_path)])
    assert rc == 2
    assert "manifest" in capsys.readouterr().err.lower()


# ---------------------------------------------------------------------------
# fetch script
# ---------------------------------------------------------------------------
def test_fetch_script_exists_and_is_executable() -> None:
    script = REPO_ROOT / "scripts" / "fetch_phase1_data.sh"
    assert script.exists(), "scripts/fetch_phase1_data.sh must exist"
    assert os.access(script, os.X_OK), "scripts/fetch_phase1_data.sh must be executable"


def test_fetch_script_has_shebang_and_pinned_versions() -> None:
    script = REPO_ROOT / "scripts" / "fetch_phase1_data.sh"
    content = script.read_text(encoding="utf-8")
    assert content.startswith("#!"), "fetch script must begin with a shebang"
    assert "set -euo pipefail" in content
    assert "UNIDIC_VERSION=" in content
    assert "WIKIPEDIA_DUMP_DATE=" in content


# ---------------------------------------------------------------------------
# make_parser exposes a stable interface — sanity check
# ---------------------------------------------------------------------------
def test_make_parser_lists_subcommands() -> None:
    buf = io.StringIO()
    parser = make_parser()
    with redirect_stdout(buf):
        parser.print_help()
    text = buf.getvalue()
    for sub in ("build", "info", "verify-manifest"):
        assert sub in text
