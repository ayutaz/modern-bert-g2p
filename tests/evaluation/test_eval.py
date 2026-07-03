"""Unit tests for the Phase 2 evaluation harness."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from modernbert_g2p.data.schema import Row
from modernbert_g2p.evaluation import (
    bootstrap_ci,
    build_comparison_table,
    evaluate_checkpoint,
    run_eval,
    run_eval_p_a,
    run_eval_p_b,
    run_eval_p_c,
    score_hardset,
    score_jsut,
    score_jvs,
    score_rohan,
)
from modernbert_g2p.metrics import compute_cer, compute_ker

_REPO_ROOT = Path(__file__).resolve().parents[2]
_HARDSET_FIXTURE = _REPO_ROOT / "tests" / "evaluation" / "fixtures" / "tiny_hardset.jsonl"
_JSUT_FIXTURE = _REPO_ROOT / "tests" / "data" / "fixtures" / "tiny_jsut.yaml"
_ROHAN_FIXTURE = _REPO_ROOT / "tests" / "data" / "fixtures" / "tiny_rohan.txt"


def _make_row(id_: str, text: str, phonemes: tuple[str, ...], category: str = "general") -> Row:
    return Row(
        id=id_,
        source="pyopenjtalk_plus",
        source_license="BSD3",
        text=text,
        phonemes=phonemes,
        mora_accents=(),
        accent_boundaries=(),
        category=category,
    )


def _tiny_gold() -> list[Row]:
    return [
        _make_row("id-1", "今日", ("k", "y", "o", "o")),
        _make_row("id-2", "こんにちは", ("k", "o", "N", "n", "i", "ch", "i", "h", "a")),
        _make_row("id-3", "水", ("m", "i", "z", "u")),
    ]


def test_run_eval_identity_returns_zero_per() -> None:
    gold = _tiny_gold()
    lookup = {row.text: row.phonemes for row in gold}

    def prediction_fn(text: str) -> tuple[str, ...]:
        return lookup[text]

    result = run_eval(gold, prediction_fn)
    assert result["aggregate"]["per_micro"] == pytest.approx(0.0)
    assert result["aggregate"]["per_macro"] == pytest.approx(0.0)
    assert result["aggregate"]["n_rows"] == 3
    assert all(row["per"] == 0.0 for row in result["per_row"])
    assert result["scores"] == [0.0, 0.0, 0.0]
    assert "bootstrap" in result


def test_run_eval_wrong_predictions_returns_nonzero_per() -> None:
    gold = _tiny_gold()

    def prediction_fn(text: str) -> tuple[str, ...]:
        return ("x", "x", "x")

    result = run_eval(gold, prediction_fn, include_bootstrap=False)
    assert result["aggregate"]["per_micro"] > 0.0
    assert result["aggregate"]["per_macro"] > 0.0
    assert all(row["per"] > 0.0 for row in result["per_row"])


def test_run_eval_accepts_canonical_form_duck_type() -> None:
    class DummyCanonical:
        def __init__(self, phonemes: tuple[str, ...]) -> None:
            self.phonemes = phonemes

    gold = _tiny_gold()
    lookup = {row.text: DummyCanonical(row.phonemes) for row in gold}

    def prediction_fn(text: str) -> DummyCanonical:
        return lookup[text]

    result = run_eval(gold, prediction_fn, include_bootstrap=False)
    assert result["aggregate"]["per_micro"] == pytest.approx(0.0)


def test_run_eval_per_row_carries_category() -> None:
    gold = [_make_row("a", "今日", ("k", "y", "o", "o"), category="polyphone")]

    def prediction_fn(text: str) -> tuple[str, ...]:
        return "k", "y", "o", "o"

    result = run_eval(gold, prediction_fn, include_bootstrap=False)
    assert result["per_row"][0]["category"] == "polyphone"


def test_run_eval_bootstrap_ci_matches_direct_call() -> None:
    gold = _tiny_gold() * 4

    def prediction_fn(text: str) -> tuple[str, ...]:
        return ("x", "x")

    result = run_eval(gold, prediction_fn, bootstrap_seed=99, bootstrap_n_resample=500)
    direct = bootstrap_ci(result["scores"], seed=99, n_resample=500)
    assert result["bootstrap"] == direct


def test_run_eval_empty_dataset() -> None:
    def prediction_fn(text: str) -> tuple[str, ...]:
        return ()

    result = run_eval([], prediction_fn)
    assert result["aggregate"]["n_rows"] == 0
    assert result["aggregate"]["per_micro"] == 0.0
    assert result["scores"] == []


def test_score_jsut_reads_yaml_fixture() -> None:
    with _JSUT_FIXTURE.open("r", encoding="utf-8") as f:
        import yaml as _yaml

        data = _yaml.safe_load(f)
    lookup = {
        entry.get("text_level2") or entry.get("text_level0"): tuple(
            p for p in entry["phone_level3"].split("-") if p
        )
        for entry in data.values()
    }

    def prediction_fn(text: str) -> tuple[str, ...]:
        return lookup[text]

    result = score_jsut(prediction_fn, _JSUT_FIXTURE)
    assert result["aggregate"]["n_rows"] == 3
    assert result["aggregate"]["per_micro"] == pytest.approx(0.0)


def test_score_jsut_with_noise_prediction_is_nonzero() -> None:
    def prediction_fn(text: str) -> tuple[str, ...]:
        return ("a", "b", "c")

    result = score_jsut(prediction_fn, _JSUT_FIXTURE, include_bootstrap=False)
    assert result["aggregate"]["per_micro"] > 0.0


def test_score_hardset_returns_all_known_categories() -> None:
    from modernbert_g2p.data.schema import KNOWN_CATEGORIES

    def prediction_fn(text: str) -> tuple[str, ...]:
        return ()

    result = score_hardset(prediction_fn, _HARDSET_FIXTURE)
    for cat in KNOWN_CATEGORIES:
        assert cat in result


def test_score_hardset_identity_predictions_zero_per_per_category() -> None:
    import json

    lookup: dict[str, tuple[str, ...]] = {}
    with _HARDSET_FIXTURE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            lookup[obj["text"]] = tuple(obj["phonemes"])

    def prediction_fn(text: str) -> tuple[str, ...]:
        return lookup[text]

    result = score_hardset(prediction_fn, _HARDSET_FIXTURE)
    non_empty = {c: r for c, r in result.items() if r["aggregate"]["n_rows"] > 0}
    assert len(non_empty) >= 3
    for cat_result in non_empty.values():
        assert cat_result["aggregate"]["per_micro"] == pytest.approx(0.0)


def test_score_hardset_empty_category_has_zero_aggregate() -> None:
    def prediction_fn(text: str) -> tuple[str, ...]:
        return ()

    result = score_hardset(prediction_fn, _HARDSET_FIXTURE)
    absent = result["counter"]
    assert absent["aggregate"]["n_rows"] == 0
    assert absent["aggregate"]["per_micro"] == 0.0
    assert absent["per_row"] == []


def test_score_jvs_reads_csv_fixture(tmp_path: Path) -> None:
    csv_path = tmp_path / "jvs_nonpara_kana.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["base", "text", "kana"])
        writer.writerow(["jvs001", "今日は良い天気ですね", "キョウワヨイテンキデスネ"])
        writer.writerow(["jvs002", "水を買う", "ミズヲカウ"])

    lookup = {
        "今日は良い天気ですね": "キョウワヨイテンキデスネ",
        "水を買う": "ミズヲカウ",
    }

    def prediction_fn(text: str) -> str:
        return lookup[text]

    result = score_jvs(prediction_fn, csv_path)
    assert result["aggregate"]["n_rows"] == 2
    assert result["aggregate"]["per_micro"] == pytest.approx(0.0)


def test_score_jvs_wrong_prediction_uses_cer() -> None:
    csv_path = Path(__file__).parent / "fixtures" / "tmp_jvs.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["base", "text", "kana"])
        writer.writerow(["jvs001", "水を買う", "ミズヲカウ"])
    try:

        def prediction_fn(text: str) -> str:
            return "ABC"

        result = score_jvs(prediction_fn, csv_path, include_bootstrap=False)
        assert result["aggregate"]["per_micro"] > 0.0
        assert result["per_row"][0]["n"] == len("ミズヲカウ")
        expected = compute_cer("ABC", "ミズヲカウ")
        assert result["per_row"][0]["s"] == expected["s"]
    finally:
        csv_path.unlink(missing_ok=True)


def test_score_rohan_reads_fixture() -> None:
    lookup: dict[str, str] = {}
    with _ROHAN_FIXTURE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or ":" not in line:
                continue
            pair = line.split(":", 1)[1]
            text, kana = pair.split(",", 1)
            text_clean = text.replace("(注)", "")
            lookup[text_clean] = kana

    def prediction_fn(text: str) -> str:
        return lookup[text]

    result = score_rohan(prediction_fn, _ROHAN_FIXTURE)
    assert result["aggregate"]["n_rows"] == 3
    assert result["aggregate"]["per_micro"] == pytest.approx(0.0)


def test_score_rohan_ker_used() -> None:
    def prediction_fn(text: str) -> str:
        return "ワ"

    result = score_rohan(prediction_fn, _ROHAN_FIXTURE, include_bootstrap=False)
    assert result["per_row"][0]["s"] + result["per_row"][0]["d"] + result["per_row"][0]["i"] > 0
    expected = compute_ker("ワ", "コレワロハンノテストブンデス")
    assert result["per_row"][0]["s"] == expected["s"]


def test_run_eval_p_a_matches_run_eval() -> None:
    gold = _tiny_gold()
    lookup = {row.text: row.phonemes for row in gold}

    def prediction_fn(text: str) -> tuple[str, ...]:
        return lookup[text]

    result_a = run_eval_p_a(gold, prediction_fn, include_bootstrap=False)
    result_generic = run_eval(gold, prediction_fn, include_bootstrap=False)
    assert result_a["aggregate"] == result_generic["aggregate"]


def test_run_eval_p_b_and_p_c_delegate() -> None:
    gold = _tiny_gold()

    def prediction_fn(text: str) -> tuple[str, ...]:
        return ("x",)

    result_b = run_eval_p_b(gold, prediction_fn, include_bootstrap=False)
    result_c = run_eval_p_c(gold, prediction_fn, include_bootstrap=False)
    assert result_b["aggregate"]["n_rows"] == 3
    assert result_c["aggregate"]["n_rows"] == 3


def test_run_eval_p_a_requires_prediction_or_model_triplet() -> None:
    gold = _tiny_gold()
    with pytest.raises(ValueError):
        run_eval_p_a(gold, None)


def test_score_hardset_rejects_legacy_fixture(tmp_path: Path) -> None:
    bad_path = tmp_path / "legacy.jsonl"
    bad_path.write_text(
        '{"id":"x","category":"polyphone","text":"a","phonemes":["a"],'
        '"accent":["H"],"accent_phrase_boundaries":[]}\n',
        encoding="utf-8",
    )

    def prediction_fn(text: str) -> tuple[str, ...]:
        return ()

    with pytest.raises(ValueError, match="hardset JSONL"):
        score_hardset(prediction_fn, bad_path)


@dataclass(frozen=True)
class _StubData:
    jsut_yaml: str | None = None
    jvs_dir: str | None = None
    rohan_txt: str | None = None
    hard_set_path: str | None = None


@dataclass(frozen=True)
class _StubCfg:
    data: _StubData


def _dummy_checkpoint(tmp_path: Path) -> Path:
    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_bytes(b"stub")
    return ckpt


def test_evaluate_checkpoint_dispatches_jsut_and_writes_json(tmp_path: Path) -> None:
    ckpt = _dummy_checkpoint(tmp_path)
    out_json = tmp_path / "out" / "eval_jsut.json"
    cfg = _StubCfg(data=_StubData(jsut_yaml=str(_JSUT_FIXTURE)))

    with _JSUT_FIXTURE.open("r", encoding="utf-8") as f:
        import yaml as _yaml

        data = _yaml.safe_load(f)
    lookup = {
        entry.get("text_level2") or entry.get("text_level0"): tuple(
            p for p in entry["phone_level3"].split("-") if p
        )
        for entry in data.values()
    }

    def prediction_fn(text: str) -> tuple[str, ...]:
        return lookup[text]

    result = evaluate_checkpoint(
        cfg,
        pilot="p_a",
        checkpoint=ckpt,
        dataset="jsut",
        batch_size=4,
        output=out_json,
        prediction_fn=prediction_fn,
    )

    assert result["pilot"] == "p_a"
    assert result["dataset"] == "jsut"
    assert result["checkpoint"] == str(ckpt)
    assert result["aggregate"]["n_rows"] == 3
    assert result["aggregate"]["per_micro"] == pytest.approx(0.0)
    assert out_json.is_file()
    persisted = json.loads(out_json.read_text(encoding="utf-8"))
    assert persisted["aggregate"]["n_rows"] == 3
    assert persisted["pilot"] == "p_a"


def test_evaluate_checkpoint_jvs_uses_cer(tmp_path: Path) -> None:
    ckpt = _dummy_checkpoint(tmp_path)
    csv_path = tmp_path / "jvs" / "jvs_nonpara_kana.csv"
    csv_path.parent.mkdir(parents=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["base", "text", "kana"])
        writer.writerow(["jvs001", "水を買う", "ミズヲカウ"])

    cfg = _StubCfg(data=_StubData(jvs_dir=str(csv_path.parent)))

    def prediction_fn(text: str) -> str:
        return "ABC"

    result = evaluate_checkpoint(
        cfg,
        pilot="p_b",
        checkpoint=ckpt,
        dataset="jvs",
        prediction_fn=prediction_fn,
        include_bootstrap=False,
    )
    expected = compute_cer("ABC", "ミズヲカウ")
    assert result["per_row"][0]["s"] == expected["s"]
    assert result["dataset"] == "jvs"


def test_evaluate_checkpoint_rohan_uses_ker(tmp_path: Path) -> None:
    ckpt = _dummy_checkpoint(tmp_path)
    cfg = _StubCfg(data=_StubData(rohan_txt=str(_ROHAN_FIXTURE)))

    def prediction_fn(text: str) -> str:
        return "ワ"

    result = evaluate_checkpoint(
        cfg,
        pilot="p_c",
        checkpoint=ckpt,
        dataset="rohan",
        prediction_fn=prediction_fn,
        include_bootstrap=False,
    )
    expected = compute_ker("ワ", "コレワロハンノテストブンデス")
    assert result["per_row"][0]["s"] == expected["s"]


def test_evaluate_checkpoint_hardset_returns_pooled_and_by_category(tmp_path: Path) -> None:
    ckpt = _dummy_checkpoint(tmp_path)
    cfg = _StubCfg(data=_StubData(hard_set_path=str(_HARDSET_FIXTURE)))

    def prediction_fn(text: str) -> tuple[str, ...]:
        return ()

    result = evaluate_checkpoint(
        cfg,
        pilot="p_a",
        checkpoint=ckpt,
        dataset="hardset",
        prediction_fn=prediction_fn,
    )
    assert "by_category" in result
    assert "aggregate" in result
    assert result["aggregate"]["n_rows"] > 0
    from modernbert_g2p.data.schema import KNOWN_CATEGORIES

    for cat in KNOWN_CATEGORIES:
        assert cat in result["by_category"]


def test_evaluate_checkpoint_unknown_pilot_raises(tmp_path: Path) -> None:
    ckpt = _dummy_checkpoint(tmp_path)
    cfg = _StubCfg(data=_StubData(jsut_yaml=str(_JSUT_FIXTURE)))
    with pytest.raises(ValueError, match="pilot"):
        evaluate_checkpoint(
            cfg,
            pilot="p_x",
            checkpoint=ckpt,
            dataset="jsut",
            prediction_fn=lambda text: (),
        )


def test_evaluate_checkpoint_unknown_dataset_raises(tmp_path: Path) -> None:
    ckpt = _dummy_checkpoint(tmp_path)
    cfg = _StubCfg(data=_StubData(jsut_yaml=str(_JSUT_FIXTURE)))
    with pytest.raises(ValueError, match="dataset"):
        evaluate_checkpoint(
            cfg,
            pilot="p_a",
            checkpoint=ckpt,
            dataset="unknown",
            prediction_fn=lambda text: (),
        )


def test_evaluate_checkpoint_missing_data_path_raises(tmp_path: Path) -> None:
    ckpt = _dummy_checkpoint(tmp_path)
    cfg = _StubCfg(data=_StubData())
    with pytest.raises(ValueError, match="jsut_yaml"):
        evaluate_checkpoint(
            cfg,
            pilot="p_a",
            checkpoint=ckpt,
            dataset="jsut",
            prediction_fn=lambda text: (),
        )


def test_evaluate_checkpoint_missing_checkpoint_raises(tmp_path: Path) -> None:
    cfg = _StubCfg(data=_StubData(jsut_yaml=str(_JSUT_FIXTURE)))
    missing = tmp_path / "does_not_exist.pt"
    with pytest.raises(ValueError, match="checkpoint"):
        evaluate_checkpoint(
            cfg,
            pilot="p_a",
            checkpoint=missing,
            dataset="jsut",
        )


def test_evaluate_checkpoint_no_prediction_fn_raises_not_implemented(tmp_path: Path) -> None:
    ckpt = _dummy_checkpoint(tmp_path)
    cfg = _StubCfg(data=_StubData(jsut_yaml=str(_JSUT_FIXTURE)))
    with pytest.raises(NotImplementedError):
        evaluate_checkpoint(
            cfg,
            pilot="p_a",
            checkpoint=ckpt,
            dataset="jsut",
        )


def _write_eval_json(
    path: Path,
    *,
    pilot: str,
    dataset: str,
    per_micro: float,
    n_rows: int = 10,
    bootstrap: dict[str, float] | None = None,
) -> None:
    payload: dict[str, object] = {
        "pilot": pilot,
        "dataset": dataset,
        "checkpoint": "/tmp/fake.pt",
        "aggregate": {
            "n_rows": n_rows,
            "n_ref_total": n_rows * 4,
            "s_total": 0,
            "d_total": 0,
            "i_total": 0,
            "per_micro": per_micro,
            "per_macro": per_micro,
        },
    }
    if bootstrap is not None:
        payload["bootstrap"] = bootstrap
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_comparison_table_reads_per_pilot_dirs(tmp_path: Path) -> None:
    for pilot in ("p_a", "p_b", "p_c"):
        d = tmp_path / pilot
        _write_eval_json(
            d / "eval_jsut.json",
            pilot=pilot,
            dataset="jsut",
            per_micro=0.85,
            bootstrap={"lo": 0.70, "hi": 1.00, "mean": 0.85},
        )
        _write_eval_json(
            d / "eval_jvs.json",
            pilot=pilot,
            dataset="jvs",
            per_micro=1.10,
        )

    dirs = [tmp_path / p for p in ("p_a", "p_b", "p_c")]
    table = build_comparison_table(dirs)
    assert "| Pilot |" in table
    assert "P-A" in table
    assert "P-B" in table
    assert "P-C" in table
    assert "jsut" in table
    assert "jvs" in table
    assert "PER" in table
    assert "CER" in table
    assert "0.850" in table
    assert "[0.700, 1.000]" in table
    assert "n/a" in table


def test_build_comparison_table_skips_missing_dirs(tmp_path: Path) -> None:
    existing = tmp_path / "p_a"
    _write_eval_json(
        existing / "eval_jsut.json",
        pilot="p_a",
        dataset="jsut",
        per_micro=0.5,
    )
    missing = tmp_path / "p_b_missing"
    table = build_comparison_table([existing, missing])
    assert "P-A" in table
    assert "P-B" not in table


def test_build_comparison_table_empty_dirs_returns_header_only(tmp_path: Path) -> None:
    empty = tmp_path / "empty_dir"
    empty.mkdir()
    table = build_comparison_table([empty])
    lines = [ln for ln in table.strip().splitlines() if ln]
    assert len(lines) == 2
    assert lines[0].startswith("| Pilot")


def test_build_comparison_table_ignores_bad_json(tmp_path: Path) -> None:
    d = tmp_path / "p_a"
    d.mkdir()
    (d / "eval_broken.json").write_text("not-json{", encoding="utf-8")
    _write_eval_json(
        d / "eval_jsut.json",
        pilot="p_a",
        dataset="jsut",
        per_micro=0.5,
    )
    table = build_comparison_table([d])
    assert "0.500" in table
    lines = [ln for ln in table.strip().splitlines() if ln.startswith("| P-A")]
    assert len(lines) == 1


def test_build_comparison_table_infers_dataset_from_filename(tmp_path: Path) -> None:
    d = tmp_path / "p_a"
    d.mkdir()
    (d / "eval_rohan.json").write_text(
        json.dumps(
            {
                "aggregate": {"n_rows": 4, "per_micro": 1.5},
            }
        ),
        encoding="utf-8",
    )
    table = build_comparison_table([d])
    assert "rohan" in table
    assert "KER" in table
    assert "p_a" in table or "P-A" in table
