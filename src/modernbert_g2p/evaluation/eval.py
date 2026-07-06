"""Generic evaluation harness for Phase 2 pilots.

Design: model inference is decoupled from evaluation via a user-supplied
``prediction_fn: text -> hypothesis``. Real model wrappers construct a
``prediction_fn`` from ``(model, tokenizer, canonicalize_fn)`` and delegate
to :func:`run_eval`; tests inject dummy predictions to exercise the loop
without loading torch / transformers.

Aggregate schema returned by :func:`run_eval`::

    {
        "scores": [float, ...],       # per-row primary metric value (e.g. PER)
        "per_row": [{"id", "hyp_len", "ref_len", "s", "d", "i", "per",
                      "category"}, ...],
        "aggregate": {
            "n_rows": int,
            "n_ref_total": int,
            "s_total": int,
            "d_total": int,
            "i_total": int,
            "per_micro": float,       # (s+d+i) / n_ref * 100
            "per_macro": float,       # mean of per_row.per
        },
        "bootstrap": {                # when include_bootstrap=True
            "mean", "lo", "hi", "n", "n_resample", "ci",
        },
    }

The key ``per`` is used uniformly across PER / CER / KER because the canonical
metric routines in :mod:`modernbert_g2p.metrics` all expose the ratio under
that same key (see ``metrics/_edit_distance._summarize``).
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import yaml

from modernbert_g2p.data.schema import Row, from_dict
from modernbert_g2p.evaluation.bootstrap import bootstrap_ci
from modernbert_g2p.metrics import compute_cer, compute_ker, compute_per

PredictionFn = Callable[[str], Any]
MetricFn = Callable[[Any, Any], dict[str, float | int]]
RefExtractor = Callable[[Row], Any]

_VALID_PILOTS: frozenset[str] = frozenset({"p_a", "p_c"})
_VALID_DATASETS: frozenset[str] = frozenset({"jsut", "jvs", "rohan", "hardset"})


def _default_ref_extractor(row: Row) -> tuple[str, ...]:
    return row.phonemes


def _extract_hyp_for_metric(prediction: Any) -> Any:
    """Return the payload metric_fn should score.

    Accepts:
    - Plain sequences / strings (returned as-is).
    - Duck-typed canonical forms exposing a ``phonemes`` attribute
      (e.g. ``modernbert_g2p.models.canonical.CanonicalForm`` when Track 1
      lands). We extract the ``phonemes`` tuple so metric_fn sees a Sequence.
    """
    if isinstance(prediction, str | list | tuple):
        return prediction
    phonemes = getattr(prediction, "phonemes", None)
    if phonemes is not None:
        return phonemes
    return prediction


def run_eval(
    gold_dataset: Iterable[Row],
    prediction_fn: PredictionFn,
    metric_fn: MetricFn = compute_per,
    *,
    ref_extractor: RefExtractor | None = None,
    include_bootstrap: bool = True,
    bootstrap_seed: int = 42,
    bootstrap_n_resample: int = 10_000,
    bootstrap_ci_level: float = 0.95,
) -> dict[str, Any]:
    """Score ``prediction_fn`` against a gold Phase-1 dataset.

    Args:
        gold_dataset: Iterable of :class:`Row` (JSONL / Parquet reader output,
            or a fixture-generated list).
        prediction_fn: ``text -> hypothesis``. The hypothesis is passed to
            ``metric_fn`` as its first argument. Objects exposing a
            ``phonemes`` attribute are auto-unpacked for PER-shaped metrics.
        metric_fn: One of :func:`compute_per` / :func:`compute_cer` /
            :func:`compute_ker` (or a caller-supplied equivalent returning
            keys ``{n, s, d, i, per}``).
        ref_extractor: Row → reference payload. Defaults to
            ``row.phonemes`` (PER schema).
        include_bootstrap: If True, add a ``bootstrap`` block with 95% CI
            over per-row primary metric values.
        bootstrap_seed: RNG seed for the bootstrap.
        bootstrap_n_resample: Resample count.
        bootstrap_ci_level: CI confidence level (default 0.95).

    Returns:
        Dict with keys ``scores``, ``per_row``, ``aggregate`` and, when
        requested, ``bootstrap``.
    """
    if ref_extractor is None:
        ref_extractor = _default_ref_extractor

    scores: list[float] = []
    per_row: list[dict[str, Any]] = []
    s_total = d_total = i_total = n_total = 0

    for row in gold_dataset:
        raw_pred = prediction_fn(row.text)
        hyp = _extract_hyp_for_metric(raw_pred)
        ref = ref_extractor(row)
        result = metric_fn(hyp, ref)
        per_val = float(result.get("per", 0.0))
        scores.append(per_val)
        per_row.append(
            {
                "id": row.id,
                "category": row.category,
                "n": int(result.get("n", 0)),
                "s": int(result.get("s", 0)),
                "d": int(result.get("d", 0)),
                "i": int(result.get("i", 0)),
                "per": per_val,
            }
        )
        s_total += int(result.get("s", 0))
        d_total += int(result.get("d", 0))
        i_total += int(result.get("i", 0))
        n_total += int(result.get("n", 0))

    n_rows = len(per_row)
    per_micro = (s_total + d_total + i_total) / n_total * 100 if n_total > 0 else 0.0
    per_macro = sum(scores) / n_rows if n_rows > 0 else 0.0

    out: dict[str, Any] = {
        "scores": scores,
        "per_row": per_row,
        "aggregate": {
            "n_rows": n_rows,
            "n_ref_total": n_total,
            "s_total": s_total,
            "d_total": d_total,
            "i_total": i_total,
            "per_micro": per_micro,
            "per_macro": per_macro,
        },
    }
    if include_bootstrap:
        out["bootstrap"] = bootstrap_ci(
            scores,
            n_resample=bootstrap_n_resample,
            ci=bootstrap_ci_level,
            seed=bootstrap_seed,
        )
    return out


def _pilot_prediction_fn(
    prediction_fn: PredictionFn | None,
    model: Any,
    tokenizer: Any,
    canonicalize_fn: Callable[..., Any] | None,
    pilot: str,
) -> PredictionFn:
    if prediction_fn is not None:
        return prediction_fn
    if model is None or tokenizer is None or canonicalize_fn is None:
        raise ValueError(
            f"run_eval_{pilot}: either prediction_fn OR (model, tokenizer, "
            "canonicalize_fn) must be provided"
        )

    def _fn(text: str) -> Any:
        raise NotImplementedError(
            f"run_eval_{pilot}: automatic inference from (model, tokenizer, "
            "canonicalize_fn) is not yet wired; supply prediction_fn directly."
        )

    return _fn


def run_eval_p_a(
    gold_dataset: Iterable[Row],
    prediction_fn: PredictionFn | None = None,
    metric_fn: MetricFn = compute_per,
    *,
    model: Any = None,
    tokenizer: Any = None,
    canonicalize_fn: Callable[..., Any] | None = None,
    ref_extractor: RefExtractor | None = None,
    include_bootstrap: bool = True,
) -> dict[str, Any]:
    """Pilot P-A (seq2seq) wrapper around :func:`run_eval`."""
    fn = _pilot_prediction_fn(prediction_fn, model, tokenizer, canonicalize_fn, "p_a")
    return run_eval(
        gold_dataset,
        fn,
        metric_fn,
        ref_extractor=ref_extractor,
        include_bootstrap=include_bootstrap,
    )


def run_eval_p_c(
    gold_dataset: Iterable[Row],
    prediction_fn: PredictionFn | None = None,
    metric_fn: MetricFn = compute_per,
    *,
    model: Any = None,
    tokenizer: Any = None,
    canonicalize_fn: Callable[..., Any] | None = None,
    ref_extractor: RefExtractor | None = None,
    include_bootstrap: bool = True,
) -> dict[str, Any]:
    """Pilot P-C (char-level BERT) wrapper around :func:`run_eval`."""
    fn = _pilot_prediction_fn(prediction_fn, model, tokenizer, canonicalize_fn, "p_c")
    return run_eval(
        gold_dataset,
        fn,
        metric_fn,
        ref_extractor=ref_extractor,
        include_bootstrap=include_bootstrap,
    )


_JSUT_SOURCE = "pyopenjtalk_plus"
_JSUT_LICENSE = "BSD3"
_JVS_SOURCE = "pyopenjtalk_plus"
_JVS_LICENSE = "BSD3"
_ROHAN_SOURCE = "pyopenjtalk_plus"
_ROHAN_LICENSE = "BSD3"


def _row_from_pieces(
    *,
    id_: str,
    text: str,
    phonemes: Sequence[str],
    source: str,
    source_license: str,
    category: str = "general",
) -> Row:
    return Row(
        id=id_,
        source=source,
        source_license=source_license,
        text=text,
        phonemes=tuple(phonemes),
        mora_accents=(),
        accent_boundaries=(),
        category=category,
    )


def _load_jsut_rows(jsut_yaml_path: Path) -> list[Row]:
    with jsut_yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    rows: list[Row] = []
    for utt_id, entry in data.items():
        text = entry.get("text_level2") or entry.get("text_level0")
        if not text:
            continue
        phone_str = entry.get("phone_level3", "") or ""
        phonemes = [p for p in phone_str.split("-") if p]
        rows.append(
            _row_from_pieces(
                id_=str(utt_id),
                text=text,
                phonemes=phonemes,
                source=_JSUT_SOURCE,
                source_license=_JSUT_LICENSE,
            )
        )
    return rows


def score_jsut(
    prediction_fn: PredictionFn,
    jsut_yaml_path: Path,
    *,
    ref_extractor: RefExtractor | None = None,
    include_bootstrap: bool = True,
    bootstrap_seed: int = 42,
) -> dict[str, Any]:
    """Score JSUT Basic5000 PER using the yaml at ``jsut_yaml_path``.

    Reads ``text_level2`` (falling back to ``text_level0``) as input, and
    ``phone_level3.split("-")`` as the reference phoneme sequence. Matches
    ``scripts/eval_haqumei_jsut.py`` (haqumei baseline reproducer).
    """
    rows = _load_jsut_rows(Path(jsut_yaml_path))
    return run_eval(
        rows,
        prediction_fn,
        compute_per,
        ref_extractor=ref_extractor,
        include_bootstrap=include_bootstrap,
        bootstrap_seed=bootstrap_seed,
    )


def _load_jvs_rows(jvs_source: Path) -> list[Row]:
    jvs_source = Path(jvs_source)
    csv_path = jvs_source if jvs_source.is_file() else jvs_source / "jvs_nonpara_kana.csv"
    rows: list[Row] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            next(reader)
        except StopIteration:
            return rows
        for record in reader:
            if len(record) < 3:
                continue
            base, text, ref_kana = record[0], record[1], record[2]
            if not text:
                continue
            rows.append(
                Row(
                    id=str(base),
                    source=_JVS_SOURCE,
                    source_license=_JVS_LICENSE,
                    text=text,
                    phonemes=(),
                    mora_accents=(),
                    accent_boundaries=(),
                    extra={"ref_kana": ref_kana},
                )
            )
    return rows


def _jvs_ref_extractor(row: Row) -> str:
    return str(row.extra.get("ref_kana", ""))


def score_jvs(
    prediction_fn: PredictionFn,
    jvs_source: Path,
    *,
    include_bootstrap: bool = True,
    bootstrap_seed: int = 42,
) -> dict[str, Any]:
    """Score JVS-3000 kana CER. ``jvs_source`` is either the CSV or its dir.

    Reference is the third CSV column (kana). Prediction is a kana string
    returned by ``prediction_fn(text)``.
    """
    rows = _load_jvs_rows(Path(jvs_source))
    return run_eval(
        rows,
        prediction_fn,
        compute_cer,
        ref_extractor=_jvs_ref_extractor,
        include_bootstrap=include_bootstrap,
        bootstrap_seed=bootstrap_seed,
    )


def _strip_rohan_parens(text: str) -> str:
    out: list[str] = []
    in_paren = False
    for ch in text:
        if ch == "(":
            in_paren = True
        elif ch == ")":
            in_paren = False
        elif not in_paren:
            out.append(ch)
    return "".join(out)


def _load_rohan_rows(path: Path) -> list[Row]:
    rows: list[Row] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.rstrip("\n")
            if not line:
                continue
            id_sep = line.find(":")
            if id_sep < 0:
                continue
            utt_id = line[:id_sep]
            pair = line[id_sep + 1 :]
            comma = pair.find(",")
            if comma < 0:
                continue
            text = _strip_rohan_parens(pair[:comma])
            kana = pair[comma + 1 :]
            if not text:
                continue
            rows.append(
                Row(
                    id=utt_id or f"rohan-{lineno}",
                    source=_ROHAN_SOURCE,
                    source_license=_ROHAN_LICENSE,
                    text=text,
                    phonemes=(),
                    mora_accents=(),
                    accent_boundaries=(),
                    extra={"ref_kana": kana},
                )
            )
    return rows


def score_rohan(
    prediction_fn: PredictionFn,
    rohan_txt_path: Path,
    *,
    include_bootstrap: bool = True,
    bootstrap_seed: int = 42,
) -> dict[str, Any]:
    """Score ROHAN 4600 katakana KER.

    Line format ``ID:text,kana`` (see ``scripts/eval_haqumei_rohan.py``).
    Parenthesized annotations in ``text`` are stripped before prediction.
    """
    rows = _load_rohan_rows(Path(rohan_txt_path))
    return run_eval(
        rows,
        prediction_fn,
        compute_ker,
        ref_extractor=_jvs_ref_extractor,
        include_bootstrap=include_bootstrap,
        bootstrap_seed=bootstrap_seed,
    )


def _load_hardset_rows(hardset_jsonl_path: Path) -> list[Row]:
    rows: list[Row] = []
    with hardset_jsonl_path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            obj = json.loads(line)
            try:
                rows.append(from_dict(obj))
            except ValueError as e:
                raise ValueError(
                    f"hardset JSONL line {lineno}: {e}. "
                    "Fixture must follow the Phase-1 Row schema "
                    "(mora_accents / accent_boundaries)."
                ) from e
    return rows


def score_hardset(
    prediction_fn: PredictionFn,
    hardset_jsonl_path: Path,
    *,
    ref_extractor: RefExtractor | None = None,
    include_bootstrap: bool = False,
    bootstrap_seed: int = 42,
) -> dict[str, dict[str, Any]]:
    """Score a hard-set JSONL, grouped by ``category`` key.

    Returns a mapping ``category -> run_eval-result``. Every category listed
    in :data:`modernbert_g2p.data.schema.KNOWN_CATEGORIES` is present in the
    output (empty per_row / zero aggregate if no samples fell in it), so
    downstream reporting code can index unconditionally.
    """
    from modernbert_g2p.data.schema import KNOWN_CATEGORIES

    rows = _load_hardset_rows(Path(hardset_jsonl_path))
    by_category: dict[str, list[Row]] = {c: [] for c in KNOWN_CATEGORIES}
    for row in rows:
        by_category.setdefault(row.category, []).append(row)

    out: dict[str, dict[str, Any]] = {}
    for category, cat_rows in by_category.items():
        out[category] = run_eval(
            cat_rows,
            prediction_fn,
            compute_per,
            ref_extractor=ref_extractor,
            include_bootstrap=include_bootstrap,
            bootstrap_seed=bootstrap_seed,
        )
    return out


def _resolve_dataset_path(cfg: Any, dataset: str) -> Path:
    data = getattr(cfg, "data", None)
    if data is None:
        raise ValueError(
            f"cfg is missing a `data` attribute; cannot resolve dataset={dataset!r}"
        )
    key = {
        "jsut": "jsut_yaml",
        "jvs": "jvs_dir",
        "rohan": "rohan_txt",
        "hardset": "hard_set_path",
    }[dataset]
    raw = getattr(data, key, None)
    if not raw:
        raise ValueError(
            f"cfg.data.{key} is not set; cannot evaluate dataset={dataset!r}"
        )
    return Path(raw)


def _hardset_aggregate_summary(
    by_category: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    n_rows = 0
    n_ref_total = 0
    s_total = 0
    d_total = 0
    i_total = 0
    scores: list[float] = []
    for cat_result in by_category.values():
        agg = cat_result.get("aggregate", {})
        n_rows += int(agg.get("n_rows", 0))
        n_ref_total += int(agg.get("n_ref_total", 0))
        s_total += int(agg.get("s_total", 0))
        d_total += int(agg.get("d_total", 0))
        i_total += int(agg.get("i_total", 0))
        scores.extend(cat_result.get("scores", []))
    per_micro = (s_total + d_total + i_total) / n_ref_total * 100 if n_ref_total > 0 else 0.0
    per_macro = sum(scores) / len(scores) if scores else 0.0
    return {
        "n_rows": n_rows,
        "n_ref_total": n_ref_total,
        "s_total": s_total,
        "d_total": d_total,
        "i_total": i_total,
        "per_micro": per_micro,
        "per_macro": per_macro,
    }


def _build_prediction_fn_from_checkpoint(
    cfg: Any,
    *,
    pilot: str,
    checkpoint: Path,
    batch_size: int,
) -> PredictionFn:
    """Load a saved checkpoint and return a text -> CanonicalForm callable."""
    if pilot == "p_a":
        return _build_pa_prediction_fn(cfg, checkpoint=checkpoint, batch_size=batch_size)
    if pilot == "p_c":
        return _build_pc_prediction_fn(cfg, checkpoint=checkpoint, batch_size=batch_size)
    raise NotImplementedError(
        f"evaluate_checkpoint: automatic model loading for pilot={pilot!r} "
        f"from checkpoint={checkpoint} is not yet wired. Currently implemented: p_a, p_c. "
        "Pass a `prediction_fn` keyword to bypass model construction."
    )


def _build_pa_prediction_fn(cfg: Any, *, checkpoint: Path, batch_size: int) -> PredictionFn:
    import warnings

    import torch

    from modernbert_g2p.models.canonical import Vocab, build_default_vocab, p_a_to_canonical
    from modernbert_g2p.models.p_a.config import PAConfig
    from modernbert_g2p.models.p_a.model import build_p_a
    from modernbert_g2p.models.tokenization.p_a_tokenizer import PATokenizer

    model_cfg = cfg.model if hasattr(cfg, "model") else {}
    if not isinstance(model_cfg, dict):
        model_cfg = {}
    infer_cfg = getattr(cfg, "inference", None) or {}
    if not isinstance(infer_cfg, dict):
        infer_cfg = {}
    defaults = PAConfig()
    pa_config = PAConfig(
        encoder_name=model_cfg.get("encoder_name", defaults.encoder_name),
        decoder_layers=int(model_cfg.get("decoder_layers", defaults.decoder_layers)),
        decoder_hidden=int(model_cfg.get("decoder_hidden", defaults.decoder_hidden)),
        decoder_heads=int(model_cfg.get("decoder_heads", defaults.decoder_heads)),
        decoder_ffn=int(model_cfg.get("decoder_ffn", defaults.decoder_ffn)),
        decoder_dropout=float(model_cfg.get("decoder_dropout", defaults.decoder_dropout)),
        phoneme_vocab_size=int(model_cfg.get("phoneme_vocab_size", defaults.phoneme_vocab_size)),
        label_smoothing=float(model_cfg.get("label_smoothing", defaults.label_smoothing)),
        pad_id=int(model_cfg.get("pad_id", defaults.pad_id)),
        bos_id=int(model_cfg.get("bos_id", defaults.bos_id)),
        eos_id=int(model_cfg.get("eos_id", defaults.eos_id)),
        max_target_length=int(model_cfg.get("max_target_length", defaults.max_target_length)),
        max_decode_len=int(infer_cfg.get("max_decode_len", defaults.max_decode_len)),
        beam_size=int(infer_cfg.get("beam_size", 1)),
        length_penalty=float(infer_cfg.get("length_penalty", defaults.length_penalty)),
        coverage_penalty=float(infer_cfg.get("coverage_penalty", defaults.coverage_penalty)),
    )

    model = build_p_a(pa_config)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict):
        sd = state.get("model_state_dict") or state.get("model") or state
    else:
        sd = state
    load_result = model.load_state_dict(sd, strict=False)
    unexpected = list(getattr(load_result, "unexpected_keys", []) or [])
    if unexpected:
        warnings.warn(
            f"_build_pa_prediction_fn: {len(unexpected)} unexpected key(s) in checkpoint "
            f"were ignored (first 5: {unexpected[:5]}). This may indicate a config mismatch "
            "(e.g. differing phoneme_vocab_size or decoder_hidden between train and eval).",
            stacklevel=2,
        )
    model.eval()
    _pa_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(_pa_device)

    vocab: Vocab = build_default_vocab()
    tokenizer = PATokenizer(
        tokenizer_name=pa_config.encoder_name,
        phoneme_vocab=vocab,
        max_input_length=int(model_cfg.get("max_input_length", 512)),
        max_target_length=pa_config.max_target_length,
    )

    def predict(text: str):
        if not text:
            return p_a_to_canonical([], vocab)
        with torch.no_grad():
            enc = tokenizer.encode_input(text)
            input_ids = torch.tensor([enc["input_ids"]], dtype=torch.long, device=_pa_device)
            attn = torch.tensor([enc["attention_mask"]], dtype=torch.long, device=_pa_device)
            out_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attn,
                max_len=pa_config.max_decode_len,
                beam=pa_config.beam_size,
            )
            token_ids = out_ids[0].tolist()
        return p_a_to_canonical(token_ids, vocab)

    del batch_size
    return predict


def _build_pc_prediction_fn(cfg: Any, *, checkpoint: Path, batch_size: int) -> PredictionFn:
    import torch

    from modernbert_g2p.models.canonical import Vocab, build_default_vocab, p_c_to_canonical
    from modernbert_g2p.models.p_c.config import PCConfig
    from modernbert_g2p.models.p_c.model import build_p_c
    from modernbert_g2p.models.tokenization.p_c_tokenizer import PCTokenizer

    model_cfg = cfg.model if hasattr(cfg, "model") else {}
    if not isinstance(model_cfg, dict):
        model_cfg = {}
    pc_config = PCConfig(
        encoder_name=model_cfg.get("encoder_name", "tohoku-nlp/bert-base-japanese-char-v2"),
        head_variant=model_cfg.get("head_variant", "C1"),
        max_slot=int(model_cfg.get("max_slot", 8)),
        phoneme_vocab_size=int(model_cfg.get("phoneme_vocab_size", 68)),
        apbp_alpha=float(model_cfg.get("apbp_alpha", 0.2)),
        label_smoothing=float(model_cfg.get("label_smoothing", 0.05)),
    )
    model = build_p_c(pc_config)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict):
        sd = state.get("model_state_dict") or state.get("model") or state
    else:
        sd = state
    model.load_state_dict(sd, strict=False)
    model.eval()
    _pc_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(_pc_device)

    vocab: Vocab = build_default_vocab()
    tokenizer = PCTokenizer(
        tokenizer_name=pc_config.encoder_name,
        phoneme_vocab=vocab,
    )
    bio_labels = ("B", "I", "O")

    def predict(text: str):
        with torch.no_grad():
            enc = tokenizer.encode([text])
            input_ids = torch.tensor(enc["input_ids"], dtype=torch.long, device=_pc_device)
            attn = torch.tensor(enc["attention_mask"], dtype=torch.long, device=_pc_device)
            out = model(input_ids=input_ids, attention_mask=attn)
            phon_logits = out["phoneme_logits"] if "phoneme_logits" in out else out["phon_logits"]
            hl_logits = out["hl_logits"]
            apbp_logits = out["apbp_logits"]
            phon_ids = phon_logits.argmax(dim=-1)[0].tolist()
            hl_ids = hl_logits.argmax(dim=-1)[0].tolist()
            apbp_ids = apbp_logits.argmax(dim=-1)[0].tolist()
            mask = attn[0].bool().tolist()
        phon_slots: list[list[int]] = []
        hl_slots: list[list[str]] = []
        bio_tags: list[str] = []
        for i, m in enumerate(mask):
            if not m:
                continue
            phon_slots.append(list(phon_ids[i]) if isinstance(phon_ids[i], list) else [phon_ids[i]])
            hl_row = hl_ids[i] if isinstance(hl_ids[i], list) else [hl_ids[i]]
            hl_slots.append(["H" if x == 0 else "L" for x in hl_row])
            bio_tags.append(bio_labels[apbp_ids[i]] if apbp_ids[i] < len(bio_labels) else "O")
        return p_c_to_canonical(phon_slots, hl_slots, bio_tags, vocab)

    del batch_size
    return predict


def evaluate_checkpoint(
    cfg: Any,
    *,
    pilot: str,
    checkpoint: Path,
    dataset: str,
    batch_size: int = 32,
    output: Path | None = None,
    prediction_fn: PredictionFn | None = None,
    include_bootstrap: bool = True,
    bootstrap_seed: int = 42,
) -> dict[str, Any]:
    """Evaluate a trained pilot checkpoint on one of the standard datasets.

    Args:
        cfg: A :class:`~modernbert_g2p.config.Phase2Config` (or duck-typed
            equivalent) with ``.data.jsut_yaml`` / ``.data.jvs_dir`` /
            ``.data.rohan_txt`` / ``.data.hard_set_path`` set.
        pilot: One of ``"p_a"``, ``"p_c"``.
        checkpoint: Path to a checkpoint saved by
            :meth:`~modernbert_g2p.training.Trainer.save_checkpoint`.
        dataset: One of ``"jsut"``, ``"jvs"``, ``"rohan"``, ``"hardset"``.
        batch_size: Inference batch size (only consumed by the model-loading
            path, ignored when ``prediction_fn`` is provided).
        output: If given, JSON-dump the result dict to this path (with any
            missing parent directories created).
        prediction_fn: Optional pre-built ``text -> hypothesis`` function. When
            supplied, checkpoint loading is skipped entirely — this is the
            recommended path for unit tests and for callers that already own a
            constructed inference wrapper.
        include_bootstrap: Forwarded to the underlying scorer.
        bootstrap_seed: Forwarded to the underlying scorer.

    Returns:
        For ``jsut``/``jvs``/``rohan``: the standard :func:`run_eval` payload.
        For ``hardset``: ``{"by_category": {...}, "aggregate": {...}}`` where
        ``aggregate`` is the pooled cross-category summary. In both cases
        ``pilot`` / ``dataset`` / ``checkpoint`` metadata fields are attached
        at the top level for downstream table assembly.

    Raises:
        ValueError: If ``pilot`` / ``dataset`` are unknown, if the required
            path is missing from ``cfg.data``, or if the checkpoint file does
            not exist.
        NotImplementedError: If ``prediction_fn`` is ``None`` and the
            pilot-specific automatic inference wiring has not landed yet.
    """
    if pilot not in _VALID_PILOTS:
        raise ValueError(
            f"Unknown pilot={pilot!r}. Must be one of {sorted(_VALID_PILOTS)}"
        )
    if dataset not in _VALID_DATASETS:
        raise ValueError(
            f"Unknown dataset={dataset!r}. Must be one of {sorted(_VALID_DATASETS)}"
        )
    checkpoint = Path(checkpoint)
    if prediction_fn is None and not checkpoint.exists():
        raise ValueError(f"checkpoint not found: {checkpoint}")

    dataset_path = _resolve_dataset_path(cfg, dataset)

    if prediction_fn is None:
        prediction_fn = _build_prediction_fn_from_checkpoint(
            cfg,
            pilot=pilot,
            checkpoint=checkpoint,
            batch_size=batch_size,
        )

    if dataset == "jsut":
        result: dict[str, Any] = score_jsut(
            prediction_fn,
            dataset_path,
            include_bootstrap=include_bootstrap,
            bootstrap_seed=bootstrap_seed,
        )
    elif dataset == "jvs":
        result = score_jvs(
            prediction_fn,
            dataset_path,
            include_bootstrap=include_bootstrap,
            bootstrap_seed=bootstrap_seed,
        )
    elif dataset == "rohan":
        result = score_rohan(
            prediction_fn,
            dataset_path,
            include_bootstrap=include_bootstrap,
            bootstrap_seed=bootstrap_seed,
        )
    else:
        by_category = score_hardset(
            prediction_fn,
            dataset_path,
            include_bootstrap=False,
            bootstrap_seed=bootstrap_seed,
        )
        result = {
            "by_category": by_category,
            "aggregate": _hardset_aggregate_summary(by_category),
        }

    result["pilot"] = pilot
    result["dataset"] = dataset
    result["checkpoint"] = str(checkpoint)

    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, sort_keys=True)

    return result


_METRIC_NAME_BY_DATASET: dict[str, str] = {
    "jsut": "PER",
    "jvs": "CER",
    "rohan": "KER",
    "hardset": "PER",
}


def _iter_eval_json_files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.glob("eval_*.json") if p.is_file())


def _format_pct(value: float) -> str:
    return f"{value:.3f}"


def _pilot_label(pilot: Any) -> str:
    if isinstance(pilot, str) and pilot.startswith("p_"):
        return "P-" + pilot.split("_", 1)[1].upper()
    return str(pilot)


def build_comparison_table(directories: Sequence[Path]) -> str:
    """Aggregate per-pilot eval JSONs from ``directories`` into a Markdown table.

    Each directory is scanned for ``eval_<dataset>.json`` files written by
    :func:`evaluate_checkpoint` (or any caller that follows the same JSON
    schema). The output is a single Markdown table with columns:

    ``| Pilot | Dataset | N | Metric | Value (%) | 95% CI (%) |``

    Missing / unreadable JSON files are skipped with a stderr note but do not
    fail the call. When ``directories`` contains no eval JSONs at all, a
    stub table with the header only is returned so downstream tooling can
    still write it.

    Args:
        directories: List of per-pilot checkpoint directories.

    Returns:
        Markdown string (newline-terminated).
    """
    header = "| Pilot | Dataset | N | Metric | Value (%) | 95% CI (%) |"
    separator = "| --- | --- | ---: | --- | ---: | --- |"
    rows: list[str] = []

    for directory in directories:
        directory = Path(directory)
        if not directory.is_dir():
            continue
        for json_path in _iter_eval_json_files(directory):
            try:
                with json_path.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            aggregate = payload.get("aggregate") or {}
            n_rows = int(aggregate.get("n_rows", 0))
            per_micro = float(aggregate.get("per_micro", 0.0))
            dataset = payload.get("dataset") or json_path.stem.replace("eval_", "", 1)
            metric_name = _METRIC_NAME_BY_DATASET.get(str(dataset), "PER")
            pilot = _pilot_label(payload.get("pilot") or directory.name)
            bootstrap = payload.get("bootstrap") or {}
            if "lo" in bootstrap and "hi" in bootstrap:
                ci_str = (
                    f"[{_format_pct(float(bootstrap['lo']))}, "
                    f"{_format_pct(float(bootstrap['hi']))}]"
                )
            else:
                ci_str = "n/a"
            rows.append(
                f"| {pilot} | {dataset} | {n_rows} | {metric_name} | "
                f"{_format_pct(per_micro)} | {ci_str} |"
            )

    lines = [header, separator, *rows]
    return "\n".join(lines) + "\n"


__all__ = [
    "build_comparison_table",
    "evaluate_checkpoint",
    "run_eval",
    "run_eval_p_a",
    "run_eval_p_c",
    "score_hardset",
    "score_jsut",
    "score_jvs",
    "score_rohan",
]
