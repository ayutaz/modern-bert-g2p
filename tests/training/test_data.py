"""Tests for the Phase 2 training data loader and pilot collators (Track T6)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from modernbert_g2p.training.data import (
    G2PDataset,
    PACollator,
    PCCollator,
    make_dummy_row,
)


class _FakePATokenizer:
    """Duck-typed stand-in for T2's PATokenizer.

    ``encode`` produces one input id per character, ``encode_target`` returns
    ``[bos] + phoneme_ids + [eos]`` where each phoneme is mapped to a
    deterministic integer via ord() hashing.
    """

    pad_token_id: int = 0
    bos_id: int = 1
    eos_id: int = 2

    def encode(self, texts: list[str]) -> dict[str, Any]:
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        for t in texts:
            ids = [100 + ord(ch) % 5000 for ch in t]
            input_ids.append(ids)
            attention_mask.append([1] * len(ids))
        return {"input_ids": input_ids, "attention_mask": attention_mask, "extras": {}}

    def encode_target(self, canonical: Any) -> list[int]:
        phon = canonical.phonemes
        return [self.bos_id] + [200 + (hash(p) % 500) for p in phon] + [self.eos_id]


class _FakePCTokenizer:
    """Duck-typed stand-in for T2's PCTokenizer (1 char = 1 token)."""

    pad_token_id: int = 0
    max_slot: int = 8

    def encode(self, texts: list[str]) -> dict[str, Any]:
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        char_positions: list[list[int]] = []
        for t in texts:
            ids = [7]
            for ch in t:
                ids.append(50 + ord(ch) % 400)
            ids.append(8)
            input_ids.append(ids)
            attention_mask.append([1] * len(ids))
            char_positions.append(list(range(1, 1 + len(t))))
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "extras": {"char_positions": char_positions},
        }

    def encode_target(
        self,
        phoneme_slots: list[list[str]],
        hl_slots: list[list[str]],
        bio: list[str],
    ) -> dict[str, list[list[int]] | list[int]]:
        return {"phon": [[0] * self.max_slot for _ in phoneme_slots], "bio": [0 for _ in bio]}


def _torch() -> Any:
    return pytest.importorskip("torch")


def test_make_dummy_row_defaults_are_valid() -> None:
    from modernbert_g2p.data.schema import validate_row

    row = make_dummy_row()
    validate_row(row)
    assert row.text == "こんにちは"
    assert row.phonemes == ("k", "o", "N", "n", "i", "ch", "i", "h", "a")
    assert row.mora_accents == ("L", "H", "H", "H", "L")
    assert row.category == "general"
    assert row.sample_weight == 1.0


def test_g2pdataset_from_list_of_rows() -> None:
    rows = [
        make_dummy_row(text="桜", phonemes=("s", "a", "k", "u", "r", "a"), mora_accents=("L", "H", "H")),
        make_dummy_row(text="コーヒー", phonemes=("k", "o", "o", "h", "i", "i"), mora_accents=("H", "L", "L", "L")),
        make_dummy_row(),
    ]
    ds = G2PDataset(rows)
    assert len(ds) == 3
    assert ds[0].text == "桜"
    assert ds[1].mora_accents == ("H", "L", "L", "L")
    assert list(ds)[2].text == "こんにちは"


def test_g2pdataset_from_list_type_check() -> None:
    with pytest.raises(TypeError):
        G2PDataset([make_dummy_row(), "not-a-row"])  # type: ignore[list-item]


def test_g2pdataset_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        G2PDataset(tmp_path / "nope.jsonl")


def test_g2pdataset_unknown_suffix(tmp_path: Path) -> None:
    p = tmp_path / "corpus.txt"
    p.write_text("ignored", encoding="utf-8")
    with pytest.raises(ValueError):
        G2PDataset(p)


def test_g2pdataset_from_jsonl(tmp_path: Path) -> None:
    from modernbert_g2p.data.output import write_jsonl

    rows = [
        make_dummy_row(id_="row-a"),
        make_dummy_row(
            id_="row-b",
            text="桜",
            phonemes=("s", "a", "k", "u", "r", "a"),
            mora_accents=("L", "H", "H"),
        ),
    ]
    path = tmp_path / "corpus.jsonl"
    write_jsonl(rows, path)
    ds = G2PDataset(path)
    assert len(ds) == 2
    assert ds[0].id == "row-a"
    assert ds[1].id == "row-b"


def test_g2pdataset_from_parquet_roundtrip(tmp_path: Path) -> None:
    pq = pytest.importorskip("pyarrow.parquet")
    from modernbert_g2p.data.output import HAS_PYARROW, write_parquet

    if not HAS_PYARROW:
        pytest.skip("pyarrow not available")

    rows = [
        make_dummy_row(id_="parq-1"),
        make_dummy_row(id_="parq-2", text="桜"),
    ]
    path = tmp_path / "corpus.parquet"
    write_parquet(rows, path)
    assert pq.ParquetFile(str(path)).metadata.num_rows == 2

    ds = G2PDataset(path)
    assert len(ds) == 2
    assert {r.id for r in ds} == {"parq-1", "parq-2"}


def test_pa_collator_batches_two_rows() -> None:
    torch = _torch()
    tok = _FakePATokenizer()
    coll = PACollator(tok)
    batch = [
        make_dummy_row(id_="a", text="桜", phonemes=("s", "a", "k", "u"), mora_accents=("L", "H")),
        make_dummy_row(id_="b"),
    ]
    out = coll(batch)
    for key in (
        "input_ids",
        "attention_mask",
        "decoder_input_ids",
        "decoder_attention_mask",
        "labels",
        "sample_weights",
        "ids",
    ):
        assert key in out, f"missing key {key}"
    assert isinstance(out["input_ids"], torch.Tensor)
    assert out["input_ids"].dtype == torch.long
    assert out["input_ids"].shape[0] == 2
    assert out["attention_mask"].shape == out["input_ids"].shape
    assert out["decoder_input_ids"].shape[0] == 2
    assert out["decoder_attention_mask"].shape == out["decoder_input_ids"].shape
    assert out["labels"].shape == out["decoder_input_ids"].shape
    assert out["sample_weights"].dtype == torch.float32
    assert out["ids"] == ["a", "b"]


def test_pa_collator_padding_and_multiple_of() -> None:
    _torch()
    tok = _FakePATokenizer()
    coll = PACollator(tok, pad_to_multiple_of=8)
    batch = [
        make_dummy_row(id_="short", text="桜"),
        make_dummy_row(id_="long", text="こんにちは世界"),
    ]
    out = coll(batch)
    assert out["input_ids"].shape[1] % 8 == 0
    assert out["input_ids"].shape[1] >= max(len("桜"), len("こんにちは世界"))
    assert (out["attention_mask"][0].sum().item() == len("桜"))
    assert (out["attention_mask"][1].sum().item() == len("こんにちは世界"))


def test_pa_collator_labels_pad_id() -> None:
    _torch()
    tok = _FakePATokenizer()
    coll = PACollator(tok, pad_to_multiple_of=None, label_pad_id=-100)
    batch = [
        make_dummy_row(id_="a", phonemes=("k", "o", "N")),
        make_dummy_row(id_="b", phonemes=("k", "o", "N", "n", "i", "ch", "i", "h", "a")),
    ]
    out = coll(batch)
    assert (out["labels"][0][-1].item() == -100)
    assert (out["labels"][1] != -100).all().item()


def test_pa_collator_sample_weights_carried() -> None:
    torch = _torch()
    tok = _FakePATokenizer()
    coll = PACollator(tok)
    rows = [
        make_dummy_row(id_="w1", sample_weight=1.5, category="loanword"),
        make_dummy_row(id_="w2", sample_weight=2.5, category="loanword"),
    ]
    out = coll(rows)
    assert torch.allclose(out["sample_weights"], torch.tensor([1.5, 2.5], dtype=torch.float32))


def test_pa_collator_empty_batch_raises() -> None:
    _torch()
    tok = _FakePATokenizer()
    coll = PACollator(tok)
    with pytest.raises(ValueError):
        coll([])


def test_pa_collator_preserves_batch_order() -> None:
    _torch()
    tok = _FakePATokenizer()
    coll = PACollator(tok)
    ids = [f"row-{i}" for i in range(5)]
    batch = [make_dummy_row(id_=i) for i in ids]
    out = coll(batch)
    assert out["ids"] == ids


def test_pc_collator_produces_slot_labels() -> None:
    torch = _torch()
    tok = _FakePCTokenizer()
    coll = PCCollator(tok)
    batch = [
        make_dummy_row(id_="c1", text="桜", phonemes=("s", "a", "k", "u", "r", "a"), mora_accents=("L", "H", "H")),
        make_dummy_row(id_="c2"),
    ]
    out = coll(batch)
    assert isinstance(out["phoneme_labels"], torch.Tensor)
    assert out["phoneme_labels"].dtype == torch.long
    assert out["phoneme_labels"].dim() == 3
    batch_size, seq_len, slot = out["phoneme_labels"].shape
    assert batch_size == 2
    assert slot == coll.max_slot == 8
    assert out["hl_labels"].shape == (batch_size, seq_len, slot)
    assert out["apbp_labels"].shape == (batch_size, seq_len)
    assert out["apbp_labels"].max().item() <= 2


def test_pc_collator_empty_batch_raises() -> None:
    _torch()
    with pytest.raises(ValueError):
        PCCollator(_FakePCTokenizer())([])


def test_data_module_does_not_import_torch_at_module_load() -> None:
    """Loading ``modernbert_g2p.training.data`` must not drag torch into sys.modules.

    Run in a subprocess so we do not perturb torch state in the parent
    interpreter (torch's C extensions do not tolerate re-imports).
    """
    import subprocess
    import sys as _sys

    script = (
        "import sys, importlib; "
        "importlib.import_module('modernbert_g2p.training.data'); "
        "assert 'torch' not in sys.modules, sorted(m for m in sys.modules if 'torch' in m)"
    )
    proc = subprocess.run(
        [_sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"


def test_g2pdataset_hardset_fixture_incompat(tmp_path: Path) -> None:
    """The bundled hard-set fixture uses a Phase-1-tests key layout that
    intentionally does NOT match Row's field names; loading it MUST fail
    with a clear error so the caller adopts the canonical Row schema."""
    fixture = Path(__file__).resolve().parents[1] / "data" / "fixtures" / "tiny_hardset.jsonl"
    if not fixture.exists():
        pytest.skip("tiny_hardset fixture missing")
    with pytest.raises(ValueError):
        G2PDataset(fixture)


def test_g2pdataset_from_jsonl_fixture_reused(tmp_path: Path) -> None:
    """Emit a Row-schema-compliant JSONL and reload it."""
    from modernbert_g2p.data.output import write_jsonl

    rows = [
        make_dummy_row(id_="f1"),
        make_dummy_row(
            id_="f2",
            text="AI",
            phonemes=("e", "e", "a", "i"),
            mora_accents=("H", "L", "H", "L"),
            accent_boundaries=(2,),
            category="english_abbreviation",
        ),
    ]
    p = tmp_path / "row_schema.jsonl"
    write_jsonl(rows, p)

    contents = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()]
    assert contents[1]["accent_boundaries"] == [2]
    assert contents[1]["category"] == "english_abbreviation"

    ds = G2PDataset(p)
    assert len(ds) == 2
    assert ds[1].accent_boundaries == (2,)


def test_pa_collator_encode_target_min_length_guard() -> None:
    _torch()

    class _BadTokenizer(_FakePATokenizer):
        def encode_target(self, canonical: Any) -> list[int]:
            return [self.bos_id]

    with pytest.raises(ValueError):
        PACollator(_BadTokenizer())([make_dummy_row()])


class _PAEncodeInputTokenizer:
    """T2-shape stand-in: single-text ``encode_input`` + dict ``encode_target``.

    Mirrors the real :class:`PATokenizer` API where ``encode_target`` returns
    a same-length ``{decoder_input_ids, labels}`` pair (already shifted).
    """

    pad_token_id: int = 0
    bos_id: int = 1
    eos_id: int = 2

    def encode_input(self, text: str) -> dict[str, list[int]]:
        ids = [100 + ord(ch) % 5000 for ch in text]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def encode_target(self, canonical: Any) -> dict[str, list[int]]:
        phon = canonical.phonemes
        stream = [self.bos_id] + [200 + (hash(p) % 500) for p in phon]
        labels = stream[1:] + [self.eos_id]
        return {"decoder_input_ids": stream, "labels": labels}


def test_pa_collator_supports_encode_input_and_dict_target() -> None:
    torch = _torch()
    tok = _PAEncodeInputTokenizer()
    coll = PACollator(tok, pad_to_multiple_of=None)
    batch = [
        make_dummy_row(id_="ei-a", text="桜"),
        make_dummy_row(id_="ei-b"),
    ]
    out = coll(batch)
    assert isinstance(out["input_ids"], torch.Tensor)
    assert out["input_ids"].shape[0] == 2
    assert out["decoder_input_ids"].shape == out["labels"].shape
    assert out["decoder_input_ids"].shape[0] == 2
    assert out["ids"] == ["ei-a", "ei-b"]


class _PCEncodeInputTokenizer:
    """T2-shape P-C stand-in: single-text ``encode_input`` with top-level char_positions."""

    pad_token_id: int = 0
    max_slot: int = 8

    def encode_input(self, text: str) -> dict[str, Any]:
        ids = [7]
        for ch in text:
            ids.append(50 + ord(ch) % 400)
        ids.append(8)
        return {
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
            "char_positions": list(range(1, 1 + len(text))),
        }


def test_pc_collator_supports_encode_input_top_level_char_positions() -> None:
    torch = _torch()
    tok = _PCEncodeInputTokenizer()
    coll = PCCollator(tok)
    batch = [
        make_dummy_row(id_="pc-a", text="桜"),
        make_dummy_row(id_="pc-b"),
    ]
    out = coll(batch)
    assert isinstance(out["phoneme_labels"], torch.Tensor)
    assert out["phoneme_labels"].shape[0] == 2
    assert out["phoneme_labels"].shape[2] == coll.max_slot


def test_pc_collator_defaults_label_pad_to_ignore_index() -> None:
    _torch()
    coll = PCCollator(_FakePCTokenizer())
    assert coll.phoneme_pad_id == -100
    assert coll.hl_pad_id == -100
    assert coll.apbp_pad_id == -100


def test_pc_collator_empty_slot_labeled_with_pad_class_zero() -> None:
    """Empty phoneme slots at valid char positions must be labeled with class 0 (pad),
    NOT with -100 (ignore_index).

    Regression guard: the earlier bug shipped empty slots as ignore_index so the
    phoneme head never learned to predict pad at inference, producing PER > 100%.
    Now each char position must contain at least one pad-class-0 label whenever
    the real phoneme count is < max_slot.
    """
    torch = _torch()
    tok = _FakePCTokenizer()
    coll = PCCollator(tok)
    # A single char (桜) with 6 phonemes → slots [s,a,k,u,r,a, 0, 0] (last 2 are pad-class-0)
    batch = [
        make_dummy_row(id_="pc-slot", text="桜", phonemes=("s", "a", "k", "u", "r", "a")),
    ]
    out = coll(batch)
    labels = out["phoneme_labels"]  # (batch, seq_len, slot)
    # Find the character position — extras.char_positions[0] gives the char-to-token map
    extras = out.get("extras") or {}
    char_positions_all = extras.get("char_positions") or []
    if not char_positions_all:
        char_positions_all = [tok.encode([r.text for r in batch])["extras"]["char_positions"][0]]
    char_slot_labels = labels[0, char_positions_all[0][0], :].tolist()
    # First 6 slots have real phonemes (not pad, not -100), remainder must be 0 (pad_id)
    assert -100 not in char_slot_labels, (
        f"empty slots must be class 0 (pad_id), not -100 ignore_index; got {char_slot_labels}"
    )
    assert char_slot_labels[-1] == 0, (
        f"last (empty) slot must be class 0 (pad_id); got {char_slot_labels}"
    )
