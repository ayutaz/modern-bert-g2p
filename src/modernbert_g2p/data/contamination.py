"""4-layer contamination defense against JSUT/JVS/ROHAN/hard-set leakage.

Implements docs/design/phase1_data_pipeline.md §6.2.

Layer semantics (checked in order; first hit wins):
    L1 exact-set match after ``normalize_l1`` (NFKC + case-fold + punct strip)
    L2 single-hit char n-gram bloom filter on the normalized text
    L3 single-hit phoneme n-gram bloom filter (requires phonemes argument)
    L4 cosine similarity vs held-out sentence embeddings (optional,
        gracefully disabled when ``sentence_transformers`` is unavailable)

The L2/L3 policy diverges from ``docs/design/phase1_data_pipeline.md`` §6.2
which specifies a 3-hit threshold: for unit-test determinism the pure-Python
bloom filter here uses 1-hit with a low ``bloom_fp_rate``. The design-level
3-hit threshold is applied by the caller when replaying against the full
release-time corpus.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

_FNV_OFFSET_A = 0xCBF29CE484222325
_FNV_OFFSET_B = 0x84222325CBF29CE4
_FNV_PRIME = 0x100000001B3
_MASK64 = (1 << 64) - 1


def _fnv1a_64(data: bytes, offset: int) -> int:
    h = offset
    for b in data:
        h ^= b
        h = (h * _FNV_PRIME) & _MASK64
    return h


class BloomFilter:
    """Pure-Python bloom filter with FNV-1a double-hashing (Kirsch-Mitzenmacher)."""

    def __init__(self, capacity: int, fp_rate: float) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if not 0 < fp_rate < 1:
            raise ValueError("fp_rate must be in (0, 1)")
        self._capacity = capacity
        self._fp_rate = fp_rate
        bit_size = math.ceil(-capacity * math.log(fp_rate) / (math.log(2) ** 2))
        bit_size = max(8, ((bit_size + 7) // 8) * 8)
        self._bit_size = bit_size
        self._k = max(1, round((bit_size / capacity) * math.log(2)))
        self._bits = bytearray(bit_size // 8)
        self._count = 0

    def _indices(self, key: bytes) -> Iterable[int]:
        h1 = _fnv1a_64(key, _FNV_OFFSET_A)
        h2 = _fnv1a_64(key, _FNV_OFFSET_B) or 1
        m = self._bit_size
        for i in range(self._k):
            yield (h1 + i * h2) % m

    def add(self, key: bytes) -> None:
        for idx in self._indices(key):
            self._bits[idx >> 3] |= 1 << (idx & 7)
        self._count += 1

    def __contains__(self, key: bytes) -> bool:
        return all(
            self._bits[idx >> 3] & (1 << (idx & 7))
            for idx in self._indices(key)
        )

    @property
    def size(self) -> int:
        return self._count

    @property
    def bit_size(self) -> int:
        return self._bit_size


_PUNCT_CHARS = (
    "、。「」『』・…！？"
    "（）【】〈〉《》〜～"
    "，．：；／"
    "‐‑‒–—―─　"
    "!?,.:;()[]{}/~-"
)
_PUNCT_RE = re.compile("[" + re.escape(_PUNCT_CHARS) + "]")
_WS_RE = re.compile(r"\s+")


def normalize_l1(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.casefold()
    text = _PUNCT_RE.sub("", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


Encoder = Callable[[Sequence[str]], list[list[float]]]


class ContaminationFilter:
    """4-layer contamination filter (L1 exact / L2 char / L3 phoneme / L4 embedding)."""

    def __init__(
        self,
        *,
        char_ngram: int = 8,
        phoneme_ngram: int = 6,
        bloom_capacity: int = 5_000_000,
        bloom_fp_rate: float = 1e-6,
        embedding_threshold: float = 0.85,
        embedding_model_name: str = (
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        ),
    ) -> None:
        if char_ngram < 1:
            raise ValueError("char_ngram must be >= 1")
        if phoneme_ngram < 1:
            raise ValueError("phoneme_ngram must be >= 1")
        self._char_ngram = char_ngram
        self._phoneme_ngram = phoneme_ngram
        self._embedding_threshold = embedding_threshold
        self._embedding_model_name = embedding_model_name
        self._l1_set: set[str] = set()
        self._l2_bloom = BloomFilter(bloom_capacity, bloom_fp_rate)
        self._l3_bloom = BloomFilter(bloom_capacity, bloom_fp_rate)
        self._encoder: Encoder | None = None
        self._encoder_tried = False
        self._encoder_failed = False
        self._held_out_embeddings: list[list[float]] = []
        self._stats: dict[str, int] = {
            "held_out_count": 0,
            "queries": 0,
            "l1_hits": 0,
            "l2_hits": 0,
            "l3_hits": 0,
            "l4_hits": 0,
            "l4_skipped": 0,
        }

    def _ensure_encoder(self) -> Encoder | None:
        if self._encoder is not None:
            return self._encoder
        if self._encoder_failed:
            return None
        if self._encoder_tried:
            return None
        self._encoder_tried = True
        try:
            from sentence_transformers import SentenceTransformer
        except (ImportError, RuntimeError, OSError):
            self._encoder_failed = True
            return None
        try:
            model = SentenceTransformer(self._embedding_model_name)
        except (ImportError, RuntimeError, OSError):
            self._encoder_failed = True
            return None

        def encode(texts: Sequence[str]) -> list[list[float]]:
            vecs = model.encode(list(texts))
            return [list(map(float, v)) for v in vecs]

        self._encoder = encode
        return self._encoder

    def _char_ngrams(self, normalized: str) -> Iterable[bytes]:
        n = self._char_ngram
        if len(normalized) < n:
            return
        for i in range(len(normalized) - n + 1):
            yield normalized[i : i + n].encode("utf-8")

    def _phoneme_ngrams(self, phonemes: Sequence[str]) -> Iterable[bytes]:
        n = self._phoneme_ngram
        if len(phonemes) < n:
            return
        for i in range(len(phonemes) - n + 1):
            yield "|".join(phonemes[i : i + n]).encode("utf-8")

    def add_held_out(
        self,
        texts: Iterable[str],
        phonemes: Iterable[Iterable[str]] | None = None,
    ) -> None:
        text_list = list(texts)
        for text in text_list:
            normalized = normalize_l1(text)
            if normalized:
                self._l1_set.add(normalized)
            for gram in self._char_ngrams(normalized):
                self._l2_bloom.add(gram)
            self._stats["held_out_count"] += 1

        if phonemes is not None:
            for ph_seq in phonemes:
                ph_list = list(ph_seq)
                if not ph_list:
                    continue
                for gram in self._phoneme_ngrams(ph_list):
                    self._l3_bloom.add(gram)

        encoder = self._ensure_encoder()
        if encoder is not None and text_list:
            vecs = encoder(text_list)
            self._held_out_embeddings.extend(vecs)

    def is_contaminated(
        self,
        text: str,
        phonemes: Iterable[str] | None = None,
    ) -> tuple[bool, str]:
        self._stats["queries"] += 1
        normalized = normalize_l1(text)
        if normalized and normalized in self._l1_set:
            self._stats["l1_hits"] += 1
            return True, "L1"
        for gram in self._char_ngrams(normalized):
            if gram in self._l2_bloom:
                self._stats["l2_hits"] += 1
                return True, "L2"
        if phonemes is not None:
            ph_list = list(phonemes)
            for gram in self._phoneme_ngrams(ph_list):
                if gram in self._l3_bloom:
                    self._stats["l3_hits"] += 1
                    return True, "L3"
        encoder = self._ensure_encoder()
        if encoder is None or not self._held_out_embeddings:
            self._stats["l4_skipped"] += 1
            return False, ""
        query_vec = encoder([text])[0]
        for held in self._held_out_embeddings:
            if _cosine(query_vec, held) >= self._embedding_threshold:
                self._stats["l4_hits"] += 1
                return True, "L4"
        return False, ""

    def stats(self) -> dict[str, int]:
        return dict(self._stats)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)


def _strip_parens(text: str) -> str:
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


def _load_jsut(path: Path) -> tuple[list[str], list[list[str]]]:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    texts: list[str] = []
    phonemes: list[list[str]] = []
    for key in data:
        entry = data[key] or {}
        text = entry.get("text_level2") or entry.get("text_level0") or ""
        if not text:
            continue
        texts.append(text)
        phone_str = entry.get("phone_level3", "")
        phonemes.append(phone_str.split("-") if phone_str else [])
    return texts, phonemes


def _load_rohan(path: Path) -> tuple[list[str], list[list[str]]]:
    texts: list[str] = []
    phonemes: list[list[str]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line:
                continue
            id_sep = line.find(":")
            if id_sep < 0:
                continue
            pair = line[id_sep + 1 :]
            comma = pair.find(",")
            if comma < 0:
                continue
            texts.append(_strip_parens(pair[:comma]))
            phonemes.append([])
    return texts, phonemes


def _load_jvs(path: Path) -> tuple[list[str], list[list[str]]]:
    texts: list[str] = []
    phonemes: list[list[str]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n").strip()
            if not line or line.startswith("#"):
                continue
            texts.append(line)
            phonemes.append([])
    return texts, phonemes


def _load_hard_set(path: Path) -> tuple[list[str], list[list[str]]]:
    texts: list[str] = []
    phonemes: list[list[str]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("text", "")
            if not text:
                continue
            texts.append(text)
            ph = obj.get("phonemes")
            phonemes.append(list(ph) if ph else [])
    return texts, phonemes


def load_held_out_texts(
    jsut_yaml: Path | None,
    rohan_txt: Path | None,
    jvs_txt: Path | None,
    hard_set_jsonl: Path | None,
) -> tuple[list[str], list[list[str]] | None]:
    """Load held-out corpus texts (and phonemes when available) from any subset of 4 sources."""
    all_texts: list[str] = []
    all_phonemes: list[list[str]] = []
    any_phonemes = False

    for path, loader in [
        (jsut_yaml, _load_jsut),
        (rohan_txt, _load_rohan),
        (jvs_txt, _load_jvs),
        (hard_set_jsonl, _load_hard_set),
    ]:
        if path is None:
            continue
        if not path.exists():
            continue
        texts, phonemes = loader(path)
        all_texts.extend(texts)
        all_phonemes.extend(phonemes)
        if any(p for p in phonemes):
            any_phonemes = True

    return all_texts, all_phonemes if any_phonemes else None


__all__ = [
    "BloomFilter",
    "ContaminationFilter",
    "Encoder",
    "load_held_out_texts",
    "normalize_l1",
]
