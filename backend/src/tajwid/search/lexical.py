"""Lexical scoring (BM25) fused with vector search for the Arabic path.

Two lexical signals per āyah, both needed:

  * SURFACE tokens (lightly normalised words) — exact-phrase precision. A query that is an
    āyah fragment ("ولا يغتب بعضكم بعضا") must return that āyah: يغتب is essentially unique to
    49:12, so a surface match pins it. The tafsīr-augmented VECTOR can't (the long tafsīr
    dilutes the exact phrase), which is why lexical is not optional.
  * ROOT tokens (gold Quranic morphology) — morphological recall. A query noun (الغيبة) matches
    an āyah's verb form (يغتب) because both carry root غيب, which surface matching misses.

Query rooting needs no stemmer dependency: the corpus itself bridges surface -> root (الغيب
roots الغيبة -> غيب; نميم roots النميمة -> نمم), built into a normalised-surface -> gold-root map.

Surface and root tokens live in one BM25 bag; idf down-weights common tokens (بعض, particles)
and up-weights rare, decisive ones (يغتب). Lexical covers the āyah text only — the tafsīr is
already in the vector index, and keeping it out of the lexical bag preserves exact-match
precision.

# ponytail: O(N) BM25 scan over 6236 āyāt in Python, ~10ms/query. Fine at this scale.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import corpus
from .corpus import DATA_DIR
from .embeddings import strip_diacritics

_PROCLITICS = ["وال", "فال", "بال", "كال", "ال", "و", "ف", "ب", "ك", "ل"]


def _letters(w: str) -> str:
    """Diacritics off + alef/ya/hamza unified. Keeps clitics — this is the surface form, so
    بعضكم stays distinct from بعض (exact-phrase precision)."""
    w = strip_diacritics(w)
    for a, b in (("أ", "ا"), ("إ", "ا"), ("آ", "ا"), ("ى", "ي"), ("ؤ", "و"), ("ئ", "ي"), ("ـ", "")):
        w = w.replace(a, b)
    return w


def _stem(w: str) -> str:
    """Surface form with a leading proclitic and trailing ة dropped — the key into the
    surface->root map (so الغيبة and غيب land together)."""
    w = _letters(w)
    for p in _PROCLITICS:
        if w.startswith(p) and len(w) > len(p) + 1:
            w = w[len(p):]
            break
    return w[:-1] if w.endswith("ة") else w


# Surface tokens are prefixed so they never collide with root tokens in the shared bag.
def _surf(tok: str) -> str:
    return "s:" + tok


def _rt(tok: str) -> str:
    return "r:" + tok


@lru_cache(maxsize=1)
def _model():
    """Build once: (docs aligned to ids order, idf, avgdl, stem->root map)."""
    ids = [tuple(x) for x in json.loads((DATA_DIR / "ids.json").read_text(encoding="utf-8"))]
    stem_root: dict[str, Counter] = defaultdict(Counter)
    surf: dict[tuple[int, int], list[str]] = defaultdict(list)
    roots: dict[tuple[int, int], list[str]] = defaultdict(list)
    for r in corpus.connect().execute("SELECT sura, aya, uthmani, root FROM words"):
        key = (r["sura"], r["aya"])
        surf[key].append(_surf(_letters(r["uthmani"])))
        if r["root"]:
            stem_root[_stem(r["uthmani"])][r["root"]] += 1
            roots[key].append(_rt(r["root"]))
    stem2root = {k: v.most_common(1)[0][0] for k, v in stem_root.items()}

    docs = [surf.get(a, []) + roots.get(a, []) for a in ids]
    n = len(docs)
    avgdl = sum(len(d) for d in docs) / n if n else 0.0
    df: Counter = Counter()
    for d in docs:
        df.update(set(d))
    idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
    return docs, idf, avgdl, stem2root


def query_tokens(query: str) -> list[str]:
    """Surface tokens + root tokens for the query — the same two spaces the docs index."""
    _, _, _, stem2root = _model()
    toks: list[str] = []
    for w in query.split():
        if not w.strip():
            continue
        toks.append(_surf(_letters(w)))
        toks.append(_rt(stem2root.get(_stem(w), _stem(w))))
    return toks


# Kept for tests/back-compat: the roots a query maps to (root tokens, unprefixed).
def query_roots(query: str) -> list[str]:
    _, _, _, stem2root = _model()
    return [stem2root.get(_stem(w), _stem(w)) for w in query.split() if w.strip()]


def scores(query: str, k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    """BM25 score per āyah, aligned to ids.json order. Zeros if the query has no tokens."""
    docs, idf, avgdl, _ = _model()
    qset = set(query_tokens(query))
    out = np.zeros(len(docs), dtype="float32")
    if not qset or avgdl == 0:
        return out
    for i, d in enumerate(docs):
        if not d:
            continue
        dl = len(d)
        tf = Counter(t for t in d if t in qset)
        s = 0.0
        for t, f in tf.items():
            s += idf.get(t, 0.0) * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl))
        out[i] = s
    return out


def reset() -> None:
    _model.cache_clear()
