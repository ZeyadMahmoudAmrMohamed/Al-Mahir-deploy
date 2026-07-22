"""Āyah search: retrieval mode × query expansion, as two orthogonal knobs.

**mode** — where the score comes from:

  ``keyword``  BM25 over the āyah's Uthmani words AND their gold Quranic roots. No model,
               no download, ~10 ms. Arabic only (the lexical bag is built from Arabic).
  ``vector``   Cosine against a FAISS IndexFlatIP of BGE-M3 embeddings. The indexed text is
               the āyah PLUS its Muyassar tafsīr — measured Recall@10 0.393 with tafsīr vs
               0.264 without, because a bare āyah is too short to embed a topic. The tafsīr
               is embed-only; only the āyah is ever displayed.
  ``hybrid``   ``cosine + alpha * (bm25 / bm25.max())``. The default, and the measured best:
               the vector finds meaning, BM25 pins exact wording. An āyah fragment like
               "ولا يغتب بعضكم بعضا" ranks its own āyah #1 only because of the lexical term —
               the long tafsīr in the vector dilutes the exact phrase.

**hyde** — whether the query is rewritten before embedding (see hyde.py). Orthogonal to
mode by construction: it changes what gets EMBEDDED, not how scores combine. Meaningless
for ``keyword`` (there is nothing to embed), so it is ignored there and the response says
so rather than pretending it ran.

English (``lang=en``) has no lexical bag, so ``hybrid`` there IS ``vector``. The response
reports the mode that actually ran — a silently degraded request is a lie you debug twice.

Scores are returned RAW, with no relevance threshold: we have no data to calibrate one and
inventing a cutoff is false confidence. The client decides how to render a weak hit.
"""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from ..config import get_settings
from . import corpus, lexical
from .corpus import DATA_DIR
from .embeddings import encode, strip_diacritics
from .models import SearchHit

# Languages with their own index. A declared language we don't index falls back to ar.
INDEXED_LANGS = {"ar", "en"}
MODES = ("keyword", "vector", "hybrid")


class IndexMissingError(RuntimeError):
    pass


@dataclass
class SearchResult:
    """Hits plus what actually ran — mode/lang/hyde may differ from what was asked."""

    hits: list[SearchHit]
    matched_lang: str
    mode: str
    hyde_used: bool


@lru_cache(maxsize=None)
def _load_index(lang: str):
    import faiss

    path = DATA_DIR / f"{lang}.faiss"
    if not path.exists():
        raise IndexMissingError(f"No {lang} index at {path}.")
    return faiss.read_index(str(path))


@lru_cache(maxsize=1)
def _load_ids() -> list[tuple[int, int]]:
    path = DATA_DIR / "ids.json"
    if not path.exists():
        raise IndexMissingError(f"No ids.json at {path}.")
    return [tuple(x) for x in json.loads(path.read_text(encoding="utf-8"))]


def _has_arabic(text: str) -> bool:
    # ponytail: script check, not a langdetect dep. Arabic script (U+0600-06FF plus the
    # presentation forms) is unambiguous for routing our two indexes.
    return any("؀" <= c <= "ۿ" or "ﭐ" <= c <= "﻿" for c in text)


def _route(q: str, lang: str | None) -> str:
    """Which index answers. Explicit lang wins; else Arabic script -> ar; else en."""
    if lang:
        return lang if lang in INDEXED_LANGS else "ar"
    return "ar" if _has_arabic(q) else "en"


def _hits(ranked: list[tuple[int, float]]) -> list[SearchHit]:
    ids = _load_ids()
    out: list[SearchHit] = []
    for row, score in ranked:
        sura, aya = ids[row]
        ayah = corpus.get_ayah(sura, aya)
        tr = corpus.get_translation(sura, aya)
        out.append(
            SearchHit(
                sura=sura,
                aya=aya,
                text_uthmani=ayah["text_uthmani"] if ayah else "",
                translation=tr["text"] if tr else None,
                score=float(score),
            )
        )
    return out


def _vector_scores(embed_text: str, lang: str):
    """Full cosine over the whole index, as an array aligned to ids order.

    A full scan (all 6236 rows) rather than a top-k search, because hybrid has to add the
    lexical score to EVERY row before ranking — you cannot fuse what you truncated. At this
    size the scan is ~2 ms, so both modes share one path.
    """
    index = _load_index(lang)
    # The Arabic index is built on diacritics-stripped text (BGE-M3 embeds tashkīl-heavy
    # classical Arabic near-randomly). Strip the query so it lands in the same space.
    qvec = encode([strip_diacritics(embed_text) if lang == "ar" else embed_text])
    sims, rows = index.search(qvec, index.ntotal)
    cos = np.zeros(index.ntotal, dtype="float32")
    cos[rows[0]] = sims[0]
    return cos


def _top(scores, limit: int, drop_zero: bool = False) -> list[tuple[int, float]]:
    order = np.argsort(-scores)[:limit]
    return [(int(i), float(scores[i])) for i in order if not drop_zero or scores[i] > 0]


def search(
    q: str,
    lang: str | None = None,
    limit: int = 10,
    mode: str | None = None,
    hyde: bool | None = None,
    alpha: float | None = None,
) -> SearchResult:
    """Search the muṣḥaf. `mode`/`hyde`/`alpha` default to the configured deployment values.

    The Arabic āyah is always the primary result whatever the query language; the
    translation rides along as secondary.
    """
    s = get_settings()
    mode = mode or s.search_mode
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    alpha = s.search_hybrid_alpha if alpha is None else alpha
    want_hyde = s.search_hyde if hyde is None else hyde

    # Normalise the query exactly as the corpus was normalised at embed time.
    q = unicodedata.normalize("NFC", q.strip())

    if mode == "keyword":
        # drop_zero: BM25 0 means "shares no token with the query" — that is genuinely no
        # result, not a weak one, and padding the list to `limit` with them would be noise.
        return SearchResult(_hits(_top(lexical.scores(q), limit, drop_zero=True)), "ar", mode, False)

    matched = _route(q, lang)

    # HyDE steers the EMBEDDING only. The lexical term below still scores the raw query —
    # the whole point of the lexical signal is the wording the reciter actually typed, and
    # BM25 over an LLM's paraphrase throws that away. (Upstream fed the expansion to both;
    # this is a deliberate divergence. One line to flip if an eval disagrees.)
    embed_text, hyde_used = q, False
    if want_hyde:
        from . import hyde as _hyde

        expanded = _hyde.expand(q, matched)
        if expanded:
            embed_text, hyde_used = expanded, True

    scores = _vector_scores(embed_text, matched)

    # hybrid on a language with no lexical bag is vector — say so, don't fake it.
    applied = mode
    if mode == "hybrid" and matched == "ar" and alpha > 0:
        bm = lexical.scores(q)
        peak = float(bm.max())
        if peak > 0:
            scores = scores + alpha * (bm / peak)
    elif mode == "hybrid":
        applied = "vector"

    return SearchResult(_hits(_top(scores, limit)), matched, applied, hyde_used)


def reset() -> None:
    """Drop every cache. For tests and for a reload after rebuilding the index."""
    _load_index.cache_clear()
    _load_ids.cache_clear()
    lexical.reset()
    from . import hyde

    hyde.reset()
