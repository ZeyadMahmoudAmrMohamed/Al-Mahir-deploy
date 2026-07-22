"""Response models for āyah search."""

from pydantic import BaseModel


class SearchHit(BaseModel):
    sura: int
    aya: int
    # The Arabic āyah is ALWAYS the primary result, whatever the query language.
    text_uthmani: str
    translation: str | None = None
    # Cosine similarity (semantic) or BM25 (keyword), RAW — see service.py on thresholds.
    score: float


class SearchResponse(BaseModel):
    hits: list[SearchHit]
    # What ACTUALLY ran, which is not always what was asked for:
    #   matched_lang — the index that answered (an unindexed lang falls back to ar).
    #   mode         — "hybrid" on English degrades to "vector" (no Arabic lexical bag).
    #   hyde_used    — false when HyDE was off, unavailable (no key / upstream down), or
    #                  meaningless (keyword mode). A silent degradation is a lie you
    #                  debug twice; these three fields are how the client sees it.
    matched_lang: str
    mode: str
    hyde_used: bool
