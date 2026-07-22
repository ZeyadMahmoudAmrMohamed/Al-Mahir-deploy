"""Search checks.

Keyword (BM25) runs everywhere — no model, no download — so it guards the wiring: data
files present, ids.json aligned to the corpus, tokenisation alive. The vector/hybrid cases
need the 2 GB BGE-M3 download, so they are opt-in via TAJWID_TEST_SEMANTIC=1. The HyDE case
additionally needs an LLM key and is skipped without one — its FALLBACK, however, is tested
unconditionally, because "no key" must never break search.

Run:  python -m pytest backend/tests/test_search.py
      TAJWID_TEST_SEMANTIC=1 python backend/tests/test_search.py
"""

import os

from tajwid.config import get_settings
from tajwid.search.service import search

SEMANTIC = bool(os.environ.get("TAJWID_TEST_SEMANTIC"))


def test_keyword_finds_the_exact_ayah():
    # يغتب occurs once in the Qur'an (49:12), so an exact fragment must pin that āyah #1.
    r = search("ولا يغتب بعضكم بعضا", mode="keyword", limit=5)
    assert r.matched_lang == "ar" and r.mode == "keyword"
    assert (r.hits[0].sura, r.hits[0].aya) == (49, 12), r.hits[0]


def test_keyword_nonsense_returns_nothing():
    assert search("زقزقتليب", mode="keyword").hits == []


def test_hyde_is_ignored_for_keyword():
    # Nothing is embedded in keyword mode, so there is nothing for HyDE to steer. It must
    # report that it did not run rather than quietly billing an LLM call.
    assert search("الصبر", mode="keyword", hyde=True).hyde_used is False


def test_hybrid_beats_vector_on_an_exact_fragment():
    """The reason hybrid is the default: the lexical term is what pins exact wording.

    If someone sets alpha to 0, hybrid IS vector and this test says so.
    """
    if not SEMANTIC:
        return
    q = "ولا يغتب بعضكم بعضا"
    assert (search(q, mode="hybrid").hits[0].sura, search(q, mode="hybrid").hits[0].aya) == (49, 12)
    top_vector = search(q, mode="vector").hits[0]
    assert (top_vector.sura, top_vector.aya) != (49, 12), (
        "vector alone now pins the fragment — hybrid may no longer be earning its keep"
    )


def test_hybrid_degrades_to_vector_on_english_and_says_so():
    if not SEMANTIC:
        return
    r = search("those who are patient in hardship", mode="hybrid", limit=5)
    assert r.matched_lang == "en"
    assert r.mode == "vector", "English has no lexical bag; the response must not claim hybrid"
    assert len(r.hits) == 5 and all(h.text_uthmani for h in r.hits)


def test_hyde_falls_back_to_the_raw_query_without_a_key():
    """No key must mean 'HyDE didn't run', never 'search failed'."""
    if not SEMANTIC:
        return
    if get_settings().llm_api_key:
        return  # a key is configured; see test_hyde_disambiguates below
    r = search("الغيبة", hyde=True, limit=5)
    assert r.hyde_used is False and r.hits, "search must survive an unavailable LLM"


def test_hyde_disambiguates_a_polysemous_query():
    """الغيبة (backbiting) vs الغيب (the unseen): same letters, different concept.

    Raw, the query drowns in "unseen" āyāt and misses 49:12. This is the case HyDE exists
    for, so it is the case that proves it is wired up.
    """
    if not SEMANTIC or not get_settings().llm_api_key:
        return
    r = search("الغيبة", hyde=True, limit=10)
    assert r.hyde_used is True, "a key is set but the expansion did not run"
    assert any((h.sura, h.aya) == (49, 12) for h in r.hits), [(h.sura, h.aya) for h in r.hits]


if __name__ == "__main__":
    for name, fn in sorted(dict(globals()).items()):
        if name.startswith("test_"):
            fn()
            print(f"  {name}")
    print("ok" + ("" if SEMANTIC else "  (vector/hybrid skipped: set TAJWID_TEST_SEMANTIC=1)"))
