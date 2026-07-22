from functools import lru_cache

from quran_transcript.phonetics.search import (
    NoPhonemesSearchResult,
    PhoneticSearch,
)

from .types import Candidate, LocateResult, Span


# Past this many matches, a candidate list stops being a shortlist and becomes a
# confession. `ambiguous` is only useful if the frontend can act on it — show the
# options, or wait for the next chunk to disambiguate. 1,599 candidates (which is
# what a single normalised `ز` returns) is not something anyone can act on, so the
# honest answer is `no_match`: we cannot identify this, recite more.
MAX_CANDIDATES = 10


@lru_cache(maxsize=1)
def _search_engine() -> PhoneticSearch:
    """PhoneticSearch loads ~4.6MB from disk. Build it once."""
    return PhoneticSearch()


def _to_span(search_span) -> Span:
    return Span(
        sura=search_span.sura_idx,
        aya=search_span.aya_idx,
        word_idx=search_span.uthmani_word_idx,
    )


def locate(
    phonemes: str,
    error_ratio: float = 0.1,
    max_candidates: int = MAX_CANDIDATES,
) -> LocateResult:
    """Identify where in the Quran these phonemes come from, with no prior context.

    Unlike obad's app, which takes results[0] blindly, this reports ambiguity when
    the query matches several places. That is not a rare edge: `ٱلْحَمْدُ لِلَّهِ رَبِّ
    ٱلْعَٰلَمِينَ` matches six, and Alif-Lam-Meem eight. Picking the first would score a
    learner reciting Al-Fatiha against Al-An'am.

    The query is matched against the madd-NORMALISED index, so identification is
    style-invariant (FR-005): a reciter whose madd lengths differ from the default
    still lands on the right verse. Scoring, by contrast, uses their real lengths.
    """
    if not phonemes:
        return LocateResult(status="no_match")

    engine = _search_engine()
    try:
        results = engine.search(phonemes, error_ratio=error_ratio)
    except (NoPhonemesSearchResult, ValueError):
        return LocateResult(status="no_match")

    if not results:
        return LocateResult(status="no_match")

    if len(results) > max_candidates:
        # Too short or too common to identify at all. The query is normalised, so a
        # gibberish `زززززز` collapses to one `ز` and "matches" 1,599 places. That is
        # not ambiguity worth reporting; it is a failure to identify.
        return LocateResult(status="no_match")

    # Carry each candidate's TEXT, not just its coordinates. A caller shown
    # `[(2, 147), (3, 60)]` still has to go and look both up before they can act;
    # that is a reconstruction burden, and this contract exists to abolish those.
    # Cheap, because the list is capped at max_candidates.
    candidates = [
        Candidate(
            sura=r.start.sura_idx,
            aya=r.start.aya_idx,
            word_idx=r.start.uthmani_word_idx,
            end=_to_span(r.end),
            uthmani_text=engine.get_uthmani_from_result(r),
        )
        for r in results
    ]

    if len(results) > 1:
        # Assert nothing: no span and no uthmani_text at the top level, so there is no
        # way for a caller to accidentally score against a guess. The candidates are
        # there to be SHOWN, not scored.
        return LocateResult(status="ambiguous", candidates=candidates)

    best = results[0]
    return LocateResult(
        status="ok",
        span=candidates[0].span,
        end=_to_span(best.end),
        uthmani_text=engine.get_uthmani_from_result(best),
        candidates=candidates,
    )
