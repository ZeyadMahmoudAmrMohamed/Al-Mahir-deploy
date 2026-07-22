"""Phonetic tasmea: obad's tasmeea_sura control flow, ported from imlaei text to the
phonetic script.

Why this exists: `tasmeea_sura()` takes `sura_idx` as a REQUIRED argument and compares
imlaei text to imlaei text. Our model emits phonemes and there is no imlaei at runtime,
so it cannot be used as-is. But its control flow — cursor, backward overlap, the
`(start x window)` grid, `acceptance_ratio` — is exactly what session tracking needs. We
keep the flow and change the alphabet.

The load-bearing insight: `ph_index.npy` already stores
`[sura, aya, word_idx, uth_char_start, uth_char_end, ph_start, ph_end]` for EVERY phoneme
group in the Quran, and `ref_norm_ph.txt` is the row-aligned normalised string. So the
normalised phonemes of any span are a SLICE LOOKUP, not a computation (FR-011) — which is
the only reason the grid is affordable. The phonetizer is never called inside the loop.

Note on upstream: `PhoneticSearch.search()` accepts `start` and `window` arguments and
SILENTLY IGNORES THEM (its body carries a `TODO: Add boudary to search resutls with start
and window`). A caller passing them believes they have a windowed search and gets a
full-Quran one. We do not use them.
"""

from functools import lru_cache

import Levenshtein as lv
import numpy as np
import quran_transcript.alphabet as alph
from quran_transcript import Aya, MoshafAttributes

from .locate import _search_engine
from .types import LocateResult, Span


@lru_cache(maxsize=1)
def _word_starts() -> np.ndarray:
    """Row index in the phoneme index at which each Quran word begins.

    Rows are in mushaf order, so word ORDINALS are a flat sequence over the whole
    Quran. Working in ordinals rather than (sura, aya, word) means a span that crosses
    an aya boundary needs no special case at all — it is just a wider slice.
    """
    index = _search_engine().index
    keys = index[:, :3]
    changed = np.any(keys[1:] != keys[:-1], axis=1)
    return np.concatenate(([0], np.flatnonzero(changed) + 1))


@lru_cache(maxsize=1)
def _ordinal_of_word() -> dict[tuple[int, int, int], int]:
    """(sura, aya, word_idx) -> flat word ordinal."""
    index = _search_engine().index
    starts = _word_starts()
    return {
        (int(index[r, 0]), int(index[r, 1]), int(index[r, 2])): i
        for i, r in enumerate(starts)
    }


@lru_cache(maxsize=1)
def _word_of_ordinal() -> list[tuple[int, int, int]]:
    """Flat word ordinal -> (sura, aya, word_idx)."""
    index = _search_engine().index
    return [
        (int(index[r, 0]), int(index[r, 1]), int(index[r, 2]))
        for r in _word_starts()
    ]


def _slice_by_ordinal(start_ord: int, n_words: int) -> str:
    starts = _word_starts()
    engine = _search_engine()

    if start_ord < 0 or start_ord >= len(starts):
        return ""

    r0 = int(starts[start_ord])
    end_ord = start_ord + n_words
    r1 = int(starts[end_ord]) if end_ord < len(starts) else len(engine.ref_ph_norm)
    return engine.ref_ph_norm[r0:r1]


def normalized_phonemes_for_span(
    sura: int, aya: int, word_idx: int, n_words: int
) -> str:
    """Normalised phonemes for a word span — a slice lookup, not a computation.

    Crosses aya boundaries naturally: a chunk does not politely stop where an aya does.
    """
    start_ord = _ordinal_of_word().get((sura, aya, word_idx))
    if start_ord is None:
        return ""
    return _slice_by_ordinal(start_ord, n_words)


def _match_ratio(ref: str, other: str) -> float:
    """Identical to tasmeea.get_match_ratio — kept so scores stay comparable."""
    if not ref:
        return 0.0
    return 1 - (min(lv.distance(ref, other), len(ref)) / len(ref))


def _uthmani_for_ordinals(start_ord: int, n_words: int) -> str:
    """The Uthmani text of a word span, walking aya boundaries as needed."""
    words_by_ord = _word_of_ordinal()
    out: list[str] = []

    for o in range(start_ord, min(start_ord + n_words, len(words_by_ord))):
        sura, aya, widx = words_by_ord[o]
        out.append(Aya(sura, aya).get().uthmani_words[widx])

    return alph.uthmani.space.join(out)


def track(
    phonemes: str,
    cursor: Span,
    moshaf: MoshafAttributes,
    overlap_words: int = 6,
    window_words: int = 30,
    acceptance_ratio: float = 0.5,
    penalty: int = 0,
) -> LocateResult:
    """Find this chunk in a window around the cursor.

    The window reaches BACKWARDS by `overlap_words` as well as forwards, because
    reciters legitimately repeat and back up while memorising. A forward-only cursor
    would flag every repetition as a jump.

    This is what structurally eliminates the mutashabihat problem: `ٱلْحَمْدُ لِلَّهِ رَبِّ
    ٱلْعَٰلَمِينَ` matches six places, so a cold search can only say `ambiguous` — but a
    reciter who was just at Al-Fatiha 1:1 is obviously at 1:2, not Al-An'am 6:45. The
    cursor answers the question the phonemes alone cannot.

    `penalty` WIDENS the window (FR-007). It is carried on the session and grows each
    time a chunk fails to match, so a reciter we have briefly lost is searched for over
    a progressively larger area rather than being given up on. Widening is not the same
    as resetting: resetting discards a cursor that is probably still roughly right.

    Returns `no_match` below `acceptance_ratio` so the caller can fall back to a cold
    search rather than force a wrong answer.
    """
    engine = _search_engine()
    query = engine._normalize_query(phonemes)
    if not query:
        return LocateResult(status="no_match")

    # The penalty widens the search in BOTH directions: someone we have lost may have
    # gone backwards as easily as forwards.
    overlap_words += penalty
    window_words += penalty

    cursor_ord = _ordinal_of_word().get((cursor.sura, cursor.aya, cursor.word_idx))
    if cursor_ord is None:
        return LocateResult(status="no_match")

    # Window bounds, following tasmeea.estimate_window_len: a chunk of N normalised
    # chars spans somewhere between N/9 and N/2 words.
    min_window = max(1, len(query) // 9)
    max_window = max(min_window, len(query) // 2 + 1)

    n_total = len(_word_starts())
    best_ratio = 0.0
    best: tuple[int, int] | None = None  # (start_ordinal, n_words)

    for offset in range(-overlap_words, window_words):
        start_ord = cursor_ord + offset
        if start_ord < 0 or start_ord >= n_total:
            continue

        for n_words in range(min_window, max_window + 1):
            candidate = _slice_by_ordinal(start_ord, n_words)
            if not candidate:
                continue

            ratio = _match_ratio(candidate, query)
            if ratio > best_ratio:
                best_ratio = ratio
                best = (start_ord, n_words)

    if best is None or best_ratio < acceptance_ratio:
        return LocateResult(status="no_match")

    start_ord, n_words = best
    words_by_ord = _word_of_ordinal()

    sura, aya, widx = words_by_ord[start_ord]
    # `end` is the LAST word of the match, inclusive — matching locate()'s semantics.
    # An exclusive end would leave the cursor sitting on the next aya before the
    # reciter has said a word of it.
    end_ord = min(start_ord + n_words - 1, len(words_by_ord) - 1)
    e_sura, e_aya, e_widx = words_by_ord[end_ord]

    return LocateResult(
        status="ok",
        span=Span(sura=sura, aya=aya, word_idx=widx),
        end=Span(sura=e_sura, aya=e_aya, word_idx=e_widx),
        uthmani_text=_uthmani_for_ordinals(start_ord, n_words),
    )
