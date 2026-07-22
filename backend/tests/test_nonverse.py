"""Task 11 — istiʿādhah / basmalah / ṣadaqa (FR-009).

The first thing that happens on a real recording, and the source plan left it as a
follow-up. A learner opens with أعوذ بالله; if we have no concept of non-verse text, it
gets diffed against the verse as though they had recited it, and every word of it comes
back red. Their very first feedback is a wall of errors for doing exactly the right
thing.
"""

import quran_transcript.alphabet as alph
from quran_transcript import Aya

from conftest import AL_FATIHA_1_2_MUTASHABIH

from tajwid.feedback import MuaalemOutput, SessionState, analyse_session
from tajwid.feedback.nonverse import phonetize_marker, strip_non_verse
from tajwid.feedback.types import Span


def _istiaatha(moshaf):
    return phonetize_marker(alph.istiaatha.uthmani, moshaf)


def _basmalah(moshaf):
    return phonetize_marker(Aya(1, 1).get().uthmani, moshaf)


def _sadaka(moshaf):
    return phonetize_marker(alph.sadaka.uthmani, moshaf)


def test_istiaatha_is_stripped_from_the_head(moshaf):
    chunk = _istiaatha(moshaf) + AL_FATIHA_1_2_MUTASHABIH
    remainder, found, _s, _e = strip_non_verse(chunk, moshaf)

    assert "istiaatha" in found
    assert remainder == AL_FATIHA_1_2_MUTASHABIH


def test_sadaka_is_stripped_from_the_tail(moshaf):
    chunk = AL_FATIHA_1_2_MUTASHABIH + _sadaka(moshaf)
    remainder, found, _s, _e = strip_non_verse(chunk, moshaf)

    assert "sadaka" in found
    assert remainder == AL_FATIHA_1_2_MUTASHABIH


def test_istiaatha_and_basmalah_together_open_a_session(moshaf):
    """The ordinary opening: أعوذ بالله، بسم الله، then the verse."""
    chunk = _istiaatha(moshaf) + _basmalah(moshaf) + AL_FATIHA_1_2_MUTASHABIH
    remainder, found, _s, _e = strip_non_verse(chunk, moshaf)

    assert found == ["istiaatha", "basmalah"]
    assert remainder == AL_FATIHA_1_2_MUTASHABIH


def test_a_verse_alone_strips_nothing(moshaf):
    remainder, found, _s, _e = strip_non_verse(AL_FATIHA_1_2_MUTASHABIH, moshaf)
    assert found == []
    assert remainder == AL_FATIHA_1_2_MUTASHABIH


def test_basmalah_alone_is_not_stripped(moshaf):
    """Basmalah IS Al-Fatiha 1:1. Stripping a chunk that is nothing but basmalah would
    delete a legitimately recited verse.

    It is only non-verse when it PRECEDES something else — the opening of one of the
    other 112 suras.
    """
    remainder, found, _s, _e = strip_non_verse(_basmalah(moshaf), moshaf)
    assert found == []
    assert remainder == _basmalah(moshaf)


def test_stripping_reports_the_span_it_kept_so_probs_can_follow(moshaf):
    """`phonemes.probs` is indexed over the string we just cut.

    If the head is stripped and the probability array is not cut identically, every
    finding gets scored against some OTHER phoneme's probability — silently, and only
    for learners who said the istiʿādhah. That is the same coordinate-system trap that
    made `ph_pos` unusable for scoring (D3), wearing a different hat.
    """
    verse = AL_FATIHA_1_2_MUTASHABIH
    chunk = _istiaatha(moshaf) + verse

    remainder, found, start, end = strip_non_verse(chunk, moshaf)

    assert found == ["istiaatha"]
    assert chunk[start:end] == remainder      # the offsets really do select the verse
    assert start == len(_istiaatha(moshaf))   # ...and they skip exactly the istiaatha


def test_a_chunk_that_is_only_istiaatha_scores_nothing_and_moves_no_cursor(moshaf):
    """The learner said something correct and non-Quranic. Say so; accuse nothing."""
    cursor = Span(sura=1, aya=1, word_idx=0)
    state = SessionState(moshaf=moshaf, session_id="s", cursor=cursor)

    response, after = analyse_session(
        MuaalemOutput.from_phonemes(_istiaatha(moshaf)), state
    )

    assert response.non_verse == ["istiaatha"]
    assert response.words == []          # nothing to mark: no Quran was recited
    assert after.cursor == cursor        # it is not a verse, so there is nothing to pass


def test_istiaatha_before_a_verse_does_not_poison_the_verses_feedback(moshaf):
    """THE BUG THIS TASK EXISTS TO PREVENT.

    Without stripping, the istiʿādhah is diffed against the verse and the learner's
    first feedback of the session is a wall of red for doing the right thing.
    """
    cursor = Span(sura=1, aya=1, word_idx=0)
    state = SessionState(moshaf=moshaf, session_id="s", cursor=cursor)
    chunk = _istiaatha(moshaf) + AL_FATIHA_1_2_MUTASHABIH

    response, _ = analyse_session(MuaalemOutput.from_phonemes(chunk), state)

    assert response.status == "ok"
    assert response.non_verse == ["istiaatha"]
    assert (response.span.sura, response.span.aya) == (1, 2)
    # The verse itself was recited perfectly. Not one word may be marked wrong.
    assert all(w.status == "correct" for w in response.words)
    assert all(w.errors == [] for w in response.words)
