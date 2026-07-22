"""Task 10 — chunk-boundary hygiene: the window penalty (FR-007) and edge-word
trimming (FR-010). Both are properties of STREAMING, not of a single recitation:
they only bite once a reciter's audio is cut into chunks.
"""

from conftest import AL_FATIHA_1_2_MUTASHABIH, AL_FATIHA_1_5, AL_IKHLAS_112_1

from tajwid.feedback import MuaalemOutput, SessionState, analyse_session
from tajwid.feedback.session import MAX_PENALTY, advance
from tajwid.feedback.track import track
from tajwid.feedback.types import Span


# --- FR-007: the window WIDENS on failure; it does not reset ------------------


def test_a_chunk_beyond_the_window_is_missed_until_the_window_widens(moshaf):
    """The penalty is the difference between 'I lost you' and 'let me look harder'.

    Resetting on failure throws away a cursor that is probably still roughly right and
    sends the reciter back to a full-Quran cold search. Widening keeps the cursor and
    simply looks further — which is what a human listener does when they briefly lose
    their place.
    """
    cursor = Span(sura=1, aya=1, word_idx=0)

    # Al-Fatiha 1:5 begins ~13 words after the cursor — well past a tiny window.
    # (`window_words` bounds the START OFFSETS the grid tries, not the span length, so
    # the target has to be genuinely far away for this test to mean anything.)
    missed = track(AL_FATIHA_1_5, cursor, moshaf, window_words=2)
    assert missed.status == "no_match"

    # The same chunk, the same cursor — found once the penalty widens the window.
    found = track(AL_FATIHA_1_5, cursor, moshaf, window_words=2, penalty=20)
    assert found.status == "ok"
    assert (found.span.sura, found.span.aya) == (1, 5)


def test_the_penalty_grows_on_failure_and_is_capped(moshaf):
    state = SessionState(
        moshaf=moshaf, session_id="s", cursor=Span(sura=1, aya=1, word_idx=0)
    )
    failed = track(AL_IKHLAS_112_1, state.cursor, moshaf)
    assert failed.status == "no_match"

    for _ in range(50):
        state = advance(state, failed)

    assert state.penalty == MAX_PENALTY  # grows, but does not run away


def test_a_successful_match_clears_the_penalty(moshaf):
    state = SessionState(
        moshaf=moshaf,
        session_id="s",
        cursor=Span(sura=1, aya=1, word_idx=0),
        penalty=30,
    )
    ok = track(AL_FATIHA_1_2_MUTASHABIH, state.cursor, moshaf)
    assert ok.status == "ok"

    state = advance(state, ok)
    assert state.penalty == 0  # we found them again; stop searching so wide


# --- FR-010: do not bill the learner for words OUR chunker cut ---------------


def test_a_word_cut_by_the_chunk_boundary_is_not_scored(moshaf):
    """The worst error class we have: one WE manufactured.

    The audio is sliced mid-word, the ASR emits a mangled fragment, the diff faithfully
    reports a mismatch — and the learner is billed for a mistake that exists only
    because of where we cut. Every component behaved correctly and the product still
    lied to its user.
    """
    # A chunk sliced through its first word: the reciter was already going when we cut.
    cursor = Span(sura=1, aya=2, word_idx=0)
    partial = AL_FATIHA_1_2_MUTASHABIH[6:]

    response, _ = analyse_session(
        MuaalemOutput.from_phonemes(partial),
        SessionState(moshaf=moshaf, session_id="s", cursor=cursor),
    )

    assert response.status == "ok"
    # The span begins mid-aya, so its leading word is OUR artefact, not their mistake.
    assert response.span.word_idx > 0

    edge = response.words[0]
    assert edge.trimmed is True
    assert edge.errors == []


def test_trimmed_words_are_still_returned_for_rendering(moshaf):
    """Trimmed means UNVERIFIED, not deleted. The frontend still draws the word."""
    cursor = Span(sura=1, aya=2, word_idx=0)
    response, _ = analyse_session(
        MuaalemOutput.from_phonemes(AL_FATIHA_1_2_MUTASHABIH[6:]),
        SessionState(moshaf=moshaf, session_id="s", cursor=cursor),
    )
    assert response.status == "ok"
    assert response.words, "a trimmed word is not a dropped word"
    assert any(w.trimmed for w in response.words)
    # ...and it still carries its text, or the frontend has nothing to draw.
    assert all(w.uthmani for w in response.words)


def test_a_complete_recitation_trims_nothing(moshaf):
    """Trimming must only fire at a REAL chunk boundary.

    A span the reciter genuinely began and genuinely finished is not an artefact of our
    chunking, and silently declining to score it would be its own kind of lie.
    """
    cursor = Span(sura=1, aya=1, word_idx=0)
    response, _ = analyse_session(
        MuaalemOutput.from_phonemes(AL_FATIHA_1_2_MUTASHABIH),
        SessionState(moshaf=moshaf, session_id="s", cursor=cursor),
    )
    assert response.status == "ok"
    assert not any(w.trimmed for w in response.words)
