from conftest import (
    AL_BAQARAH_2_1_CORRECT,
    AL_FATIHA_1_2_MUTASHABIH,
    AL_FATIHA_1_5,
    AL_IKHLAS_112_1,
)

from tajwid.feedback.locate import locate
from tajwid.feedback.session import SessionState, advance
from tajwid.feedback.track import normalized_phonemes_for_span, track
from tajwid.feedback.types import Span


def test_normalized_span_lookup_needs_no_phonetizer():
    # Al-Fatiha 1:1, first 2 words. Comes straight out of ph_index.npy (FR-011).
    out = normalized_phonemes_for_span(sura=1, aya=1, word_idx=0, n_words=2)
    assert out
    assert isinstance(out, str)
    # Normalized => one char per phoneme group, so no repeated madd letters.
    assert "اا" not in out


def test_span_lookup_crosses_aya_boundaries():
    """A chunk does not politely stop at the end of an aya."""
    within = normalized_phonemes_for_span(sura=1, aya=1, word_idx=0, n_words=4)
    across = normalized_phonemes_for_span(sura=1, aya=1, word_idx=0, n_words=8)
    # Al-Fatiha 1:1 is 4 words, so 8 words must reach into 1:2.
    assert len(across) > len(within)


def test_tracking_finds_the_next_chunk_ahead_of_the_cursor(moshaf):
    cursor = Span(sura=1, aya=1, word_idx=0)
    result = track(AL_FATIHA_1_2_MUTASHABIH, cursor, moshaf)
    assert result.status == "ok"
    assert result.span.sura == 1
    assert result.span.aya == 2


def test_tracking_resolves_a_mutashabih_that_cold_search_cannot(moshaf):
    """THE POINT OF THE WHOLE PHASE.

    `ٱلْحَمْدُ لِلَّهِ رَبِّ ٱلْعَٰلَمِينَ` matches six places, so a COLD search can only
    shrug and say `ambiguous`. But a reciter who was just at Al-Fatiha 1:1 is
    obviously reciting 1:2 — not Al-An'am 6:45. The cursor turns an unanswerable
    question into a trivial one.

    This is what it means to say tracking *structurally* eliminates the mutashabihat
    problem, rather than guessing its way past it.
    """
    assert locate(AL_FATIHA_1_2_MUTASHABIH).status == "ambiguous"

    cursor = Span(sura=1, aya=1, word_idx=0)
    tracked = track(AL_FATIHA_1_2_MUTASHABIH, cursor, moshaf)
    assert tracked.status == "ok"
    assert (tracked.span.sura, tracked.span.aya) == (1, 2)


def test_tracking_tolerates_the_reciter_repeating_a_word(moshaf):
    """Reciters back up while memorising. The window looks BEHIND the cursor, so a
    repeat must still match rather than being flagged as a jump."""
    cursor = Span(sura=1, aya=2, word_idx=1)
    result = track(AL_FATIHA_1_2_MUTASHABIH, cursor, moshaf)  # re-reciting 1:2 from the top
    assert result.status == "ok"
    assert result.span.aya == 2


def test_tracking_rejects_a_chunk_far_outside_the_window(moshaf):
    """Below acceptance_ratio inside the window => no_match, so the caller can fall
    back to a cold search rather than force a wrong answer.

    NOTE the fixture is Al-Ikhlas (sura 112), not Al-Baqarah. The source plan used
    Al-Baqarah 2:1 here and expected `no_match` — but Al-Fatiha is only ~29 words
    long, so a 30-word forward window REACHES Al-Baqarah, and tracking finds it with
    a perfect score. That is correct behaviour, not a bug (see the next test), so the
    fixture is wrong, not the assertion.
    """
    cursor = Span(sura=1, aya=1, word_idx=0)
    result = track(AL_IKHLAS_112_1, cursor, moshaf)  # thousands of words away
    assert result.status == "no_match"


def test_a_reciter_continuing_past_the_end_of_a_sura_is_followed(moshaf):
    """Recitation does not stop at sura boundaries, and neither does the window.

    A learner who finishes Al-Fatiha and carries straight on into Al-Baqarah has not
    "jumped" — they are doing the most ordinary thing in the world. The window is in
    flat word ordinals over the whole Quran, so it follows them without a special case.
    """
    cursor = Span(sura=1, aya=7, word_idx=0)
    result = track(AL_BAQARAH_2_1_CORRECT, cursor, moshaf)
    assert result.status == "ok"
    assert (result.span.sura, result.span.aya) == (2, 1)


def test_session_advances_its_cursor(moshaf):
    state = SessionState(moshaf=moshaf, session_id="s1", cursor=None)
    result = track(
        AL_FATIHA_1_2_MUTASHABIH, Span(sura=1, aya=1, word_idx=0), moshaf
    )
    state = advance(state, result)
    assert state.cursor is not None
    assert state.cursor.aya == 2


def test_a_failed_match_leaves_the_cursor_alone(moshaf):
    """The reciter may simply have coughed. Resetting on every miss would thrash."""
    cursor = Span(sura=1, aya=2, word_idx=0)
    state = SessionState(moshaf=moshaf, session_id="s1", cursor=cursor)
    failed = track(AL_IKHLAS_112_1, cursor, moshaf)
    assert failed.status == "no_match"

    after = advance(state, failed)
    assert after.cursor == cursor


def test_session_id_is_carried(moshaf):
    """FR-001. We keep no session store — the caller holds the state."""
    state = SessionState(moshaf=moshaf, session_id="learner-42", cursor=None)
    result = track(AL_FATIHA_1_5, Span(sura=1, aya=4, word_idx=0), moshaf)
    assert advance(state, result).session_id == "learner-42"
