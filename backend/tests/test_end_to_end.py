"""The whole pipeline, on a COMPLETE model output (text + probs + sifat).

Every other test feeds the system a bare phoneme string, which is what obad's HTTP API
would give us. These feed it what the model actually produces — and so they are the
first tests in which US2 (sifat) and US3 (confidence) are reachable at all.

This is our half of the integration, verified without the model.
"""

from tajwid.feedback.mock import mock_output, phonemes_of

from tajwid.feedback import SessionState, analyse, analyse_session
from tajwid.feedback.types import Span


def _errors(response):
    return [e for w in response.words for e in w.errors]


def test_a_perfect_recitation_is_marked_correct_throughout(moshaf):
    out = mock_output(1, 5, moshaf)
    response = analyse(out, moshaf)

    assert response.status == "ok"
    assert all(w.status == "correct" for w in response.words)
    assert _errors(response) == []


def test_a_confident_madd_error_is_reported_as_an_error(moshaf):
    """The model heard the mistake clearly. Say so."""
    perfect = phonemes_of(1, 5, moshaf)
    short = perfect.replace("ۦۦۦۦ", "ۦ")  # نَسْتَعِينُ: 4 counts recited as 1
    assert short != perfect

    response = analyse(mock_output(1, 5, moshaf, recited=short, prob=0.96), moshaf)

    bad = [w for w in response.words if w.status == "error"]
    assert len(bad) == 1
    assert bad[0].word_idx == 3               # نَسْتَعِينُ, the last word
    err = bad[0].errors[0]
    assert err.expected_len == 4 and err.predicted_len == 1
    assert err.confidence == 0.96


def test_the_same_mistake_heard_badly_is_only_almost(moshaf):
    """THE PRODUCT'S WHOLE STANCE ON UNCERTAINTY, in one test.

    Identical recitation, identical error — the only thing that changed is that the
    model was unsure of what it heard. We must hedge, not accuse.
    """
    perfect = phonemes_of(1, 5, moshaf)
    short = perfect.replace("ۦۦۦۦ", "ۦ")

    unsure = mock_output(
        1, 5, moshaf, recited=short, prob=0.96, unsure_span=(0, len(short))
    )
    response = analyse(unsure, moshaf)

    assert not any(w.status == "error" for w in response.words)
    assert any(w.status == "almost" for w in response.words)


def test_an_articulation_slip_is_surfaced_on_the_right_word(moshaf):
    """US2 — feedback the model computes and obad's API drops on the floor."""
    out = mock_output(
        1, 5, moshaf, wrong_sifat={0: {"tafkheem_or_taqeeq": "mofakham"}}
    )
    response = analyse(out, moshaf)

    sifa = [e for e in _errors(response) if e.error_type == "sifa"]
    assert len(sifa) == 1
    assert sifa[0].expected_ph == "tafkheem_or_taqeeq=moraqaq"
    assert sifa[0].predicted_ph == "tafkheem_or_taqeeq=mofakham"

    # ...and it landed on a word, not on a floating character offset (SC-001).
    owner = [w for w in response.words if sifa[0] in w.errors]
    assert len(owner) == 1


def test_an_unsure_articulation_slip_does_not_accuse(moshaf):
    """Sifat get their OWN threshold (FR-019): different head, different calibration."""
    out = mock_output(
        1,
        5,
        moshaf,
        wrong_sifat={0: {"tafkheem_or_taqeeq": "mofakham"}},
        sifa_prob=0.40,
    )
    response = analyse(out, moshaf)

    sifa = [e for e in _errors(response) if e.error_type == "sifa"]
    assert sifa and sifa[0].confidence == 0.40
    assert not any(w.status == "error" for w in response.words)


def test_a_cold_session_cannot_identify_the_basmalah_alone(moshaf):
    """A session that starts with NO cursor cannot resolve its own first chunk.

    Al-Fatiha 1:1 *is* the basmalah, and the basmalah also appears at 27:30 — so the
    opening words of the most-recited sura in the world are ambiguous with no context.
    The system correctly refuses to guess.

    The cure is not cleverness, it is CONTEXT: a real app knows which sura the learner
    chose before they pressed record. Seed the cursor. (Recorded in docs/INTEGRATION.md.)
    """
    cold = SessionState(moshaf=moshaf, session_id="learner-42")  # no cursor
    response, _ = analyse_session(mock_output(1, 1, moshaf), cold)
    assert response.status == "ambiguous"
    assert response.words == []


def test_a_full_session_walks_al_fatiha(moshaf):
    """Four chunks, in order, tracked by the cursor — the streaming path end to end.

    The cursor is SEEDED at 1:1, because the learner chose Al-Fatiha before reciting.
    """
    state = SessionState(
        moshaf=moshaf, session_id="learner-42", cursor=Span(sura=1, aya=1, word_idx=0)
    )
    seen = []

    for aya in (1, 2, 3, 4):
        out = mock_output(1, aya, moshaf)
        response, state = analyse_session(out, state)

        assert response.status == "ok", f"lost the reciter at 1:{aya}"
        assert all(
            w.status == "correct" for w in response.words if not w.trimmed
        ), f"a perfect recitation was marked wrong at 1:{aya}"
        seen.append((response.span.sura, response.span.aya))

    assert seen == [(1, 1), (1, 2), (1, 3), (1, 4)]
    # The cursor followed them the whole way.
    assert state.cursor.sura == 1 and state.cursor.aya == 4
    assert state.penalty == 0
