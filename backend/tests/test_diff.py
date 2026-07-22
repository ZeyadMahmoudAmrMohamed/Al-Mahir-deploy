from conftest import AL_BAQARAH_2_1_CORRECT, AL_BAQARAH_2_1_SHORT_MADDS

from tajwid.feedback.diff import diff_recitation

ALM = "الٓمٓ"


def test_correct_recitation_yields_no_errors(moshaf):
    assert diff_recitation(ALM, AL_BAQARAH_2_1_CORRECT, moshaf) == []


def test_short_madd_is_a_tajweed_error_with_counts(moshaf):
    errors = diff_recitation(ALM, AL_BAQARAH_2_1_SHORT_MADDS, moshaf)
    tajweed = [e for e in errors if e.error_type == "tajweed"]
    assert tajweed, "expected madd errors"

    first = tajweed[0]
    assert first.expected_len == 6      # madd lazim
    assert first.predicted_len == 2     # recited far too short
    assert any(r.name_en == "Lazem Madd" for r in first.tajweed_rules)


def test_error_carries_positions(moshaf):
    errors = diff_recitation(ALM, AL_BAQARAH_2_1_SHORT_MADDS, moshaf)
    e = errors[0]
    assert e.uthmani_pos[0] <= e.uthmani_pos[1]
    assert e.ph_pos[0] <= e.ph_pos[1]


def test_confidence_defaults_to_none_not_one(moshaf):
    """(D2) Unknown is not certain.

    No probabilities have been supplied here, so every finding must be UNSCORED.
    The source plan defaulted `confidence` to 1.0, which makes an unscored finding
    grade as a hard `error` — maximally accusatory, and invisible in its own tests.
    """
    errors = diff_recitation(ALM, AL_BAQARAH_2_1_SHORT_MADDS, moshaf)
    assert all(e.confidence is None for e in errors)


def test_replace_error_carries_a_predicted_span(moshaf):
    """(D3) `ph_pos` is in REFERENCE coordinates and cannot index `phonemes.probs`,
    which is an array over the PREDICTED string. The two coincide only when the
    reciter was correct — i.e. exactly when there is no error to score.

    So the diff must capture the predicted-side span here, where both coordinate
    systems are in scope.
    """
    errors = diff_recitation(ALM, AL_BAQARAH_2_1_SHORT_MADDS, moshaf)
    e = errors[0]

    assert e.pred_ph_pos is not None
    start, end = e.pred_ph_pos
    assert start < end

    # The predicted span must actually select the predicted phonemes the error is
    # about — a short madd of two alifs, not the six-alif reference.
    assert AL_BAQARAH_2_1_SHORT_MADDS[start:end] == e.predicted_ph

    # And it must NOT be the same span as the reference one, or we have simply
    # relabelled reference coordinates and fixed nothing.
    assert e.pred_ph_pos != e.ph_pos


def test_pure_deletion_has_no_predicted_span(moshaf):
    """A reciter who said nothing gives us nothing to read a probability from.

    (D3/D2) No predicted span => no confidence => `almost`, never `error`.
    """
    # Recite only the first phoneme group and stop dead.
    errors = diff_recitation(ALM, "ءَ", moshaf)
    deletions = [e for e in errors if e.speech_error_type == "delete"]
    assert deletions, "expected deletions when the reciter stops early"
    assert all(e.pred_ph_pos is None for e in deletions)
