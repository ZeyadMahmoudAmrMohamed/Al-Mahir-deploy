import pytest

from tajwid.feedback.confidence import STRICTNESS, grade, score_errors
from tajwid.feedback.types import FeedbackError


def _error(conf=None, pred_ph_pos=(0, 2), error_type="normal"):
    return FeedbackError(
        error_type=error_type,
        speech_error_type="replace",
        uthmani_pos=(0, 1),
        ph_pos=(0, 2),
        pred_ph_pos=pred_ph_pos,
        expected_ph="قَ",
        predicted_ph="كَ",
        confidence=conf,
    )


def test_high_confidence_error_is_a_real_error():
    assert grade(_error(conf=0.95), STRICTNESS["normal"]) == "error"


def test_low_confidence_error_is_downgraded_to_almost():
    assert grade(_error(conf=0.40), STRICTNESS["normal"]) == "almost"


def test_strict_mode_asserts_errors_the_lenient_mode_softens():
    err = _error(conf=0.60)
    assert grade(err, STRICTNESS["strict"]) == "error"
    assert grade(err, STRICTNESS["lenient"]) == "almost"


def test_missing_probs_degrades_to_almost():
    """(D2) THE INVERTED TEST. The source plan asserted the opposite.

    Its `test_missing_probs_defaults_to_confident` asserted `confidence == 1.0` when
    the model supplies no probabilities — so an unscored finding graded as a hard
    `error`. A teammate who ships `phonemes.text` without `probs` would silently get a
    maximally accusatory system, and nothing in the suite would notice.

    FR-018 / Constitution VI: absence of confidence MUST NOT be treated as high
    confidence. Unknown degrades to `almost`.
    """
    scored = score_errors([_error()], phoneme_probs=[])
    assert scored[0].confidence is None
    assert grade(scored[0], STRICTNESS["normal"]) == "almost"
    # Even at the harshest setting, an unscored finding is never an accusation.
    assert grade(scored[0], STRICTNESS["strict"]) == "almost"


def test_score_errors_reads_probability_from_the_predicted_phonemes():
    """(D3) Confidence is sliced with pred_ph_pos, NOT ph_pos.

    `phonemes.probs` is an array over the PREDICTED string. Slicing it with reference
    coordinates reads the probability of phonemes the reciter never said.
    """
    err = _error(pred_ph_pos=(0, 2))
    # The model was very unsure about the phonemes this error actually covers.
    scored = score_errors([err], phoneme_probs=[0.3, 0.2, 0.99, 0.99])
    assert scored[0].confidence == pytest.approx(0.2)  # weakest link


def test_confidence_is_the_weakest_phoneme_not_the_average():
    """An error is only as trustworthy as the least certain phoneme under it."""
    err = _error(pred_ph_pos=(0, 3))
    scored = score_errors([err], phoneme_probs=[0.9, 0.9, 0.1])
    assert scored[0].confidence == pytest.approx(0.1)


def test_a_deletion_has_no_predicted_span_so_it_cannot_be_scored():
    """(D3/D2) The reciter said nothing; there is no probability to read."""
    err = _error(pred_ph_pos=None)
    scored = score_errors([err], phoneme_probs=[0.9, 0.9, 0.9])
    assert scored[0].confidence is None
    assert grade(scored[0], STRICTNESS["normal"]) == "almost"


def test_sifa_and_phoneme_errors_use_separate_thresholds():
    """(D4 / FR-019) The two probabilities come from DIFFERENT HEADS of the model and
    are not assumed to share a calibration.

    A single global threshold, tuned to suppress false sifat accusations, would also
    dull genuine word-error detection. So the same confidence can legitimately grade
    differently depending on which signal produced it.
    """
    level = STRICTNESS["normal"]
    phoneme_thr, sifa_thr = level
    assert phoneme_thr != sifa_thr, "a shared threshold defeats the purpose of FR-019"

    # Pick a confidence that falls between the two thresholds.
    between = (min(phoneme_thr, sifa_thr) + max(phoneme_thr, sifa_thr)) / 2
    phoneme_err = _error(conf=between, error_type="normal")
    sifa_err = _error(conf=between, error_type="sifa")

    assert grade(phoneme_err, level) != grade(sifa_err, level)
