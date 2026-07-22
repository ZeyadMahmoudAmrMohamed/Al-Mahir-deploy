from typing import Literal, Optional

from .types import FeedbackError

# Minimum model confidence at which we are willing to call something a mistake rather
# than soften it to "almost". A LOWER threshold is a harsher teacher.
#
# (D4 / FR-019) Each level is a (phoneme_threshold, sifa_threshold) PAIR, not one
# number. The two probabilities come from different heads of the model and are not
# assumed to share a calibration: a single global threshold, tuned low enough to
# suppress false sifat accusations, would also dull genuine word-error detection.
#
# THESE VALUES ARE PLACEHOLDERS AWAITING CALIBRATION (T068-T070). The constitution
# requires them to be "calibrated against a labelled set rather than hard-coded to a
# guess" — until that lands, this IS the guess, and it is marked as one.
# The DIRECTION of failure is not a tunable: where the curves force a trade, buy fewer
# false accusations at the cost of more misses (Constitution VI).
STRICTNESS: dict[str, tuple[float, float]] = {
    #            phoneme,  sifa
    "lenient": (0.90, 0.95),
    "normal": (0.70, 0.85),
    "strict": (0.50, 0.65),
}


def score_errors(
    errors: list[FeedbackError],
    phoneme_probs: Optional[list[float]],
) -> list[FeedbackError]:
    """Attach the model's confidence to each error.

    An error spans a range of phonemes; its confidence is the WEAKEST of them, because
    a single uncertain phoneme is enough to make the whole call doubtful. Averaging
    would let one confident phoneme launder an unsure one into an accusation.

    (D3) The slice is taken with `pred_ph_pos` — a span in the PREDICTED string —
    because that is what `phonemes.probs` is an array over. `ph_pos` is in REFERENCE
    coordinates, and the two diverge on every insert, delete and wrong-length madd:
    precisely the errors worth scoring. They agree only when the recitation was
    correct, i.e. when there is nothing to score at all.

    An error with no predicted span (a pure deletion — the reciter said nothing) has
    no probability to read. Its confidence stays None, which grades to `almost`.
    """
    if not phoneme_probs:
        # No probabilities from the model: everything stays UNSCORED (None), which is
        # not the same as confident. See grade().
        return errors

    for err in errors:
        if err.pred_ph_pos is None:
            continue

        start, end = err.pred_ph_pos
        window = phoneme_probs[start:end]
        if window:
            err.confidence = min(window)

    return errors


def grade(
    error: FeedbackError, thresholds: tuple[float, float]
) -> Literal["almost", "error"]:
    """A mistake we are unsure about is reported as `almost`, never as `error`.

    Constitution VI: false accusations destroy trust faster than missed errors. A user
    falsely corrected on a verse they recited perfectly has been told the product
    cannot hear, and that user does not come back.

    (D2) Unknown confidence is NOT high confidence. An unscored finding degrades to
    `almost` at every strictness level — there is no setting harsh enough to turn a
    guess into an accusation.
    """
    if error.confidence is None:
        return "almost"

    phoneme_threshold, sifa_threshold = thresholds
    threshold = sifa_threshold if error.error_type == "sifa" else phoneme_threshold

    return "error" if error.confidence >= threshold else "almost"
