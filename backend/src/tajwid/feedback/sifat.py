from quran_transcript import chunck_phonemes
from quran_transcript.phonetics.error_explainer import align_phonemes_groups

from .types import SIFA_ATTRS, FeedbackError, PredictedSifa


def compare_sifat(
    ref_sifat,
    predicted_sifat: list[PredictedSifa],
    ref_phonemes: str,
    predicted_phonemes: str,
) -> list[FeedbackError]:
    """Report articulation (sifa/makhraj) errors on correctly-identified letters.

    The model already predicts 10 articulation attributes per phoneme group, and
    upstream already compares them against the expected ones — but it renders the
    result to a Rich terminal table that nothing calls. This is a working capability
    sitting unused; here it becomes JSON (FR-017).

    Only 'equal' alignments are compared. If the reciter said a DIFFERENT letter, that
    is already a phoneme error, and flagging its sifat as wrong too would bill one
    mistake twice — the learner sees two red marks for one slip and loses faith in the
    count.
    """
    ref_groups = chunck_phonemes(ref_phonemes)
    pred_groups = chunck_phonemes(predicted_phonemes)
    alignments = align_phonemes_groups(ref_groups, pred_groups)

    errors: list[FeedbackError] = []

    for align in alignments:
        if align.op_type != "equal":
            continue
        if align.ref_idx >= len(ref_sifat) or align.pred_idx >= len(predicted_sifat):
            # The reciter stopped early: there is no prediction for this group. The
            # diff already reports it as a deletion; inventing articulation errors for
            # phonemes nobody uttered would be a pure fabrication.
            continue

        expected = ref_sifat[align.ref_idx]
        actual = predicted_sifat[align.pred_idx]

        for attr in SIFA_ATTRS:
            expected_value = getattr(expected, attr, None)
            actual_value = actual.attrs.get(attr)

            if expected_value is None or actual_value is None:
                continue
            if actual_value == expected_value:
                continue

            errors.append(
                FeedbackError(
                    error_type="sifa",
                    speech_error_type="replace",
                    uthmani_pos=(0, 0),  # filled in by the pipeline; see _place_sifa_error
                    ph_pos=(align.ref_idx, align.ref_idx + 1),
                    expected_ph=f"{attr}={expected_value}",
                    predicted_ph=f"{attr}={actual_value}",
                    # The model's own certainty about THIS attribute. Absent => None,
                    # which grades to `almost` rather than `error` (FR-018).
                    confidence=actual.probs.get(attr),
                )
            )

    return errors
