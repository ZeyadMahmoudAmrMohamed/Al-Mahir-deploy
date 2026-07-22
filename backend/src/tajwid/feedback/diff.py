from collections import defaultdict

from quran_transcript import MoshafAttributes, chunck_phonemes, explain_error
from quran_transcript.phonetics.error_explainer import align_phonemes_groups

from .reference import build_reference
from .types import FeedbackError, TajweedRuleInfo


def _rule_to_info(rule) -> TajweedRuleInfo:
    return TajweedRuleInfo(
        name_ar=rule.name.ar,
        name_en=rule.name.en,
        golden_len=rule.golden_len,
        correctness_type=rule.correctness_type,
        tag=rule.tag,
    )


def _group_bounds(groups: list[str]) -> list[tuple[int, int]]:
    """Char span of each phoneme group within the joined phoneme string."""
    bounds, pos = [], 0
    for g in groups:
        bounds.append((pos, pos + len(g)))
        pos += len(g)
    return bounds


def _predicted_span(
    ph_pos: tuple[int, int],
    ref_bounds: list[tuple[int, int]],
    pred_bounds: list[tuple[int, int]],
    alignments,
) -> tuple[int, int] | None:
    """Translate a REFERENCE char span into a PREDICTED char span.

    Alignment is by phoneme GROUP (matched on each group's first character, which is
    what makes it madd-invariant), so the journey is:

        ref char span -> ref group indices -> aligned pred group indices -> pred chars

    Returns None when the reference groups align to nothing predicted — a pure
    deletion. There is no probability to read for a phoneme the reciter never said,
    and inventing one would be the exact false confidence FR-018 forbids.
    """
    start, end = ph_pos

    ref_idxs = [
        i for i, (gs, ge) in enumerate(ref_bounds) if gs < end and ge > start
    ]
    if not ref_idxs:
        return None

    # `delete` alignments carry a pred_idx too, but it is only an insertion POINT —
    # no predicted group lives there — so they are deliberately excluded.
    aligned: dict[int, list[int]] = defaultdict(list)
    for a in alignments:
        if a.op_type in ("equal", "replace"):
            aligned[a.ref_idx].append(a.pred_idx)
        elif a.op_type == "insert":
            # Extra phonemes the reciter added; they belong to the reference group
            # they were inserted before.
            aligned[a.ref_idx].append(a.pred_idx)

    pred_idxs = [p for i in ref_idxs for p in aligned.get(i, [])]
    pred_idxs = [p for p in pred_idxs if 0 <= p < len(pred_bounds)]
    if not pred_idxs:
        return None

    return (
        min(pred_bounds[p][0] for p in pred_idxs),
        max(pred_bounds[p][1] for p in pred_idxs),
    )


def diff_recitation(
    uthmani_text: str, predicted_phonemes: str, moshaf: MoshafAttributes
) -> list[FeedbackError]:
    """Diff a predicted phoneme string against the canonical reference.

    Alignment is madd-invariant: phoneme groups are matched on their FIRST character
    only, so a madd of the wrong length still aligns and its length error is caught by
    counting repetitions inside the matched group (FR-013/FR-014). A six-count madd
    recited as two is one length error, not a cascade of desynchronised substitutions.
    """
    ref = build_reference(uthmani_text, moshaf)

    raw_errors = explain_error(
        uthmani_text=uthmani_text,
        ref_ph_text=ref.phonemes,
        predicted_ph_text=predicted_phonemes,
        mappings=ref.mappings,
    )

    ref_groups = chunck_phonemes(ref.phonemes)
    pred_groups = chunck_phonemes(predicted_phonemes)
    ref_bounds = _group_bounds(ref_groups)
    pred_bounds = _group_bounds(pred_groups)
    alignments = align_phonemes_groups(ref_groups, pred_groups)

    out = []
    for e in raw_errors:
        # Upstream splits tajweed rules across four fields. For feedback we only need
        # what SHOULD have applied — that is `ref_tajweed_rules`.
        rules = [_rule_to_info(r) for r in (e.ref_tajweed_rules or [])]
        ph_pos = tuple(e.ph_pos)

        out.append(
            FeedbackError(
                error_type=e.error_type,
                speech_error_type=e.speech_error_type,
                uthmani_pos=tuple(e.uthmani_pos),
                ph_pos=ph_pos,
                pred_ph_pos=_predicted_span(
                    ph_pos, ref_bounds, pred_bounds, alignments
                ),
                expected_ph=e.expected_ph,
                predicted_ph=e.preditected_ph,  # sic: upstream typo
                expected_len=e.expected_len,
                predicted_len=e.predicted_len,
                tajweed_rules=rules,
            )
        )
    return out
