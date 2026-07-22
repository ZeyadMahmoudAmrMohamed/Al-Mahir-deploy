from quran_transcript import MoshafAttributes, chunck_phonemes
from quran_transcript.phonetics.error_explainer import extract_ref_phonetic_to_uthmani

from .confidence import STRICTNESS, score_errors
from .diff import diff_recitation
from .locate import locate
from .reference import build_reference
from .rules import filter_rules
from .session import SessionState, advance
from .sifat import compare_sifat
from .track import track
from .types import FeedbackError, FeedbackResponse, LocateResult, MuaalemOutput
from .nonverse import strip_non_verse
from .words import aggregate, trim_edges


def _place_sifa_error(err: FeedbackError, ref, ref_groups: list[str]) -> FeedbackError:
    """Convert a sifa error's phoneme-GROUP index into an Uthmani char offset.

    Sifat come out of the model per phoneme group, but `aggregate` places errors on
    words by Uthmani character offset. `mappings` is the spine that connects the two
    (DF-006): phoneme index -> uthmani char -> word.
    """
    group_idx = err.ph_pos[0]

    # Char offset of this group's first phoneme within the phonetic string.
    ph_char_start = sum(len(g) for g in ref_groups[:group_idx])

    ph_to_uth = extract_ref_phonetic_to_uthmani(ref.mappings)
    uth_idx = ph_to_uth.get(ph_char_start, 0)

    return err.model_copy(
        update={
            "uthmani_pos": (uth_idx, uth_idx + 1),
            "ph_pos": (ph_char_start, ph_char_start + 1),
        }
    )


def analyse(
    output: MuaalemOutput,
    moshaf: MoshafAttributes,
    strictness: str = "normal",
    error_ratio: float = 0.1,
    rules: frozenset[str] | None = None,
) -> FeedbackResponse:
    """Model phonetic output -> per-word feedback. Cold (no session cursor).

    locate -> reference -> diff (+ sifat) -> score -> aggregate.

    When the verse cannot be pinned down, this returns `ambiguous` or `no_match` and
    asserts NOTHING against the learner: no words, no errors. Guessing would mean
    scoring someone against a verse they were not reciting, which is the worst thing
    this system can do (Constitution VI).

    If `output.phonemes.probs` is absent, every finding is UNSCORED and therefore
    reported as `almost`, never `error`. That is deliberate: unknown is not certain.

    `rules` restricts which tajwid/sifa rules are graded (see feedback.rules); None
    grades everything.
    """
    found = locate(output.phonemes.text, error_ratio=error_ratio)
    return _analyse_located(found, output, moshaf, strictness, rules=rules)


def analyse_session(
    output: MuaalemOutput,
    state: SessionState,
    error_ratio: float = 0.1,
) -> tuple[FeedbackResponse, SessionState]:
    """Session-aware analysis: track around the cursor, fall back to a cold search.

    This is the path that matters for streaming. `analyse` runs a full-Quran fuzzy
    search on every call, which is fine for record-then-submit; tracking searches a
    window around where the reciter actually is.

    It also structurally eliminates the mutashabihat problem. A cold search for
    `ٱلْحَمْدُ لِلَّهِ رَبِّ ٱلْعَٰلَمِينَ` can only say `ambiguous` — it matches six places. A
    reciter who was just at Al-Fatiha 1:1 is at 1:2, and the cursor knows it.

    Returns the feedback plus the advanced session state. The caller holds the state.
    """
    # (FR-009) Strip istiaatha / basmalah / sadaka BEFORE anything else. They are
    # correct, non-Quranic, and would otherwise be diffed against the verse.
    verse_phonemes, non_verse, start, end = strip_non_verse(
        output.phonemes.text, state.moshaf
    )

    if not verse_phonemes:
        # The learner recited only non-verse text. Nothing to score, nothing to
        # accuse, and no verse to pass — so the cursor does not move.
        return (
            FeedbackResponse(
                status="ok",
                predicted_phonemes=output.phonemes.text,
                non_verse=non_verse,
            ),
            state,
        )

    # `probs` is indexed over the string we just cut, so it must be cut identically.
    # Leaving it whole would score every finding against some other phoneme's
    # probability — silently, and only when the learner said the istiaatha.
    probs = output.phonemes.probs
    stripped = output.model_copy(
        update={
            "phonemes": output.phonemes.model_copy(
                update={
                    "text": verse_phonemes,
                    "probs": probs[start:end] if probs else None,
                }
            )
        }
    )

    if state.cursor is None:
        found = locate(verse_phonemes, error_ratio=error_ratio)
    else:
        found = track(
            verse_phonemes, state.cursor, state.moshaf, penalty=state.penalty
        )
        if found.status != "ok":
            # The reciter jumped somewhere else entirely. Re-locate cold rather than
            # force a wrong answer out of the window.
            found = locate(verse_phonemes, error_ratio=error_ratio)

    response = _analyse_located(
        found, stripped, state.moshaf, state.strictness, trim=True, rules=state.rules
    )
    response.non_verse = non_verse
    return response, advance(state, found)


def _analyse_located(
    found: LocateResult,
    output: MuaalemOutput,
    moshaf: MoshafAttributes,
    strictness: str,
    trim: bool = False,
    rules: frozenset[str] | None = None,
) -> FeedbackResponse:
    """Everything downstream of "we know which words these are".

    `trim` is for the STREAMING path only (FR-010). A record-then-submit recitation was
    not cut by us, so it has no boundary artefacts to forgive.
    """
    predicted = output.phonemes.text

    if found.status != "ok":
        return FeedbackResponse(
            status=found.status,
            predicted_phonemes=predicted,
            candidates=found.candidates,
        )

    ref = build_reference(found.uthmani_text, moshaf)
    errors = diff_recitation(found.uthmani_text, predicted, moshaf)

    # Phoneme-level findings are scored from phonemes.probs. Sifa findings already
    # carry their own confidence (each attribute's own probability), so they are
    # scored at birth in compare_sifat and must not be re-scored here.
    errors = score_errors(errors, output.phonemes.probs)

    if output.sifat:
        ref_groups = chunck_phonemes(ref.phonemes)
        sifa_errors = compare_sifat(ref.sifat, output.sifat, ref.phonemes, predicted)
        errors.extend(
            _place_sifa_error(e, ref, ref_groups) for e in sifa_errors
        )

    # Leniency: drop findings for rules this reciter is not working on. It happens HERE,
    # upstream of `aggregate`, because word `status` is derived from the error list —
    # filtering afterwards would leave a word painted red over a mistake we then refuse
    # to show, which is the worst of both.
    errors = filter_rules(errors, rules)

    words = aggregate(
        found.uthmani_text, found.span, errors, thresholds=STRICTNESS[strictness]
    )

    if trim and found.end is not None:
        words = trim_edges(words, found.span, found.end)

    return FeedbackResponse(
        status="ok",
        span=found.span,
        end=found.end,
        uthmani_text=found.uthmani_text,
        predicted_phonemes=predicted,
        reference_phonemes=ref.phonemes,
        words=words,
    )
