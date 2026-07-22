"""A stand-in for the Muaalem model.

The real model is a multi-level CTC network behind torch and a GPU. We need none of
that: our input is plain data — a phoneme string, one float per character, and ten
articulation attributes per phoneme group — so the model can be faked *exactly*, not
approximated, using the phonetizer we already ship.

This is shipped (not hidden in tests/) on purpose: **the frontend team can build against
it today**, without waiting for the model to be wired in-process. Everything here comes
from real ayat, phonetized by the real phonetizer in the reciter's real moshaf. The only
fiction is the confidence numbers, chosen so a caller can demonstrate what they do.
"""

import re

from quran_transcript import Aya, MoshafAttributes, chunck_phonemes

from .reference import build_reference
from .types import SIFA_ATTRS, MuaalemOutput, Phonemes, PredictedSifa


def phonemes_of(sura: int, aya: int, moshaf: MoshafAttributes) -> str:
    """What a PERFECT recitation of this aya sounds like, in this reciter's style."""
    return build_reference(Aya(sura, aya).get().uthmani, moshaf).phonemes


def shorten_a_madd(phonemes: str, keep: int = 1) -> str:
    """Cut the LAST long vowel run short — the classic beginner mistake.

    A madd is encoded as the same phoneme repeated (a 6-count madd is the letter six
    times), so shortening one is literally deleting repetitions. There are no durations
    anywhere in this system and none are needed (DF-003).
    """
    runs = list(re.finditer(r"(.)\1{2,}", phonemes))
    if not runs:
        return phonemes

    run = runs[-1]
    return phonemes[: run.start()] + run.group(1) * keep + phonemes[run.end() :]


def substitute_a_letter(phonemes: str, old: str, new: str) -> str:
    """Recite the wrong letter — e.g. ق where ك was written."""
    return phonemes.replace(old, new, 1)


def mock_output(
    sura: int,
    aya: int,
    moshaf: MoshafAttributes,
    *,
    recited: str | None = None,
    prob: float = 0.97,
    sifa_prob: float = 0.97,
    unsure_span: tuple[int, int] | None = None,
    unsure_prob: float = 0.35,
    wrong_sifat: dict[int, dict[str, str]] | None = None,
) -> MuaalemOutput:
    """Fake a MuaalemOutput for `sura:aya`, exactly as the real model would emit it.

    recited:     what the learner ACTUALLY said (defaults to a perfect recitation).
    prob:        the model's confidence in each predicted phoneme.
    unsure_span: char range of `recited` the model was NOT sure about — "the audio was
                 muddy here", the case that must degrade to `almost` rather than accuse.
    wrong_sifat: {group_idx: {attr: wrong_value}} — an articulation slip.

    `probs` is ONE FLOAT PER CHARACTER of the phoneme string, matching the real model.
    Getting that length wrong is the easiest way to silently score every finding against
    the wrong phoneme, so this is deliberate about it.
    """
    ref = build_reference(Aya(sura, aya).get().uthmani, moshaf)
    text = recited if recited is not None else ref.phonemes

    probs = [prob] * len(text)
    if unsure_span:
        start, end = unsure_span
        for i in range(max(0, start), min(len(text), end)):
            probs[i] = unsure_prob

    wrong_sifat = wrong_sifat or {}
    sifat: list[PredictedSifa] = []

    for idx, group in enumerate(chunck_phonemes(text)):
        # A correct recitation aligns 1:1 with the reference, and so does a wrong-length
        # madd — grouping collapses repeated letters, which is exactly why the alignment
        # is madd-invariant to begin with.
        source = ref.sifat[idx] if idx < len(ref.sifat) else ref.sifat[-1]

        attrs = {attr: getattr(source, attr) for attr in SIFA_ATTRS}
        attrs.update(wrong_sifat.get(idx, {}))

        sifat.append(
            PredictedSifa(
                phonemes_group=group,
                attrs=attrs,
                probs={attr: sifa_prob for attr in SIFA_ATTRS},
            )
        )

    return MuaalemOutput(phonemes=Phonemes(text=text, probs=probs), sifat=sifat)
