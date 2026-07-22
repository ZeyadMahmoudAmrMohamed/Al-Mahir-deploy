"""Adapt the real Muaalem model's output to ours.

The model (`quran_muaalem`) returns its own dataclasses — `Unit`, `SingleUnit`,
`Sifa` — carrying torch tensors. We take plain pydantic data. Both hold the SAME
information; this is a shape change, not a conversion.

Kept here (not in a script) because it is real integration code that deserves a test,
and because it is the single place the #1 silent bug can be caught: `probs` must be one
float per CHARACTER of the phoneme text. Get that wrong and every confidence lands on
the wrong phoneme — invisibly. So we assert it, loudly, at the boundary.

`from_muaalem` is duck-typed: it never imports `quran_muaalem` (which needs torch), it
just reads the attributes the real objects are documented to have. That keeps our
component torch-free and lets this run and be tested with a stand-in.
"""

from .types import SIFA_ATTRS, MuaalemOutput, Phonemes, PredictedSifa


def from_muaalem(raw) -> MuaalemOutput:
    """Convert a `quran_muaalem.MuaalemOutput` into ours.

    `raw` must look like the model's output:
        raw.phonemes.text  : str
        raw.phonemes.probs : sequence[float] | None   (one per CHARACTER of text)
        raw.sifat          : list, each with .phonemes_group and 10 attributes,
                             each attribute a `SingleUnit(.text, .prob)` or None.
    """
    text = raw.phonemes.text
    probs = raw.phonemes.probs

    if probs is not None:
        probs = [float(p) for p in probs]
        if len(probs) != len(text):
            raise ValueError(
                f"phonemes.probs has {len(probs)} values but phonemes.text has "
                f"{len(text)} characters. We slice probs by character offset, so they "
                "MUST align 1:1. If the model emits one probability per phoneme GROUP "
                "instead of per character, expand it before calling us — otherwise "
                "every confidence silently attaches to the wrong phoneme."
            )

    sifat = []
    for s in raw.sifat:
        attrs: dict[str, str] = {}
        sprobs: dict[str, float] = {}
        for attr in SIFA_ATTRS:
            unit = getattr(s, attr, None)
            if unit is not None:
                attrs[attr] = unit.text
                sprobs[attr] = float(unit.prob)
        sifat.append(
            PredictedSifa(
                phonemes_group=s.phonemes_group, attrs=attrs, probs=sprobs
            )
        )

    return MuaalemOutput(phonemes=Phonemes(text=text, probs=probs), sifat=sifat)
