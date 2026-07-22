"""The adapter, tested against stand-ins that MIRROR the real model's dataclasses.

The stand-ins below reproduce the exact attribute names from obad's
`quran_muaalem/muaalem_typing.py` (Unit / SingleUnit / Sifa). We cannot import the real
ones — they pull in torch — but if obad renames a field, these tests are what should
catch the drift, so they are written to look exactly like the real thing.
"""

from dataclasses import dataclass

import pytest

from tajwid.feedback.adapt import from_muaalem
from tajwid.feedback.types import SIFA_ATTRS


# --- mirrors of quran_muaalem.muaalem_typing (do not "simplify") -------------
@dataclass
class _Unit:
    text: str
    probs: list

@dataclass
class _SingleUnit:
    text: str
    prob: float

@dataclass
class _Sifa:
    phonemes_group: str
    hams_or_jahr: object = None
    shidda_or_rakhawa: object = None
    tafkheem_or_taqeeq: object = None
    itbaq: object = None
    safeer: object = None
    qalqla: object = None
    tikraar: object = None
    tafashie: object = None
    istitala: object = None
    ghonna: object = None

@dataclass
class _MuaalemOutput:
    phonemes: _Unit
    sifat: list


def _sifa(group, **overrides):
    attrs = {a: _SingleUnit(text="x", prob=0.9) for a in SIFA_ATTRS}
    attrs.update(overrides)
    return _Sifa(phonemes_group=group, **attrs)


def test_adapts_text_and_probs():
    raw = _MuaalemOutput(
        phonemes=_Unit(text="ءَلِ", probs=[0.9, 0.8, 0.7, 0.6]),
        sifat=[_sifa("ءَ"), _sifa("لِ")],
    )
    out = from_muaalem(raw)
    assert out.phonemes.text == "ءَلِ"
    assert out.phonemes.probs == [0.9, 0.8, 0.7, 0.6]
    assert len(out.sifat) == 2
    assert out.sifat[0].phonemes_group == "ءَ"
    assert out.sifat[0].attrs["hams_or_jahr"] == "x"
    assert out.sifat[0].probs["hams_or_jahr"] == 0.9


def test_none_probs_pass_through():
    raw = _MuaalemOutput(phonemes=_Unit(text="ءَ", probs=None), sifat=[_sifa("ءَ")])
    assert from_muaalem(raw).phonemes.probs is None


def test_a_none_sifa_attribute_is_simply_absent():
    """Not every phoneme has every attribute; the model sends None for those."""
    raw = _MuaalemOutput(
        phonemes=_Unit(text="ءَ", probs=[0.9, 0.9]),
        sifat=[_sifa("ءَ", safeer=None)],
    )
    out = from_muaalem(raw)
    assert "safeer" not in out.sifat[0].attrs   # absent, not guessed


def test_mismatched_probs_length_fails_loudly():
    """THE #1 INTEGRATION BUG, turned from silent corruption into a crash.

    `probs` must be one float per CHARACTER. If the model hands us one per phoneme GROUP
    (here: 2 groups vs 4 chars), we must refuse rather than quietly score every finding
    against the wrong phoneme's probability.
    """
    raw = _MuaalemOutput(
        phonemes=_Unit(text="ءَلِ", probs=[0.9, 0.8]),  # 2 probs, 4 chars
        sifat=[_sifa("ءَ"), _sifa("لِ")],
    )
    with pytest.raises(ValueError, match="MUST align"):
        from_muaalem(raw)


def test_the_adapted_output_actually_runs_through_analyse(moshaf):
    """End to end: a model-shaped object → adapter → real feedback."""
    from tajwid.feedback import analyse
    from tajwid.feedback.mock import mock_output

    # Borrow the mock to get a REAL phoneme string + sifat, then re-wrap it in the
    # model's dataclass shape so the adapter is genuinely exercised.
    ours = mock_output(1, 5, moshaf)
    raw = _MuaalemOutput(
        phonemes=_Unit(text=ours.phonemes.text, probs=ours.phonemes.probs),
        sifat=[
            _sifa(s.phonemes_group,
                  **{a: _SingleUnit(text=s.attrs[a], prob=s.probs[a])
                     for a in s.attrs})
            for s in ours.sifat
        ],
    )

    response = analyse(from_muaalem(raw), moshaf)
    assert response.status == "ok"
    assert all(w.status == "correct" for w in response.words)
