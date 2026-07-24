"""Snapshots of untouched quran-transcript behaviour.

These lock in upstream's output so later refactors (and the Task 8 vendoring)
have something to prove they did not break the phonetizer.
"""

import json
import os
from dataclasses import asdict

import pytest
from quran_transcript import Aya, explain_error, quran_phonetizer

from conftest import AL_BAQARAH_2_1_CORRECT, AL_FATIHA_1_2_MUTASHABIH, AL_FATIHA_1_5

# (case_id, uthmani_text, predicted_phonemes)
# Predicted strings are taken from obad's own docstrings and README so we are
# snapshotting known-real model output, not invented input.
CASES = [
    # Alif-Lam-Meem recited with all three madds far too short.
    ("alm_short_madds", "الٓمٓ", "ءَلِفلَااممِۦۦم"),
    # Same verse recited correctly — must produce zero errors.
    ("alm_correct", "الٓمٓ", "ءَلِفلَااااااممممِۦۦۦۦۦۦم"),
    # Qaf substituted for kaf, and the madd shortened.
    ("qalu_wrong_letter", "قَالُوٓا۟", "كالۥۥ"),
]


def _phonetize(uthmani_text, moshaf):
    return quran_phonetizer(uthmani_text, moshaf, remove_spaces=True)


def _jsonable(obj):
    """Make upstream's dataclasses JSON-safe, and — crucially — deterministic.

    `TajweedRule.available_tags` is a `set[str]`. Python randomizes string hashing
    per process, so a set's iteration order differs between runs: dumped as-is, the
    snapshot would pass or fail depending on the interpreter's mood. Sets are
    therefore sorted, not merely listed.
    """
    if isinstance(obj, (set, frozenset)):
        return sorted(_jsonable(v) for v in obj)
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


@pytest.mark.parametrize("case_id,uthmani,predicted", CASES)
def test_golden_explain_error(case_id, uthmani, predicted, moshaf, golden_path):
    ref = _phonetize(uthmani, moshaf)
    errors = explain_error(
        uthmani_text=uthmani,
        ref_ph_text=ref.phonemes,
        predicted_ph_text=predicted,
        mappings=ref.mappings,
    )

    actual = _jsonable(
        {
            "uthmani": uthmani,
            "predicted_phonemes": predicted,
            "reference_phonemes": ref.phonemes,
            "errors": [asdict(e) for e in errors],
        }
    )

    snapshot = golden_path / f"{case_id}.json"

    if os.environ.get("UPDATE_GOLDEN") or not snapshot.exists():
        golden_path.mkdir(parents=True, exist_ok=True)
        snapshot.write_text(
            json.dumps(actual, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        pytest.skip(f"wrote golden snapshot {snapshot.name}")

    expected = json.loads(snapshot.read_text(encoding="utf-8"))
    assert actual == expected


def test_phoneme_fixtures_match_the_phonetizer(moshaf):
    """The shared phoneme constants are real phonetizer output, not reconstructions.

    T008 / plan D8. The source plan's hand-built Al-Fatiha string was wrong (it
    dropped a madd repetition) and was used by three separate tasks. Asserting the
    constants against the phonetizer means a wrong input string fails HERE, once,
    saying so — instead of surfacing as three unrelated failures elsewhere.
    """
    def phonemes_of(sura, aya):
        return quran_phonetizer(
            Aya(sura, aya).get().uthmani, moshaf, remove_spaces=True
        ).phonemes

    assert phonemes_of(1, 5) == AL_FATIHA_1_5
    assert phonemes_of(1, 2) == AL_FATIHA_1_2_MUTASHABIH

    # A correctly-recited verse must diff to nothing. This is the same invariant
    # tests/golden/alm_correct.json locks in, asserted against the constant.
    assert phonemes_of(2, 1) == AL_BAQARAH_2_1_CORRECT


def test_leen_madd_at_length_two_does_not_crash_the_phonetizer():
    """madd_alleen_len=2 made the leen substitution NET-DELETING (the pattern captures
    the sukun after the madd letter, the replacement does not restore it, and at length
    2 only one madd letter is added). That took get_mappings' `delete` branch, which
    attaches the rule to a zero-width mapping pointing at the FOLLOWING consonant, and
    the waw/yaa tag lookup raised KeyError -- discarding the reciter's whole chunk.

    Observed live on 6:143 (ٱثْنَيْنِ) as KeyError('ن'), four chunks lost in one session.
    """
    from quran_transcript import MoshafAttributes, quran_phonetizer

    from tajwid.session import default_moshaf

    cfg = default_moshaf().model_dump()
    cfg["madd_aared_len"] = 2
    cfg["madd_alleen_len"] = 2
    uthmani = "ثَمَـٰنِيَةَ أَزْوَٰجٍۢ مِّنَ ٱلضَّأْنِ ٱثْنَيْنِ وَمِنَ ٱلْمَعْزِ ٱثْنَيْنِ"

    result = quran_phonetizer(uthmani, MoshafAttributes(**cfg), remove_spaces=True)
    assert result.phonemes

    # The tag must still be recovered, not merely not-crash: the madd letter sits one
    # position before the zero-width mapping.
    tags = {
        rule.tag
        for m in result.mappings
        for rule in (m.tajweed_rules or [])
        if rule.name.en == "Leen Madd"
    }
    assert tags == {"yaa"}


def test_leen_madd_at_length_four_is_unchanged():
    """The fix must be a no-op wherever the old code worked."""
    from quran_transcript import MoshafAttributes, quran_phonetizer

    from tajwid.session import default_moshaf

    uthmani = "ثَمَـٰنِيَةَ أَزْوَٰجٍۢ مِّنَ ٱلضَّأْنِ ٱثْنَيْنِ وَمِنَ ٱلْمَعْزِ ٱثْنَيْنِ"
    result = quran_phonetizer(uthmani, default_moshaf(), remove_spaces=True)

    # Four-count leen: three extra yaas, versus one at length two.
    assert result.phonemes.endswith("ثنَييين")
    tags = {
        rule.tag
        for m in result.mappings
        for rule in (m.tajweed_rules or [])
        if rule.name.en == "Leen Madd"
    }
    assert tags == {"yaa"}
