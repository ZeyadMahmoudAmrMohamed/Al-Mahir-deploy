from pathlib import Path

import pytest
from quran_transcript import MoshafAttributes

# --- Verified phoneme fixtures (T008 / plan D8) ------------------------------
#
# These are REAL phonetizer output, not reconstructions. Every string here was
# printed from `quran_phonetizer` and checked before any test relied on it.
#
# The source plan carried a hand-built Al-Fatiha 1:2 string, `ءَلحَمدُلِللَاهِرَبب`,
# which is WRONG: it drops a madd repetition (`لِللَا` where the phonetizer emits
# `لِللَاا`). It is used in three different tasks, so the mistake would have shown
# up as three unrelated-looking failures, none of them pointing at the input.
#
# `test_phoneme_fixtures_match_the_phonetizer` in test_golden.py guards these:
# if upstream ever changes, we find out here rather than three tasks downstream.

# Al-Fatiha 1:5 (إِيَّاكَ نَعْبُدُ وَإِيَّاكَ نَسْتَعِينُ) — GENUINELY UNIQUE in the Quran
# (the search returns exactly 1 result). This is the fixture for "unambiguous".
AL_FATIHA_1_5 = "ءِييَااكَنَعبُدُوَءِييَااكَنَستَعِۦۦۦۦن"

# Al-Fatiha 1:2 (ٱلْحَمْدُ لِلَّهِ رَبِّ ٱلْعَٰلَمِينَ) — a MUTASHABIH: it matches SIX places
# (1:2, 6:45, 10:10, 37:182, 39:75, 40:65).
#
# The source plan used this as its *unambiguous* example, commented "occurs once".
# It does not. Praising the Lord of the Worlds is, unsurprisingly, something the Quran
# does more than once. Tightening `error_ratio` cannot rescue it — the phrase genuinely
# recurs, so the fuzzy search is right and the premise was wrong.
# Kept, because it is an excellent ambiguity fixture: a phrase a human would *swear*
# is unique to Al-Fatiha.
AL_FATIHA_1_2_MUTASHABIH = "ءَلحَمدُلِللَااهِرَببِلعَاالَمِۦۦۦۦن"

# Al-Baqarah 2:1 (الٓمٓ) recited correctly — madd lazim, 6 counts.
# Taken from obad's own docstrings, and confirmed by tests/golden/alm_correct.json
# producing zero errors.
AL_BAQARAH_2_1_CORRECT = "ءَلِفلَااااااممممِۦۦۦۦۦۦم"

# The same verse with all three madds recited far too short.
AL_BAQARAH_2_1_SHORT_MADDS = "ءَلِفلَااممِۦۦم"

# Al-Ikhlas 112:1 (قُلْ هُوَ ٱللَّهُ أَحَدٌ) — unique, and thousands of words away from
# Al-Fatiha. The "reciter jumped somewhere else entirely" fixture.
AL_IKHLAS_112_1 = "قُلهُوَللَااهُءَحَدڇ"


@pytest.fixture
def moshaf() -> MoshafAttributes:
    """The default Hafs moshaf. Matches obad's DEFAULT_MOSHAF in app/types.py."""
    return MoshafAttributes(
        rewaya="hafs",
        madd_monfasel_len=4,
        madd_mottasel_len=4,
        madd_mottasel_waqf=4,
        madd_aared_len=4,
    )


@pytest.fixture
def golden_path() -> Path:
    return Path(__file__).parent / "golden"
