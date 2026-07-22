import quran_transcript.alphabet as alph
from quran_transcript import MoshafAttributes

from tajwid.feedback.reference import build_reference, word_index_of_char


def test_reference_of_alm_has_six_alifs(moshaf):
    # Alif-Lam-Meem is madd lazim: 6 counts, so 6 repeated alifs.
    ref = build_reference("الٓمٓ", moshaf)
    assert ref.phonemes.count(alph.phonetics.alif) == 6


def test_reference_is_cached(moshaf):
    a = build_reference("الٓمٓ", moshaf)
    b = build_reference("الٓمٓ", moshaf)
    assert a is b  # same object => cache hit, not recomputed


def test_different_moshaf_is_not_a_cache_hit():
    short = MoshafAttributes(
        rewaya="hafs", madd_monfasel_len=2, madd_mottasel_len=4,
        madd_mottasel_waqf=4, madd_aared_len=2,
    )
    long_ = MoshafAttributes(
        rewaya="hafs", madd_monfasel_len=5, madd_mottasel_len=4,
        madd_mottasel_waqf=4, madd_aared_len=6,
    )
    assert build_reference("الٓمٓ", short) is not build_reference("الٓمٓ", long_)


def test_word_index_of_char():
    text = alph.uthmani.space.join(["بِسْمِ", "ٱللَّهِ", "ٱلرَّحْمَٰنِ"])
    assert word_index_of_char(text, 0) == 0
    assert word_index_of_char(text, len("بِسْمِ") + 1) == 1
    assert word_index_of_char(text, len(text) - 1) == 2
