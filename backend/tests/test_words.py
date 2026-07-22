from conftest import (
    AL_BAQARAH_2_1_CORRECT,
    AL_BAQARAH_2_1_SHORT_MADDS,
    AL_FATIHA_1_5,
)

from tajwid.feedback import MuaalemOutput, analyse
from tajwid.feedback.diff import diff_recitation
from tajwid.feedback.types import Span
from tajwid.feedback.words import aggregate

ALM = "الٓمٓ"


def test_every_word_appears_even_when_correct(moshaf):
    errors = diff_recitation(ALM, AL_BAQARAH_2_1_CORRECT, moshaf)
    words = aggregate(ALM, Span(sura=2, aya=1, word_idx=0), errors)

    assert len(words) == 1
    assert words[0].status == "correct"
    assert words[0].errors == []
    assert words[0].sura == 2 and words[0].aya == 1


def test_errors_land_on_the_right_word(moshaf):
    errors = diff_recitation(ALM, AL_BAQARAH_2_1_SHORT_MADDS, moshaf)
    words = aggregate(ALM, Span(sura=2, aya=1, word_idx=0), errors)

    assert words[0].status != "correct"
    assert len(words[0].errors) >= 1


def test_word_idx_is_offset_by_the_span_start(moshaf):
    # A span that starts at word 3 of the aya must report ABSOLUTE indices, so the
    # frontend can highlight word 3 without knowing where the span began.
    errors = diff_recitation(ALM, AL_BAQARAH_2_1_CORRECT, moshaf)
    words = aggregate(ALM, Span(sura=2, aya=1, word_idx=3), errors)
    assert words[0].word_idx == 3


def test_multi_word_verse_reports_every_word_in_order(moshaf):
    """The frontend renders the whole verse from `words[]` alone (FR-020/FR-023)."""
    response = analyse(MuaalemOutput.from_phonemes(AL_FATIHA_1_5), moshaf)

    assert response.status == "ok"
    assert [w.word_idx for w in response.words] == list(range(len(response.words)))
    assert all(w.sura == 1 and w.aya == 5 for w in response.words)
    # Al-Fatiha 1:5 is four words: إِيَّاكَ نَعْبُدُ وَإِيَّاكَ نَسْتَعِينُ
    assert len(response.words) == 4
    assert " ".join(w.uthmani for w in response.words) == response.uthmani_text


def test_analyse_end_to_end_on_a_correct_recitation(moshaf):
    response = analyse(MuaalemOutput.from_phonemes(AL_FATIHA_1_5), moshaf)
    assert response.status == "ok"
    assert all(w.status == "correct" for w in response.words)
    assert all(w.errors == [] for w in response.words)


def test_analyse_reports_ambiguity_instead_of_guessing(moshaf):
    """FR-006 / Constitution VI: never silently take results[0]."""
    response = analyse(MuaalemOutput.from_phonemes(AL_BAQARAH_2_1_CORRECT), moshaf)
    assert response.status == "ambiguous"
    assert response.words == []          # assert NOTHING against the learner
    assert len(response.candidates) > 1


def test_analyse_on_gibberish_asserts_nothing(moshaf):
    response = analyse(MuaalemOutput.from_phonemes("زززززز"), moshaf)
    assert response.status == "no_match"
    assert response.words == []


def test_every_error_carries_a_word_index(moshaf):
    """SC-001: the frontend performs ZERO boundary reconstruction.

    This is the gap that motivated the whole project — obad returns a flat errors[]
    keyed by character offsets with no word index anywhere.
    """
    response = analyse(MuaalemOutput.from_phonemes(AL_FATIHA_1_5), moshaf)
    for word in response.words:
        for err in word.errors:
            assert isinstance(word.word_idx, int)
            assert err.uthmani_pos is not None
