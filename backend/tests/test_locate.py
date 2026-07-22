from quran_transcript import Aya, MoshafAttributes, quran_phonetizer

from conftest import AL_BAQARAH_2_1_CORRECT, AL_FATIHA_1_2_MUTASHABIH, AL_FATIHA_1_5
from tajwid.feedback.locate import locate


def test_unambiguous_phrase_resolves_to_one_span():
    # Al-Fatiha 1:5 — genuinely unique in the Quran (verified: the search returns
    # exactly one result). NOT 1:2, which the source plan believed was unique and
    # which in fact matches six places.
    result = locate(AL_FATIHA_1_5, error_ratio=0.1)
    assert result.status == "ok"
    assert result.span.sura == 1
    assert result.span.aya == 5
    # Derive the expected text; do not hand-type Arabic. A retyped literal differs
    # from the real one in invisible ways (diacritic codepoints and their order),
    # which is how the source plan's phoneme string went wrong in the first place.
    assert result.uthmani_text == Aya(1, 5).get().uthmani


def test_mutashabih_phrase_reports_ambiguous_with_candidates():
    # Alif-Lam-Meem opens 6 suras; obad's app silently returns the first.
    result = locate(AL_BAQARAH_2_1_CORRECT, error_ratio=0.1)
    assert result.status == "ambiguous"
    assert len(result.candidates) > 1
    assert {c.sura for c in result.candidates} >= {2, 3}
    # An ambiguous result asserts nothing: no span, no text to score against.
    assert result.span is None


def test_candidates_carry_their_text_not_just_coordinates():
    """`[(2, 147), (3, 60)]` is not an answer anyone can act on.

    A caller shown bare coordinates has to go and look each one up before they can
    decide anything — the same reconstruction burden this contract abolishes for
    words. Ship the verse.
    """
    result = locate(AL_BAQARAH_2_1_CORRECT, error_ratio=0.1)
    assert result.status == "ambiguous"

    for c in result.candidates:
        assert c.uthmani_text, "a candidate with no text is a lookup, not an answer"
        assert c.end is not None
        # The text really is the verse it claims to be.
        assert c.uthmani_text in Aya(c.sura, c.aya).get().uthmani


def test_a_phrase_that_feels_unique_but_is_not():
    """`ٱلْحَمْدُ لِلَّهِ رَبِّ ٱلْعَٰلَمِينَ` matches SIX places, not one.

    This is the exact case obad's `results[0]` silently gets wrong: a learner
    reciting Al-Fatiha would be located to Al-An'am 6:45 if the index happened to
    order it first, and then scored against the wrong verse.
    """
    result = locate(AL_FATIHA_1_2_MUTASHABIH, error_ratio=0.1)
    assert result.status == "ambiguous"
    assert {c.sura for c in result.candidates} >= {1, 6, 10}


def test_gibberish_returns_no_match():
    """Gibberish must not come back as a 1,599-way "ambiguity".

    The search query is madd-NORMALISED, so repeated letters collapse: `زززززز`
    becomes a single `ز`, which "matches" 1,599 places in the Quran. Reporting that
    as `ambiguous` would be technically true and operationally useless — a candidate
    list nobody can act on. Above `max_candidates` we say `no_match` instead: we
    cannot identify this, recite more.
    """
    result = locate("زززززز", error_ratio=0.0)
    assert result.status == "no_match"
    assert result.span is None
    assert result.candidates == []


def test_a_real_ambiguity_still_reports_its_candidates():
    """The `no_match` ceiling must not swallow genuine, actionable ambiguity."""
    result = locate(AL_BAQARAH_2_1_CORRECT, error_ratio=0.1)
    assert result.status == "ambiguous"
    assert 1 < len(result.candidates) <= 10


def test_empty_input_returns_no_match():
    assert locate("", error_ratio=0.1).status == "no_match"


def test_identification_is_style_invariant():
    """FR-005 / SC-002: the same verse recited in a DIFFERENT style must locate to
    the SAME span.

    The search index is madd-NORMALISED; scoring uses the reciter's real madd
    lengths. If identification ever started matching on full-length phonemes, every
    reciter whose style differs from the default would be silently mislocated — and
    then scored against the wrong verse. Nothing else in the suite guards this.
    """
    short = MoshafAttributes(
        rewaya="hafs", madd_monfasel_len=2, madd_mottasel_len=4,
        madd_mottasel_waqf=4, madd_aared_len=2,
    )
    long_ = MoshafAttributes(
        rewaya="hafs", madd_monfasel_len=5, madd_mottasel_len=5,
        madd_mottasel_waqf=6, madd_aared_len=6,
    )

    uthmani = Aya(1, 5).get().uthmani
    ph_short = quran_phonetizer(uthmani, short, remove_spaces=True).phonemes
    ph_long = quran_phonetizer(uthmani, long_, remove_spaces=True).phonemes

    # Different styles really do produce different phoneme strings...
    assert ph_short != ph_long

    # ...and must still land on the same verse.
    a = locate(ph_short, error_ratio=0.1)
    b = locate(ph_long, error_ratio=0.1)
    assert a.status == "ok" and b.status == "ok"
    assert (a.span.sura, a.span.aya) == (1, 5)
    assert (a.span.sura, a.span.aya) == (b.span.sura, b.span.aya)
