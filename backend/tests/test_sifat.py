from tajwid.feedback.reference import build_reference
from tajwid.feedback.sifat import compare_sifat
from tajwid.feedback.types import SIFA_ATTRS, PredictedSifa

ALM = "الٓمٓ"


def _predicted_from_reference(ref, overrides=None, prob=1.0):
    """A predicted sifat list identical to the reference, optionally corrupted.

    overrides: {group_idx: {attr: wrong_value}}
    """
    overrides = overrides or {}
    out = []
    for idx, s in enumerate(ref.sifat):
        attrs = {attr: getattr(s, attr) for attr in SIFA_ATTRS}
        attrs.update(overrides.get(idx, {}))
        out.append(
            PredictedSifa(
                phonemes_group=s.phonemes,
                attrs=attrs,
                probs={k: prob for k in attrs},
            )
        )
    return out


def test_matching_sifat_produce_no_errors(moshaf):
    ref = build_reference(ALM, moshaf)
    predicted = _predicted_from_reference(ref)

    errors = compare_sifat(ref.sifat, predicted, ref.phonemes, ref.phonemes)
    assert errors == []


def test_wrong_tafkheem_is_a_sifa_error(moshaf):
    ref = build_reference(ALM, moshaf)
    # Recite a moraqaq letter as mofakham — a classic makhraj/sifa slip.
    assert ref.sifat[0].tafkheem_or_taqeeq == "moraqaq"  # guard: not a vacuous test
    predicted = _predicted_from_reference(
        ref, {0: {"tafkheem_or_taqeeq": "mofakham"}}
    )

    errors = compare_sifat(ref.sifat, predicted, ref.phonemes, ref.phonemes)
    assert len(errors) == 1

    err = errors[0]
    assert err.error_type == "sifa"
    # (D6) The source plan wrote:
    #     assert "tafkheem_or_taqeeq" in errors[0].expected_ph or errors[0].predicted_ph
    # which parses as (... in expected_ph) OR (predicted_ph). `predicted_ph` is a
    # non-empty string, so the whole expression is ALWAYS true — the assertion could
    # not fail no matter what the code did. Assert on each field separately.
    assert err.expected_ph == "tafkheem_or_taqeeq=moraqaq"
    assert err.predicted_ph == "tafkheem_or_taqeeq=mofakham"


def test_only_the_wrong_attribute_is_reported(moshaf):
    """Nine attributes match; exactly one does not. Report one error, not ten."""
    ref = build_reference(ALM, moshaf)
    predicted = _predicted_from_reference(ref, {2: {"hams_or_jahr": "jahr"}})

    errors = compare_sifat(ref.sifat, predicted, ref.phonemes, ref.phonemes)
    assert len(errors) == 1
    assert errors[0].expected_ph.startswith("hams_or_jahr=")


def test_sifa_confidence_comes_from_the_models_probability(moshaf):
    """The model's per-attribute probability must survive to the finding (FR-002)."""
    ref = build_reference(ALM, moshaf)
    predicted = _predicted_from_reference(
        ref, {0: {"tafkheem_or_taqeeq": "mofakham"}}, prob=0.42
    )

    errors = compare_sifat(ref.sifat, predicted, ref.phonemes, ref.phonemes)
    assert errors[0].confidence == 0.42


def test_truncated_prediction_invents_no_errors_for_the_missing_tail(moshaf):
    """A reciter who stopped early made a DELETION, not ten articulation mistakes.

    (D6) The source plan asserted `all(e.error_type == "sifa" ...)` over a list whose
    every element is `sifa` by construction — another test that cannot fail. The real
    intent is that the unaligned tail produces NO sifa errors at all: the missing
    phonemes are already reported as deletions by the diff, and re-billing them as
    articulation errors would charge the learner twice for one mistake.
    """
    ref = build_reference(ALM, moshaf)
    predicted = _predicted_from_reference(ref)[:1]  # reciter stopped after one group

    errors = compare_sifat(ref.sifat, predicted, ref.phonemes, ref.phonemes)
    assert len(errors) == 0


def test_a_substituted_letter_is_not_also_a_sifa_error(moshaf):
    """If the reciter said a DIFFERENT letter, that is a phoneme error.

    Its sifat will of course differ too — but reporting that as well would bill one
    mistake twice. Sifat are compared only where the phoneme groups align.
    """
    ref = build_reference(ALM, moshaf)
    predicted = _predicted_from_reference(ref)

    # Pretend the reciter said a completely different phoneme string.
    errors = compare_sifat(ref.sifat, predicted, ref.phonemes, "قققق")
    assert errors == []
