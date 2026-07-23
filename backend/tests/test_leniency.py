"""Leniency: grade only the rules the reciter is working on (feedback.rules).

The load-bearing claim is not "the error disappears from the list" — it is that the
WORD stops being marked, because the frontend paints from `status`. A filter that hid
the error but left the word red would be worse than no filter at all.
"""

import pytest
from conftest import AL_FATIHA_1_5
from fastapi.testclient import TestClient

from tajwid.asr.engine import MockEngine
from tajwid.config import Settings
from tajwid.feedback import MuaalemOutput, analyse
from tajwid.feedback.rules import catalogue, filter_rules, rule_key
from tajwid.feedback.types import FeedbackError
from tajwid.main import create_app
from tajwid.session import LiveSession

# ءِيَّاكَ نَعْبُدُ وَإِيَّاكَ نَسْتَعِينُ with two independent madd faults: a normal madd
# (word 0) held for 1 instead of 2, and the aared madd (word 3) held for 2 instead of 4.
# Two rules, two different words — so a filter that keeps one and drops the other is
# visible in the statuses, not just the error count.
TWO_MADD_FAULTS = AL_FATIHA_1_5.replace("عِۦۦۦۦن", "عِۦۦن").replace("ءِييَاا", "ءِييَا", 1)


def _rules_of(response) -> set[str]:
    return {
        rule_key(r.name_en)
        for w in response.words
        for e in w.errors
        for r in e.tajweed_rules
    }


def test_no_selection_grades_everything(moshaf):
    """The default. Absent leniency must behave exactly as before the feature existed."""
    response = analyse(MuaalemOutput.from_phonemes(TWO_MADD_FAULTS), moshaf)

    assert _rules_of(response) == {"normal_madd", "aared_madd"}
    assert [w.status for w in response.words] == ["almost", "correct", "correct", "almost"]


def test_selecting_one_rule_silences_the_other_word(moshaf):
    response = analyse(
        MuaalemOutput.from_phonemes(TWO_MADD_FAULTS),
        moshaf,
        rules=frozenset({"aared_madd"}),
    )

    assert _rules_of(response) == {"aared_madd"}
    # Word 0's normal-madd fault is not merely hidden: the word is CORRECT again.
    assert response.words[0].status == "correct"
    assert response.words[0].errors == []
    assert response.words[3].status == "almost"


def test_empty_selection_still_grades_hifz_and_tashkeel(moshaf):
    """`rules=[]` narrows tajwid to nothing. It does not stop checking the recitation."""
    wrong_haraka = AL_FATIHA_1_5.replace("نَعبُدُ", "نَعبُدَ")
    extra_letter = AL_FATIHA_1_5.replace("نَعبُدُ", "نَعبُدُس")

    for phonemes, expected in ((wrong_haraka, "tashkeel"), (extra_letter, "normal")):
        response = analyse(
            MuaalemOutput.from_phonemes(phonemes), moshaf, rules=frozenset()
        )
        kinds = {e.error_type for w in response.words for e in w.errors}
        assert kinds == {expected}

    # ...while the madd faults from the same empty selection are gone.
    lenient = analyse(
        MuaalemOutput.from_phonemes(TWO_MADD_FAULTS), moshaf, rules=frozenset()
    )
    assert all(w.status == "correct" for w in lenient.words)


def test_sifa_findings_filter_on_their_attribute():
    """Sifa errors carry their attribute in `expected_ph` as "<attr>=<value>"."""
    ghonna = FeedbackError(
        error_type="sifa",
        speech_error_type="replace",
        uthmani_pos=(0, 1),
        ph_pos=(0, 1),
        expected_ph="ghonna=maghnoon",
        predicted_ph="ghonna=not_maghnoon",
    )
    safeer = ghonna.model_copy(
        update={"expected_ph": "safeer=safeer", "predicted_ph": "safeer=no_safeer"}
    )

    assert filter_rules([ghonna, safeer], frozenset({"ghonna"})) == [ghonna]
    assert filter_rules([ghonna, safeer], frozenset()) == []
    assert filter_rules([ghonna, safeer], None) == [ghonna, safeer]


def test_unattributed_tajweed_findings_survive_any_selection():
    """A skipped rule-bearing phoneme is marked `tajweed` with NO rule attached
    (error_explainer's `delete` branch). There is no key to match it against, and
    dropping it would hide a real miss behind a filter set for something else."""
    orphan = FeedbackError(
        error_type="tajweed",
        speech_error_type="delete",
        uthmani_pos=(0, 1),
        ph_pos=(0, 1),
        expected_ph="اا",
        predicted_ph="",
    )
    assert filter_rules([orphan], frozenset({"qalqalah"})) == [orphan]
    assert filter_rules([orphan], frozenset()) == [orphan]


def test_an_unknown_strictness_costs_the_setting_not_the_session():
    """`STRICTNESS[strictness]` is read per chunk, deep in the pipeline — so a bad value
    used to raise KeyError mid-recitation and close the socket with code 1000 ("OK"),
    indistinguishable from a normal end. Resolve it at the boundary instead."""
    from tajwid.session import resolve_strictness

    assert resolve_strictness("strict") == "strict"
    assert resolve_strictness(None) == "normal"
    assert resolve_strictness("") == "normal"
    assert resolve_strictness("Normal") == "normal"  # capital N used to be fatal
    assert resolve_strictness("medium") == "normal"

    # And the resolved value is always one the pipeline can actually index.
    from tajwid.feedback.confidence import STRICTNESS

    for raw in ("strict", None, "", "Normal", "medium", "bogus"):
        assert resolve_strictness(raw) in STRICTNESS


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("TAJWID_ASR_ENGINE", "mock")
    from tajwid.config import get_settings

    get_settings.cache_clear()
    with TestClient(create_app()) as c:
        yield c
    get_settings.cache_clear()


def test_the_catalogue_endpoint_serves_what_the_filter_understands(client):
    """The chips the frontend draws must be keys the backend can act on."""
    body = client.get("/tajweed-rules").json()
    keys = {r["key"] for r in body["rules"]}

    assert keys == {e["key"] for e in catalogue()}
    assert "aared_madd" in keys and "ghonna" in keys


def test_an_empty_rules_list_is_a_choice_not_a_missing_value():
    """The one place this could quietly break: `[]` and absent both look falsy, and
    collapsing them would turn "grade nothing" into "grade everything" — the exact
    opposite of what the reciter asked for."""
    engine = MockEngine(Settings(asr_engine="mock"))
    make = lambda rules: LiveSession(engine, session_id="t", rules=rules)  # noqa: E731

    # An ABSENT selection no longer means "grade everything": sifat are withheld by
    # default (Settings.grade_sifat, see FINDINGS.md), so it resolves to every tajwid
    # rule and no sifa. The distinction this test exists to protect is unchanged --
    # absent still grades the tajwid rules, `[]` still grades none of them.
    absent = make(None).state.rules
    assert absent and "aared_madd" in absent and "ghonna" not in absent
    assert make(frozenset()).state.rules == frozenset()
    assert make(frozenset({"ghonna"})).state.rules == frozenset({"ghonna"})

    # And the same distinction as api/ws.py parses it off the start message.
    for raw, expected in (({}, None), ({"rules": None}, None), ({"rules": []}, frozenset())):
        got = raw.get("rules")
        assert (frozenset(got) if got is not None else None) == expected


def test_the_documented_rule_table_matches_the_real_catalogue():
    """`GET /tajweed-rules`'s docstring reproduces the whole mapping so the frontend can
    be built without a running server. That is duplicated data, and duplicated data
    drifts — a table that quietly disagreed with the catalogue would hand the frontend
    team keys the backend does not accept. Parse it back and compare."""
    import re

    from tajwid.api.rest import tajweed_rules

    rows = re.findall(
        r"^\s*\|\s*`([a-z_]+)`\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*(tajweed|sifa)\s*\|$",
        tajweed_rules.__doc__,
        re.MULTILINE,
    )
    documented = {k: (ar, en, kind) for k, ar, en, kind in rows}
    actual = {e["key"]: (e["name_ar"], e["name_en"], e["kind"]) for e in catalogue()}

    assert documented == actual


def test_every_catalogue_key_is_one_a_filter_can_actually_match():
    """The catalogue is the contract the frontend builds its chips from. A key in it
    that nothing emits would be a checkbox that silently does nothing."""
    entries = catalogue()
    keys = [e["key"] for e in entries]

    assert len(keys) == len(set(keys)), "duplicate key — two chips, one meaning"
    assert "aared_madd" in keys and "ghonna" in keys
    assert all(e["name_ar"] and e["name_en"] for e in entries)
    assert {e["kind"] for e in entries} == {"tajweed", "sifa"}
