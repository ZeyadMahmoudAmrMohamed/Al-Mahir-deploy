"""Sifat are withheld by default, but an explicit selection is still honoured whole."""

from tajwid.config import Settings
from tajwid.feedback.rules import catalogue
from tajwid.session import resolve_rules

SIFA_KEYS = {r["key"] for r in catalogue() if r["kind"] == "sifa"}
TAJWEED_KEYS = {r["key"] for r in catalogue() if r["kind"] != "sifa"}


def test_unspecified_grades_tajweed_but_no_sifat():
    got = resolve_rules(None, Settings(grade_sifat=False))
    assert got == TAJWEED_KEYS
    assert not (got & SIFA_KEYS)


def test_explicit_selection_is_honoured_including_sifat():
    # A reciter drilling ghunnah asked for it. The default being off must not veto that.
    asked = frozenset({"ghonna", "aared_madd"})
    assert resolve_rules(asked, Settings(grade_sifat=False)) == asked


def test_empty_selection_stays_empty():
    # `[]` is "hifz and tashkeel only" -- a real choice, not a missing one.
    assert resolve_rules(frozenset(), Settings(grade_sifat=False)) == frozenset()


def test_flag_on_restores_grade_everything():
    assert resolve_rules(None, Settings(grade_sifat=True)) is None


if __name__ == "__main__":
    test_unspecified_grades_tajweed_but_no_sifat()
    test_explicit_selection_is_honoured_including_sifat()
    test_empty_selection_stays_empty()
    test_flag_on_restores_grade_everything()
    print("ok")
