"""The client's moshaf choice must actually reach the model.

Regression for a real bug: `/moshaf-schema` hides any field with fewer than 2 options
(nothing for the reciter to choose) -- and `rewaya` is Literal["hafs"], exactly one
option, so it's hidden. The frontend's config therefore never includes `rewaya`. But
`rewaya` is a REQUIRED field on MoshafAttributes with no default, so validating the
client's payload as-is always raised, and the old code caught that broadly and silently
fell back to `default_moshaf(settings)` -- discarding every field the reciter DID set,
including madd_monfasel_len. A reciter choosing "2 harakat" was always scored against
the settings-wide default (4) instead.
"""

from __future__ import annotations

from tajwid.config import Settings
from tajwid.session import default_moshaf, resolve_moshaf

# Exactly what the frontend sends: every /moshaf-schema field (>=2 options), never
# `rewaya` (1 option, filtered out of the schema).
FRONTEND_SHAPED_PAYLOAD = {
    "recitation_speed": "murattal",
    "takbeer": "no_takbeer",
    "madd_monfasel_len": 2,
    "madd_mottasel_len": 4,
    "madd_mottasel_waqf": 4,
    "madd_aared_len": 4,
    "ghonna_lam_and_raa": "no_ghonna",
}


def test_a_field_missing_from_the_ui_schema_does_not_erase_the_users_choice():
    m = resolve_moshaf(FRONTEND_SHAPED_PAYLOAD, Settings())
    assert m.madd_monfasel_len == 2, "the reciter's own choice must win"
    assert m.rewaya == "hafs", "the field the UI never shows still resolves to its default"


def test_no_moshaf_from_the_client_uses_the_settings_default():
    assert resolve_moshaf(None, Settings()) == default_moshaf(Settings())


def test_a_genuinely_invalid_combination_still_falls_back_safely():
    # madd al-leen may not exceed madd al-aared -- a real invalid combination, not a
    # payload-shape bug. This must still degrade to the default rather than crash the
    # session (the case ws.py's try/except was actually meant to catch).
    bad = {**FRONTEND_SHAPED_PAYLOAD, "madd_alleen_len": 99}
    assert resolve_moshaf(bad, Settings()) == default_moshaf(Settings())


if __name__ == "__main__":
    test_a_field_missing_from_the_ui_schema_does_not_erase_the_users_choice()
    test_no_moshaf_from_the_client_uses_the_settings_default()
    test_a_genuinely_invalid_combination_still_falls_back_safely()
    print("ok")
