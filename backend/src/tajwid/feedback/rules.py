"""The catalogue of gradeable rules, and the leniency filter over it.

A learner working on madd al-aared does not want to be told about qalqalah. This is
the "focus" knob: the session names the rules it wants graded, and findings for every
other rule are dropped before they can colour a word.

Keys come from the rule CLASSES upstream actually constructs (operations.py), not from
a hand-written list. A rename upstream then moves the catalogue and the filter
together; a hand-written list would drift silently, and a key that no longer matches
means a rule quietly stops being graded — leniency failing open, which is the false
reassurance this codebase exists to avoid.
"""

from __future__ import annotations

from quran_transcript.phonetics.tajweed_rules import (
    AaredMaddRule,
    LazemMaddRule,
    LeenMaddRule,
    MonfaselMaddRule,
    MottaselMaddPauseRule,
    MottaselMaddRule,
    NormalMaddRule,
    Qalqalah,
)

from .types import SIFA_ATTRS, FeedbackError

# Every rule `operations.py` builds. Ghonnah is deliberately absent: it is defined
# upstream as a TajweedRule but never constructed as one — in this pipeline ghunnah
# reaches the learner as the `ghonna` SIFA, which is why it lives in the sifa half of
# the catalogue below.
RULE_CLASSES = (
    NormalMaddRule,
    MonfaselMaddRule,
    MottaselMaddRule,
    MottaselMaddPauseRule,
    LazemMaddRule,
    AaredMaddRule,
    LeenMaddRule,
    Qalqalah,
)

# The 10 articulation attributes, in Arabic. There is no Arabic source for these
# upstream (SIFA_ATTRS is bare identifiers), so unlike the tajweed half this map is
# hand-written — SIFA_ATTRS is still the source of truth for WHICH keys exist.
#
# `qalqla` carries the "(صفة)" suffix because the tajweed rule Qalqalah shares its
# Arabic name. Two chips reading القلقلة would be an unanswerable choice.
SIFA_NAMES_AR = {
    "hams_or_jahr": "الهمس والجهر",
    "shidda_or_rakhawa": "الشدة والرخاوة",
    "tafkheem_or_taqeeq": "التفخيم والترقيق",
    "itbaq": "الإطباق",
    "safeer": "الصفير",
    "qalqla": "القلقلة (صفة)",
    "tikraar": "التكرار",
    "tafashie": "التفشي",
    "istitala": "الاستطالة",
    "ghonna": "الغنة",
}


def rule_key(name_en: str) -> str:
    """A tajweed rule's stable key: `"Aared Madd"` -> `"aared_madd"`."""
    return name_en.lower().replace(" ", "_")


def catalogue() -> list[dict]:
    """Every rule a session can ask to be graded on, for the settings panel."""
    entries = []
    for cls in RULE_CLASSES:
        rule = cls()
        entries.append(
            {
                "key": rule_key(rule.name.en),
                "name_ar": rule.name.ar,
                "name_en": rule.name.en,
                "kind": "tajweed",
            }
        )
    entries.extend(
        {
            "key": attr,
            "name_ar": SIFA_NAMES_AR.get(attr, attr),
            "name_en": attr.replace("_", " ").title(),
            "kind": "sifa",
        }
        for attr in SIFA_ATTRS
    )
    return entries


def filter_rules(
    errors: list[FeedbackError], rules: frozenset[str] | None
) -> list[FeedbackError]:
    """Keep only the findings the reciter asked to be graded on.

    `rules is None` grades everything — the default, and what every caller that never
    heard of this feature gets. An EMPTY set is a real choice, not a missing one: it
    means hifz and tashkeel only.

    Three kinds of finding are never filtered, whatever the selection:

      * `normal` — a wrong or missing letter. That is hifz, not tajwid.
      * `tashkeel` — a wrong haraka. Same.
      * `tajweed` findings carrying NO rule. `explain_error` marks a deletion as
        `tajweed` when the phoneme it fell on bore a rule, but attaches no rule object
        (error_explainer.py's `delete` branch) — the reciter skipped the letter
        outright. There is no key to match it against, and dropping a finding we
        cannot attribute would hide a real miss behind a filter the learner set for
        something else.
    """
    if rules is None:
        return errors

    kept: list[FeedbackError] = []
    for e in errors:
        if e.error_type == "tajweed" and e.tajweed_rules:
            if not any(rule_key(r.name_en) in rules for r in e.tajweed_rules):
                continue
        elif e.error_type == "sifa":
            # sifat.compare_sifat writes `expected_ph` as "<attr>=<value>"; the attr is
            # the catalogue key. Sniffing the string rather than adding a field only
            # this filter would read.
            if e.expected_ph.split("=", 1)[0] not in rules:
                continue
        kept.append(e)
    return kept
