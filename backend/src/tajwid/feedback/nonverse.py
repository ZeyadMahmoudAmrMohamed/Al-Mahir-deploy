"""Recited-but-not-verse text: istiʿādhah, basmalah, ṣadaqa (FR-009).

Learners bracket essentially every real session with these. They are correct, pious,
and not Quran — so they must be recognised and excluded from the diff, never scored.

Without this, the istiʿādhah is aligned against the verse the learner is about to
recite, mismatches on every phoneme, and the session's FIRST feedback is a wall of red
for doing exactly the right thing. That is a false accusation with no learner error
behind it (Constitution VI).
"""

from functools import lru_cache

import quran_transcript.alphabet as alph
import Levenshtein as lv
from quran_transcript import Aya, MoshafAttributes, chunck_phonemes

from .reference import build_reference

# How closely the head/tail of a chunk must match a marker before we believe it is
# there. Generous, because these are recited from memory at speed and the ASR will
# mangle them at least as readily as it mangles Quran.
MATCH_THRESHOLD = 0.75


def phonetize_marker(uthmani: str, moshaf: MoshafAttributes) -> str:
    """Phonemes for a non-verse marker, in the reciter's own style."""
    return build_reference(uthmani, moshaf).phonemes


@lru_cache(maxsize=32)
def _markers(moshaf_key: str) -> dict[str, str]:
    moshaf = MoshafAttributes.model_validate_json(moshaf_key)
    return {
        "istiaatha": phonetize_marker(alph.istiaatha.uthmani, moshaf),
        # Basmalah is ALSO Al-Fatiha 1:1 — see strip_non_verse for why that matters.
        "basmalah": phonetize_marker(Aya(1, 1).get().uthmani, moshaf),
        "sadaka": phonetize_marker(alph.sadaka.uthmani, moshaf),
    }


def _norm(groups: list[str]) -> str:
    """One character per phoneme group — madd-length invariant, like the search index."""
    return "".join(g[0] for g in groups)


def _ratio(a: str, b: str) -> float:
    if not a:
        return 0.0
    return 1 - (min(lv.distance(a, b), len(a)) / len(a))


def _strip_head(groups: list[str], marker: str) -> int:
    """How many groups to drop from the head, or 0 if the marker is not there."""
    marker_groups = chunck_phonemes(marker)
    n = len(marker_groups)
    if n == 0 or len(groups) < n:
        return 0

    if _ratio(_norm(marker_groups), _norm(groups[:n])) >= MATCH_THRESHOLD:
        return n
    return 0


def _strip_tail(groups: list[str], marker: str) -> int:
    marker_groups = chunck_phonemes(marker)
    n = len(marker_groups)
    if n == 0 or len(groups) < n:
        return 0

    if _ratio(_norm(marker_groups), _norm(groups[-n:])) >= MATCH_THRESHOLD:
        return n
    return 0


def strip_non_verse(
    phonemes: str, moshaf: MoshafAttributes
) -> tuple[str, list[str], int, int]:
    """Remove istiʿādhah / basmalah / ṣadaqa, returning what is left and what was found.

    Order matters: the ordinary opening is istiʿādhah, then basmalah, then the verse.

    BASMALAH IS THE ONE WITH TEETH. It opens 113 suras as non-verse text, but it *is*
    Al-Fatiha 1:1 — an actual verse a learner may be reciting and expecting to be
    marked on. So it is only stripped when something FOLLOWS it. A chunk that is
    nothing but basmalah is left alone, because deleting it could silently erase a
    legitimately recited verse, and locate() can identify it perfectly well.

    Returns `(remainder, found, start_char, end_char)`. The char offsets are into the
    ORIGINAL phoneme string, and the caller MUST use them to slice `phonemes.probs` to
    match: that array is indexed over the string we just cut, so failing to slice it
    would score every error against some other phoneme's probability.
    """
    groups = chunck_phonemes(phonemes)
    markers = _markers(moshaf.model_dump_json())
    found: list[str] = []

    start = 0
    end = len(phonemes)

    n = _strip_head(groups, markers["istiaatha"])
    if n:
        start += sum(len(g) for g in groups[:n])
        groups = groups[n:]
        found.append("istiaatha")

    n = _strip_head(groups, markers["basmalah"])
    # Only non-verse if the reciter carried on into something else.
    if n and len(groups) > n:
        start += sum(len(g) for g in groups[:n])
        groups = groups[n:]
        found.append("basmalah")

    n = _strip_tail(groups, markers["sadaka"])
    if n:
        end -= sum(len(g) for g in groups[-n:])
        groups = groups[:-n]
        found.append("sadaka")

    return "".join(groups), found, start, end
