from functools import lru_cache

import quran_transcript.alphabet as alph
from quran_transcript import (
    MoshafAttributes,
    QuranPhoneticScriptOutput,
    quran_phonetizer,
)


@lru_cache(maxsize=512)
def _phonetize_cached(uthmani_text: str, moshaf_key: str) -> QuranPhoneticScriptOutput:
    moshaf = MoshafAttributes.model_validate_json(moshaf_key)
    # remove_spaces=True is required: it marks each space's mapping as `deleted`,
    # which is what keeps word boundaries recoverable from the mappings.
    return quran_phonetizer(uthmani_text, moshaf, remove_spaces=True)


def build_reference(
    uthmani_text: str, moshaf: MoshafAttributes
) -> QuranPhoneticScriptOutput:
    """Phonetize an Uthmani span with the user's moshaf. Cached.

    The phonetizer runs 25+ ordered regex transformations. A learner drilling one
    aya calls it identically on every attempt, so the cache is not a micro-
    optimisation — it is the difference between re-deriving Arabic phonology on
    every keystroke and not (FR-024).

    Returns an object with .phonemes (str), .sifat (list[SifaOutput]) and
    .mappings (uthmani char idx -> phonetic span + tajweed rules).
    """
    # MoshafAttributes is a Pydantic model and therefore unhashable, so it cannot be
    # an lru_cache key directly. Key on its serialised form instead.
    return _phonetize_cached(uthmani_text, moshaf.model_dump_json())


def word_index_of_char(uthmani_text: str, char_idx: int) -> int:
    """0-based word index within `uthmani_text` for a character offset.

    `uthmani_text` is always a span of words joined by alph.uthmani.space, so
    counting separators before the offset gives the word index.
    """
    return uthmani_text[:char_idx].count(alph.uthmani.space)
