from dataclasses import dataclass, replace
from typing import Optional

from quran_transcript import MoshafAttributes

from .types import LocateResult, Span

# How much wider the tracking window gets after each failed chunk, and how far that
# widening is allowed to go before we admit we have simply lost the reciter and let the
# cold-search fallback take over.
PENALTY_STEP = 15
MAX_PENALTY = 90


@dataclass(frozen=True)
class SessionState:
    """A session is a cursor and a moshaf. No audio, no history buffer.

    `session_id` is the caller's (FR-001) — for their logs, metrics and reconnects. We
    keep NO session store: you hold this object and hand it back with each chunk.
    Owning eviction, TTLs and concurrency for state the caller already has in hand is a
    service's job, and this is a library.

    `penalty` is how far the tracking window has been widened after failures (FR-007).

    `rules` is the leniency selection (feedback.rules): the tajwid/sifa rules this
    reciter asked to be graded on. None — the default — grades everything.
    """

    moshaf: MoshafAttributes
    session_id: str = ""
    cursor: Optional[Span] = None
    strictness: str = "normal"
    penalty: int = 0
    rules: Optional[frozenset[str]] = None


def advance(state: SessionState, result: LocateResult) -> SessionState:
    """Move the cursor to the end of what was just matched, and adjust the penalty.

    On success: the cursor moves and the penalty clears — we have found the reciter
    again, so there is no reason to keep searching so wide.

    On failure: the cursor STAYS PUT and the penalty grows (FR-007). The reciter may
    simply have coughed. Resetting the cursor on every miss would make tracking thrash —
    the learner would be relocated to the far side of the Quran for clearing their
    throat. Widening the window instead says "you are probably still about here; let me
    look harder."
    """
    if result.status != "ok" or result.end is None:
        return replace(state, penalty=min(state.penalty + PENALTY_STEP, MAX_PENALTY))

    return replace(state, cursor=result.end, penalty=0)
