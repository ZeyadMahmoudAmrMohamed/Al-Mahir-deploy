"""The live-session orchestrator: where the two halves of the pipeline meet.

Audio frames come in; per-word feedback events go out. Per finalized VAD chunk:

    StreamSession (silero endpointing)          [asr half]
      -> engine.transcribe_chunk                [asr half: real GPU model or mock]
      -> transcript_to_output                   [the in-process adapter INTEGRATION.md demands]
      -> analyse_session                        [feedback half: track/locate, diff, sifat, score]
      -> one JSON-able event

The adapter is the load-bearing piece: the old HTTP handoff dropped per-character
probs and sifat, killing confidence grading and sifat feedback silently. Here the
model output crosses the boundary whole.

Each LiveSession owns a fresh silero instance (stateful RNN — sharing one across
concurrent sessions would mix their hidden states; the pre-merge server did share it).
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

from quran_transcript import MoshafAttributes

from .asr.contract import ChunkResult
from .asr.engine import AsrEngine, ChunkContext
from .asr.stream import FinalizedChunk, StreamSession
from .asr.transcribe import ChunkTranscript
from .asr.vad import load_vad
from .config import Settings, get_settings
from .feedback.confidence import STRICTNESS
from .feedback.pipeline import analyse_session
from .feedback.rules import catalogue
from .feedback.session import SessionState
from .feedback.types import (
    SIFA_ATTRS,
    MuaalemOutput,
    Phonemes,
    PredictedSifa,
    Span,
)


def transcript_to_output(t: ChunkTranscript) -> MuaalemOutput:
    """ChunkTranscript -> the feedback half's input type. A shape change, not a
    conversion: text + per-CHAR probs + all 10 sifat attrs with their own probs."""
    sifat: list[PredictedSifa] = []
    for s in t.sifat:
        attrs: dict[str, str] = {}
        probs: dict[str, float] = {}
        for attr in SIFA_ATTRS:
            unit = getattr(s, attr, None)
            if unit is not None and unit.text is not None:
                attrs[attr] = unit.text
                probs[attr] = float(unit.prob)
        sifat.append(
            PredictedSifa(phonemes_group=s.phonemes_group, attrs=attrs, probs=probs)
        )
    return MuaalemOutput(
        phonemes=Phonemes(text=t.phonemes_text, probs=t.char_probs or None),
        sifat=sifat,
    )


def default_moshaf(settings: Settings | None = None) -> MoshafAttributes:
    s = settings or get_settings()
    return MoshafAttributes(
        rewaya="hafs",
        madd_monfasel_len=s.madd_monfasel_len,
        madd_mottasel_len=s.madd_mottasel_len,
        madd_mottasel_waqf=s.madd_mottasel_waqf,
        madd_aared_len=s.madd_aared_len,
    )


def resolve_moshaf(raw: dict | None, settings: Settings | None = None) -> MoshafAttributes:
    """The client's moshaf choice, filled in over the default rather than validated bare.

    ``/moshaf-schema`` (api/rest.py) only shows fields with 2+ options -- nothing to pick
    when there's only one -- so a field like ``rewaya`` (Literal["hafs"], fixed) never
    appears in what the frontend sends. But ``rewaya`` has no default on MoshafAttributes,
    so validating the client's dict AS THE WHOLE MODEL always raised, and used to fall
    back to `default_moshaf()` outright -- discarding every field the reciter DID set,
    including the one this bug was reported over (madd_monfasel_len). Layering the
    client's dict over the resolved default's dump means an omitted-because-hidden field
    resolves to its default while a provided field still overrides it.

    A genuinely invalid combination (madd al-leen longer than madd al-aared, an
    out-of-range value) still raises after the merge and still falls back to the
    default -- that's the real "don't let a bad config kill the session" case.
    """
    s = settings or get_settings()
    if not raw:
        return default_moshaf(s)
    try:
        return MoshafAttributes(**{**default_moshaf(s).model_dump(), **raw})
    except Exception:  # noqa: BLE001 — any bad-config shape, not just ValidationError
        return default_moshaf(s)


def resolve_strictness(raw: str | None, settings: Settings | None = None) -> str:
    """The client's strictness, or the default — never an unusable value.

    `STRICTNESS[strictness]` is read once per finalized chunk, deep inside the feedback
    pipeline, long after the start message was accepted. An unrecognised value therefore
    did not fail at the boundary where it arrived: it raised KeyError mid-recitation,
    the exception escaped the WebSocket handler, and the socket closed with code 1000 —
    "OK". The reciter's feedback simply stopped, and the client could not tell that from
    a normal end of session. `"Normal"` with a capital N was enough to do it.

    Falling back mirrors what this boundary already does for `engine` (unbuilt name ->
    server default) and `moshaf` (invalid combination -> default). A bad setting costs
    you the setting, not the session.
    """
    s = settings or get_settings()
    if raw in STRICTNESS:
        return raw
    return s.strictness if s.strictness in STRICTNESS else "normal"


def resolve_rules(
    raw: frozenset[str] | None, settings: Settings | None = None
) -> frozenset[str] | None:
    """The session's leniency selection, with sifat withheld unless asked for.

    `None` means "grade everything", which is the right default for tajwid and the
    wrong one for sifat today: the sifa comparison is measurably unreliable (see
    `Settings.grade_sifat` and FINDINGS.md) and its findings arrive at confidence
    1.000, so they cannot soften to `almost` — they land as confident accusations on
    correct recitation. Until the reference derivation is audited, an unspecified
    selection resolves to every TAJWID rule and no sifa.

    An EXPLICIT selection is honoured whole, sifa keys included. Turning the default
    off is not the same as removing the feature, and a client that asks to be graded
    on الغنة is making a real choice — that is how this gets re-evaluated once the
    reference side is fixed.
    """
    s = settings or get_settings()
    if raw is not None or s.grade_sifat:
        return raw
    return frozenset(
        r["key"] for r in catalogue() if r["kind"] != "sifa"
    )


class LiveSession:
    """One reciter's live stream: endpointing + ASR + tracking feedback."""

    def __init__(
        self,
        engine: AsrEngine,
        *,
        session_id: str,
        moshaf: MoshafAttributes | None = None,
        start: Optional[Span] = None,
        strictness: str | None = None,
        include_units: bool = False,
        rules: frozenset[str] | None = None,
        settings: Settings | None = None,
        zipformer_engine: object | None = None,
    ):
        self.s = settings or get_settings()
        self.engine = engine
        self.stream = StreamSession(load_vad(), self.s)
        self.state = SessionState(
            moshaf=moshaf or default_moshaf(self.s),
            session_id=session_id,
            cursor=start,
            strictness=resolve_strictness(strictness, self.s),
            rules=resolve_rules(rules, self.s),
        )
        self.include_units = include_units
        self.seq = 0
        # For chunk overlap: the previous chunk's span-final word (re-sent as this
        # chunk's head, so not to be re-graded) and whether that chunk ended on a
        # forced cut (in which case its last word was chopped, not paused, and the
        # overlap SHOULD rescue it). None until the first chunk has been scored.
        self._prev_end: Optional[Span] = None
        self._prev_forced = False

        # Tier 1 (streaming-zipformer live word-fill). Companion-only: on iff enabled,
        # zipformer was built (files present), and the grader is Muaalem (real/remote).
        # The live tier does NOT go through `engine` — it owns its own CPU-local stream,
        # so it stays responsive even when Muaalem grades on a remote GPU.
        self._live = None
        self._samples_since_live = 0
        if (
            self.s.live_feedback
            and zipformer_engine is not None
            and getattr(engine, "name", "") in ("real", "remote")
        ):
            from .asr.live_aligner import LiveAligner

            self._live = LiveAligner(zipformer_engine._recognizer, self.s)
            if start is not None:
                self._live.reanchor(start)

    # -- public API -------------------------------------------------------
    def feed(self, samples: np.ndarray) -> list[dict]:
        """Append audio; return one feedback event per finalized waqf chunk, plus any
        interleaved live `progress` events (Tier 1)."""
        events: list[dict] = []
        fins = self.stream.feed(samples)
        if self._live is not None:
            self._live.feed(samples)
        for fin in fins:
            if e := self._process(fin):
                events.append(e)
        if fins:
            # A waqf was graded; re-anchor the live tier to the authoritative cursor.
            if self._live is not None and self.state.cursor is not None:
                self._live.reanchor(self.state.cursor)
            self._samples_since_live = 0
        elif self._live is not None:
            self._samples_since_live += int(np.asarray(samples).size)
            if self._samples_since_live >= self.s.live_interval_samples:
                self._samples_since_live = 0
                try:
                    confirmed, skipped = self._live.progress()
                    if confirmed or skipped:
                        events.append(self._live_event(confirmed, skipped))
                except Exception as err:  # noqa: BLE001 — the provisional tier is a
                    # nicety; a failure in it must never take down the session or block
                    # the authoritative feedback. Drop this tick and carry on.
                    logger.warning("Live word-fill tick failed; skipping it: %r", err)
        return events

    def _live_event(self, confirmed: list[Span], skipped: list[Span]) -> dict:
        """Provisional: coordinates only, never a verdict."""
        tip = confirmed[-1] if confirmed else self.state.cursor
        return {
            "type": "progress",
            "confirmed": [w.model_dump() for w in confirmed],
            "skipped": [w.model_dump() for w in skipped],
            "cursor": tip.model_dump() if tip else None,
        }

    def flush(self) -> list[dict]:
        """End of stream: finalize and score any in-progress utterance."""
        return [e for fin in self.stream.flush() if (e := self._process(fin))]

    def seek(self, span: Span) -> None:
        """The user repositioned (picked a different sura/aya). Reset the cursor."""
        self.state = replace(self.state, cursor=span, penalty=0)
        if self._live is not None:
            self._live.reanchor(span)

    @property
    def cursor(self) -> Optional[Span]:
        return self.state.cursor

    # -- internals --------------------------------------------------------
    def _process(self, fin: FinalizedChunk) -> dict | None:
        sr = self.s.sample_rate
        ctx = ChunkContext(
            duration_s=(fin.end_sample - fin.start_sample) / sr,
            cursor=self.state.cursor,
            moshaf=self.state.moshaf,
        )
        transcript = self.engine.transcribe_chunk(fin.wave, sr, ctx)
        if not transcript.phonemes_text:
            return None

        output = transcript_to_output(transcript)

        # Under overlap, the previous chunk's pausal span-final word is re-sent as this
        # chunk's head. Don't re-grade it (it was already scored, in pausal form) --
        # unless that chunk was force-cut, where the word was chopped mid-flow and the
        # overlap is genuinely rescuing it. No overlap (=0) -> the word was never
        # re-sent, so _prev_end simply won't be in this chunk's span and the blank is a
        # harmless no-op.
        seam = (
            self._prev_end
            if self.s.chunk_overlap_ms and not self._prev_forced
            else None
        )
        try:
            feedback, self.state = analyse_session(
                output, self.state, forced=fin.forced, overlap_seam=seam
            )
        except Exception as err:  # noqa: BLE001 — a vendored-phonetizer crash on ONE
            # span must not kill the whole recitation. Skip this chunk, keep the cursor
            # where it was, and log enough (cursor + predicted phonemes) to reproduce the
            # exact failing span offline via analyse_session. See the KeyError:'ء' in
            # quran_transcript's get_mappings.
            logger.error(
                "Feedback analysis crashed on a chunk; skipping it (session continues). "
                "cursor=%s phonemes=%r error=%r",
                self.state.cursor,
                transcript.phonemes_text,
                err,
            )
            return None
        self._prev_end = self.state.cursor
        self._prev_forced = fin.forced

        event: dict = {
            "type": "feedback",
            "chunk_seq": self.seq,
            "audio_span_sec": [fin.start_sample / sr, fin.end_sample / sr],
            "forced_cut": fin.forced,
            "phonemes": transcript.phonemes_text,
            "feedback": feedback.model_dump(),
            "cursor": self.state.cursor.model_dump() if self.state.cursor else None,
        }
        if self.include_units:
            event["units"] = [
                u.model_dump()
                for u in ChunkResult.from_transcript(
                    transcript,
                    session_id=self.state.session_id,
                    chunk_seq=self.seq,
                    audio_span_sec=tuple(event["audio_span_sec"]),
                ).units
            ]
        self.seq += 1
        return event
