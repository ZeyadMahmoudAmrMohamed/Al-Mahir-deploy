"""Diagnostic capture: record a session's INPUT so the pipeline can be replayed offline.

Deliberately records the input rather than the intermediates. The pipeline is
deterministic given identical frames -- silero is a TorchScript module in eval() fed
fixed 1536-sample windows, ``track``/``diff_recitation``/``trim_edges``/``match_forward``
are pure, and the live tier's cadence is SAMPLE-driven (session.py's
``_samples_since_live``), not wall-clock. So every intermediate regenerates by replay,
and -- the part that matters -- a saved input can be re-run under parameters the original
session never used. Saved intermediates are frozen at one parameter set forever.

Muaalem on a remote GPU is the one component that may not be bit-exact, which is why
``events.jsonl`` is recorded too: it is the ground truth the replay is CHECKED against,
not an assumption that replay is faithful.

Capture is double-gated (see ``open_capture``) and every write is swallowed on failure:
a diagnostic must never take down a recitation.
"""

from __future__ import annotations

import json
import logging
import time
import wave as wavemod
from pathlib import Path

from .config import Settings

logger = logging.getLogger(__name__)

# The Settings fields that alter what the pipeline DOES. Replay reproduces a session by
# restoring these, so a field omitted here is a field whose effect silently disappears
# from every replayed run. Anything that changes chunking, endpointing, the live tier or
# grading belongs in this tuple.
CAPTURED_SETTINGS: tuple[str, ...] = (
    "sample_rate",
    "vad_threshold",
    "vad_window_samples",
    "min_silence_endpoint_ms",
    "min_speech_ms",
    "max_chunk_s",
    "chunk_lead_pad_ms",
    "chunk_trail_pad_ms",
    "chunk_overlap_ms",
    "live_feedback",
    "live_interval_ms",
    "live_lookahead_words",
    "live_window_words",
    "grade_sifat",
)


class SessionCapture:
    """Writes one session's four capture files. Never raises to its caller."""

    def __init__(self, root: Path, session_id: str, sample_rate: int = 16000):
        self.dir = Path(root) / session_id
        self.dead = False
        self._t0 = time.monotonic()
        self._wav = None
        self._frames = None
        self._events = None
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            self._wav = wavemod.open(str(self.dir / "input.wav"), "wb")
            self._wav.setnchannels(1)
            self._wav.setsampwidth(2)
            self._wav.setframerate(sample_rate)
            self._frames = (self.dir / "frames.jsonl").open("w", encoding="utf-8")
            self._events = (self.dir / "events.jsonl").open("w", encoding="utf-8")
        except OSError as err:
            self._die(err)

    def start(self, cfg: dict, resolved: dict) -> None:
        """Record the session config. ``resolved`` is what the server actually chose."""
        if self.dead:
            return
        from .config import get_settings

        s = get_settings()
        try:
            (self.dir / "start.json").write_text(
                json.dumps(
                    {
                        "cfg": cfg,
                        "resolved": resolved,
                        "settings": {k: getattr(s, k) for k in CAPTURED_SETTINGS},
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except OSError as err:
            self._die(err)

    def frame(self, pcm: bytes) -> None:
        """One inbound audio frame: its samples, and where its boundary fell.

        The boundary is load-bearing, not bookkeeping -- sherpa's streaming decode
        depends on how audio was split across ``accept_waveform`` calls, so a replay
        that re-splits the same bytes differently produces different live-tier output.
        ``t_ms`` is diagnostic only: it exposes network stalls that starved the
        endpointer, which replay cannot reproduce and should not try to.
        """
        if self.dead or not pcm:
            return
        try:
            self._wav.writeframes(pcm)
            self._frames.write(
                json.dumps(
                    {
                        "n": len(pcm) // 2,
                        "t_ms": round((time.monotonic() - self._t0) * 1000, 1),
                    }
                )
                + "\n"
            )
        except (OSError, ValueError, AttributeError) as err:
            self._die(err)

    def events(self, events: list[dict]) -> None:
        if self.dead or not events:
            return
        try:
            for e in events:
                self._events.write(json.dumps(e, ensure_ascii=False) + "\n")
            self._events.flush()
        except (OSError, TypeError, AttributeError) as err:
            self._die(err)

    def close(self) -> None:
        for handle in (self._wav, self._frames, self._events):
            if handle is None:
                continue
            try:
                handle.close()
            except (OSError, ValueError):
                pass

    def _die(self, err: Exception) -> None:
        """Disable capture for this session. Logged once, never re-raised."""
        if not self.dead:
            logger.warning("Session capture disabled after a write error: %r", err)
        self.dead = True


def open_capture(
    settings: Settings, cfg: dict, session_id: str
) -> SessionCapture | None:
    """A capture for this session, or None.

    Two keys, both required:
      * ``settings.capture_dir`` -- the OPERATOR opted this server in. Without it no
        client message can cause recording, however it is crafted.
      * ``cfg["capture"] is True`` -- the RECITER pressed Diagnose. Without it the
        server never records, even where it is allowed to.
    """
    if not settings.capture_dir or cfg.get("capture") is not True:
        return None
    return SessionCapture(Path(settings.capture_dir), session_id, settings.sample_rate)
