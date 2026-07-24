"""Reproduce every intermediate of the live pipeline, for a file or a captured session.

Two entry points, one return type:

    probe(wav_path)      stream a file through a LiveSession
    replay(session_dir)  re-run a captured session, frame boundary for frame boundary

Everything is obtained by WRAPPING production functions, never editing them: the VAD is
proxied to record its probabilities, ``engine.transcribe_chunk`` is proxied to capture the
model output and the cursor as it stood going in, and ``stream.feed``/``flush`` are proxied
to capture the finalized chunks before the engine can swallow them (a dead remote GPU
returns an empty transcript, ``LiveSession._process`` then returns None, and the chunk
would vanish from an events-derived count -- losing VAD numbers that never needed a GPU).

Deliberately NOT merged into measure_chunks.py. That is a CLI appending JSON metrics to
runs.jsonl; this returns Python objects holding float arrays. Same subject, different
interface.
"""

from __future__ import annotations

import json
import wave as wavemod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tajwid.asr.batch import load_audio
from tajwid.config import Settings, get_settings
from tajwid.feedback.types import Span


class ProbeVad:
    """Records every silero speech probability. The VAD is only ever called as
    ``vad(window, sr)`` and reset via ``reset_states()``, so a two-method proxy is the
    whole shim (same trick as measure_chunks.py)."""

    def __init__(self, vad):
        self._vad = vad
        self.probs: list[float] = []

    def __call__(self, window, sr):
        p = self._vad(window, sr)
        self.probs.append(float(p))
        return p

    def reset_states(self):
        self._vad.reset_states()


@dataclass
class ChunkProbe:
    """One finalized chunk, with everything needed to see and hear what happened to it."""

    seq: int
    start_s: float
    end_s: float
    forced: bool
    wave: np.ndarray
    predicted_phonemes: str = ""
    groups: list[str] = field(default_factory=list)
    group_probs: list[float] = field(default_factory=list)
    cursor_before: Span | None = None
    match_status: str | None = None
    words: list[dict] = field(default_factory=list)


@dataclass
class ProbeResult:
    audio: np.ndarray
    sample_rate: int
    vad_probs: np.ndarray
    chunks: list[ChunkProbe]
    live_ticks: list[dict]
    settings: Settings
    recorded_events: list[dict] = field(default_factory=list)

    def context(self, seq: int, pad_s: float = 1.0) -> np.ndarray:
        """The chunk's audio with ``pad_s`` seconds either side, clamped to the recording.

        This is what makes a boundary AUDIBLE. The chunk alone cannot tell you whether a
        final consonant was cut, because the cut is exactly what is missing from it -- you
        have to hear past the edge to hear what was lost.
        """
        c = next(x for x in self.chunks if x.seq == seq)
        sr = self.sample_rate
        a = max(0, int((c.start_s - pad_s) * sr))
        b = min(self.audio.size, int((c.end_s + pad_s) * sr))
        return self.audio[a:b]


def _instrument(session):
    """Wrap a LiveSession's VAD, stream and engine. Returns (probe_vad, endpointed, seen)."""
    probe_vad = ProbeVad(session.stream.vad)
    session.stream.vad = probe_vad

    endpointed: list = []
    for name in ("feed", "flush"):
        orig = getattr(session.stream, name)

        def wrapped(*a, _orig=orig, **kw):
            got = _orig(*a, **kw)
            endpointed.extend(got)
            return got

        setattr(session.stream, name, wrapped)

    # (cursor_at_entry, transcript) per chunk. The cursor is read at TRANSCRIBE time --
    # before analyse_session advances it -- so this is the cursor the aligner actually
    # searched from, which is what re-running track() offline needs.
    seen: list[tuple[Span | None, object]] = []
    real_transcribe = session.engine.transcribe_chunk

    def traced(*a, **kw):
        cursor = session.state.cursor
        t = real_transcribe(*a, **kw)
        seen.append((cursor, t))
        return t

    session.engine.transcribe_chunk = traced
    return probe_vad, endpointed, seen


def _assemble(audio, sr, probe_vad, endpointed, seen, events, settings) -> ProbeResult:
    """Join the endpointed chunks (the spine) to the model output and the feedback.

    Endpointed chunks are the spine rather than the events, because a chunk the engine
    failed on still endpointed -- and its VAD numbers are exactly as real as any other's.
    """
    by_start = {
        round(e["audio_span_sec"][0], 3): e
        for e in events
        if e.get("type") == "feedback"
    }

    chunks: list[ChunkProbe] = []
    for i, fin in enumerate(endpointed):
        start_s = round(fin.start_sample / sr, 3)
        c = ChunkProbe(
            seq=i,
            start_s=start_s,
            end_s=round(fin.end_sample / sr, 3),
            forced=fin.forced,
            wave=np.asarray(fin.wave, dtype=np.float32),
        )
        if i < len(seen):
            cursor, t = seen[i]
            c.cursor_before = cursor
            c.predicted_phonemes = getattr(t, "phonemes_text", "")
            c.groups = list(getattr(t, "groups", []))
            c.group_probs = [float(p) for p in getattr(t, "group_probs", [])]
        e = by_start.get(start_s)
        if e is not None:
            c.match_status = e["feedback"]["status"]
            c.words = e["feedback"].get("words") or []
        chunks.append(c)

    return ProbeResult(
        audio=audio,
        sample_rate=sr,
        vad_probs=np.asarray(probe_vad.probs, dtype=np.float32),
        chunks=chunks,
        live_ticks=[e for e in events if e.get("type") == "progress"],
        settings=settings,
    )


def probe(
    wav_path,
    *,
    start: Span | None = None,
    settings: Settings | None = None,
    frame_ms: int = 100,
    engine=None,
) -> ProbeResult:
    """Stream an audio file through a LiveSession and capture every intermediate."""
    from tajwid.asr.engine import make_engine
    from tajwid.session import LiveSession

    s = settings or get_settings()
    sr = s.sample_rate
    audio = load_audio(wav_path, sr).numpy()
    frame = int(frame_ms * sr / 1000)

    session = LiveSession(
        engine or make_engine(),
        session_id=Path(wav_path).stem,
        start=start,
        settings=s,
    )
    probe_vad, endpointed, seen = _instrument(session)

    events: list[dict] = []
    for i in range(0, len(audio), frame):
        events.extend(session.feed(audio[i : i + frame]))
    events.extend(session.flush())

    return _assemble(audio, sr, probe_vad, endpointed, seen, events, s)


# --- Replaying a captured session ----------------------------------------------


def _read_capture(session_dir) -> tuple[np.ndarray, list[int], dict, list[dict]]:
    """Read the four capture files: audio, frame sizes, start config, recorded events."""
    d = Path(session_dir)
    with wavemod.open(str(d / "input.wav"), "rb") as w:
        if w.getsampwidth() != 2 or w.getnchannels() != 1:
            raise ValueError("capture must be 16-bit mono")
        raw = w.readframes(w.getnframes())
    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0

    sizes = [
        json.loads(line)["n"]
        for line in (d / "frames.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    cfg = json.loads((d / "start.json").read_text(encoding="utf-8"))

    events_path = d / "events.jsonl"
    events = (
        [
            json.loads(line)
            for line in events_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if events_path.exists()
        else []
    )
    return audio, sizes, cfg, events


def settings_from_capture(session_dir, **overrides) -> Settings:
    """The captured session's settings, with any field overridden.

    This is what makes a sweep possible: the same recitation re-run at a different
    vad_threshold, trail pad or overlap. Fields absent from the capture fall back to
    this process's own environment.
    """
    _, _, cfg, _ = _read_capture(session_dir)
    return Settings(**{**cfg.get("settings", {}), **overrides})


def replay(session_dir, *, settings: Settings | None = None, engine=None) -> ProbeResult:
    """Re-run a captured session, frame boundary for frame boundary.

    Splitting at the RECORDED boundaries is not fussiness: sherpa's streaming decode
    depends on how audio was handed to ``accept_waveform``, so re-splitting the same
    bytes at a different size changes the live tier's output. The authoritative tier
    would survive it; the thing under investigation would not.

    ``recorded_events`` carries what the LIVE session emitted, so the replay can be
    CHECKED rather than trusted -- Muaalem on a remote GPU is the one component that
    may not be bit-exact, and a divergence is a finding rather than an embarrassment.
    """
    from tajwid.asr.engine import make_engine
    from tajwid.session import LiveSession

    audio, sizes, cfg, recorded = _read_capture(session_dir)
    s = settings or Settings(**cfg.get("settings", {}))
    sr = s.sample_rate

    raw_cfg = cfg.get("cfg", {})
    start = (
        Span(
            sura=int(raw_cfg["sura"]),
            aya=int(raw_cfg["aya"]),
            word_idx=int(raw_cfg.get("word_idx", 0)),
        )
        if "sura" in raw_cfg and "aya" in raw_cfg
        else None
    )

    session = LiveSession(
        engine or make_engine(),
        session_id=Path(session_dir).name,
        start=start,
        settings=s,
    )
    probe_vad, endpointed, seen = _instrument(session)

    events: list[dict] = []
    pos = 0
    for n in sizes:
        events.extend(session.feed(audio[pos : pos + n]))
        pos += n
    if pos < audio.size:  # a tail the capture never got to record a boundary for
        events.extend(session.feed(audio[pos:]))
    events.extend(session.flush())

    result = _assemble(audio, sr, probe_vad, endpointed, seen, events, s)
    result.recorded_events = recorded
    return result


def _self_check() -> None:
    """Replay a captured session and assert it reproduces the recorded boundaries."""
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("usage: python probe_stream.py <capture_session_dir>")
    r = replay(sys.argv[1])
    recorded = sorted(
        round(e["audio_span_sec"][0], 2)
        for e in r.recorded_events
        if e["type"] == "feedback"
    )
    got = sorted(round(c.start_s, 2) for c in r.chunks if c.match_status)
    assert got == recorded, f"replay diverged:\n  live   {recorded}\n  replay {got}"
    print(f"OK - {len(r.chunks)} chunks, boundaries match the recorded session")


if __name__ == "__main__":
    _self_check()
