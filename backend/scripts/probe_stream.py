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


# --- Boundary measurement ------------------------------------------------------
#
# The decisive test for "the last letter is missing". `stream.py` ends a chunk at
# `_last_speech_end_abs` -- the end of the last window scoring above `vad_threshold`,
# which is 0.6, raised from silero's 0.3 default so shallow waqf dips register as
# silence. A raised threshold truncates trailing low-energy phonemes EARLIEST, and an
# unvoiced fricative or a sukun-final consonant carries little energy.
# `chunk_trail_pad_ms` (240) exists to cover the difference. Whether it does is what
# these functions measure.


def _frame_rms(audio: np.ndarray, sr: int, hop_ms: float) -> np.ndarray:
    hop = max(1, int(hop_ms * sr / 1000))
    n = audio.size // hop
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    framed = audio[: n * hop].reshape(n, hop).astype(np.float64)
    return np.sqrt((framed**2).mean(axis=1)).astype(np.float32)


def noise_floor(audio: np.ndarray, sr: int, hop_ms: float = 10.0) -> float:
    """The session's noise floor: the 10th percentile of frame RMS.

    A percentile, not an absolute threshold, because `autoGainControl` is on in
    `mic.ts` -- the same room at the same loudness lands at a different absolute level
    depending on what the AGC did, so any fixed number is wrong for some session.
    """
    rms = _frame_rms(audio, sr, hop_ms)
    return float(np.percentile(rms, 10)) if rms.size else 0.0


def sound_end_sample(
    audio: np.ndarray,
    sr: int,
    from_sample: int,
    floor: float,
    hop_ms: float = 10.0,
    margin_db: float = 15.0,
    limit_ms: float = 2000.0,
) -> int:
    """Where the sound that was still going at ``from_sample`` actually stops.

    Scans forward and returns the FIRST frame to drop below the floor, not the last
    frame above it. That distinction is the whole measurement: on continuous recitation
    the next utterance begins a few hundred milliseconds later, so "last frame above the
    floor within a window" lands inside the FOLLOWING word and reports the entire window
    as this chunk's tail. Measured on A_processed.wav, that mistake reported 22 of 23
    chunks cut short with every gap pinned at the search limit -- a dramatic result that
    was purely an artefact of the wrong bound.

    "Above the floor" is ``margin_db`` above ``noise_floor``. The default of 15 dB is
    chosen to sit between breath and speech, not at the edge of audibility: measured on
    A_processed.wav the noise floor (p10) is 0.00065 and speech (p50) is 0.064, a 40 dB
    separation, so an inter-word breath clears 6 dB easily while a trailing unvoiced
    consonant lands nearer 15-20 dB. At 6 dB this function reports breath as continuing
    speech. Callers that care should measure at two margins rather than trust one --
    see ``tail_report``.

    ``limit_ms`` remains as a safety cap for sound that genuinely never stops.
    """
    hop = max(1, int(hop_ms * sr / 1000))
    limit = int(limit_ms * sr / 1000)
    a = max(0, from_sample)
    b = min(audio.size, a + limit)
    if b <= a:
        return a
    rms = _frame_rms(audio[a:b], sr, hop_ms)
    threshold = floor * (10 ** (margin_db / 20.0))
    below = np.flatnonzero(rms <= threshold)
    if below.size == 0:
        return b  # still sounding at the cap
    return a + int(below[0]) * hop


def band_energy(
    audio: np.ndarray, sr: int, lo_hz: float = 4000.0, hi_hz: float = 8000.0
) -> float:
    """Mean spectral energy in a band -- 4-8 kHz by default, where the unvoiced
    fricatives live.

    Reported beside the RMS tail measure because a fricative dies in the BAND before it
    dies in broadband RMS, so an energy-only measure systematically under-reports
    exactly the phonemes under suspicion.
    """
    if audio.size == 0:
        return 0.0
    spectrum = np.abs(np.fft.rfft(audio.astype(np.float64)))
    freqs = np.fft.rfftfreq(audio.size, 1.0 / sr)
    sel = (freqs >= lo_hz) & (freqs < hi_hz)
    return float((spectrum[sel] ** 2).mean()) if sel.any() else 0.0


def tail_gap_ms(
    result: ProbeResult,
    seq: int,
    floor: float | None = None,
    margin_db: float = 15.0,
) -> float:
    """Milliseconds of sound left OUTSIDE a chunk after its end.

    Exceeding ``chunk_trail_pad_ms`` means the chunk provably ends before the sound does.
    """
    c = next(x for x in result.chunks if x.seq == seq)
    sr = result.sample_rate
    if floor is None:
        floor = noise_floor(result.audio, sr)
    end_sample = int(c.end_s * sr)
    sounding = sound_end_sample(
        result.audio, sr, end_sample, floor, margin_db=margin_db
    )
    return max(0.0, (sounding - end_sample) / sr * 1000.0)


def tail_report(result: ProbeResult) -> list[dict]:
    """One row per chunk: is its tail cut, and did the model lose confidence there?

    Two INDEPENDENT witnesses to the same event. ``cut_short`` is an acoustic claim
    (sound continued past the chunk); ``prob_drop`` is a model claim (the final phoneme
    group scored below the chunk's median CTC confidence). Agreement is strong evidence.
    A disagreement means the reported symptom has another cause, and saying so is as
    much a result as confirming it.

    The acoustic claim is reported at TWO thresholds, 15 dB and a strict 25 dB above the
    noise floor, because a single threshold hides the distinction that matters: a chunk
    cut short at both is genuinely losing a phoneme, while one that qualifies only at
    the lenient threshold is trailing breath. Do not collapse these into one number
    without saying which was used.
    """
    pad = result.settings.chunk_trail_pad_ms
    floor = noise_floor(result.audio, result.sample_rate)
    rows = []
    for c in result.chunks:
        gap = tail_gap_ms(result, c.seq, floor=floor, margin_db=15.0)
        gap_strict = tail_gap_ms(result, c.seq, floor=floor, margin_db=25.0)
        final = c.group_probs[-1] if c.group_probs else None
        median = float(np.median(c.group_probs)) if c.group_probs else None
        rows.append(
            {
                "seq": c.seq,
                "start_s": c.start_s,
                "end_s": c.end_s,
                "forced": c.forced,
                "tail_gap_ms": round(gap, 1),
                "tail_gap_ms_strict": round(gap_strict, 1),
                "trail_pad_ms": pad,
                "cut_short": gap > pad,
                # The claim that survives a threshold 10 dB stricter. This is the one
                # to quote: it cannot be trailing breath.
                "cut_short_strict": gap_strict > pad,
                "final_group": c.groups[-1] if c.groups else None,
                "final_group_prob": round(final, 3) if final is not None else None,
                "median_group_prob": round(median, 3) if median is not None else None,
                "prob_drop": round(median - final, 3)
                if final is not None and median is not None
                else None,
            }
        )
    return rows


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
