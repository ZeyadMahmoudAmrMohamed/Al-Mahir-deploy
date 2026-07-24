"""The offline probe: does it reproduce what the live pipeline did?"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from probe_stream import ProbeResult, probe  # noqa: E402

ASSET = Path(__file__).resolve().parent / "assets" / "test.wav"


@pytest.fixture(scope="module")
def result():
    import os

    os.environ["TAJWID_ASR_ENGINE"] = "mock"
    from tajwid.config import get_settings

    get_settings.cache_clear()
    from tajwid.asr.engine import make_engine
    from tajwid.feedback.types import Span

    # A start span is not optional for the MOCK engine: it fabricates its output from
    # the phonetizer at the cursor, so with no cursor it has nothing to transcribe and
    # returns an empty string. The real engine reads the audio and needs no seed.
    out = probe(ASSET, start=Span(sura=1, aya=1, word_idx=0), engine=make_engine())
    yield out
    get_settings.cache_clear()


def test_probe_returns_audio_and_vad_probs(result):
    assert isinstance(result, ProbeResult)
    assert result.sample_rate == 16000
    assert result.audio.ndim == 1
    assert result.audio.size > 0
    # One silero probability per whole 1536-sample window fed.
    assert result.vad_probs.size > 0
    assert float(result.vad_probs.min()) >= 0.0
    assert float(result.vad_probs.max()) <= 1.0


def test_chunks_carry_their_audio_and_span(result):
    assert result.chunks, "test.wav should endpoint into at least one chunk"
    for c in result.chunks:
        assert c.end_s > c.start_s
        assert c.wave.size > 0
        # The wave is the chunk's own samples, so its length matches its span
        # (within the lead/trail pad the extractor adds).
        assert abs(c.wave.size / result.sample_rate - (c.end_s - c.start_s)) < 0.5


def test_context_widens_the_slice(result):
    c = result.chunks[0]
    wide = result.context(c.seq, pad_s=1.0)
    assert wide.size >= c.wave.size
    assert wide.size <= c.wave.size + 2 * result.sample_rate + 1


def test_context_clamps_at_the_edges(result):
    """A chunk cannot be padded past the ends of the recording."""
    wide = result.context(result.chunks[0].seq, pad_s=1000.0)
    assert wide.size == result.audio.size


def test_scored_chunks_carry_the_model_output(result):
    """The point of the probe: the model's own output, per chunk, beside its audio."""
    scored = [c for c in result.chunks if c.match_status]
    assert scored, "the mock engine should score at least one chunk"
    for c in scored:
        assert c.predicted_phonemes
        assert len(c.groups) == len(c.group_probs)
        assert all(0.0 <= p <= 1.0 for p in c.group_probs)
        # The cursor is captured BEFORE analyse_session advances it, so re-running
        # track() offline searches from where the aligner actually searched.
        assert c.cursor_before is not None


# --- Replay -------------------------------------------------------------------


def _make_capture(tmp_path):
    """Drive a mock session through the REAL WS handler, so the capture is genuine
    rather than a fixture that happens to have the right filenames."""
    import json
    import os

    from fastapi.testclient import TestClient

    from tajwid.asr.batch import load_audio
    from tajwid.config import get_settings

    os.environ["TAJWID_ASR_ENGINE"] = "mock"
    os.environ["TAJWID_CAPTURE_DIR"] = str(tmp_path)
    get_settings.cache_clear()

    from tajwid.main import create_app

    wave = load_audio(ASSET, 16000).numpy()
    pcm = (np.clip(wave, -1, 1) * 32767).astype("<i2")
    with TestClient(create_app()) as c:
        with c.websocket_connect("/ws/session") as ws:
            ws.send_text(
                json.dumps(
                    {
                        "type": "start",
                        "sura": 1,
                        "aya": 1,
                        "word_idx": 0,
                        "capture": True,
                    }
                )
            )
            ack = json.loads(ws.receive_text())
            for i in range(0, len(pcm), 1600):
                ws.send_bytes(pcm[i : i + 1600].tobytes())
            ws.send_text(json.dumps({"type": "end"}))
            while json.loads(ws.receive_text()).get("type") != "done":
                pass
    return tmp_path / ack["session_id"]


@pytest.fixture(scope="module")
def capture_dir(tmp_path_factory):
    from tajwid.config import get_settings

    d = _make_capture(tmp_path_factory.mktemp("cap"))
    yield d
    get_settings.cache_clear()


def test_replay_reproduces_the_captured_chunk_boundaries(capture_dir):
    from probe_stream import replay

    r = replay(capture_dir)
    recorded = [e for e in r.recorded_events if e["type"] == "feedback"]
    assert r.chunks, "the capture should contain at least one chunk"
    assert recorded, "the live session should have scored at least one chunk"

    live_starts = sorted(round(e["audio_span_sec"][0], 2) for e in recorded)
    replay_starts = sorted(round(c.start_s, 2) for c in r.chunks if c.match_status)
    assert replay_starts == live_starts


def test_replay_restores_the_captured_settings(capture_dir):
    from probe_stream import replay

    from tajwid.config import Settings

    r = replay(capture_dir)
    assert r.settings.sample_rate == 16000
    assert r.settings.vad_threshold == Settings().vad_threshold


def test_replay_honours_a_settings_override(capture_dir):
    """The whole point of replay: re-run the same audio under different parameters.

    Asserted on `chunk_lead_pad_ms` rather than `vad_threshold` because its effect is
    arithmetic and therefore guaranteed -- `stream._extract` starts the chunk at
    `speech_start - lead_pad`, so doubling the pad MUST move the start by exactly that
    much. A `vad_threshold` change is only guaranteed to move a boundary on audio with
    a soft onset, and test.wav is one clean utterance whose silero probability jumps at
    onset: the override reaches the pipeline (checked below) but has nothing to bite on.
    Testing the plumbing on an asset that cannot exercise it would be a test that passes
    for the wrong reason.
    """
    from probe_stream import replay, settings_from_capture

    base = replay(capture_dir)
    padded = replay(
        capture_dir, settings=settings_from_capture(capture_dir, chunk_lead_pad_ms=600)
    )

    assert base.settings.chunk_lead_pad_ms == 120
    assert padded.settings.chunk_lead_pad_ms == 600
    assert len(padded.chunks) == len(base.chunks)
    moved = base.chunks[0].start_s - padded.chunks[0].start_s
    # 600 ms - 120 ms = 480 ms earlier, unless clamped by the start of the recording.
    assert moved == pytest.approx(min(0.48, base.chunks[0].start_s), abs=0.02)


def test_settings_override_reaches_the_vad(capture_dir):
    """`vad_threshold` is plumbed through to the endpointer, whatever the audio does
    with it."""
    from probe_stream import replay, settings_from_capture

    loose = replay(
        capture_dir, settings=settings_from_capture(capture_dir, vad_threshold=0.05)
    )
    assert loose.settings.vad_threshold == 0.05


def test_replay_splits_at_the_recorded_frame_boundaries(capture_dir):
    """Frame boundaries are load-bearing, not bookkeeping: sherpa's streaming decode
    depends on how audio was handed to accept_waveform."""
    import json

    from probe_stream import _read_capture

    audio, sizes, cfg, events = _read_capture(capture_dir)
    assert sizes, "frames.jsonl must not be empty"
    assert sum(sizes) == audio.size
    assert cfg["resolved"]["engine"] == "mock"
    assert any(e["type"] == "feedback" for e in events)
