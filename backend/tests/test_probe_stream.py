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


# --- Boundary measurement -----------------------------------------------------


def test_sound_end_finds_a_tone_that_stops_at_one_second():
    """A tone that stops at 1.0 s must be detected as ending at ~1.0 s -- not at
    whatever moment an endpointer stopped looking."""
    from probe_stream import noise_floor, sound_end_sample

    sr = 16000
    t = np.arange(int(1.5 * sr)) / sr
    audio = np.zeros_like(t, dtype=np.float32)
    audio[:sr] = (np.sin(2 * np.pi * 300 * t[:sr]) * 0.3).astype(np.float32)
    audio += np.random.default_rng(0).normal(0, 1e-4, audio.size).astype(np.float32)

    floor = noise_floor(audio, sr)
    assert floor < 0.01
    end = sound_end_sample(audio, sr, from_sample=int(0.5 * sr), floor=floor)
    assert abs(end - sr) < 0.05 * sr  # within 50 ms of the true end


def test_sound_end_reports_nothing_after_a_span_that_already_covers_the_sound():
    """A chunk whose end is past the audio must not report a positive gap."""
    from probe_stream import noise_floor, sound_end_sample

    sr = 16000
    audio = np.zeros(sr, dtype=np.float32)
    audio[: sr // 2] = 0.3
    floor = noise_floor(audio, sr)
    end = sound_end_sample(audio, sr, from_sample=int(0.9 * sr), floor=floor)
    assert end <= int(0.9 * sr) + 1


def test_sound_end_stops_looking_before_the_next_utterance():
    """Two utterances separated by 3 s of silence: measuring the tail of the first
    must not run into the second and report a 3-second 'tail'."""
    from probe_stream import noise_floor, sound_end_sample

    sr = 16000
    audio = np.zeros(6 * sr, dtype=np.float32)
    audio[: sr] = 0.3  # utterance A, 0-1 s
    audio[4 * sr : 5 * sr] = 0.3  # utterance B, 4-5 s
    floor = noise_floor(audio, sr)
    end = sound_end_sample(audio, sr, from_sample=sr, floor=floor, limit_ms=2000.0)
    assert end < 4 * sr


def test_breath_between_words_does_not_count_as_continuing_speech():
    """The failure that a 6 dB margin caused on real audio: an inter-word breath sits
    a few dB above the noise floor, so a near-silence threshold reports it as the word
    still being articulated. Speech here is 40 dB above the floor; breath is ~8 dB."""
    from probe_stream import noise_floor, sound_end_sample

    sr = 16000
    rng = np.random.default_rng(0)
    floor_level = 6.5e-4
    audio = rng.normal(0, floor_level, 3 * sr).astype(np.float32)
    audio[: sr] += rng.normal(0, 0.064, sr).astype(np.float32)  # speech, ~40 dB up
    audio[sr : 2 * sr] += rng.normal(0, floor_level * 1.6, sr).astype(np.float32)  # breath

    floor = noise_floor(audio, sr)
    end = sound_end_sample(audio, sr, from_sample=sr, floor=floor)
    # The word ended at 1.0 s. The breath must not extend it into the second 1 s.
    assert end < sr + int(0.15 * sr)


def test_tail_report_carries_both_thresholds(result):
    """A single threshold hides whether a 'cut' is a phoneme or a breath."""
    from probe_stream import tail_report

    for r in tail_report(result):
        assert r["tail_gap_ms_strict"] <= r["tail_gap_ms"]
        assert r["cut_short_strict"] == (r["tail_gap_ms_strict"] > r["trail_pad_ms"])
        # A strict cut is necessarily also a lenient one.
        assert not r["cut_short_strict"] or r["cut_short"]


def test_band_energy_separates_the_fricative_band_from_a_low_tone():
    from probe_stream import band_energy

    sr = 16000
    t = np.arange(sr) / sr
    low = np.sin(2 * np.pi * 200 * t).astype(np.float32)
    high = np.sin(2 * np.pi * 6000 * t).astype(np.float32)
    assert band_energy(high, sr) > 10 * band_energy(low, sr)


def test_tail_report_shape(result):
    from probe_stream import tail_report

    rows = tail_report(result)
    assert len(rows) == len(result.chunks)
    for r in rows:
        assert r["tail_gap_ms"] >= 0
        assert r["trail_pad_ms"] == result.settings.chunk_trail_pad_ms
        assert r["cut_short"] == (r["tail_gap_ms"] > r["trail_pad_ms"])


# --- Alignment tracing --------------------------------------------------------


def test_track_grid_argmax_agrees_with_track():
    """The instrumented grid must be the SAME algorithm, not a lookalike."""
    from probe_stream import track_grid

    from tajwid.feedback.track import _ordinal_of_word, _word_of_ordinal, track
    from tajwid.feedback.types import Span
    from tajwid.session import default_moshaf

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from conftest import AL_FATIHA_1_5

    cursor = Span(sura=1, aya=4, word_idx=0)
    moshaf = default_moshaf()

    found = track(AL_FATIHA_1_5, cursor, moshaf)
    grid = track_grid(AL_FATIHA_1_5, cursor, moshaf)

    assert found.status == "ok"
    assert grid["best"] is not None
    assert grid["best_ratio"] == pytest.approx(float(grid["ratios"].max()), abs=1e-6)

    cur_ord = _ordinal_of_word()[(cursor.sura, cursor.aya, cursor.word_idx)]
    best_offset, _ = grid["best"]
    assert _word_of_ordinal()[cur_ord + best_offset] == (
        found.span.sura,
        found.span.aya,
        found.span.word_idx,
    )


def test_track_grid_margin_is_against_a_different_start():
    """Taking the second-highest CELL would report ~0 margin for every chunk, because
    (best_offset, best_n + 1) is the same start with one more word. The margin has to
    be against a different PLACE in the mushaf to mean anything."""
    from probe_stream import track_grid

    from tajwid.feedback.types import Span
    from tajwid.session import default_moshaf

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from conftest import AL_FATIHA_1_5

    grid = track_grid(AL_FATIHA_1_5, Span(sura=1, aya=4, word_idx=0), default_moshaf())
    ratios = grid["ratios"]
    bi, bj = np.unravel_index(int(ratios.argmax()), ratios.shape)

    # The neighbouring length at the SAME offset is near-identical...
    if bj + 1 < ratios.shape[1]:
        assert ratios[bi, bj + 1] > 0.5 * ratios[bi, bj]
    # ...yet the reported margin is not driven by it, because that row is excluded.
    assert grid["runner_up_ratio"] <= grid["best_ratio"]
    assert grid["margin"] == pytest.approx(
        grid["best_ratio"] - grid["runner_up_ratio"], abs=1e-6
    )
    # A genuinely unique passage should win clearly.
    assert grid["margin"] > 0.05


def test_track_grid_survives_an_unfindable_cursor():
    from probe_stream import track_grid

    from tajwid.feedback.types import Span
    from tajwid.session import default_moshaf

    grid = track_grid("", Span(sura=1, aya=1, word_idx=0), default_moshaf())
    assert grid["best"] is None
    assert grid["margin"] == 0.0


def test_live_trace_is_empty_without_the_live_tier(result):
    """The mock grader gets no live tier (session.py gates it to real/remote), so the
    trace must be empty rather than fabricated."""
    from probe_stream import live_trace

    assert live_trace(result) == []
