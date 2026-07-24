import json
import wave as wavemod
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tajwid.capture import CAPTURED_SETTINGS, open_capture
from tajwid.config import Settings

ASSETS = Path(__file__).resolve().parent / "assets"


def test_gate_needs_both_keys(tmp_path):
    """Neither key alone records audio."""
    off = Settings(capture_dir=None)
    on = Settings(capture_dir=str(tmp_path))
    assert open_capture(off, {"capture": True}, "s1") is None
    assert open_capture(on, {}, "s2") is None
    assert open_capture(on, {"capture": False}, "s3") is None
    assert open_capture(on, {"capture": True}, "s4") is not None


def test_writes_four_files(tmp_path):
    s = Settings(capture_dir=str(tmp_path))
    cap = open_capture(s, {"capture": True, "sura": 1, "aya": 1}, "sid")
    cap.start({"capture": True, "sura": 1, "aya": 1}, {"engine": "remote"})
    # 320 samples of silence = 640 bytes of PCM16.
    cap.frame(b"\x00\x00" * 320)
    cap.frame(b"\x00\x00" * 320)
    cap.events([{"type": "feedback", "chunk_seq": 0}])
    cap.close()

    d = tmp_path / "sid"
    assert {p.name for p in d.iterdir()} == {
        "input.wav",
        "frames.jsonl",
        "events.jsonl",
        "start.json",
    }

    with wavemod.open(str(d / "input.wav"), "rb") as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getframerate() == 16000
        assert w.getnframes() == 640

    frames = [json.loads(x) for x in (d / "frames.jsonl").read_text().splitlines()]
    assert [f["n"] for f in frames] == [320, 320]
    assert all("t_ms" in f for f in frames)

    start = json.loads((d / "start.json").read_text(encoding="utf-8"))
    assert start["cfg"]["sura"] == 1
    assert start["resolved"]["engine"] == "remote"
    assert set(start["settings"]) == set(CAPTURED_SETTINGS)


def test_settings_snapshot_covers_pipeline_behaviour():
    """Every field that changes chunking or grading must be recorded, or replay lies."""
    for field in (
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
        "sample_rate",
    ):
        assert field in CAPTURED_SETTINGS


def test_write_failure_does_not_raise(tmp_path, monkeypatch):
    """A capture error must never take down a recitation."""
    s = Settings(capture_dir=str(tmp_path))
    cap = open_capture(s, {"capture": True}, "sid")
    cap.start({}, {})

    def boom(*a, **kw):
        raise OSError("disk full")

    monkeypatch.setattr(cap._wav, "writeframes", boom)
    cap.frame(b"\x00\x00" * 320)  # must not raise
    assert cap.dead is True
    cap.frame(b"\x00\x00" * 320)  # still silent once dead
    cap.events([{"type": "x"}])
    cap.close()


# --- The WS/REST surface ------------------------------------------------------


@pytest.fixture
def capture_client(monkeypatch, tmp_path):
    """A mock-engine app with capture enabled, plus the directory it records into."""
    monkeypatch.setenv("TAJWID_ASR_ENGINE", "mock")
    monkeypatch.setenv("TAJWID_CAPTURE_DIR", str(tmp_path))
    from tajwid.config import get_settings
    from tajwid.main import create_app

    get_settings.cache_clear()
    with TestClient(create_app()) as c:
        yield c, tmp_path
    get_settings.cache_clear()


def test_health_reports_capture_availability(capture_client):
    client, _ = capture_client
    assert client.get("/health").json()["capture_available"] is True


def test_health_reports_capture_unavailable_without_a_dir(monkeypatch):
    monkeypatch.setenv("TAJWID_ASR_ENGINE", "mock")
    # Set empty rather than delete: `backend/.env` may define TAJWID_CAPTURE_DIR on a
    # developer's machine, and pydantic-settings reads it, so deleting the process env
    # var does not disable capture. An explicit "" outranks the .env and is falsy.
    monkeypatch.setenv("TAJWID_CAPTURE_DIR", "")
    from tajwid.config import get_settings
    from tajwid.main import create_app

    get_settings.cache_clear()
    with TestClient(create_app()) as c:
        assert c.get("/health").json()["capture_available"] is False
    get_settings.cache_clear()


def _stream(client, *, capture: bool):
    """Drive one session through the real WS handler. Returns the `session` ack."""
    from tajwid.asr.batch import load_audio

    wave = load_audio(ASSETS / "test.wav", 16000).numpy()
    pcm = (np.clip(wave, -1, 1) * 32767).astype("<i2")
    start = {"type": "start", "sura": 1, "aya": 1, "word_idx": 0}
    if capture:
        start["capture"] = True
    with client.websocket_connect("/ws/session") as ws:
        ws.send_text(json.dumps(start))
        ack = json.loads(ws.receive_text())
        for i in range(0, len(pcm), 1600):
            ws.send_bytes(pcm[i : i + 1600].tobytes())
        ws.send_text(json.dumps({"type": "end"}))
        while True:
            msg = json.loads(ws.receive_text())
            if msg.get("type") == "done":
                break
    return ack


def test_ws_records_when_asked(capture_client):
    client, root = capture_client
    ack = _stream(client, capture=True)

    assert ack["capture"] is True
    d = root / ack["session_id"]
    assert {p.name for p in d.iterdir()} == {
        "input.wav",
        "frames.jsonl",
        "events.jsonl",
        "start.json",
    }
    frames = [json.loads(x) for x in (d / "frames.jsonl").read_text().splitlines()]
    assert frames and sum(f["n"] for f in frames) > 0
    with wavemod.open(str(d / "input.wav"), "rb") as w:
        assert w.getnframes() == sum(f["n"] for f in frames)
    # The resolved config must record what the SERVER chose, not what was requested.
    start = json.loads((d / "start.json").read_text(encoding="utf-8"))
    assert start["resolved"]["engine"] == "mock"
    assert start["resolved"]["strictness"] in ("lenient", "normal", "strict")


def test_ws_records_nothing_when_not_asked(capture_client):
    """A normal session is byte-identical to today's behaviour."""
    client, root = capture_client
    ack = _stream(client, capture=False)
    assert ack["capture"] is False
    assert list(root.iterdir()) == []
