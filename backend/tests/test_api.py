"""The WebSocket + REST surface, driven with the mock engine (no GPU)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tajwid.asr.batch import load_audio
from tajwid.main import create_app

ASSETS = Path(__file__).resolve().parent / "assets"
FATIHA = ASSETS / "fatiha_long_track.wav"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("TAJWID_ASR_ENGINE", "mock")
    from tajwid.config import get_settings

    get_settings.cache_clear()
    with TestClient(create_app()) as c:
        yield c
    get_settings.cache_clear()


def test_health_reports_the_engine(client):
    body = client.get("/health").json()
    assert body["status"] == "healthy"
    assert body["engine"] == "mock"


def _pcm16(wave: np.ndarray) -> bytes:
    return (np.clip(wave, -1, 1) * 32767).astype("<i2").tobytes()


@pytest.mark.skipif(not FATIHA.exists(), reason="fatiha asset missing")
def test_ws_session_streams_feedback(client):
    wave = load_audio(FATIHA, 16000).numpy()[: 16000 * 20]  # first 20 s
    pcm = _pcm16(wave)
    frame = int(0.1 * 16000) * 2

    with client.websocket_connect("/ws/session") as ws:
        ws.send_json({"type": "start", "sura": 1, "aya": 1, "word_idx": 0})
        hello = ws.receive_json()
        assert hello["type"] == "session" and hello["engine"] == "mock"

        for i in range(0, len(pcm), frame):
            ws.send_bytes(pcm[i : i + frame])
        ws.send_json({"type": "end"})

        events = []
        while True:
            msg = ws.receive_json()
            if msg["type"] == "done":
                break
            events.append(msg)

    assert events, "no feedback events came back"
    fb = events[0]
    assert fb["type"] == "feedback"
    assert fb["phonemes"]
    assert fb["feedback"]["status"] in ("ok", "ambiguous", "no_match")
    assert fb["cursor"] is not None


def test_ws_rejects_a_non_json_first_message(client):
    """The first message must be the start config — it seeds the cursor."""
    with pytest.raises(Exception):
        with client.websocket_connect("/ws/session") as ws:
            ws.send_text("not json")
            ws.receive_json()


def test_ws_session_can_select_zipformer_engine_per_session(client):
    """Server default engine is 'mock' (see the client fixture), but a session
    can ask for 'zipformer' specifically in its start message and get it —
    per-session engine selection, not the server-wide default. Skips if the
    zipformer model files aren't staged (same convention as test_zipformer_engine.py)."""
    from pathlib import Path

    model = Path(__file__).resolve().parents[1] / "models" / "asr_zipformer" / "quran_phoneme_zipformer.int8.onnx"
    tokens = Path(__file__).resolve().parents[1] / "models" / "asr_zipformer" / "tokens.txt"
    if not model.exists() or not tokens.exists():
        pytest.skip("zipformer model/tokens not present under models/asr_zipformer/")

    wave = load_audio(FATIHA, 16000).numpy()[: 16000 * 15]
    pcm = _pcm16(wave)
    frame = int(0.1 * 16000) * 2

    with client.websocket_connect("/ws/session") as ws:
        ws.send_json({"type": "start", "sura": 1, "aya": 1, "word_idx": 0, "engine": "zipformer"})
        hello = ws.receive_json()
        assert hello["type"] == "session"
        assert hello["engine"] == "zipformer", "requested engine wasn't honored"

        for i in range(0, len(pcm), frame):
            ws.send_bytes(pcm[i : i + frame])
        ws.send_json({"type": "end"})

        events = []
        while True:
            msg = ws.receive_json()
            if msg["type"] == "done":
                break
            events.append(msg)

    assert events, "no feedback events came back from the zipformer engine"


def test_ws_session_unknown_engine_falls_back_to_default(client):
    """An unrecognized engine name doesn't reject the connection — it falls
    back to the server default, and the ack says so honestly."""
    with client.websocket_connect("/ws/session") as ws:
        ws.send_json({"type": "start", "sura": 1, "aya": 1, "word_idx": 0, "engine": "not_a_real_engine"})
        hello = ws.receive_json()
        assert hello["type"] == "session"
        assert hello["engine"] == "mock"
        ws.send_json({"type": "end"})
        ws.receive_json()


def test_moshaf_schema_hides_the_inert_recitation_speed(client):
    """It is declared by MoshafAttributes but read by nothing; offering it as a control
    would promise an effect that never arrives."""
    keys = [f["key"] for f in client.get("/moshaf-schema").json()["fields"]]
    assert "recitation_speed" not in keys
    assert "madd_monfasel_len" in keys, "the real grading knobs must still be listed"


def test_a_config_carrying_recitation_speed_is_still_accepted():
    """Hiding it must not break a client that stored it before."""
    from tajwid.session import resolve_moshaf

    m = resolve_moshaf({"recitation_speed": "hadr", "madd_monfasel_len": 5})
    assert m.madd_monfasel_len == 5, "the rest of the config must survive"
