"""The remote engine: the wire format, and the fallbacks around it.

The one test that matters here is that the round trip is LOSSLESS. The pre-merge HTTP
handoff dropped per-character probs and ṣifāt, which silently disabled confidence grading
and articulation feedback (see tajwid/session.py's docstring). Moving inference back onto
a wire reintroduces exactly that hazard, so it is asserted rather than assumed.

No GPU and no network: the transport is exercised against a local echo server.
"""

from __future__ import annotations

import json
import threading

import numpy as np
import pytest
import torch

from tajwid.asr.remote import (
    RemoteAsrEngine,
    pcm16_to_wave,
    transcript_to_wire,
    wave_to_pcm16,
    wire_to_transcript,
)
from tajwid.asr.transcribe import ChunkTranscript


class _Sifa:
    """Stands in for quran_muaalem.Sifa: .phonemes_group plus per-attribute units."""

    def __init__(self, group, **attrs):
        self.phonemes_group = group
        for name, value in attrs.items():
            setattr(self, name, value)


class _Unit:
    def __init__(self, text, prob):
        self.text = text
        self.prob = prob


def _transcript() -> ChunkTranscript:
    return ChunkTranscript(
        phonemes_text="بِسمِ",
        char_probs=[0.99, 0.97, 0.95, 0.93, 0.91],
        groups=["بِ", "سمِ"],
        group_probs=[0.98, 0.92],
        sifat=[
            _Sifa(
                "بِ",
                hams_or_jahr=_Unit("jahr", 0.97),
                shidda_or_rakhawa=_Unit("shadeed", 0.95),
                qalqla=_Unit("qalqla", 0.91),
                ghonna=None,
            ),
            _Sifa("سمِ", hams_or_jahr=_Unit("hams", 0.88), ghonna=None),
        ],
    )


def test_round_trip_preserves_char_probs_and_sifat():
    """Every field the feedback half reads must survive the wire, not just the phonemes."""
    original = _transcript()
    restored = wire_to_transcript(json.loads(json.dumps(transcript_to_wire(original))))

    assert restored.phonemes_text == original.phonemes_text
    # The confidence grader slices char_probs by CHARACTER offset, so a dropped or
    # reordered entry is a wrong accusation, not a rounding error.
    assert restored.char_probs == original.char_probs
    assert len(restored.char_probs) == len(restored.phonemes_text)
    assert restored.groups == original.groups
    assert restored.group_probs == original.group_probs

    assert len(restored.sifat) == 2
    assert restored.sifat[0].hams_or_jahr.text == "jahr"
    assert restored.sifat[0].hams_or_jahr.prob == pytest.approx(0.97)
    assert restored.sifat[0].qalqla.text == "qalqla"
    # An absent ṣifā must come back as None, not as a fabricated neutral value.
    assert restored.sifat[0].ghonna is None
    assert restored.sifat[1].hams_or_jahr.text == "hams"


def test_missing_sifa_stays_none_for_every_attribute():
    """A group with no ṣifāt at all round-trips to all-None, never to defaults."""
    t = ChunkTranscript(
        phonemes_text="ب", char_probs=[0.9], groups=["ب"], group_probs=[0.9],
        sifat=[_Sifa("ب")],
    )
    restored = wire_to_transcript(transcript_to_wire(t))
    for attr in ("hams_or_jahr", "itbaq", "safeer", "ghonna", "istitala"):
        assert getattr(restored.sifat[0], attr) is None


def test_audio_survives_the_pcm16_hop():
    wave = torch.from_numpy(
        (np.sin(np.linspace(0, 40, 1600)) * 0.5).astype(np.float32)
    )
    restored = pcm16_to_wave(wave_to_pcm16(wave))
    assert restored.shape == wave.shape
    # PCM16 quantisation is ~3e-5; anything larger means a byte-order or scaling bug.
    assert torch.max(torch.abs(restored - wave)).item() < 1e-3


def test_engine_requires_a_url():
    from tajwid.config import Settings

    with pytest.raises(ValueError, match="TAJWID_REMOTE_URL"):
        RemoteAsrEngine(settings=Settings(remote_url=None))


def test_engine_rejects_non_16k_audio():
    engine = RemoteAsrEngine(url="ws://127.0.0.1:1/infer")
    with pytest.raises(ValueError, match="16kHz"):
        engine.transcribe_chunk(torch.zeros(160), 8000)


def test_unreachable_remote_returns_empty_rather_than_killing_the_session():
    """A dead tunnel must cost one chunk, not the recitation.

    An empty transcript is what LiveSession already drops without emitting an event, so
    the reciter sees no feedback for that waqf and the session continues.
    """
    engine = RemoteAsrEngine(url="ws://127.0.0.1:1/infer")
    engine.timeout = 0.5
    result = engine.transcribe_chunk(torch.zeros(16000), 16000)
    assert result.phonemes_text == ""
    assert result.char_probs == []
    assert result.sifat == []


def test_engine_talks_to_a_server_speaking_the_wire_format():
    """End to end over a real socket, with a stub standing in for the GPU."""
    from websockets.sync.server import serve

    expected = transcript_to_wire(_transcript())
    received: list[int] = []

    def handler(ws):
        for message in ws:
            received.append(len(message))
            ws.send(json.dumps(expected))

    server = serve(handler, "127.0.0.1", 0)
    port = server.socket.getsockname()[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        engine = RemoteAsrEngine(url=f"ws://127.0.0.1:{port}/infer")
        assert engine.name == "remote"
        out = engine.transcribe_chunk(torch.zeros(16000), 16000)
        engine.close()
    finally:
        server.shutdown()

    # 1 s of 16 kHz PCM16 is 32000 bytes on the wire.
    assert received == [32000]
    assert out.phonemes_text == "بِسمِ"
    assert out.char_probs == [0.99, 0.97, 0.95, 0.93, 0.91]
    assert out.sifat[0].hams_or_jahr.text == "jahr"
