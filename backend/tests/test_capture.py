import json
import wave as wavemod

from tajwid.capture import CAPTURED_SETTINGS, open_capture
from tajwid.config import Settings


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
