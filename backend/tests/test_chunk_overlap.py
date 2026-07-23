"""Chunk overlap re-sends the previous chunk's tail without disturbing endpointing."""

import numpy as np
import torch

from tajwid.asr.stream import StreamSession
from tajwid.config import Settings

SR = 16000


class FakeVad:
    """Speech iff the window has any energy. Deterministic, no model download."""

    def __call__(self, window, sr):
        return 1.0 if float(window.abs().max()) > 0.01 else 0.0

    def reset_states(self):
        pass


def _stream(overlap_ms: int):
    """Two 2 s utterances separated by 1 s of silence."""
    s = Settings(chunk_overlap_ms=overlap_ms, min_speech_ms=200)
    speech = np.sin(np.linspace(0, 400, 2 * SR)).astype(np.float32)
    silence = np.zeros(SR, dtype=np.float32)
    wave = np.concatenate([silence, speech, silence, speech, silence])

    session = StreamSession(FakeVad(), s)
    chunks = []
    for i in range(0, len(wave), 1600):
        chunks.extend(session.feed(wave[i : i + 1600]))
    chunks.extend(session.flush())
    return chunks


def test_overlap_off_is_unchanged():
    chunks = _stream(0)
    assert len(chunks) == 2
    for c in chunks:
        assert c.wave.numel() == c.end_sample - c.start_sample


def test_second_chunk_carries_the_previous_tail():
    overlap_ms = 500
    off, on = _stream(0), _stream(overlap_ms)

    # Endpointing itself must not change: same chunk count. If the retained tail were
    # re-fed to the VAD it would re-finalize as extra chunks (or loop).
    assert len(on) == len(off) == 2

    extra = int(overlap_ms * SR / 1000)
    assert on[0].wave.numel() == off[0].wave.numel(), "first chunk has no predecessor"
    assert on[1].wave.numel() == off[1].wave.numel() + extra
    # start_sample moves back with the audio, so audio_span describes what was sent.
    assert on[1].start_sample == off[1].start_sample - extra
    # The prepended samples are literally the previous chunk's last ones.
    assert torch.equal(on[1].wave[:extra], on[0].wave[-extra:])


def _stream_n(overlap_ms: int, n_utterances: int):
    s = Settings(chunk_overlap_ms=overlap_ms, min_speech_ms=200)
    speech = np.sin(np.linspace(0, 400, 2 * SR)).astype(np.float32)
    silence = np.zeros(SR, dtype=np.float32)
    wave = np.concatenate([silence] + [speech, silence] * n_utterances)

    session = StreamSession(FakeVad(), s)
    chunks = []
    for i in range(0, len(wave), 1600):
        chunks.extend(session.feed(wave[i : i + 1600]))
    chunks.extend(session.flush())
    return chunks


def test_tail_does_not_accumulate_across_chunks():
    """A tail taken AFTER prepending would grow every chunk. Take it before.

    Compared against the overlap-off run rather than against a constant: the final
    chunk is legitimately shorter (the stream ends before its trailing pad), so equal
    absolute sizes is the wrong invariant. A CONSTANT DELTA is the right one.
    """
    extra = int(500 * SR / 1000)
    off, on = _stream_n(0, 4), _stream_n(500, 4)
    assert len(on) == len(off) == 4

    deltas = [b.wave.numel() - a.wave.numel() for a, b in zip(off, on)]
    assert deltas[0] == 0, "the first chunk has no predecessor to overlap with"
    assert deltas[1:] == [extra] * 3, f"overlap accumulating: {deltas}"


def test_overlap_stays_inside_the_max_chunk_cap():
    """A capped chunk plus its overlap must not exceed max_chunk_s.

    Muaalem was trained on <=20 s segments; emitting cap + overlap breaks the limit the
    cap exists to enforce. Measured at 20.34 s before this was fixed.
    """
    s = Settings(chunk_overlap_ms=2700, min_speech_ms=200, max_chunk_s=19.0)
    speech = np.sin(np.linspace(0, 8000, 40 * SR)).astype(np.float32)  # 40 s, no pause
    session = StreamSession(FakeVad(), s)
    chunks = []
    for i in range(0, len(speech), 1600):
        chunks.extend(session.feed(speech[i : i + 1600]))
    chunks.extend(session.flush())

    assert any(c.forced for c in chunks), "expected the cap to force a cut"
    for c in chunks:
        assert c.wave.numel() <= s.max_chunk_samples, (
            f"chunk of {c.wave.numel() / SR:.2f}s exceeds the {s.max_chunk_s}s cap"
        )


if __name__ == "__main__":
    test_overlap_stays_inside_the_max_chunk_cap()
    test_overlap_off_is_unchanged()
    test_second_chunk_carries_the_previous_tail()
    test_tail_does_not_accumulate_across_chunks()
    print("ok")
