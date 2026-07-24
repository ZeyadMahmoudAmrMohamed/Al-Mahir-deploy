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
