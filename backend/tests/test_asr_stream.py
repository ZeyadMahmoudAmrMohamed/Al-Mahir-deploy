"""ASR-half unit tests: prob aggregation, PCM decode, and the streaming endpointer
driven by a stubbed VAD. Model tests are opt-in with ``RUN_MODEL_TESTS=1`` (GPU box).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from tajwid.config import get_settings
from tajwid.asr.stream import StreamSession

ASSETS = Path(__file__).resolve().parent / "assets"
RUN_MODEL_TESTS = os.environ.get("RUN_MODEL_TESTS") == "1"


def test_group_probs_aggregates_multi_token_groups():
    from tajwid.asr.transcribe import _group_probs

    id_to_phoneme = {1: "ن", 2: "َ", 3: "ل"}
    ids = [1, 1, 1, 1, 2, 3, 3, 2]
    probs = [0.9, 0.9, 0.9, 0.9, 1.0, 0.8, 0.8, 0.6]
    groups = ["ننننَ", "للَ"]

    out = _group_probs(id_to_phoneme, ids, probs, groups)
    assert len(out) == 2
    assert out[0] == pytest.approx(np.mean([0.9, 0.9, 0.9, 0.9, 1.0]))
    assert out[1] == pytest.approx(np.mean([0.8, 0.8, 0.6]))


def test_char_probs_expand_to_one_per_character():
    from quran_muaalem.muaalem_typing import Unit
    from tajwid.asr.transcribe import _char_probs

    id_to_phoneme = {1: "ن", 2: "َ"}
    unit = Unit(
        text="ننَ",
        probs=torch.tensor([0.9, 0.8, 0.7]),
        ids=torch.tensor([1, 1, 2]),
    )
    out = _char_probs(id_to_phoneme, unit)
    assert out == pytest.approx([0.9, 0.8, 0.7])


def test_pcm16_to_float_roundtrip():
    from tajwid.api.ws import _pcm16_to_float

    pcm = np.array([0, 32767, -32768, 16384], dtype="<i2").tobytes()
    out = _pcm16_to_float(pcm)
    assert out.dtype == np.float32
    assert out[0] == pytest.approx(0.0)
    assert out[1] == pytest.approx(32767 / 32768.0)
    assert out[2] == pytest.approx(-1.0)


class _StubVAD:
    """Fake silero VAD: speech iff the window's mean abs amplitude exceeds a threshold."""

    def __init__(self, amp_threshold: float = 0.1):
        self.amp_threshold = amp_threshold
        self.resets = 0

    def reset_states(self):
        self.resets += 1

    def __call__(self, window: torch.Tensor, _sr: int) -> float:
        return 1.0 if window.abs().mean().item() > self.amp_threshold else 0.0


def _tone(n: int, amp: float = 0.5) -> torch.Tensor:
    return torch.full((n,), amp, dtype=torch.float32)


def test_endpointing_splits_on_silence():
    s = get_settings()
    sr = s.sample_rate
    session = StreamSession(_StubVAD(), s)

    silence_gap = s.min_silence_endpoint_samples + 2 * s.vad_window_samples
    wave = torch.cat([_tone(sr), torch.zeros(silence_gap), _tone(sr)])

    chunks = session.feed(wave)
    chunks += session.flush()

    assert len(chunks) == 2, f"expected 2 utterances, got {len(chunks)}"
    assert chunks[0].end_sample <= sr + silence_gap
    assert chunks[0].start_sample < chunks[0].end_sample
    assert (
        chunks[1].start_sample
        >= sr + silence_gap - s.chunk_lead_pad_samples - s.vad_window_samples
    )
    assert all(not c.forced for c in chunks)


def test_endpointing_force_cuts_long_speech():
    s = get_settings()
    sr = s.sample_rate
    session = StreamSession(_StubVAD(), s)

    wave = _tone(int(25 * sr))  # ~25 s continuous speech, no pause
    chunks = session.feed(wave) + session.flush()

    assert len(chunks) >= 2
    assert chunks[0].forced
    dur = chunks[0].end_sample - chunks[0].start_sample
    assert dur <= s.max_chunk_samples + s.chunk_lead_pad_samples + s.vad_window_samples


def test_chunk_keeps_edge_padding_around_speech():
    """The finalized region must extend before onset and after offset so the first and
    last phonemes (soft ء, trailing madd) are not clipped."""
    s = get_settings()
    sr = s.sample_rate
    pre_sil = int(0.5 * sr)
    speech = sr
    post_sil = s.min_silence_endpoint_samples + 4 * s.vad_window_samples
    wave = torch.cat([torch.zeros(pre_sil), _tone(speech), torch.zeros(post_sil)])

    session = StreamSession(_StubVAD(), s)
    chunks = session.feed(wave) + session.flush()

    assert len(chunks) == 1
    c = chunks[0]
    onset, offset = pre_sil, pre_sil + speech
    assert c.start_sample < onset
    assert onset - c.start_sample >= s.vad_window_samples
    assert c.end_sample > offset
    assert c.end_sample - offset >= s.vad_window_samples


def test_feed_incrementally_matches_single_feed():
    s = get_settings()
    sr = s.sample_rate
    silence_gap = s.min_silence_endpoint_samples + 2 * s.vad_window_samples
    wave = torch.cat([_tone(sr), torch.zeros(silence_gap), _tone(sr)])

    once = StreamSession(_StubVAD(), s)
    a = once.feed(wave) + once.flush()

    drip = StreamSession(_StubVAD(), s)
    b = []
    frame = int(0.1 * sr)
    for start in range(0, wave.numel(), frame):
        b += drip.feed(wave[start : start + frame])
    b += drip.flush()

    assert len(a) == len(b) == 2


# --------------------------------------------------------------------------- #
# Model tests (opt in: RUN_MODEL_TESTS=1, needs the GPU box / model downloads)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not RUN_MODEL_TESTS, reason="set RUN_MODEL_TESTS=1 to run model tests")
def test_reference_free_matches_stock_on_phonemes_and_sifat():
    from librosa.core import load
    from quran_transcript import Aya, MoshafAttributes, quran_phonetizer

    from tajwid.asr.models import get_models
    from tajwid.asr.transcribe import transcribe_reference_free
    from tajwid.feedback.types import SIFA_ATTRS

    bundle = get_models()
    sr = bundle.settings.sample_rate
    wave, _ = load(str(ASSETS / "test.wav"), sr=sr, mono=True)

    uthmani_ref = Aya(8, 75).get_by_imlaey_words(17, 9).uthmani
    moshaf = MoshafAttributes(
        rewaya="hafs",
        madd_monfasel_len=2,
        madd_mottasel_len=4,
        madd_mottasel_waqf=4,
        madd_aared_len=2,
    )
    ref = quran_phonetizer(uthmani_ref, moshaf, remove_spaces=True)

    stock = bundle.muaalem([wave], [ref], sampling_rate=sr)[0]
    free = transcribe_reference_free(bundle.muaalem, [wave], sr)[0]

    assert free.phonemes_text == stock.phonemes.text.replace("[PAD]", "")
    assert len(free.char_probs) == len(free.phonemes_text)

    matches = total = 0
    for sf_free, sf_stock in zip(free.sifat, stock.sifat):
        if sf_free.phonemes_group != sf_stock.phonemes_group:
            continue
        for level in SIFA_ATTRS:
            a = getattr(sf_free, level)
            b = getattr(sf_stock, level)
            total += 1
            if (a is None) == (b is None) and (a is None or a.text == b.text):
                matches += 1
    assert total > 0
    assert matches / total >= 0.95, f"sifat agreement {matches}/{total} too low"


@pytest.mark.skipif(not RUN_MODEL_TESTS, reason="set RUN_MODEL_TESTS=1 to run model tests")
def test_batch_end_to_end():
    from tajwid.asr.batch import transcribe_file

    results = transcribe_file(ASSETS / "dussary_002282.mp3")

    assert len(results) >= 1
    assert results[-1].is_final
    for r in results:
        assert r.predicted_phonemes
        for u in r.units:
            assert 0.0 <= u.prob <= 1.0
            for feat in u.sifat.values():
                assert feat is None or 0.0 <= feat.prob <= 1.0


@pytest.mark.skipif(not RUN_MODEL_TESTS, reason="set RUN_MODEL_TESTS=1 to run model tests")
def test_streaming_live_session_end_to_end():
    """The full merged pipeline on real audio: hussary reciting An-Najm 53:1."""
    from tajwid.asr.batch import stream_file
    from tajwid.asr.engine import RealMuaalemEngine
    from tajwid.feedback.types import Span

    events = stream_file(
        ASSETS / "hussary_053001.mp3",
        start=Span(sura=53, aya=1, word_idx=0),
        engine=RealMuaalemEngine(),
    )
    assert events
    ok = [e for e in events if e["feedback"]["status"] == "ok"]
    assert ok, "no chunk matched the seeded location"
    assert ok[0]["feedback"]["span"]["sura"] == 53
