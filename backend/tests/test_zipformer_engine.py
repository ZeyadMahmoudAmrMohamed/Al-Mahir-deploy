"""ZipformerAsrEngine, driven directly (no session/VAD in between) against real audio."""
from __future__ import annotations

from pathlib import Path

import pytest

from tajwid.asr.batch import load_audio

ASSETS = Path(__file__).resolve().parent / "assets"
FATIHA = ASSETS / "fatiha_long_track.wav"

MODEL = Path(__file__).resolve().parents[1] / "models" / "asr_zipformer" / "quran_phoneme_zipformer.int8.onnx"
TOKENS = Path(__file__).resolve().parents[1] / "models" / "asr_zipformer" / "tokens.txt"


@pytest.fixture
def zipformer_engine():
    from tajwid.asr.engine import ZipformerAsrEngine

    if not MODEL.exists() or not TOKENS.exists():
        pytest.skip("zipformer model/tokens not present under models/asr_zipformer/")
    return ZipformerAsrEngine(model_path=str(MODEL), tokens_path=str(TOKENS))


@pytest.mark.skipif(not FATIHA.exists(), reason="fatiha asset missing")
def test_transcribe_chunk_produces_phonemes(zipformer_engine):
    wave = load_audio(FATIHA, 16000)[: 16000 * 15]  # first 15s

    transcript = zipformer_engine.transcribe_chunk(wave, 16000)

    assert transcript.phonemes_text
    assert transcript.groups
    # Every group is a real phoneme-unit string, concatenating back to the full text.
    assert "".join(transcript.groups) == transcript.phonemes_text


def test_char_probs_are_honestly_unscored_not_fabricated(zipformer_engine):
    """See ZipformerAsrEngine's docstring: no real per-character confidence is
    available from this decode path, so char_probs must come back empty —
    which feedback.confidence treats as genuinely UNSCORED (None), not as a
    fabricated confident/unconfident number."""
    wave = load_audio(FATIHA, 16000)[: 16000 * 5]

    transcript = zipformer_engine.transcribe_chunk(wave, 16000)

    assert transcript.char_probs == []


def test_sifat_are_all_none(zipformer_engine):
    """Zipformer has no tajweed/sifat detection — every attribute of every
    group's Sifa must be None, not a guessed value."""
    from tajwid.asr.engine import SIFA_ATTRS

    wave = load_audio(FATIHA, 16000)[: 16000 * 5]

    transcript = zipformer_engine.transcribe_chunk(wave, 16000)

    assert transcript.sifat, "expected at least one group from 5s of real recitation"
    for sifa in transcript.sifat:
        for attr in SIFA_ATTRS:
            assert getattr(sifa, attr) is None


def test_empty_audio_returns_empty_transcript(zipformer_engine):
    import torch

    transcript = zipformer_engine.transcribe_chunk(torch.zeros(1600), 16000)
    assert transcript.phonemes_text == ""
    assert transcript.groups == []


def test_rejects_wrong_sample_rate(zipformer_engine):
    import torch

    with pytest.raises(ValueError):
        zipformer_engine.transcribe_chunk(torch.zeros(8000), 8000)
