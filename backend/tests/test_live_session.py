"""The merged pipeline, end to end, without a GPU.

Real silero VAD endpointing over REAL recitation audio + the mock ASR engine. Only
the acoustic model is faked; the endpointer, the adapter, tracking, the diff, sifat
comparison, scoring and word aggregation are all the production code paths.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tajwid.asr.batch import load_audio, stream_file
from tajwid.asr.engine import (
    _MAX_MOCK_WORDS,
    ChunkContext,
    MockEngine,
    _zipformer_files_present,
)
from tajwid.config import Settings
from tajwid.feedback.types import Span
from tajwid.session import LiveSession, transcript_to_output

ASSETS = Path(__file__).resolve().parent / "assets"
FATIHA = ASSETS / "fatiha_long_track.wav"


@pytest.fixture
def mock_engine():
    return MockEngine(Settings(asr_engine="mock"))


def test_mock_engine_produces_per_char_probs(mock_engine):
    """The #1 silent bug: probs must be one float per CHARACTER of the phoneme text."""
    from tajwid.session import default_moshaf

    t = mock_engine.transcribe_chunk(
        np.zeros(16000, dtype=np.float32),
        16000,
        ChunkContext(duration_s=3.0, cursor=Span(sura=1, aya=1, word_idx=0), moshaf=default_moshaf()),
    )
    assert t.phonemes_text
    assert len(t.char_probs) == len(t.phonemes_text)
    assert len(t.sifat) == len(t.groups)


def test_transcript_to_output_carries_all_three_fields(mock_engine):
    """INTEGRATION.md's whole point: text AND probs AND sifat cross the boundary."""
    from tajwid.session import default_moshaf

    t = mock_engine.transcribe_chunk(
        np.zeros(16000, dtype=np.float32),
        16000,
        ChunkContext(duration_s=3.0, cursor=Span(sura=1, aya=2, word_idx=0), moshaf=default_moshaf()),
    )
    out = transcript_to_output(t)
    assert out.phonemes.text == t.phonemes_text
    assert out.phonemes.probs is not None
    assert len(out.phonemes.probs) == len(out.phonemes.text)
    assert out.sifat and out.sifat[0].attrs and out.sifat[0].probs


@pytest.mark.skipif(not FATIHA.exists(), reason="fatiha asset missing")
def test_live_session_tracks_through_al_fatiha(mock_engine):
    """A seeded session over real audio: chunks match, the cursor advances, no ayah
    is skipped, and words come back graded and renderable."""
    events = stream_file(
        FATIHA, start=Span(sura=1, aya=1, word_idx=0), engine=mock_engine
    )

    assert events, "silero found no speech in the recitation"
    ok = [e for e in events if e["feedback"]["status"] == "ok"]
    assert ok, "no chunk was located"

    first = ok[0]["feedback"]
    assert first["span"]["sura"] == 1
    assert first["words"], "an ok match must return renderable words"
    for w in first["words"]:
        assert {"sura", "aya", "word_idx", "uthmani", "status"} <= set(w)
        assert w["status"] in ("correct", "almost", "error")

    # The cursor advances and never skips ahead. Compared in FLAT WORD ORDINALS, not
    # ayah numbers: recitation runs off the end of Al-Fatiha into Al-Baqarah, where the
    # ayah number resets to 1 while the position has plainly moved forward.
    from tajwid.feedback.track import _ordinal_of_word

    ordinals = [
        _ordinal_of_word()[(e["cursor"]["sura"], e["cursor"]["aya"], e["cursor"]["word_idx"])]
        for e in ok
        if e["cursor"]
    ]
    assert ordinals == sorted(ordinals), f"cursor went backwards: {ordinals}"
    assert len(set(ordinals)) > 1, "the session never moved past the first chunk"
    # No ayah is silently skipped: each chunk resumes within a word or two of where the
    # last one ended (the tracker's backward overlap makes small rewinds legitimate).
    gaps = [b - a for a, b in zip(ordinals, ordinals[1:])]
    assert all(g <= _MAX_MOCK_WORDS + 2 for g in gaps), f"a gap skipped words: {gaps}"


def test_seek_resets_the_cursor(mock_engine):
    session = LiveSession(
        mock_engine, session_id="t", start=Span(sura=1, aya=1, word_idx=0)
    )
    session.seek(Span(sura=2, aya=255, word_idx=0))
    assert session.cursor.sura == 2 and session.cursor.aya == 255
    assert session.state.penalty == 0


def test_perfect_recitation_is_never_accused(mock_engine):
    """The rule the whole design bends around: a perfect recitation gets no `error`."""
    from tajwid.session import default_moshaf

    session = LiveSession(
        mock_engine, session_id="t", start=Span(sura=112, aya=1, word_idx=0)
    )
    t = mock_engine.transcribe_chunk(
        np.zeros(16000, dtype=np.float32),
        16000,
        ChunkContext(duration_s=4.0, cursor=session.cursor, moshaf=default_moshaf()),
    )
    from tajwid.feedback.pipeline import analyse_session

    feedback, _ = analyse_session(transcript_to_output(t), session.state)
    assert feedback.status == "ok"
    assert not [w for w in feedback.words if w.status == "error"]


# --- Tier 1 gating (streaming-zipformer live tier) ---------------------------


class _FakeRealEngine:
    """A Muaalem-named grader that hears nothing. The live tier no longer routes audio
    through the grading engine, so the gate can be exercised with a hollow stand-in."""

    name = "real"

    def transcribe_chunk(self, wave, sample_rate, ctx=None):
        from tajwid.asr.transcribe import ChunkTranscript

        return ChunkTranscript(
            phonemes_text="", char_probs=[], groups=[], group_probs=[], sifat=[]
        )


def test_no_live_tier_without_a_zipformer_engine():
    # Pin live_feedback explicitly so an ambient .env/TAJWID_LIVE_FEEDBACK can't be what
    # disables the tier (init kwargs outrank env sources in pydantic-settings).
    s = Settings(asr_engine="real", live_feedback=True)
    sess = LiveSession(
        _FakeRealEngine(),
        session_id="t",
        start=Span(sura=1, aya=1, word_idx=0),
        settings=s,
        zipformer_engine=None,
    )
    assert sess._live is None


def test_no_live_tier_on_mock_even_with_zipformer(mock_engine):
    # Companion-only: a mock/zipformer grader gets no live tier regardless of zipformer.
    s = Settings(asr_engine="mock", live_feedback=True)
    sentinel = object()  # a non-None "zipformer engine" that must still be ignored
    sess = LiveSession(
        mock_engine,
        session_id="t",
        start=Span(sura=1, aya=1, word_idx=0),
        settings=s,
        zipformer_engine=sentinel,
    )
    assert sess._live is None


@pytest.mark.skipif(
    not _zipformer_files_present(Settings()), reason="zipformer model files absent"
)
def test_live_tier_built_on_real_with_zipformer():
    from tajwid.asr.engine import ZipformerAsrEngine

    s = Settings(asr_engine="real", live_feedback=True)
    zf = ZipformerAsrEngine(settings=s)
    sess = LiveSession(
        _FakeRealEngine(),
        session_id="t",
        start=Span(sura=1, aya=1, word_idx=0),
        settings=s,
        zipformer_engine=zf,
    )
    assert sess._live is not None


def test_process_survives_a_feedback_analysis_crash(mock_engine, monkeypatch):
    """A vendored-phonetizer KeyError on one span must skip the chunk, not kill the
    session (see the KeyError:'ء' from quran_transcript on some Al-An'am spans)."""
    import torch

    import tajwid.session as sess_mod
    from tajwid.asr.stream import FinalizedChunk

    sess = LiveSession(mock_engine, session_id="t", start=Span(sura=1, aya=1, word_idx=0))

    def boom(*_a, **_k):
        raise KeyError("ء")

    monkeypatch.setattr(sess_mod, "analyse_session", boom)
    fin = FinalizedChunk(
        wave=torch.zeros(16000, dtype=torch.float32),
        start_sample=0,
        end_sample=16000,
        forced=False,
    )
    # No exception escapes, the chunk is dropped, and the cursor is left untouched.
    assert sess._process(fin) is None
    assert sess.cursor == Span(sura=1, aya=1, word_idx=0)


# --- The per-session `live` choice -------------------------------------------


def test_resolve_live_defaults_to_the_env_setting():
    from tajwid.session import resolve_live

    assert resolve_live(None, Settings(live_feedback=True)) is True
    assert resolve_live(None, Settings(live_feedback=False)) is False


def test_resolve_live_honours_an_explicit_choice():
    from tajwid.session import resolve_live

    # An explicit pick outranks the env default in BOTH directions.
    assert resolve_live(False, Settings(live_feedback=True)) is False
    assert resolve_live(True, Settings(live_feedback=False)) is True


@pytest.mark.skipif(
    not _zipformer_files_present(Settings()), reason="zipformer model files absent"
)
def test_live_false_turns_the_tier_off_even_when_it_could_run():
    from tajwid.asr.engine import ZipformerAsrEngine

    s = Settings(asr_engine="real", live_feedback=True)
    sess = LiveSession(
        _FakeRealEngine(),
        session_id="t",
        start=Span(sura=1, aya=1, word_idx=0),
        settings=s,
        zipformer_engine=ZipformerAsrEngine(settings=s),
        live=False,
    )
    assert sess._live is None


def test_live_true_cannot_force_the_tier_onto_a_mock_grader(mock_engine):
    """Companion-only is an invariant, not a default: the client can only turn it OFF."""
    s = Settings(asr_engine="mock", live_feedback=False)
    sess = LiveSession(
        mock_engine,
        session_id="t",
        start=Span(sura=1, aya=1, word_idx=0),
        settings=s,
        zipformer_engine=object(),
        live=True,
    )
    assert sess._live is None
