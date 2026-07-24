from pathlib import Path

import numpy as np
import pytest

from tajwid.asr.engine import _zipformer_files_present
from tajwid.asr.live_aligner import match_forward
from tajwid.config import get_settings
from tajwid.feedback.types import Span

ASSETS = Path(__file__).resolve().parent / "assets"
FATIHA = ASSETS / "fatiha_long_track.wav"

# Synthetic muṣḥaf window: 4 words "aa|bb|cc|dd", boundaries at 2,4,6,8.
EXP = "aabbccdd"
ENDS = [2, 4, 6, 8]


def test_forward_recitation_confirms_minus_lookahead():
    # Reciter has said words 0,1,2; the last is held back by the 1-word lookahead.
    confirmed, skipped = match_forward(EXP, ENDS, "aabbcc", lookahead_words=1)
    assert confirmed == [0, 1]
    assert skipped == []


def test_full_partial_holds_back_only_the_last_word():
    confirmed, skipped = match_forward(EXP, ENDS, "aabbccdd", lookahead_words=1)
    assert confirmed == [0, 1, 2]
    assert skipped == []


def test_skipped_leading_word_is_flagged():
    # Reciter skipped word 0 and said 1,2,3.
    confirmed, skipped = match_forward(EXP, ENDS, "bbccdd", lookahead_words=1)
    assert skipped == [0]
    assert confirmed == [1, 2]  # word 3 held back


def test_garbage_partial_stalls():
    confirmed, skipped = match_forward(EXP, ENDS, "xyzxyzxyz", lookahead_words=1)
    assert confirmed == []
    assert skipped == []


def test_empty_partial_stalls():
    assert match_forward(EXP, ENDS, "", lookahead_words=1) == ([], [])


# --- The streaming component (needs the zipformer model files) ----------------


@pytest.mark.skipif(not _zipformer_files_present(get_settings()), reason="zipformer model files absent")
def test_live_aligner_confirms_forward_through_al_fatiha():
    import sherpa_onnx

    from tajwid.asr.batch import load_audio
    from tajwid.asr.live_aligner import LiveAligner
    from tajwid.feedback.track import _ordinal_of_word

    s = get_settings()
    rec = sherpa_onnx.OnlineRecognizer.from_zipformer2_ctc(
        tokens=s.zipformer_tokens_path,
        model=s.zipformer_model_path,
        num_threads=2,
        sample_rate=16000,
        feature_dim=80,
        decoding_method="greedy_search",
    )
    aligner = LiveAligner(rec, s)
    aligner.reanchor(Span(sura=1, aya=1, word_idx=0))

    wave = np.asarray(load_audio(FATIHA, 16000), dtype=np.float32).reshape(-1)
    seen: list[Span] = []
    for i in range(0, len(wave), 8000):  # ~0.5s frames
        aligner.feed(wave[i : i + 8000])
        confirmed, _skipped = aligner.progress()
        if confirmed:
            seen = confirmed

    assert seen, "the aligner should confirm words as the recitation streams in"
    # Al-Fatiha starts at sura 1; confirmed words are real, ordered coordinates.
    assert seen[0].sura == 1
    ordinals = [_ordinal_of_word()[(w.sura, w.aya, w.word_idx)] for w in seen]
    assert ordinals == sorted(ordinals), "confirmed words must be in muṣḥaf order"
