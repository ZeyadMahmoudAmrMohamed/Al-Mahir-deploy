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


# --- Over-confirmation guards ------------------------------------------------
#
# The synthetic 4-word window above is too small to expose the failure these pin:
# on a real 40-word window a few stray characters used to drag the match extent to
# the far end, confirming ~30 words the reciter never said. In hidden (hifz) mode a
# confirmed word is REVEALED, so over-confirmation spoils the memorization exercise.
# Under-confirming is safe; over-confirming is not.


def _real_window(sura=1, aya=1, word_idx=0, words=40):
    from tajwid.feedback.track import normalized_phonemes_for_span

    expected = normalized_phonemes_for_span(sura, aya, word_idx, words)
    ends = [
        len(normalized_phonemes_for_span(sura, aya, word_idx, k))
        for k in range(1, words + 1)
    ]
    return expected, ends


def test_trailing_noise_does_not_run_the_extent_ahead():
    """8 chars of ASR noise after 5 real words must not confirm the rest of the page."""
    expected, ends = _real_window()
    said = expected[: ends[4]]  # 5 words genuinely recited
    confirmed, _skipped = match_forward(expected, ends, said + "لالالالم", lookahead_words=1)
    assert len(confirmed) <= 5, f"over-confirmed {len(confirmed)} words from 5 recited"


def test_repeated_word_does_not_run_the_extent_ahead():
    """A stutter (word repeated in the partial) must not confirm ahead either."""
    expected, ends = _real_window()
    said = expected[: ends[4]]
    stutter = said + expected[ends[3] : ends[4]]
    confirmed, _skipped = match_forward(expected, ends, stutter, lookahead_words=1)
    assert len(confirmed) <= 5, f"over-confirmed {len(confirmed)} words from 5 recited"


def test_noisy_partial_never_confirms_far_past_what_was_said():
    """Substitutions/insertions/deletions at a realistic rate must not fan out."""
    import random

    expected, ends = _real_window()
    alphabet = sorted(set(expected))
    rng = random.Random(7)
    said_words = 5
    truth = expected[: ends[said_words - 1]]

    worst = 0
    for _ in range(40):
        out = []
        for ch in truth:
            r = rng.random()
            if r < 0.05:
                continue
            if r < 0.10:
                out.append(rng.choice(alphabet))
            elif r < 0.15:
                out.append(ch)
                out.append(rng.choice(alphabet))
            else:
                out.append(ch)
        confirmed, _sk = match_forward(expected, ends, "".join(out), lookahead_words=1)
        worst = max(worst, len(confirmed))
    # A couple of words of slack for a genuinely ambiguous boundary; not 30.
    assert worst <= said_words + 2, f"a noisy partial confirmed {worst} of {said_words} words"


def test_garbage_against_a_real_window_stalls():
    expected, ends = _real_window()
    assert match_forward(expected, ends, "زززززززززززز", lookahead_words=1) == ([], [])


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
