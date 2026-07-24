"""Streaming-zipformer live tier: a persistent stream + a forward-only matcher.

The matcher aligns the growing zipformer partial against the KNOWN expected phonemes from
the cursor with a single Levenshtein pass -- no locate, no grid search. Forward-only: it
confirms words the reciter has reached and flags clean skips, and stalls (returns nothing)
when the partial stops lining up, deferring to the Muaalem waqf grade to re-anchor.
"""

from __future__ import annotations

import threading

import Levenshtein as lv
import numpy as np

from ..config import Settings
from ..feedback.locate import _search_engine
from ..feedback.track import (
    _ordinal_of_word,
    _word_of_ordinal,
    normalized_phonemes_for_span,
)
from ..feedback.types import Span

# One recognizer is shared across sessions (the ONNX model is 263 MB); each session owns
# its own stream. sherpa-onnx decode/reset mutate state, so serialize them.
# ponytail: global decode lock, ~50 ms per 0.5 s tick. Per-session recognizers or sherpa
# batched decode if throughput ever demands it.
_DECODE_LOCK = threading.Lock()


def match_forward(
    expected_norm: str,
    word_char_ends: list[int],
    partial_norm: str,
    lookahead_words: int = 1,
    min_match_ratio: float = 0.5,
) -> tuple[list[int], list[int]]:
    """Align the normalized partial against expected phonemes; return (confirmed, skipped)
    word indices (0-based from the anchor).

    ``word_char_ends[k]`` is the cumulative char length of ``expected_norm`` through word k.
    A word is CONFIRMED when it lies inside the best-matching expected PREFIX, minus a
    ``lookahead_words`` tail (the last word is still in flight). A word is SKIPPED when it
    sits before the point the partial actually starts matching. ``min_match_ratio`` is the
    stall guard: too little literal agreement and we assert nothing.

    Why a prefix scan rather than one ``opcodes`` pass over the whole window: an edit-
    distance alignment MAXIMISES matched characters, so it behaves like an LCS across the
    entire window. Arabic phoneme text has a small alphabet where ل ا م ن recur in every
    word, so a few stray characters at the tail of a noisy partial get paired with letters
    far ahead, and the last such pairing reads as "the reciter got this far". Measured: 5
    words recited, 10% ASR error -> 35 of 40 words confirmed. A char-ratio guard cannot
    catch it (that case scores 100% agreement); the extent itself has to be constrained.

    Over-confirming is the dangerous direction, not under-confirming: in hidden (hifz)
    mode a confirmed word is revealed on the page, so running ahead hands the reciter the
    words they were trying to recall. A stalled tick costs nothing -- the Muaalem waqf
    grade re-anchors moments later.
    """
    if not partial_norm or not expected_norm or not word_char_ends:
        return [], []

    # 1. Which expected PREFIX best explains the partial? Raw edit distance is minimised
    #    at the true extent: past it each surplus expected char costs an insertion,
    #    before it each missing one costs a deletion. Only word boundaries are candidates,
    #    so this is <=len(window) C-level distance calls.
    last = min(range(len(word_char_ends)), key=lambda k: lv.distance(partial_norm, expected_norm[: word_char_ends[k]]))
    end = word_char_ends[last]

    # 2. Stall guard, over the chosen prefix: normalised agreement, not raw overlap.
    if lv.distance(partial_norm, expected_norm[:end]) > (1.0 - min_match_ratio) * max(
        len(partial_norm), end
    ):
        return [], []

    # 3. Did the reciter start partway in? Trimming leading words only wins if it makes
    #    the alignment strictly cheaper, so an on-cue start reports no skip.
    starts = [0, *word_char_ends[:-1]]
    best = lv.distance(partial_norm, expected_norm[:end])
    first = 0
    for k in range(1, last + 1):
        d = lv.distance(partial_norm, expected_norm[starts[k] : end])
        if d < best:
            best, first = d, k

    covered = list(range(first, last + 1))
    if lookahead_words:
        covered = covered[:-lookahead_words] if lookahead_words < len(covered) else []
    return covered, list(range(first))


class LiveAligner:
    """One session's streaming live tier: a persistent zipformer stream + the matcher."""

    def __init__(self, recognizer, settings: Settings):
        self.rec = recognizer
        self.s = settings
        self.stream = recognizer.create_stream()
        self.anchor_ord: int | None = None
        self.expected_norm = ""
        self.word_char_ends: list[int] = []

    def reanchor(self, cursor: Span) -> None:
        """Reset the stream and recompute the expected window from the authoritative cursor."""
        with _DECODE_LOCK:
            self.rec.reset(self.stream)
        self.anchor_ord = _ordinal_of_word().get((cursor.sura, cursor.aya, cursor.word_idx))
        w = self.s.live_window_words
        self.expected_norm = normalized_phonemes_for_span(
            cursor.sura, cursor.aya, cursor.word_idx, w
        )
        # Cumulative normalized-char length through each word, for mapping a matched char
        # extent back to a confirmed word count.
        self.word_char_ends = [
            len(normalized_phonemes_for_span(cursor.sura, cursor.aya, cursor.word_idx, k))
            for k in range(1, w + 1)
        ]

    def feed(self, samples: np.ndarray) -> None:
        wav = np.asarray(samples, dtype=np.float32).reshape(-1)
        if not wav.size:
            return
        with _DECODE_LOCK:
            self.stream.accept_waveform(self.s.sample_rate, wav)
            while self.rec.is_ready(self.stream):
                self.rec.decode_stream(self.stream)

    def progress(self) -> tuple[list[Span], list[Span]]:
        if self.anchor_ord is None or not self.expected_norm:
            return [], []
        with _DECODE_LOCK:
            partial = self.rec.get_result(self.stream)
        partial_norm = _search_engine()._normalize_query(partial)
        conf_idx, skip_idx = match_forward(
            self.expected_norm,
            self.word_char_ends,
            partial_norm,
            lookahead_words=self.s.live_lookahead_words,
        )
        return self._to_spans(conf_idx), self._to_spans(skip_idx)

    def _to_spans(self, idxs: list[int]) -> list[Span]:
        words = _word_of_ordinal()
        out: list[Span] = []
        for k in idxs:
            o = (self.anchor_ord or 0) + k
            if 0 <= o < len(words):
                sura, aya, widx = words[o]
                out.append(Span(sura=sura, aya=aya, word_idx=widx))
        return out
