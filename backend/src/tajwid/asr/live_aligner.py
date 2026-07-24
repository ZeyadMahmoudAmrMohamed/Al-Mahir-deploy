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
    A word is CONFIRMED when its expected chars are behind the matched extent, minus a
    ``lookahead_words`` tail (the last word is still in flight). A word is SKIPPED when its
    expected chars sit entirely before the first matched char. ``min_match_ratio`` is the
    stall guard: too little literal agreement and we assert nothing.
    """
    if not partial_norm or not expected_norm:
        return [], []

    ops = lv.opcodes(partial_norm, expected_norm)
    equal_chars = sum(j2 - j1 for tag, _i1, _i2, j1, j2 in ops if tag == "equal")
    consumed = [(j1, j2) for tag, _i1, _i2, j1, j2 in ops if tag in ("equal", "replace")]

    # Stall: the partial doesn't line up with the expected text well enough to trust.
    if not consumed or equal_chars < min_match_ratio * len(partial_norm):
        return [], []

    first_match = consumed[0][0]
    matched_extent = consumed[-1][1]

    skipped = [k for k, end in enumerate(word_char_ends) if end <= first_match]
    covered = [k for k, end in enumerate(word_char_ends) if end <= matched_extent]
    if lookahead_words:
        covered = covered[:-lookahead_words]
    skipset = set(skipped)
    confirmed = [k for k in covered if k not in skipset]
    return confirmed, skipped


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
