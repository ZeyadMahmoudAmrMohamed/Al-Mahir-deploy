"""Per-session streaming buffer + silero-VAD endpointing.

The session accepts arbitrary-length 16 kHz mono float frames and decides *when* a waqf
(pause) has ended, emitting finalized speech regions. It does no phoneme/sifat inference --
that is done downstream on each finalized region (see ``pipeline.run``).

Endpointing rule (per silero window, ~96 ms):
  * mark speech start on the first window whose speech prob > threshold;
  * once in speech, a silence run >= ``min_silence_endpoint_ms`` finalizes the utterance;
  * a hard cap ``max_chunk_s`` force-cuts an over-long utterance (the Muaalem model was
    trained on <=20 s waqf segments).

silero VAD is stateful (an RNN); we reset its state at each finalized-by-silence boundary so
successive utterances are scored independently, mirroring how waqf segments are independent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..config import Settings, get_settings


@dataclass
class FinalizedChunk:
    """A finalized speech region, in absolute stream sample coordinates."""

    wave: torch.FloatTensor  # the (unpadded) speech samples
    start_sample: int  # absolute, inclusive
    end_sample: int  # absolute, exclusive
    forced: bool  # True if cut by the max-length cap rather than a waqf


class StreamSession:
    """Stateful endpointer for a single recitation stream.

    Depends only on the (tiny, CPU) silero VAD module — not on the full model
    bundle — so live endpointing works identically under the mock ASR engine.
    """

    def __init__(self, vad: torch.jit.ScriptModule, settings: Settings | None = None):
        self.vad = vad
        self.s = settings or get_settings()
        self.window = self.s.vad_window_samples

        # Absolute sample index of buffer[0].
        self._buffer_start_abs = 0
        self._buffer = torch.zeros(0, dtype=torch.float32)
        # Number of full windows already fed to silero (relative to buffer start).
        self._fed_windows = 0

        # Utterance state machine.
        self._in_speech = False
        self._speech_start_abs = 0
        self._last_speech_end_abs = 0
        self._trailing_silence = 0

        # Tail of the previously emitted chunk, re-sent as the head of the next one
        # (see Settings.chunk_overlap_ms). Empty when overlap is disabled.
        self._overlap_tail = torch.zeros(0, dtype=torch.float32)

        self.vad.reset_states()

    # -- public API -------------------------------------------------------
    def feed(self, samples: np.ndarray | torch.Tensor) -> list[FinalizedChunk]:
        """Append samples (16 kHz mono float in [-1, 1]) and return any finalized chunks."""
        wave = _as_float_tensor(samples)
        if wave.numel():
            self._buffer = torch.cat([self._buffer, wave])
        return self._process()

    def flush(self) -> list[FinalizedChunk]:
        """End of stream: finalize any in-progress utterance. Marks nothing as forced."""
        chunks = self._process()
        speech_len = self._last_speech_end_abs - self._speech_start_abs
        if self._in_speech and speech_len >= self.s.min_speech_samples:
            chunks.append(
                self._extract(
                    self._speech_start_abs, self._last_speech_end_abs, forced=False
                )
            )
            self._reset_after_finalize(self._last_speech_end_abs, reset_vad=True)
        return chunks

    # -- internals --------------------------------------------------------
    def _process(self) -> list[FinalizedChunk]:
        chunks: list[FinalizedChunk] = []
        n_windows = self._buffer.numel() // self.window

        while self._fed_windows < n_windows:
            w_start_abs = self._buffer_start_abs + self._fed_windows * self.window
            rel = self._fed_windows * self.window
            window = self._buffer[rel : rel + self.window]
            with torch.no_grad():
                prob = float(self.vad(window, self.s.sample_rate))
            self._fed_windows += 1
            w_end_abs = w_start_abs + self.window
            is_speech = prob > self.s.vad_threshold

            if is_speech:
                if not self._in_speech:
                    self._in_speech = True
                    self._speech_start_abs = w_start_abs
                self._last_speech_end_abs = w_end_abs
                self._trailing_silence = 0
            elif self._in_speech:
                self._trailing_silence += self.window

            # Finalize on a long-enough waqf. Speech shorter than min_speech is dropped as
            # noise (a breath/click) rather than emitted as a chunk.
            if (
                self._in_speech
                and self._trailing_silence >= self.s.min_silence_endpoint_samples
            ):
                speech_len = self._last_speech_end_abs - self._speech_start_abs
                if speech_len >= self.s.min_speech_samples:
                    chunks.append(
                        self._extract(
                            self._speech_start_abs,
                            self._last_speech_end_abs,
                            forced=False,
                        )
                    )
                self._reset_after_finalize(w_end_abs, reset_vad=True)
                n_windows = self._buffer.numel() // self.window
                continue

            # Hard cap: force-cut an over-long utterance. The overlap tail is prepended
            # AFTER this check, so the budget has to make room for it -- otherwise a
            # capped 19 s chunk goes out at 19 s + overlap and breaks the very limit the
            # cap exists to enforce (Muaalem was trained on <=20 s segments).
            if (
                self._in_speech
                and (w_end_abs - self._speech_start_abs)
                >= self.s.max_chunk_samples
                - self.s.chunk_overlap_samples
                - self.s.chunk_lead_pad_samples
            ):
                chunks.append(
                    self._extract(self._speech_start_abs, w_end_abs, forced=True)
                )
                # Continue the stream as a fresh utterance from the cut point.
                self._reset_after_finalize(w_end_abs, reset_vad=False)
                n_windows = self._buffer.numel() // self.window
                continue

        self._trim()
        return chunks

    def _extract(self, start_abs: int, end_abs: int, *, forced: bool) -> FinalizedChunk:
        lead = self.s.chunk_lead_pad_samples
        # No trailing pad on a forced (mid-speech) cut: it would duplicate the next chunk's
        # onset. A waqf cut pads into the following silence to keep the trailing madd whole.
        #
        # `chunk_overlap_ms` duplicates audio DELIBERATELY, which does not contradict the
        # rule above but supersedes its reasoning: the objection to a trailing pad was an
        # unreconciled duplicate word, and overlap is paired with a dedup rule that keeps
        # the scored verdict. The trailing pad stays off regardless -- it is an unmanaged
        # duplicate of unknown length, where the overlap is a fixed, deliberate window.
        trail = 0 if forced else self.s.chunk_trail_pad_samples
        a = max(self._buffer_start_abs, start_abs - lead)
        b = min(self._buffer_start_abs + self._buffer.numel(), end_abs + trail)
        rel_a = a - self._buffer_start_abs
        rel_b = b - self._buffer_start_abs
        wave = self._buffer[rel_a:rel_b].clone()

        # Stash this chunk's tail BEFORE prepending the previous one's, so the overlap
        # is a fixed window of new audio rather than a tail that grows each chunk.
        overlap = self.s.chunk_overlap_samples
        tail, self._overlap_tail = self._overlap_tail, (
            wave[-overlap:].clone() if overlap else self._overlap_tail
        )
        if overlap and tail.numel():
            # The re-sent audio is real audio the reciter uttered, so `start_sample`
            # moves back with it: audio_span_sec must describe what was transcribed,
            # not what was newly endpointed. The duplicated WORDS are reconciled
            # downstream by key (frontend lib/marks.ts keeps the scored verdict over a
            # later `unverified` re-emission), which is what makes the duplication safe
            # -- see the note in this method about forced cuts.
            wave = torch.cat([tail, wave])
            a -= tail.numel()

        return FinalizedChunk(
            wave=wave,
            start_sample=a,
            end_sample=b,
            forced=forced,
        )

    def _reset_after_finalize(self, cut_abs: int, *, reset_vad: bool) -> None:
        """Drop buffer up to ``cut_abs`` and reset the utterance state machine."""
        rel = cut_abs - self._buffer_start_abs
        self._buffer = self._buffer[rel:].clone()
        self._buffer_start_abs = cut_abs
        self._fed_windows = 0
        self._in_speech = False
        self._speech_start_abs = cut_abs
        self._last_speech_end_abs = cut_abs
        self._trailing_silence = 0
        if reset_vad:
            self.vad.reset_states()

    def _trim(self) -> None:
        """Discard already-scored leading silence to keep the buffer bounded.

        A lookback of ``chunk_lead_pad`` (rounded up to whole windows, plus one) worth of
        already-scored silence is retained so that when speech onset is next detected, the
        lead pad in ``_extract`` has real samples to extend into (otherwise soft onsets get
        clipped). Only fully-scored windows beyond the lookback are dropped.
        """
        if self._in_speech:
            return
        keep_windows = self.s.chunk_lead_pad_samples // self.window + 2
        drop_windows = self._fed_windows - keep_windows
        if drop_windows <= 0:
            return
        drop = drop_windows * self.window
        self._buffer = self._buffer[drop:].clone()
        self._buffer_start_abs += drop
        self._fed_windows -= drop_windows
        self._speech_start_abs = self._buffer_start_abs
        self._last_speech_end_abs = self._buffer_start_abs


def _as_float_tensor(samples: np.ndarray | torch.Tensor) -> torch.FloatTensor:
    if isinstance(samples, torch.Tensor):
        t = samples.detach().to(torch.float32).cpu().reshape(-1)
    else:
        t = torch.as_tensor(np.asarray(samples), dtype=torch.float32).reshape(-1)
    return t
