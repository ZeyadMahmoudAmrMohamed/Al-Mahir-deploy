"""Waqf segmentation wrapper over the recitations-segmenter model.

Used to (optionally) refine a finalized silero chunk into 20 ms-accurate waqf sub-segments
before Muaalem inference, matching the distribution the Muaalem model was trained on.
Also usable standalone to segment a whole file in the batch/offline path.
"""

from __future__ import annotations

import torch

from recitations_segmenter import (
    NoSpeechIntervals,
    TooHighMinSpeechDuration,
    clean_speech_intervals,
    segment_recitations,
)

from .models import ModelBundle


def segment_wave(
    bundle: ModelBundle,
    wave: torch.FloatTensor,
) -> list[tuple[torch.FloatTensor, tuple[int, int]]]:
    """Segment a single 16 kHz mono wave into waqf-bounded slices.

    Args:
        bundle: Loaded models.
        wave: 1-D float tensor on CPU.

    Returns:
        A list of ``(slice_wave, (start_sample, end_sample))`` where the sample offsets are
        relative to the start of ``wave``. If the segmenter finds no speech (silence) or
        filtering removes everything, returns an empty list.
    """
    s = bundle.settings
    if wave.numel() == 0:
        return []

    outputs = segment_recitations(
        [wave.cpu()],
        bundle.segmenter,
        bundle.segmenter_processor,
        batch_size=s.segmenter_batch_size,
        device=bundle.segmenter_device,
        dtype=bundle.segmenter_dtype,
        sample_rate=s.sample_rate,
    )
    out = outputs[0]

    try:
        clean = clean_speech_intervals(
            out.speech_intervals,
            out.is_complete,
            min_silence_duration_ms=s.min_silence_duration_ms,
            min_speech_duration_ms=s.min_speech_duration_ms,
            pad_duration_ms=s.pad_duration_ms,
            return_seconds=False,
        )
    except (NoSpeechIntervals, TooHighMinSpeechDuration):
        return []

    slices: list[tuple[torch.FloatTensor, tuple[int, int]]] = []
    total = wave.numel()
    for start, end in clean.clean_speech_intervals.tolist():
        start_i = max(0, int(start))
        end_i = min(total, int(end))
        if end_i <= start_i:
            continue
        slices.append((wave[start_i:end_i], (start_i, end_i)))
    return slices
