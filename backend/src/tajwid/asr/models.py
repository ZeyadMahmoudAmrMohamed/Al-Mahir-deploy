"""Load and cache the three models the pipeline needs, exactly once.

* silero VAD  -- tiny, stateful, real-time streaming endpoint gate.
* segmenter   -- Wav2Vec2-BERT frame classifier for 20 ms-accurate waqf cuts.
* Muaalem     -- Multi-level CTC model: chunk audio -> phonemes + 10 sifat levels.

The models are downloaded from Hugging Face on first use.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioFrameClassification,
)

from quran_muaalem.inference import Muaalem
from .vad import load_vad

from ..config import Settings, get_settings


@dataclass
class ModelBundle:
    settings: Settings
    vad: torch.jit.ScriptModule
    segmenter: AutoModelForAudioFrameClassification
    segmenter_processor: AutoFeatureExtractor
    muaalem: Muaalem
    # Resolved placement for the segmenter (used by segment.py).
    segmenter_device: torch.device
    segmenter_dtype: torch.dtype


def load_models(settings: Settings | None = None) -> ModelBundle:
    """Instantiate every model once and move each to its configured device/dtype.

    Devices are resolved per component so a small-VRAM GPU can host the per-chunk Muaalem
    model while the segmenter runs on CPU (see Settings.*_device).
    """
    settings = settings or get_settings()

    vad_device = settings.resolved_vad_device
    seg_device = settings.resolved_segmenter_device
    seg_dtype = settings.dtype_for(seg_device)
    muaalem_device = settings.resolved_muaalem_device

    vad = load_vad()
    vad.to(torch.device(vad_device))

    segmenter_processor = AutoFeatureExtractor.from_pretrained(settings.segmenter_model_id)
    segmenter = AutoModelForAudioFrameClassification.from_pretrained(
        settings.segmenter_model_id
    )
    segmenter.to(torch.device(seg_device), dtype=seg_dtype)
    segmenter.eval()

    muaalem = Muaalem(
        model_name_or_path=settings.muaalem_model_id,
        device=muaalem_device,
        dtype=settings.dtype_for(muaalem_device),
    )

    return ModelBundle(
        settings=settings,
        vad=vad,
        segmenter=segmenter,
        segmenter_processor=segmenter_processor,
        muaalem=muaalem,
        segmenter_device=torch.device(seg_device),
        segmenter_dtype=seg_dtype,
    )


@lru_cache
def get_models() -> ModelBundle:
    """Process-wide singleton bundle (used by the server)."""
    return load_models()
