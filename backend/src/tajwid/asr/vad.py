"""Load the silero VAD (v4.0 TorchScript) shipped inside recitations_segmenter.

Loaded via the package's file location rather than ``import recitations_segmenter``,
because that import drags in torchaudio + transformers — heavy modules the mock
engine never needs. The real engine imports them anyway via models.py.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


def load_vad() -> torch.jit.ScriptModule:
    """Deliberately NOT cached: silero is a stateful RNN, so every live session must
    own a fresh instance — a shared module would mix the streams' hidden states."""
    spec = importlib.util.find_spec("recitations_segmenter")
    if spec is None or spec.origin is None:
        raise RuntimeError("recitations_segmenter package not found")
    path = Path(spec.origin).parent / "data" / "silero_vad_v4.0.jit"
    model = torch.jit.load(str(path), map_location="cpu")
    model.eval()
    return model
