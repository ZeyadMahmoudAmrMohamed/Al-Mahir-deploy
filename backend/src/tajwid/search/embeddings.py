"""BGE-M3 text embeddings for āyah search.

The one rule that matters: **vectors are L2-normalised**. The FAISS index is IndexFlatIP,
so with unit-norm vectors inner product IS cosine similarity. Forget to normalise and the
rankings look plausible and are silently wrong.

The model is ~2 GB and takes ~10 s to load. It is loaded lazily, on the first semantic
query — the recitation service must boot fast and most sessions never search.
"""

from __future__ import annotations

import unicodedata

import numpy as np

MODEL_NAME = "BAAI/bge-m3"
DIM = 1024

_model = None

# Tatweel is non-combining, so combining()==0 misses it.
_TATWEEL = "ـ"  # U+0640


def strip_diacritics(s: str) -> str:
    """Remove Arabic tashkīl (harakat, shadda, sukun, superscript alef, …) and tatweel.

    BGE-M3 was not trained on heavily-diacritised classical Arabic: embedding text_uthmani
    directly gives near-random vectors (measured Recall@10 0.10 vs 0.26 undiacritised), so
    the Arabic index — and therefore the Arabic query — lives in stripped space.
    """
    s = unicodedata.normalize("NFC", s)
    return "".join(c for c in s if c != _TATWEEL and unicodedata.combining(c) == 0)


def get_model():
    """Load BGE-M3 once. Imported inside so importing this module never drags in torch."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(MODEL_NAME)
    return _model


def encode(texts: list[str]) -> np.ndarray:
    """Encode to L2-normalised float32 vectors, shape (len(texts), DIM)."""
    vecs = get_model().encode(
        texts,
        batch_size=64,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vecs.astype("float32")
