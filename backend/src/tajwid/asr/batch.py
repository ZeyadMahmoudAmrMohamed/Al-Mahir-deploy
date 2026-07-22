"""Offline helpers: transcribe (and optionally score) a whole audio file.

The whole-file path uses the W2V-BERT segmenter as the chunker (it is designed for
complete recitations of any length), then the same engine + feedback path as live.
Real-engine only: the segmenter and Muaalem model must be loaded.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import torch

from ..config import get_settings
from .contract import ChunkResult
from .engine import ChunkContext, RealMuaalemEngine
from .models import ModelBundle, get_models
from .segment import segment_wave


def load_audio(path: str | Path, sample_rate: int = 16000) -> torch.FloatTensor:
    """Load any ffmpeg-decodable audio as a 1-D 16 kHz mono float32 tensor."""
    wave, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    return torch.from_numpy(wave).to(torch.float32)


def transcribe_file(
    path: str | Path,
    bundle: ModelBundle | None = None,
    *,
    session_id: str | None = None,
) -> list[ChunkResult]:
    """Whole-file: waqf-segment with the W2V-BERT segmenter, then transcribe each
    region reference-free (no streaming, no silero pass)."""
    bundle = bundle or get_models()
    sr = bundle.settings.sample_rate
    session_id = session_id or Path(path).stem

    engine = RealMuaalemEngine()
    wave = load_audio(path, sr).cpu()
    regions = segment_wave(bundle, wave)
    if not regions:  # very short clip / all silence: treat the whole wave as one chunk
        regions = [(wave, (0, wave.numel()))]

    results: list[ChunkResult] = []
    for seq, (sub_wave, (start, end)) in enumerate(regions):
        transcript = engine.transcribe_chunk(
            sub_wave, sr, ChunkContext(duration_s=(end - start) / sr)
        )
        results.append(
            ChunkResult.from_transcript(
                transcript,
                session_id=session_id,
                chunk_seq=seq,
                audio_span_sec=(start / sr, end / sr),
            )
        )
    if results:
        results[-1].is_final = True
    return results


def stream_file(
    path: str | Path,
    *,
    session_id: str | None = None,
    start=None,
    frame_ms: int = 100,
    engine=None,
) -> list[dict]:
    """Emulate live streaming: feed a file through a LiveSession in small frames.

    Works with either engine — with the mock it is the CPU-only end-to-end check of
    the whole merged pipeline (VAD endpointing -> ASR -> tracking -> word feedback).
    """
    from ..session import LiveSession
    from .engine import make_engine

    s = get_settings()
    sr = s.sample_rate
    wave = load_audio(path, sr).cpu().numpy()
    frame = int(frame_ms * sr / 1000)

    session = LiveSession(
        engine or make_engine(),
        session_id=session_id or Path(path).stem,
        start=start,
    )
    events: list[dict] = []
    for i in range(0, len(wave), frame):
        events.extend(session.feed(wave[i : i + frame]))
    events.extend(session.flush())
    return events


def _main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Transcribe a recitation to phonemes+sifat."
    )
    parser.add_argument("audio", help="Path to an audio file.")
    args = parser.parse_args()

    settings = get_settings()
    if settings.resolved_asr_engine != "real":
        raise SystemExit(
            "Batch transcription needs the real models (TAJWID_ASR_ENGINE=real)."
        )

    for result in transcribe_file(args.audio):
        print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()
