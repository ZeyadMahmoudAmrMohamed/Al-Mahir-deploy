"""Configuration for the whole Tajwid service (ASR half + feedback half).

All tunables live here so the transport, VAD, model, and feedback layers stay
parameter-free. Values can be overridden with ``TAJWID_`` prefixed environment
variables, e.g. ``TAJWID_DEVICE=cuda`` or ``TAJWID_ASR_ENGINE=mock``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

import torch
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TAJWID_",
        protected_namespaces=(),
        # A .env beside the process CWD is the ordinary way to hand this service an LLM
        # key without putting it in a shell profile or a systemd unit.
        env_file=".env",
        extra="ignore",
    )

    # --- Engine selection ------------------------------------------------
    # "real" loads the GPU models; "mock" fabricates model output from the
    # phonetizer (no torch models, runs anywhere); "auto" picks real iff CUDA
    # is available. The mock exists so the full backend+frontend loop can be
    # exercised on a machine with no GPU.
    asr_engine: Literal["auto", "real", "mock", "zipformer", "remote"] = "auto"
    # Probability that the mock engine injects a shortened madd into an
    # otherwise-perfect chunk, so the feedback colours can be demonstrated.
    mock_error_rate: float = 0.0

    # --- Zipformer (see asr/engine.py's ZipformerAsrEngine) --------------
    # Resolved relative to the process's CWD at runtime (sherpa-onnx takes
    # these as plain path strings) — same as everything else in this file
    # being env-overridable rather than hard-coded absolute. Override with
    # TAJWID_ZIPFORMER_MODEL_PATH / TAJWID_ZIPFORMER_TOKENS_PATH if this
    # service isn't launched from the repo root.
    zipformer_model_path: str = "models/asr_zipformer/quran_phoneme_zipformer.onnx"
    zipformer_tokens_path: str = "models/asr_zipformer/tokens.txt"

    # --- Remote GPU inference (see asr/remote.py) ------------------------
    # The WebSocket endpoint of a GPU running remote_server.py, typically a Kaggle or
    # Colab notebook exposed through ngrok or a Cloudflare tunnel, e.g.
    #   wss://<subdomain>.ngrok.app/infer
    # Unset by default and OPTIONAL: with no URL the engine is simply not built, exactly
    # as zipformer is skipped when its files are missing. Local inference is untouched.
    remote_url: str | None = None
    # Generous on purpose. A cold notebook loads ~2.4 GB of weights on the first chunk,
    # and a tunnel adds a round trip to whichever continent the GPU is in.
    remote_timeout_s: float = 120.0

    # --- Devices / dtype -------------------------------------------------
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Muaalem was trained in bfloat16; segmenter supports it too. On CPU we fall
    # back to float32 (bfloat16 matmul on CPU is slow / partially unsupported).
    dtype_str: str = "bfloat16"

    # Per-component device placement (defaults to `device`). On a small-VRAM GPU you
    # can keep the heavy per-chunk Muaalem model on GPU while offloading the segmenter
    # to CPU, e.g. TAJWID_SEGMENTER_DEVICE=cpu. silero VAD stays on CPU by default
    # (tiny, real-time on CPU, consumes the CPU-resident buffer without transfers).
    muaalem_device: str | None = None
    segmenter_device: str | None = None
    vad_device: str = "cpu"

    # --- Audio -----------------------------------------------------------
    sample_rate: int = 16000  # fixed by every model in the stack

    # --- Model ids -------------------------------------------------------
    muaalem_model_id: str = "obadx/muaalem-model-v3_2"
    segmenter_model_id: str = "obadx/recitation-segmenter-v2"

    # --- Streaming endpointing (silero VAD gate) -------------------------
    # silero v4 operates on fixed 1536-sample windows (~96 ms at 16 kHz).
    vad_window_samples: int = 1536
    # Speech-probability threshold. 0.6 (not 0.3) so the shallow, short dips at a waqf
    # in continuous recitation register as silence.
    vad_threshold: float = 0.6
    # A silence run at least this long *after* speech finalizes a chunk (a waqf).
    min_silence_endpoint_ms: int = 300
    # Discard finalized speech shorter than this as noise (breaths/clicks).
    min_speech_ms: int = 200
    # Hard cap per chunk: the Muaalem model was trained on <=20 s waqf segments.
    max_chunk_s: float = 19.0
    # Padding added around a finalized speech region before inference (see stream.py).
    chunk_lead_pad_ms: int = 120
    chunk_trail_pad_ms: int = 240
    # Re-send this much of the previous chunk's tail as the head of the next one, so a
    # word on a chunk boundary is INTERIOR to the following chunk and gets scored there
    # instead of being blanked by feedback.words.trim_edges.
    #
    # 0 disables it (the historical behaviour). Two words is the minimum that helps:
    # trim_edges blanks words[0] whenever the span starts mid-aya, so one word of
    # overlap merely re-trims the same word in the next chunk. At the measured learner
    # pace (~0.74 words/s) two words is ~2700 ms; at studio pace (~2 words/s) ~1000 ms.
    # Costs proportionally more inference per chunk -- measure before raising it.
    # See FINDINGS.md.
    chunk_overlap_ms: int = 0

    # --- Diagnostic capture (see capture.py) -----------------------------
    # Directory under which a diagnosed session's raw input is recorded for offline
    # replay. None (the default) disables capture UNCONDITIONALLY -- a client asking
    # for `capture: true` on a server without this set records nothing. Both keys are
    # required so a deployed instance cannot be made to record by a crafted client
    # message, and never records without the reciter asking.
    capture_dir: str | None = None

    # --- Live word-fill (Tier 1: provisional per-word feedback before the waqf) ---
    # Read the streaming-zipformer partial this often (ms) to fill words in live. The
    # live tier is a COMPANION to the authoritative grade, so it is gated to sessions
    # graded by Muaalem (real/remote) — a mock or zipformer grader gets no live tier.
    live_feedback: bool = True
    live_interval_ms: int = 300
    # Words held back from the END of each live match: the last word(s) of a
    # decode-so-far are still in flight (the reciter may be mid-madd), committed only
    # once later audio stabilises them.
    live_lookahead_words: int = 1
    # Words of expected context the live matcher aligns against, from the anchor forward.
    # A waqf re-anchor arrives well before an utterance could exhaust this.
    live_window_words: int = 40

    # --- W2V-BERT segmenter (chunker for the offline whole-file batch path) ---
    segmenter_batch_size: int = 8
    min_silence_duration_ms: int = 30
    min_speech_duration_ms: int = 30
    pad_duration_ms: int = 60

    # --- Feedback defaults ------------------------------------------------
    # Grade the 10 sifat attributes. OFF pending an audit of the reference-side
    # derivation: measured over three recitations, sifat produced ~95% of all findings
    # at a median confidence of 1.000, and a PROFESSIONAL recitation
    # (tests/assets/fatiha_long_track.wav) scored 80% of its words as errors. The
    # confusions are symmetric in both directions (ghonna maghnoon->not_maghnoon 16
    # times AND not_maghnoon->maghnoon 16 times), which is the signature of comparing
    # against the wrong reference groups rather than a biased model head. Since the
    # sifa confidence threshold is 0.85 and the confidences are 1.000, none of these
    # can degrade to `almost` — every one is a confident false accusation, which is
    # the single thing Constitution VI forbids. See FINDINGS.md.
    #
    # This is a DEFAULT, not a removal: a session that explicitly names sifa keys in
    # the start message's `rules` still gets them graded.
    grade_sifat: bool = False
    # The reciter's style. Overridable per session in the WS config message.
    madd_monfasel_len: int = 4
    madd_mottasel_len: int = 4
    madd_mottasel_waqf: int = 4
    madd_aared_len: int = 4
    strictness: str = "normal"

    # --- Āyah search (see search/service.py) ------------------------------
    # Weight of the lexical (surface + root BM25) signal in `mode=hybrid`:
    #   final = cosine + alpha * (bm25 / bm25.max())
    # Swept upstream: Recall@10 peaks near 0.15 (0.429), but an exact-āyah-fragment query
    # needs >= ~0.15 for its own āyah to rank #1; 0.20 keeps exact matches robust at 0.417
    # (vs 0.393 vector-only). Small on purpose — the vector stays dominant.
    search_hybrid_alpha: float = 0.20
    # Default for HyDE query expansion when a request doesn't say. Off: it costs an LLM
    # call (latency + a key), and search must work without one.
    search_hyde: bool = False
    # Default search mode when a request doesn't say. "hybrid" is the measured best on
    # the Arabic path and degrades to pure vector on English (no Arabic lexical bag).
    search_mode: Literal["keyword", "vector", "hybrid"] = "hybrid"

    # --- LLM (HyDE query expansion only, today) ---------------------------
    # Any OpenAI-compatible provider; migrating is this URL and nothing else.
    llm_base_url: str = "https://api.groq.com/openai/v1"
    # Optional ON PURPOSE. The service must start without it — recitation feedback,
    # keyword search and plain vector search need no LLM at all. search/llm.py raises at
    # call time instead, and HyDE falls back to the raw query.
    llm_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("TAJWID_LLM_API_KEY", "GROQ_API_KEY", "LLM_API_KEY"),
    )
    # HyDE is a rewrite, not reasoning — use the small/cheap model.
    llm_model_small: str = "openai/gpt-oss-20b"

    def dtype_for(self, device: str) -> torch.dtype:
        """Inference dtype for a device: configured dtype on CUDA, float32 on CPU."""
        if device.startswith("cpu"):
            return torch.float32
        return getattr(torch, self.dtype_str)

    @property
    def dtype(self) -> torch.dtype:
        return self.dtype_for(self.device)

    @property
    def resolved_asr_engine(self) -> str:
        if self.asr_engine != "auto":
            return self.asr_engine
        return "real" if torch.cuda.is_available() else "mock"

    # Resolved per-component devices (fall back to the main device).
    @property
    def resolved_muaalem_device(self) -> str:
        return self.muaalem_device or self.device

    @property
    def resolved_segmenter_device(self) -> str:
        return self.segmenter_device or self.device

    @property
    def resolved_vad_device(self) -> str:
        return self.vad_device or "cpu"

    @property
    def max_chunk_samples(self) -> int:
        return int(self.max_chunk_s * self.sample_rate)

    @property
    def min_silence_endpoint_samples(self) -> int:
        return int(self.min_silence_endpoint_ms * self.sample_rate / 1000)

    @property
    def min_speech_samples(self) -> int:
        return int(self.min_speech_ms * self.sample_rate / 1000)

    @property
    def chunk_lead_pad_samples(self) -> int:
        return int(self.chunk_lead_pad_ms * self.sample_rate / 1000)

    @property
    def chunk_trail_pad_samples(self) -> int:
        return int(self.chunk_trail_pad_ms * self.sample_rate / 1000)

    @property
    def chunk_overlap_samples(self) -> int:
        return int(self.chunk_overlap_ms * self.sample_rate / 1000)

    @property
    def live_interval_samples(self) -> int:
        return int(self.live_interval_ms * self.sample_rate / 1000)


@lru_cache
def get_settings() -> Settings:
    return Settings()
