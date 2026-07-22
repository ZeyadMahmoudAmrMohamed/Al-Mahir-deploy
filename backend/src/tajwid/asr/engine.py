"""ASR engines: the one seam between audio and the feedback half.

Three implementations of one small interface:

* ``RealMuaalemEngine`` — the GPU path: silero VAD chunks upstream, then the Muaalem
  multi-level CTC model, decoded reference-free (see transcribe.py). Byte-identical
  to the pre-merge Al-Mahir pipeline.
* ``ZipformerAsrEngine`` — streaming phoneme-CTC (Muno459/zipformer_p-quran via
  sherpa-onnx), CPU-only. StreamSession's waqf/pause endpointing already hands every
  engine one finalized utterance-sized wave at a time, so this engine decodes each
  chunk independently (a fresh sherpa-onnx stream per call) rather than keeping
  cross-chunk streaming state — the underlying model is architecturally streaming,
  but nothing here needs it to be, given how it's invoked. No per-character
  confidence or sifat detection: see ZipformerAsrEngine's docstring.
* ``MockEngine`` — no model at all. It fabricates the transcript a *perfect* reciter
  would have produced from the session cursor, using the real phonetizer. This lets
  the whole backend + frontend loop run on a machine with no GPU, and lets frontend
  work proceed without the model. The only fiction is the confidences.

Engines are built once at startup (see main.py's build_engines) and selected
PER SESSION from the ``start`` message's ``engine`` field (api/ws.py) — falling back
to Settings.resolved_asr_engine's choice for an unknown/omitted name.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

import numpy as np
import torch

from ..config import Settings, get_settings

logger = logging.getLogger(__name__)
from quran_transcript import Aya

from ..feedback.mock import shorten_a_madd
from ..feedback.reference import build_reference
from ..feedback.track import _ordinal_of_word, _uthmani_for_ordinals
from ..feedback.types import SIFA_ATTRS, Span
from .transcribe import ChunkTranscript


@dataclass
class ChunkContext:
    """What an engine may know about a live chunk beyond the audio itself.

    The real engine ignores it entirely. The mock engine — which cannot hear — uses
    the cursor and duration to decide what the reciter "said".
    """

    duration_s: float
    cursor: Optional[Span] = None
    moshaf: object | None = None  # MoshafAttributes; typed loosely to avoid an import cycle


class AsrEngine(Protocol):
    def transcribe_chunk(
        self, wave: torch.FloatTensor, sample_rate: int, ctx: ChunkContext | None = None
    ) -> ChunkTranscript: ...


class RealMuaalemEngine:
    """The GPU path. Loads models lazily via asr.models.get_models()."""

    name = "real"

    def __init__(self) -> None:
        from .models import get_models  # heavy import kept out of mock-mode startup

        self.bundle = get_models()

    def transcribe_chunk(
        self, wave: torch.FloatTensor, sample_rate: int, ctx: ChunkContext | None = None
    ) -> ChunkTranscript:
        from .transcribe import transcribe_reference_free

        return transcribe_reference_free(self.bundle.muaalem, [wave], sample_rate)[0]


# --- Mock ---------------------------------------------------------------------


@dataclass
class _FakeUnit:
    text: str
    prob: float


class _FakeSifa:
    """Duck-types quran_muaalem.Sifa: .phonemes_group + one _FakeUnit per SIFA attr."""

    def __init__(self, phonemes_group: str, attrs: dict[str, str], prob: float):
        self.phonemes_group = phonemes_group
        for attr in SIFA_ATTRS:
            value = attrs.get(attr)
            setattr(self, attr, _FakeUnit(value, prob) if value is not None else None)


# Reciting pace used to guess how many words a mock chunk covers. Murattal with madd
# runs ~1.5–2.5 words/s; the tracker forgives over- and under-shoot anyway.
_WORDS_PER_SECOND = 2.0
_MAX_MOCK_WORDS = 28


class MockEngine:
    """Fabricates the perfect recitation continuing from the session cursor.

    ``mock_error_rate`` optionally shortens a madd in some chunks so the feedback
    colours can be demonstrated end-to-end without a model.
    """

    name = "mock"

    def __init__(self, settings: Settings | None = None):
        self.s = settings or get_settings()
        self._rng = random.Random(0)

    def transcribe_chunk(
        self, wave: torch.FloatTensor, sample_rate: int, ctx: ChunkContext | None = None
    ) -> ChunkTranscript:
        if ctx is None or ctx.cursor is None or ctx.moshaf is None:
            return ChunkTranscript(
                phonemes_text="", char_probs=[], groups=[], group_probs=[], sifat=[]
            )

        n_words = max(1, min(_MAX_MOCK_WORDS, round(ctx.duration_s * _WORDS_PER_SECOND)))
        cursor_ord = _ordinal_of_word().get(
            (ctx.cursor.sura, ctx.cursor.aya, ctx.cursor.word_idx)
        )
        if cursor_ord is None:
            return ChunkTranscript(
                phonemes_text="", char_probs=[], groups=[], group_probs=[], sifat=[]
            )

        # ponytail: the basmalah seam is the mock's one hazard. Al-Fātiḥa 1:1 IS the
        # basmalah, so a span starting there and spilling into 1:2 gets phonetized as one
        # continuous phrase — which elides the hamzat-waṣl of ٱلْحَمْد mid-phrase. The
        # pipeline then strips a *standalone* basmalah and is left with ٱلْحَمْد minus its
        # opening ءَ, one phoneme short of the phrase-initial reference, and the whole
        # group alignment shifts. A real reciter says the basmalah as its own waqf (its
        # own audio chunk), so the model never hands the feedback half this seam. The mock
        # models that by not crossing it: a chunk that opens on 1:1 stops at the basmalah.
        if (ctx.cursor.sura, ctx.cursor.aya, ctx.cursor.word_idx) == (1, 1, 0):
            n_words = min(n_words, len(Aya(1, 1).get().uthmani_words))

        uthmani = _uthmani_for_ordinals(cursor_ord, n_words)
        ref = build_reference(uthmani, ctx.moshaf)

        text = ref.phonemes
        if self.s.mock_error_rate and self._rng.random() < self.s.mock_error_rate:
            text = _inject_error(text, self._rng)

        from quran_transcript import chunck_phonemes

        groups = chunck_phonemes(text)
        prob = 0.97
        # A correct (or madd-shortened) recitation aligns group-for-group with the
        # reference, so reusing the reference sifat per group is exact, not approximate.
        sifat = [
            _FakeSifa(
                group,
                {
                    attr: getattr(ref.sifat[min(i, len(ref.sifat) - 1)], attr)
                    for attr in SIFA_ATTRS
                },
                prob,
            )
            for i, group in enumerate(groups)
        ]
        return ChunkTranscript(
            phonemes_text=text,
            char_probs=[prob] * len(text),
            groups=groups,
            group_probs=[prob] * len(groups),
            sifat=sifat,
        )


def _inject_error(text: str, rng: random.Random) -> str:
    """Corrupt one phoneme group in the MIDDLE of the chunk, for the demo colours.

    Middle, not the edge: an error on the trailing (or leading) word sits on a chunk
    boundary, and the pipeline correctly trims boundary words as unscored — so a
    trailing corruption would demonstrate nothing. A substituted consonant surfaces as
    a word-level error (red); it is the most legible mark to show the loop is live.

    ponytail: demo aid only. The real engine transcribes real audio and needs none of
    this.
    """
    from quran_transcript import chunck_phonemes

    groups = chunck_phonemes(text)
    if len(groups) < 5:
        return shorten_a_madd(text)  # too short to have a safe middle; fall back
    i = rng.randint(len(groups) // 3, max(len(groups) // 3, 2 * len(groups) // 3))
    g = groups[i]
    # Swap the base consonant for a plausible confusion (ت/ط, س/ص, …) or, failing a
    # known pair, just double a letter — either reads as a recitation slip.
    swaps = {"ت": "ط", "س": "ص", "ذ": "ز", "ك": "ق", "د": "ت", "ه": "ح"}
    base = g[0]
    groups[i] = (swaps.get(base, base) + g[1:]) if base in swaps else g + g[0]
    return "".join(groups)


def make_engine(settings: Settings | None = None) -> AsrEngine:
    s = settings or get_settings()
    resolved = s.resolved_asr_engine
    if resolved == "real":
        return RealMuaalemEngine()
    if resolved == "remote":
        return _build_remote(s)
    return MockEngine(s)


# --- Zipformer -------------------------------------------------------------

# Placeholder confidence for the optional debug `units` field only (ChunkResult.
# UnitResult.prob is non-optional). The real scoring path never sees this —
# transcribe_chunk returns char_probs=[], which feedback.confidence already
# treats as "unscored" (see its docstring: "everything stays UNSCORED (None),
# which is not the same as confident") rather than a fabricated confidence
# number. 0.5 here is deliberately neutral, not a claim about accuracy.
_ZIPFORMER_UNSCORED_PROB = 0.5


class ZipformerAsrEngine:
    """Streaming phoneme-CTC (Muno459/zipformer_p-quran, via sherpa-onnx), run
    CPU-only, one finalized utterance-chunk at a time — see this module's
    docstring for why no cross-chunk streaming state is needed here despite
    the model itself being architecturally streaming.

    Two things this engine genuinely cannot provide, left honest rather than
    faked:
      - No per-character CTC confidence extracted from this decode path
        (unlike Muaalem's softmax probs) -> char_probs is always [].
      - No tajweed/sifat detection at all -> every sifat attribute is None,
        via the same _FakeSifa shim MockEngine already uses for its own
        different reason (no model at all vs. this: a model with no sifat
        head).

    tokens.txt MUST be the one shipped in the zipformer_p-quran HF repo, not
    one rebuilt from phoneme_units.json — that mapping has blank at the wrong
    id. See the Quran Companion API backend's ai/asr_streaming_session.py for
    the full story; not repeated here since this is a different repo.
    """

    name = "zipformer"

    def __init__(self, model_path: str | None = None, tokens_path: str | None = None, settings: Settings | None = None):
        import sherpa_onnx

        s = settings or get_settings()
        self._recognizer = sherpa_onnx.OnlineRecognizer.from_zipformer2_ctc(
            tokens=tokens_path or s.zipformer_tokens_path,
            model=model_path or s.zipformer_model_path,
            num_threads=2,
            sample_rate=16000,
            feature_dim=80,
            decoding_method="greedy_search",
        )

    def transcribe_chunk(
        self, wave: torch.FloatTensor, sample_rate: int, ctx: ChunkContext | None = None
    ) -> ChunkTranscript:
        from quran_transcript import chunck_phonemes

        if sample_rate != 16000:
            raise ValueError(f"ZipformerAsrEngine requires 16kHz audio, got {sample_rate}")

        samples = wave.detach().to(torch.float32).cpu().numpy()
        stream = self._recognizer.create_stream()
        stream.accept_waveform(sample_rate, samples)
        # Zipformer2-CTC is chunk-based with a lookahead window: the LAST real audio
        # frames only clear that window once enough audio follows them. Without this,
        # input_finished() lands before the tail has propagated through, and whatever
        # was still sitting in the window — reliably the last character of the last
        # word, since that is always what's most recent when the chunk ends — is
        # silently dropped rather than decoded. This is not a guess: it's the same
        # 0.66s zero-padding sherpa-onnx's own online-zipformer-ctc example feeds
        # before input_finished() for this exact model type.
        tail_paddings = np.zeros(int(0.66 * sample_rate), dtype=np.float32)
        stream.accept_waveform(sample_rate, tail_paddings)
        stream.input_finished()
        while self._recognizer.is_ready(stream):
            self._recognizer.decode_stream(stream)
        text = self._recognizer.get_result(stream)

        if not text:
            return ChunkTranscript(phonemes_text="", char_probs=[], groups=[], group_probs=[], sifat=[])

        # chunck_phonemes, not ai.phoneme_units.segment from the other repo —
        # this is the grouping the rest of THIS system's sifat alignment
        # anchors on, and it's regex/core-letter based rather than a greedy
        # vocab-longest-match, so re-deriving it independently here (instead
        # of importing a different segmenter) keeps this engine's output
        # consistent with what RealMuaalemEngine/MockEngine already produce.
        groups = chunck_phonemes(text)
        sifat = [_FakeSifa(g, {}, _ZIPFORMER_UNSCORED_PROB) for g in groups]
        return ChunkTranscript(
            phonemes_text=text,
            char_probs=[],
            groups=groups,
            group_probs=[_ZIPFORMER_UNSCORED_PROB] * len(groups),
            sifat=sifat,
        )


def _zipformer_files_present(s: Settings) -> bool:
    return Path(s.zipformer_model_path).is_file() and Path(s.zipformer_tokens_path).is_file()


# --- Remote -----------------------------------------------------------------
#
# Built whenever TAJWID_REMOTE_URL is set, on the same terms as zipformer: cheap enough
# to attempt eagerly (it loads no model, it holds a URL), skipped with a warning rather
# than a crash when unavailable, and then selectable PER SESSION alongside whatever
# engine runs locally. That is what makes LOCAL and REMOTE a runtime choice rather than
# a deployment: one server can offer both, and the client picks per recitation.
def _build_remote(s: Settings):
    from .remote import RemoteAsrEngine

    return RemoteAsrEngine(settings=s)


# Engines that are cheap enough to always build eagerly at startup alongside
# whatever Settings.resolved_asr_engine picks (main.py's lifespan calls this
# in a thread, same eager-build-to-avoid-a-GPU-race reasoning as before).
# "zipformer" is CPU/ONNX and loads in well under a second, so it's always
# INCLUDED WHEN AVAILABLE — a user can switch to it per-session even when the
# server's default engine is "real"/"mock".
#
# Its model files are not in the repo (README), so a GPU-less dev box running
# mock/real should still boot: skip zipformer with a warning rather than crash
# the whole service over an engine nobody asked for. If zipformer WAS asked
# for (TAJWID_ASR_ENGINE=zipformer), missing files are the user's actual bug —
# let ZipformerAsrEngine's own assertion surface instead of masking it.
def build_engines(settings: Settings | None = None) -> dict[str, AsrEngine]:
    s = settings or get_settings()
    default_name = s.resolved_asr_engine
    engines: dict[str, AsrEngine] = {}
    if default_name == "zipformer" or _zipformer_files_present(s):
        engines["zipformer"] = ZipformerAsrEngine(settings=s)
    else:
        logger.warning(
            "Skipping the zipformer engine: no model at %s / tokens at %s. "
            "It will not appear in /health's available_engines or be selectable "
            "per-session until TAJWID_ZIPFORMER_MODEL_PATH/TAJWID_ZIPFORMER_TOKENS_PATH "
            "point at real files.",
            s.zipformer_model_path,
            s.zipformer_tokens_path,
        )
    if default_name == "remote" or s.remote_url:
        try:
            engines["remote"] = _build_remote(s)
        except Exception as err:  # noqa: BLE001 — a bad URL must not stop local serving
            if default_name == "remote":
                raise
            logger.warning("Skipping the remote engine: %s", err)
    elif default_name != "remote":
        logger.info(
            "No remote inference configured (TAJWID_REMOTE_URL unset); local engines only."
        )
    if default_name not in engines:
        engines[default_name] = make_engine(s)
    return engines
