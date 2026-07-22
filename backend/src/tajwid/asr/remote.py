"""Remote GPU inference: the same AsrEngine contract, over a WebSocket.

The seam is ``AsrEngine.transcribe_chunk`` and nothing else. VAD endpointing, waqf
chunking, muṣḥaf tracking, the reference diff and scoring all stay in the local process;
only the acoustic model moves. Two things follow from putting the hop here rather than at
``WS /ws/session``:

* Only speech crosses the wire. StreamSession has already dropped the silence, so a
  tunnelled demo sends roughly what the reciter actually said, not the whole stream.
* The remote holds no session state. A dropped tunnel costs one chunk, not the session.

The wire format is a lossless projection of ``ChunkTranscript``: the phoneme string, the
per-CHARACTER probs the confidence grader slices, the groups, and all 10 ṣifāt per group
with their own probabilities. Dropping any of those is what the old HTTP handoff did, and
it silently killed confidence grading and articulation feedback (see session.py's
docstring). The round trip is covered by a test that asserts equality, not shape.

For demonstrations. There is no auth, no retry budget worth the name, and no attempt to
hide latency.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

import numpy as np
import torch

from ..config import Settings, get_settings
from .transcribe import ChunkTranscript

logger = logging.getLogger(__name__)

# The 10 ṣifāt, in the order everything else in this codebase uses.
_SIFA_ATTRS = (
    "hams_or_jahr",
    "shidda_or_rakhawa",
    "tafkheem_or_taqeeq",
    "itbaq",
    "safeer",
    "qalqla",
    "tikraar",
    "tafashie",
    "istitala",
    "ghonna",
)


class _WireUnit:
    """One ṣifā value. Duck-types quran_muaalem.Unit's .text/.prob."""

    __slots__ = ("text", "prob")

    def __init__(self, text: str, prob: float):
        self.text = text
        self.prob = prob


class _WireSifa:
    """Duck-types quran_muaalem.Sifa: .phonemes_group plus one attribute per ṣifā.

    Defined here rather than reusing engine.py's ``_FakeSifa`` to avoid an import cycle
    (engine.py builds this module's engine). The shim is four lines; the cycle is not.
    """

    def __init__(self, phonemes_group: str, attrs: dict, probs: dict):
        self.phonemes_group = phonemes_group
        for attr in _SIFA_ATTRS:
            value = attrs.get(attr)
            setattr(
                self,
                attr,
                _WireUnit(value, float(probs.get(attr, 0.0))) if value is not None else None,
            )


def transcript_to_wire(t: ChunkTranscript) -> dict:
    """ChunkTranscript -> JSON-able dict. Used by the remote server."""
    sifat = []
    for s in t.sifat:
        attrs: dict[str, str] = {}
        probs: dict[str, float] = {}
        for attr in _SIFA_ATTRS:
            unit = getattr(s, attr, None)
            if unit is not None and unit.text is not None:
                attrs[attr] = unit.text
                probs[attr] = float(unit.prob)
        sifat.append(
            {
                "phonemes_group": s.phonemes_group,
                "attrs": attrs,
                "probs": probs,
            }
        )
    return {
        "phonemes_text": t.phonemes_text,
        "char_probs": [float(p) for p in t.char_probs],
        "groups": list(t.groups),
        "group_probs": [float(p) for p in t.group_probs],
        "sifat": sifat,
    }


def wire_to_transcript(d: dict) -> ChunkTranscript:
    """The inverse. Used by RemoteAsrEngine."""
    return ChunkTranscript(
        phonemes_text=d.get("phonemes_text", ""),
        char_probs=[float(p) for p in d.get("char_probs", [])],
        groups=list(d.get("groups", [])),
        group_probs=[float(p) for p in d.get("group_probs", [])],
        sifat=[
            _WireSifa(s.get("phonemes_group", ""), s.get("attrs", {}), s.get("probs", {}))
            for s in d.get("sifat", [])
        ],
    )


def wave_to_pcm16(wave: torch.FloatTensor) -> bytes:
    """The wire audio format: 16 kHz mono PCM16 little-endian, same as the client sends.

    PCM16 rather than float32 halves the bytes over a tunnel, and costs nothing real: the
    audio arrived from the client as PCM16 in the first place, so this re-quantises data
    that was never finer than 16 bits.
    """
    samples = wave.detach().to(torch.float32).cpu().numpy()
    return (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def pcm16_to_wave(data: bytes) -> torch.FloatTensor:
    """The inverse, on the remote side."""
    if not data:
        return torch.zeros(0, dtype=torch.float32)
    arr = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    return torch.from_numpy(arr.copy())


_EMPTY = ChunkTranscript(
    phonemes_text="", char_probs=[], groups=[], group_probs=[], sifat=[]
)


class RemoteAsrEngine:
    """Sends each finalized chunk to a GPU elsewhere and returns what comes back.

    One persistent WebSocket, opened lazily and reopened once on failure. ``transcribe_chunk``
    is called from a worker thread (session.feed runs under ``asyncio.to_thread``), so this
    uses the synchronous websockets client rather than the async one.

    On an unrecoverable failure it logs and returns an EMPTY transcript, which the caller
    already treats as "nothing was said" and drops without emitting an event. That keeps a
    flaky tunnel from killing a live recitation. It does mean a persistently broken remote
    looks like a silent reciter, so the error is logged at ERROR, loudly, every time.
    """

    name = "remote"

    def __init__(self, url: str | None = None, settings: Settings | None = None):
        s = settings or get_settings()
        self.url = url or s.remote_url
        if not self.url:
            raise ValueError(
                "The remote engine needs a URL. Set TAJWID_REMOTE_URL to the tunnel's "
                "WebSocket endpoint, e.g. wss://<subdomain>.ngrok.app/infer"
            )
        self.timeout = s.remote_timeout_s
        self._ws: Optional[Any] = None

    # -- connection -------------------------------------------------------
    def _connect(self):
        from websockets.sync.client import connect

        logger.info("Connecting to the remote inference server at %s", self.url)
        return connect(self.url, open_timeout=self.timeout, max_size=None)

    def _ensure(self):
        if self._ws is None:
            self._ws = self._connect()
        return self._ws

    def close(self) -> None:
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:  # noqa: BLE001 — teardown, nothing to recover
                pass
            self._ws = None

    # -- the contract -----------------------------------------------------
    def transcribe_chunk(self, wave, sample_rate: int, ctx=None) -> ChunkTranscript:
        if sample_rate != 16000:
            raise ValueError(f"RemoteAsrEngine requires 16kHz audio, got {sample_rate}")

        payload = wave_to_pcm16(wave)

        # One retry, because the common failure by far is a tunnel that dropped an idle
        # connection between waqfs. A second failure is a real outage, not a stale socket.
        for attempt in (1, 2):
            try:
                ws = self._ensure()
                ws.send(payload)
                reply = ws.recv(timeout=self.timeout)
                return wire_to_transcript(json.loads(reply))
            except Exception as err:  # noqa: BLE001 — any transport/decode failure
                self.close()
                if attempt == 1:
                    logger.warning("Remote inference failed (%s); reconnecting.", err)
                    continue
                logger.error(
                    "Remote inference at %s failed twice: %s. Returning an empty "
                    "transcript for this chunk — the session continues but this audio "
                    "produced no feedback.",
                    self.url,
                    err,
                )
        return _EMPTY
