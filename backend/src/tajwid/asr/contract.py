"""Handoff contract: the JSON one finalized chunk produces for the alignment peer.

Per chunk we emit the model's *raw predicted* phonetic script + tajweed sifat, with
confidences and the absolute audio time span. The peer concatenates ``predicted_phonemes``
across chunks, runs phonetic search to locate the passage, phonetizes the reference and
calls ``explain_error`` -- none of which happens here.

The ``units`` granularity is one entry per phoneme *group* (a base letter plus its
diacritics / madd repeats, e.g. ``ننننَ``), which is exactly the granularity of
``quran_muaalem`` sifat (one ``Sifa`` per group).
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

# The 10 sifat (tajweed feature) levels emitted per phoneme group, in a stable order.
SIFAT_LEVELS: tuple[str, ...] = (
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


class SifatFeature(BaseModel):
    """One predicted tajweed feature for a phoneme group."""

    text: str = Field(..., description="Categorical label, e.g. 'jahr', 'shadeed'.")
    prob: float = Field(..., description="Model confidence for this feature in [0, 1].")


class UnitResult(BaseModel):
    """One phoneme group with its phoneme confidence and predicted sifat."""

    phonemes_group: str = Field(..., description="The phoneme group text, e.g. 'ءَ'.")
    prob: float = Field(
        ..., description="Mean phoneme (CTC) confidence for this group in [0, 1]."
    )
    sifat: dict[str, Optional[SifatFeature]] = Field(
        default_factory=dict,
        description="Predicted sifat keyed by level (see SIFAT_LEVELS); null if the "
        "model produced no feature for that level on this group.",
    )


class ChunkResult(BaseModel):
    """The full per-chunk message handed to the alignment peer."""

    session_id: str
    chunk_seq: int = Field(..., description="0-based index of this chunk in the stream.")
    is_final: bool = Field(
        False, description="True on the last chunk emitted for the stream."
    )
    audio_span_sec: tuple[float, float] = Field(
        ..., description="[start, end] of this chunk in seconds, absolute in the stream."
    )
    predicted_phonemes: str = Field(
        ..., description="Concatenatable predicted phonetic script for this chunk."
    )
    units: list[UnitResult] = Field(default_factory=list)

    @classmethod
    def from_transcript(
        cls,
        transcript: "ChunkTranscript",
        *,
        session_id: str,
        chunk_seq: int,
        audio_span_sec: tuple[float, float],
        is_final: bool = False,
    ) -> "ChunkResult":
        units: list[UnitResult] = []
        for group, group_prob, sifa in zip(
            transcript.groups, transcript.group_probs, transcript.sifat
        ):
            sifat: dict[str, Optional[SifatFeature]] = {}
            for level in SIFAT_LEVELS:
                single = getattr(sifa, level, None)
                sifat[level] = (
                    SifatFeature(text=single.text, prob=float(single.prob))
                    if single is not None
                    else None
                )
            units.append(
                UnitResult(
                    phonemes_group=group,
                    prob=float(group_prob),
                    sifat=sifat,
                )
            )
        return cls(
            session_id=session_id,
            chunk_seq=chunk_seq,
            is_final=is_final,
            audio_span_sec=audio_span_sec,
            predicted_phonemes=transcript.phonemes_text,
            units=units,
        )
