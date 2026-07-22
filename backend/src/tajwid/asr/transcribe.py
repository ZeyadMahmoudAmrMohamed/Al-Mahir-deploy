"""Reference-free Muaalem transcription.

The stock ``quran_muaalem.Muaalem.__call__`` requires a per-item reference phonetic
script, but only uses it to *fix up* the sifat decoding when a decoded sifat level has a
different length than the predicted phoneme groups. We are in the "free / target unknown"
setting, so no reference exists yet -- reference alignment is the peer's job.

This module reproduces the reference-free part of the pipeline:

* phonemes are decoded straight from the audio (already reference-free upstream via
  ``phonemes_level_greedy_decode``);
* every sifat level is CTC-decoded independently and *index-aligned* against the predicted
  phoneme groups (``inference.format_sifat`` anchors on the phoneme groups and fills
  ``None`` where a level is shorter). On the common no-length-mismatch path this is
  bit-identical to the stock decode; on a mismatch we skip the reference-based realign and
  fall back to positional alignment, leaving the true alignment to the peer.

Everything is reused from ``quran_muaalem`` -- no upstream file is modified.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from numpy.typing import NDArray

from quran_transcript import chunck_phonemes
from quran_muaalem.decode import ctc_decode, phonemes_level_greedy_decode
from quran_muaalem.inference import Muaalem, format_sifat
from quran_muaalem.modeling.vocab import PAD_TOKEN_IDX
from quran_muaalem.muaalem_typing import Sifa, Unit


def _strip_pad(unit: Unit, id_to_phoneme: dict[int, str]) -> Unit:
    """Drop PAD (blank) tokens from a decoded phonemes unit.

    ``ctc_decode`` can leave a trailing blank id (documented upstream bug); the stock engine
    filters ``idx != 0`` before rendering. We do the same so ``predicted_phonemes`` never
    contains the literal ``[PAD]`` marker.
    """
    ids = [int(i) for i in unit.ids]
    probs = [float(p) for p in unit.probs]
    kept = [(i, p) for i, p in zip(ids, probs) if i != PAD_TOKEN_IDX]
    new_ids = [i for i, _ in kept]
    new_probs = [p for _, p in kept]
    text = "".join(id_to_phoneme[i] for i in new_ids)
    return Unit(
        text=text,
        probs=torch.tensor(new_probs, dtype=torch.float32),
        ids=torch.tensor(new_ids, dtype=torch.long),
    )


@dataclass
class ChunkTranscript:
    """Reference-free transcription of a single audio chunk.

    Attributes:
        phonemes_text: Full predicted phonetic script (concatenatable across chunks).
        char_probs: One phoneme (CTC) confidence per CHARACTER of ``phonemes_text``.
            The feedback half slices this array with character offsets, so the 1:1
            alignment is a hard contract (see feedback.adapt).
        groups: Phoneme groups (base letter + its diacritics/madd), one per sifat entry.
        group_probs: Mean phoneme (CTC) confidence per group, in [0, 1].
        sifat: One ``Sifa`` per group (all 10 tajweed levels).
    """

    phonemes_text: str
    char_probs: list[float]
    groups: list[str]
    group_probs: list[float]
    sifat: list[Sifa]


def _group_probs(
    id_to_phoneme: dict[int, str],
    phoneme_ids: list[int] | torch.Tensor,
    phoneme_probs: list[float] | torch.Tensor,
    groups: list[str],
) -> list[float]:
    """Aggregate per-token phoneme probs into a mean prob per phoneme group.

    ``groups`` is a segmentation of the concatenated phoneme string, so each group is an
    exact concatenation of consecutive decoded token strings. We walk the tokens and assign
    their probs to the group they complete. Robust to multi-token groups (madd, ghonna).
    """
    token_strs = [id_to_phoneme[int(i)] for i in phoneme_ids]
    token_probs = [float(p) for p in phoneme_probs]

    result: list[float] = []
    tok_idx = 0
    for group in groups:
        acc = ""
        probs: list[float] = []
        # Consume tokens until their concatenation matches this group.
        while tok_idx < len(token_strs) and acc != group:
            acc += token_strs[tok_idx]
            probs.append(token_probs[tok_idx])
            tok_idx += 1
        result.append(sum(probs) / len(probs) if probs else 0.0)
    return result


@torch.no_grad()
def transcribe_reference_free(
    muaalem: Muaalem,
    waves: list[list[float] | torch.FloatTensor | NDArray],
    sampling_rate: int,
) -> list[ChunkTranscript]:
    """Run the Muaalem model on a batch of waves and decode phonemes + sifat, no reference.

    Args:
        muaalem: An initialized ``quran_muaalem.Muaalem`` (we use its model / processor /
            tokenizer; we do not call ``muaalem(...)`` which would demand a reference).
        waves: Batch of 16 kHz mono waveforms.
        sampling_rate: Must be 16000.

    Returns:
        One ``ChunkTranscript`` per input wave.
    """
    if sampling_rate != 16000:
        raise ValueError(f"`sampling_rate` has to be 16000 got: `{sampling_rate}`")

    tokenizer = muaalem.multi_level_tokenizer

    features = muaalem.processor(waves, sampling_rate=sampling_rate, return_tensors="pt")
    features = {
        k: v.to(muaalem.device, dtype=muaalem.dtype) for k, v in features.items()
    }
    level_to_logits = muaalem.model(**features, return_dict=False)[0]

    probs: dict[str, torch.Tensor] = {}
    for level in level_to_logits:
        probs[level] = (
            torch.nn.functional.softmax(level_to_logits[level], dim=-1)
            .cpu()
            .to(torch.float32)
        )

    # Phonemes: reference-free CTC decode (per-token text + probs), PAD-stripped.
    id_to_phoneme = tokenizer.id_to_vocab["phonemes"]
    phonemes_units: list[Unit] = [
        _strip_pad(u, id_to_phoneme)
        for u in phonemes_level_greedy_decode(probs["phonemes"], id_to_phoneme)
    ]

    chunked_phonemes_batch: list[list[str]] = [
        chunck_phonemes(u.text) for u in phonemes_units
    ]

    # Sifat: CTC-decode each level independently (no reference realignment).
    level_to_units: dict[str, list[Unit]] = {"phonemes": phonemes_units}
    for level, level_probs in probs.items():
        if level == "phonemes":
            continue
        batch_probs, batch_ids = level_probs.topk(1, dim=-1)
        decode_outs = ctc_decode(
            batch_ids.squeeze(-1), batch_probs.squeeze(-1), collapse_consecutive=True
        )
        units: list[Unit] = []
        for decode_out in decode_outs:
            text = "".join(
                tokenizer.id_to_vocab[level][int(i)] for i in decode_out.ids
            )
            units.append(Unit(text=text, probs=decode_out.p, ids=decode_out.ids))
        level_to_units[level] = units

    # Anchor sifat on the predicted phoneme groups (fills None on short levels).
    sifat_batch: list[list[Sifa]] = format_sifat(
        level_to_units, chunked_phonemes_batch, tokenizer
    )

    transcripts: list[ChunkTranscript] = []
    for seq_idx, ph_unit in enumerate(phonemes_units):
        groups = chunked_phonemes_batch[seq_idx]
        group_probs = _group_probs(id_to_phoneme, ph_unit.ids, ph_unit.probs, groups)
        transcripts.append(
            ChunkTranscript(
                phonemes_text=ph_unit.text,
                char_probs=_char_probs(id_to_phoneme, ph_unit),
                groups=groups,
                group_probs=group_probs,
                sifat=sifat_batch[seq_idx],
            )
        )
    return transcripts


def _char_probs(id_to_phoneme: dict[int, str], unit: Unit) -> list[float]:
    """Expand per-token CTC probs to one prob per CHARACTER of ``unit.text``.

    A decoded token is usually a single phoneme character, in which case this is the
    identity; a multi-character token contributes its probability to each of its
    characters. The feedback half slices this array by character offset, so the
    result must satisfy ``len(char_probs) == len(unit.text)`` exactly.
    """
    probs: list[float] = []
    for token_id, prob in zip(unit.ids, unit.probs):
        probs.extend([float(prob)] * len(id_to_phoneme[int(token_id)]))
    if len(probs) != len(unit.text):  # belt and braces: never hand over a misaligned array
        raise RuntimeError(
            f"per-char probs ({len(probs)}) misaligned with phonemes ({len(unit.text)})"
        )
    return probs
