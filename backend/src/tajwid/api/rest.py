"""REST endpoints: health, offline file transcription, and cold analysis."""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

from typing import get_args

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile

from ..config import get_settings
from ..search.models import SearchResponse

router = APIRouter()


def _literal_values(annotation) -> list:
    """The allowed values of a ``Literal`` (or ``Optional[Literal]``) field, in order."""
    values: list = []
    for arg in get_args(annotation):
        if arg is type(None):
            continue
        inner = get_args(arg)  # a nested Literal inside Optional[...]
        values.extend(inner if inner else [arg])
    return values


# Declared by MoshafAttributes but read by NO code in either half of the pipeline, so
# offering them as controls would promise an effect that never arrives.
#   recitation_speed (سرعة التلاوة): the phonetizer never branches on it, and `moshaf`
#   does not reach the acoustic model at all -- it only feeds reference derivation. It is
#   a leftover from the quran_transcript DATASET schema, where it labelled style.
# Still ACCEPTED on the wire (see session.resolve_moshaf), so a stored client config that
# carries one keeps working; this only stops advertising it.
_INERT_FIELDS = frozenset({"recitation_speed"})


@router.get(
    "/moshaf-schema",
    tags=["recitation"],
    summary="Tajwid-style fields for the settings panel",
    responses={
        200: {
            "description": "One entry per adjustable field. Fields fixed to a single "
            "value (e.g. `rewaya`, always `hafs` in this app) are intentionally NOT "
            "listed here — see the note below.",
            "content": {
                "application/json": {
                    "example": {
                        "fields": [
                            {
                                "key": "recitation_speed",
                                "name_ar": "سرعة التلاوة",
                                "description": "The recitation speed sorted from "
                                "slowest to the fastest",
                                "default": "murattal",
                                "options": [
                                    {"value": "mujawad", "label": "مجود"},
                                    {"value": "above_murattal", "label": "فويق المرتل"},
                                    {"value": "murattal", "label": "مرتل"},
                                    {"value": "hadr", "label": "حدر"},
                                ],
                            },
                            {
                                "key": "madd_monfasel_len",
                                "name_ar": "مقدار المد المنفصل",
                                "description": "Separate madd length, in harakat.",
                                "default": 4,
                                "options": [
                                    {"value": 2, "label": "2"},
                                    {"value": 3, "label": "3"},
                                    {"value": 4, "label": "4"},
                                    {"value": 5, "label": "5"},
                                ],
                            },
                        ]
                    }
                }
            },
        }
    },
)
def moshaf_schema() -> dict:
    """The recitation (moshaf) attributes the reciter can set, for the settings panel.

    Introspected from ``MoshafAttributes`` so the panel is generated from the one source
    of truth: each field's Arabic name, its allowed values with Arabic labels, and a
    sensible starting value.

    **Fields with only one possible value are deliberately omitted** — there is nothing
    to choose (e.g. ``rewaya`` is fixed to ``"hafs"`` in this app). When building the
    ``moshaf`` object for ``WS /ws/session``'s start message, send back whatever subset
    of THESE fields the reciter changed; the server fills in every field this endpoint
    doesn't show (including the fixed ones) from its own defaults — see
    ``tajwid.session.resolve_moshaf``. You do not need to reconstruct the omitted fields
    yourself.
    """
    from pydantic_core import PydanticUndefined
    from quran_transcript import MoshafAttributes

    fields = []
    for name, f in MoshafAttributes.model_fields.items():
        if name in _INERT_FIELDS:
            continue
        values = _literal_values(f.annotation)
        if len(values) < 2:
            continue  # nothing to choose (e.g. rewaya is fixed to hafs)
        extra = f.json_schema_extra or {}
        amap = extra.get("field_arabic_attrs_map") or {}
        default = f.default if f.default not in (None, PydanticUndefined) else values[0]
        fields.append(
            {
                "key": name,
                "name_ar": extra.get("field_arabic_name", name),
                "description": f.description,
                "default": default,
                "options": [
                    {"value": v, "label": amap.get(str(v), amap.get(v, str(v)))}
                    for v in values
                ],
            }
        )
    return {"fields": fields}


@router.get(
    "/tajweed-rules",
    tags=["recitation"],
    summary="Every rule a session can ask to be graded on (leniency)",
    responses={
        200: {
            "description": "One entry per gradeable rule. `kind` says which channel the "
            "rule arrives on — `tajweed` findings carry it in `tajweed_rules[]`, `sifa` "
            "findings are articulation attributes — but a client filtering by these keys "
            "does not need to care: send the keys, the server does the rest.",
            "content": {
                "application/json": {
                    "example": {
                        "rules": [
                            {
                                "key": "aared_madd",
                                "name_ar": "المد العارض للسكون",
                                "name_en": "Aared Madd",
                                "kind": "tajweed",
                            },
                            {
                                "key": "ghonna",
                                "name_ar": "الغنة",
                                "name_en": "Ghonna",
                                "kind": "sifa",
                            },
                        ]
                    }
                }
            },
        }
    },
)
def tajweed_rules() -> dict:
    """The tajwid rules and sifat a reciter can choose to be graded on.

    Send the chosen subset as `rules` in `WS /ws/session`'s start message and findings
    for every other rule are dropped before they can colour a word. Omit `rules` (or
    send null) to grade everything — that is the default.

    **Hifz and tashkeel are never affected.** A wrong or missing word, and a wrong
    haraka, are reported whatever the selection: leniency narrows which TAJWID rules
    are enforced, not whether the recitation is checked.

    Derived from the rule classes upstream actually constructs plus the 10 sifa
    attributes, so this list cannot drift from what the grader recognises.

    ## The complete mapping

    Fetch this endpoint rather than hard-coding the table — it is reproduced here so a
    UI can be designed against it without a running server, not so it can be pasted
    into one. `key` is what you send; `name_ar` is what upstream calls the rule and is
    the label this app's own settings panel shows.

    | `key` | `name_ar` | `name_en` | `kind` |
    |---|---|---|---|
    | `normal_madd` | المد الطبيعي | Normal Madd | tajweed |
    | `monfasel_madd` | المد المنفصل | Monfasel Madd | tajweed |
    | `mottasel_madd` | المد المتصل | Mottasel Madd | tajweed |
    | `mottasel_madd_at_pause` | المد المتصل وقفا | Mottasel Madd at Pause | tajweed |
    | `lazem_madd` | المد اللازم | Lazem Madd | tajweed |
    | `aared_madd` | المد العارض للسكون | Aared Madd | tajweed |
    | `leen_madd` | مد اللين | Leen Madd | tajweed |
    | `qalqalah` | قلقة | Qalqalah | tajweed |
    | `hams_or_jahr` | الهمس والجهر | Hams Or Jahr | sifa |
    | `shidda_or_rakhawa` | الشدة والرخاوة | Shidda Or Rakhawa | sifa |
    | `tafkheem_or_taqeeq` | التفخيم والترقيق | Tafkheem Or Taqeeq | sifa |
    | `itbaq` | الإطباق | Itbaq | sifa |
    | `safeer` | الصفير | Safeer | sifa |
    | `qalqla` | القلقلة (صفة) | Qalqla | sifa |
    | `tikraar` | التكرار | Tikraar | sifa |
    | `tafashie` | التفشي | Tafashie | sifa |
    | `istitala` | الاستطالة | Istitala | sifa |
    | `ghonna` | الغنة | Ghonna | sifa |

    Three of these are worth reading twice before wiring up a settings panel:

    * **`ghonna` is a `sifa`, not a tajwid rule.** Ghunnah is the rule learners ask for
      by name, so this is the key most likely to be looked for in the wrong half of the
      list. Upstream defines a `Ghonnah` TajweedRule but never constructs one; in this
      pipeline ghunnah reaches the learner as the `ghonna` articulation attribute.
    * **`qalqalah` and `qalqla` are different keys for the same phenomenon** on the two
      different channels — the tajwid rule and the sifa. Offer both, or neither; the
      Arabic name carries a "(صفة)" suffix on the sifa one purely so a UI showing both
      does not render two identical chips.
    * **`name_ar` for `qalqalah` reads قلقة, not قلقلة.** That is upstream's spelling,
      passed through unaltered rather than silently corrected here, so that this
      service and the grader always agree on what a rule is called. Override it in the
      UI if you prefer, but key off `key`.
    """
    from ..feedback.rules import catalogue

    return {"rules": catalogue()}


@router.get(
    "/health",
    tags=["health"],
    summary="Service status + which ASR engine(s) loaded",
    responses={
        200: {
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "engine": "mock",
                        "available_engines": ["mock"],
                        "device": "cpu",
                        "dtype": "bfloat16",
                        "muaalem_model": "obadx/muaalem-model-v3_2",
                        "segmenter_model": "obadx/recitation-segmenter-v2",
                    }
                }
            }
        }
    },
)
def health(request: Request) -> dict:
    """Which engine is the server DEFAULT, and which engines it actually built.

    `available_engines` is what `WS /ws/session`'s per-session `"engine"` field may
    request — anything else silently falls back to `engine`. On a GPU box with the real
    model, `engine` is `"real"` and `muaalem_device`/`segmenter_device` are also present
    (omitted here — this example is from a GPU-less dev machine running `mock`).
    `zipformer` is absent from `available_engines` whenever its model files aren't
    staged under `models/asr_zipformer/` (see the README's Prerequisites section) — the
    service still starts fine without them.
    """
    s = get_settings()
    engines = request.app.state.engines
    default_name = request.app.state.default_engine_name
    engine = engines[default_name]
    info = {
        "status": "healthy",
        "engine": getattr(engine, "name", "unknown"),
        "available_engines": sorted(engines.keys()),
        # Whether this server was started with a capture directory. The Diagnose
        # toggle greys itself out when this is false, so the reciter is never offered
        # a recording the server would silently drop.
        "capture_available": bool(s.capture_dir),
        "device": s.device,
        "dtype": s.dtype_str,
        "muaalem_model": s.muaalem_model_id,
        "segmenter_model": s.segmenter_model_id,
    }
    if getattr(engine, "name", "") == "real":
        bundle = engine.bundle
        info["muaalem_device"] = str(bundle.muaalem.device)
        info["segmenter_device"] = str(bundle.segmenter_device)
    return info


@router.get(
    "/search",
    tags=["search"],
    response_model=SearchResponse,
    summary="Find āyāt by wording, meaning, or both",
    responses={
        200: {
            "description": "Real capture: `?q=الرحمن الرحيم&mode=keyword&limit=2`. Note "
            "the fields that report what ACTUALLY ran — see the description below.",
            "content": {
                "application/json": {
                    "example": {
                        "hits": [
                            {
                                "sura": 1,
                                "aya": 3,
                                "text_uthmani": "ٱلرَّحْمَٰنِ ٱلرَّحِيمِ",
                                "translation": "The Entirely Merciful, the Especially Merciful,",
                                "score": 5.762516498565674,
                            },
                            {
                                "sura": 41,
                                "aya": 2,
                                "text_uthmani": "تَنزِيلٌ مِّنَ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ",
                                "translation": "[This is] a revelation from the Entirely "
                                "Merciful, the Especially Merciful -",
                                "score": 5.418358325958252,
                            },
                        ],
                        "matched_lang": "ar",
                        "mode": "keyword",
                        "hyde_used": False,
                    }
                }
            },
        },
        422: {
            "description": "A blank query, or an unrecognised `mode` (real captures).",
            "content": {
                "application/json": {
                    "examples": {
                        "blank query": {
                            "value": {"detail": "q must not be empty or whitespace"}
                        },
                        "bad mode": {
                            "value": {
                                "detail": [
                                    {
                                        "type": "string_pattern_mismatch",
                                        "loc": ["query", "mode"],
                                        "msg": "String should match pattern "
                                        "'^(keyword|vector|hybrid)$'",
                                        "input": "bogus",
                                    }
                                ]
                            }
                        },
                    }
                }
            },
        },
    },
)
async def search_ayahs(
    q: str = Query(..., description="Query text; Arabic or English"),
    mode: str | None = Query(
        None,
        pattern="^(keyword|vector|hybrid)$",
        description="keyword=BM25, vector=embeddings, hybrid=both (default, best)",
    ),
    hyde: bool | None = Query(
        None, description="LLM query expansion before embedding; ignored for mode=keyword"
    ),
    lang: str | None = Query(None, description="ar/en; omit to detect from the script"),
    alpha: float | None = Query(
        None, ge=0, le=2, description="Lexical weight in hybrid; omit for the tuned default"
    ),
    limit: int = Query(20, ge=1, le=50),
) -> SearchResponse:
    """Find āyāt by wording, by meaning, or both.

    ``mode`` and ``hyde`` are deliberately separate knobs rather than one enum: mode says
    where the score comes from, hyde says whether the query is rewritten before embedding.
    Every combination is legal, so there is nothing to add when a third retrieval mode or a
    second expander shows up. See ``search/service.py``.

    The response reports the mode, language and hyde flag that ACTUALLY ran.

    We do not log ``q``: a query log is user data, and this service holds none.

    Runs in a thread: the first vector call loads BGE-M3 (~10 s, ~2 GB), every call after
    scans the whole index, and HyDE waits on a network round trip — none of that belongs on
    the event loop that is also carrying live recitation WebSockets.
    """
    from ..search.service import search

    if not q.strip():
        raise HTTPException(422, "q must not be empty or whitespace")
    r = await asyncio.to_thread(search, q, lang, limit, mode, hyde, alpha)
    return SearchResponse(
        hits=r.hits, matched_lang=r.matched_lang, mode=r.mode, hyde_used=r.hyde_used
    )


@router.post(
    "/transcribe-file",
    tags=["recitation"],
    summary="Offline: upload a full recitation, get per-chunk transcripts",
    responses={
        200: {
            "description": "**Illustrative** (schema-accurate, hand-built — this "
            "endpoint is real-engine only and needs a GPU + downloaded models, so it "
            "has no mock-engine capture like the other endpoints on this page). One "
            "entry in `chunks` per waqf the W2V-BERT segmenter found; `units` is one "
            "phoneme GROUP per entry, each with its 10 predicted sifat.",
            "content": {
                "application/json": {
                    "example": {
                        "chunks": [
                            {
                                "session_id": "my_recitation",
                                "chunk_seq": 0,
                                "is_final": False,
                                "audio_span_sec": [0.0, 3.42],
                                "predicted_phonemes": "بِسمِللَااهِررَحمَاانِررَحِۦۦم",
                                "units": [
                                    {
                                        "phonemes_group": "بِ",
                                        "prob": 0.98,
                                        "sifat": {
                                            "hams_or_jahr": {"text": "jahr", "prob": 0.97},
                                            "shidda_or_rakhawa": {"text": "shadeed", "prob": 0.95},
                                            "qalqla": {"text": "qalqla", "prob": 0.91},
                                            "ghonna": None,
                                        },
                                    }
                                ],
                            }
                        ]
                    }
                }
            },
        }
    },
)
async def transcribe_file_endpoint(file: UploadFile = File(...)) -> dict:
    """Batch: transcribe a complete uploaded recitation (any ffmpeg-decodable format).

    **Real-engine only** — the W2V-BERT segmenter is the chunker here, and this endpoint
    does not fall back to `mock`/`zipformer` regardless of `TAJWID_ASR_ENGINE`. Needs a
    GPU with the Muaalem + segmenter models loaded (see README's GPU / CUDA section).

    Unlike `WS /ws/session`, this returns the RAW per-chunk model output (phonemes +
    sifat, no diff/scoring against a reference) — the same `ChunkResult` shape
    `include_units=true` on the WebSocket also exposes per live chunk. There is no
    tracking, diffing or word-level correct/error grading here; pair this with the
    feedback pipeline yourself if you need that offline.
    """
    from ..asr.batch import transcribe_file

    audio_bytes = await file.read()
    filename = file.filename or "upload"

    def _run():
        suffix = Path(filename).suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        try:
            return transcribe_file(tmp_path, session_id=filename)
        finally:
            os.unlink(tmp_path)

    results = await asyncio.to_thread(_run)
    return {"chunks": [r.model_dump() for r in results]}
