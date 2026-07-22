r"""# Tajwid recitation feedback service

Streaming Qur'an recitation feedback: microphone audio in, per-word feedback out.

    audio -> silero VAD endpointing -> Muaalem phonemes + 10 sifat -> locate/track the
    passage -> diff vs the reference -> per-word feedback, over one WebSocket.

This page is generated from the OpenAPI schema below, which only covers **REST**
endpoints. **The live recitation endpoint, `WS /ws/session`, is a WebSocket — OpenAPI
has no operation type for those, so it can never appear as a route on this page, in any
FastAPI service.** This docstring is that endpoint's documentation; the REST endpoints
below (`/health`, `/moshaf-schema`, `/search`, `/transcribe-file`) are documented as
normal via their own tags and examples.

## `WS /ws/session` — live recitation

Connect, then:

**1. First message (JSON, required)** — the start config. The socket closes with code
1002 if the first thing it receives is not JSON, so this always comes before any audio:

```json
{"type": "start", "sura": 1, "aya": 1, "word_idx": 0,
 "strictness": "normal", "moshaf": {"madd_monfasel_len": 4, "...": "..."},
 "engine": "zipformer", "rules": ["aared_madd", "monfasel_madd", "ghonna"]}
```

**Every field is optional, `sura`/`aya`/`word_idx` included.** `moshaf` is the
reciter's tajwid style — get its full schema from `GET /moshaf-schema`; omit it for the
server default. `engine` selects a different ASR engine than the server default for
THIS SESSION ONLY (see `GET /health` for which engines this server actually built; an
unbuilt or unknown name falls back to the default rather than erroring). `rules` is
leniency — see below.

### `strictness` — how sure the model must be before it accuses

One of `"lenient"`, `"normal"` (default), `"strict"`. Anything else, including a
mis-cased `"Normal"`, falls back to the default rather than failing the session.

It does NOT change what counts as a mistake — the diff finds exactly the same errors at
every level. It sets the model-confidence threshold at which a finding is reported as a
hard `error` instead of being softened to `almost`. **A lower threshold is a harsher
teacher.** Each level is a `(phoneme, sifa)` pair, because the two probabilities come
from different heads of the model and are not assumed to share a calibration:

| level | phoneme | sifa |
|---|---|---|
| `lenient` | 0.90 | 0.95 |
| `normal` | 0.70 | 0.85 |
| `strict` | 0.50 | 0.65 |

So one finding the model is 0.75 sure of grades `almost` on `lenient` and `error` on
`normal`/`strict`; at 0.55 it is `error` only on `strict`.

No setting turns a guess into an accusation: a finding with **unknown** confidence
(`confidence: null` — e.g. a pure deletion, where the reciter said nothing there is no
probability to read) grades `almost` at every level, `strict` included.

> **These thresholds are uncalibrated placeholders.** They are hand-picked guesses
> awaiting calibration against a labelled set, and are marked as such in
> `feedback/confidence.py`. Treat the ORDERING as meaningful and the absolute numbers
> as provisional; do not derive a user-facing accuracy claim from them.

### `sura`/`aya`/`word_idx` — optional, but send them if you have them

They SEED THE CURSOR. Sending them is not required, and a session without them is not
broken — it is just working harder and less certainly:

- **With a position**, each chunk is matched by `track`: a search in a window around
  where the reciter already is. Cheap, and structurally immune to mutashabihat.
- **Without one**, the first chunk is matched by `locate`: a fuzzy search over the
  whole Qur'an. If the passage is distinctive that returns `status: "ok"` and the
  cursor SELF-SEEDS from it, so every later chunk tracks normally — one cold search,
  then business as usual.

The cost of omitting it is confined to passages that genuinely occur in more than one
place, and there the answer is `ambiguous` with a candidate list, not an error:

```json
{"status": "ambiguous",
 "candidates": [{"sura": 1, "aya": 1, "uthmani_text": "بِسْمِ ٱللَّهِ ..."},
                {"sura": 27, "aya": 30, "uthmani_text": "..."}]}
```

That case is worth designing for rather than dismissing, because the single most
likely thing a reciter opens with — the basmalah — is exactly it, matching both
Al-Fatiha 1:1 and An-Naml 27:30. A client that already knows what the learner picked
(this app's muṣḥaf view always does) should send the position and skip the problem
entirely. A client that does NOT know — a "just start reciting, find me" mode — can
legitimately omit it, show the candidates, and let the next chunk disambiguate.

`ambiguous` and `no_match` assert NOTHING against the reciter: no words, no errors.
Scoring someone against a verse they were not reciting is the worst thing this service
can do, so it declines instead of guessing.

**Testing note:** the `mock` engine fabricates its phonemes FROM the cursor, so it
emits nothing at all when there isn't one. A cold, position-less session therefore
looks silent on `mock` — that is the mock's limitation, not the pipeline's. Use
`real`/`zipformer` to exercise the cold-start path.

### `rules` — grade only what the learner is working on

A learner drilling madd al-aared does not want to be corrected on qalqalah. `rules`
names the tajwid rules and sifat to grade; findings for every other rule are dropped
before they can mark a word, so `status` and `errors[]` agree — a filtered rule leaves
no red word behind it.

Keys come from `GET /tajweed-rules` (8 tajwid rules + the 10 sifat) — **that endpoint's
description on this page carries the complete `key` -> Arabic/English mapping**, so a
settings panel can be built against it without calling the server. Note ghunnah is
`ghonna`, a SIFA key, not a tajwid rule.

| `rules` | Meaning |
|---|---|
| omitted, or `null` | Grade everything. The default, and what a client that predates this feature sends. |
| `["aared_madd", "ghonna"]` | Grade these; stay silent on every other tajwid rule. |
| `[]` | A real choice, not a missing one: no tajwid rule at all, hifz and tashkeel only. |

**Hifz and tashkeel are never filtered.** A wrong or missing word (`error_type:
"normal"`) and a wrong haraka (`"tashkeel"`) are reported whatever the selection —
leniency narrows which tajwid rules are enforced, not whether the recitation is
checked. A `tajweed` finding that carries an empty `tajweed_rules[]` (the reciter
skipped a rule-bearing letter outright) is also never filtered: there is no rule to
match it against, and hiding it would bury a real miss under a filter set for
something else.

Unknown keys are not an error; they simply match nothing. Sending only unknown keys
therefore grades no tajwid rule at all, exactly as `[]` does.

**2. Server replies** with the session ack (real capture, mock engine):

```json
{"type": "session", "session_id": "a35c7c65-9b4f-4814-bbd2-3bc04a01e12d",
 "engine": "mock", "sample_rate": 16000}
```

`engine` is the engine that ACTUALLY got used — compare it to what you requested; they
differ silently if the requested one wasn't built on this server.

**3. Stream binary frames** — 16 kHz mono PCM16 little-endian audio, any chunk size.

**4. Server pushes one `feedback` event per finalized waqf chunk** (whenever the
reciter pauses), unprompted. A correct chunk (real capture):

```json
{
  "type": "feedback", "chunk_seq": 0, "audio_span_sec": [0.168, 6.0], "forced_cut": false,
  "phonemes": "بِسمِللَااهِررَحمَاانِررَحِۦۦۦۦم",
  "feedback": {
    "status": "ok",
    "span": {"sura": 1, "aya": 1, "word_idx": 0},
    "end": {"sura": 1, "aya": 1, "word_idx": 3},
    "uthmani_text": "بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ",
    "words": [
      {"sura": 1, "aya": 1, "word_idx": 0, "uthmani": "بِسْمِ", "status": "correct", "errors": [], "trimmed": false},
      {"sura": 1, "aya": 1, "word_idx": 1, "uthmani": "ٱللَّهِ", "status": "correct", "errors": [], "trimmed": false},
      {"sura": 1, "aya": 1, "word_idx": 2, "uthmani": "ٱلرَّحْمَٰنِ", "status": "correct", "errors": [], "trimmed": false},
      {"sura": 1, "aya": 1, "word_idx": 3, "uthmani": "ٱلرَّحِيمِ", "status": "correct", "errors": [], "trimmed": false}
    ]
  },
  "cursor": {"sura": 1, "aya": 1, "word_idx": 3}
}
```

A word with an actual mistake looks like this instead (real capture — the reciter held
a 2-count madd for 3 counts on the second word of "ar-Rahman"):

```json
{
  "sura": 1, "aya": 1, "word_idx": 2, "uthmani": "ٱلرَّحْمَٰنِ", "status": "error", "trimmed": false,
  "errors": [{
    "error_type": "tajweed", "speech_error_type": "replace",
    "expected_ph": "اا", "predicted_ph": "ااا",
    "expected_len": 2, "predicted_len": 3,
    "tajweed_rules": [{"name_ar": "المد الطبيعي", "name_en": "Normal Madd",
                         "golden_len": 2, "correctness_type": "count", "tag": "alif"}],
    "confidence": 0.97
  }]
}
```

That finding is a `Normal Madd`, so a session started with `"rules": ["aared_madd"]`
would not receive it at all and the word would come back `correct` — not `error` with
an empty `errors[]`.

Three statuses, and what a client MUST do with each (see README's WebSocket section for
the full rationale):

- **`correct`** — no issue.
- **`almost`** — the model wasn't confident enough to accuse. Render as a HINT, never a
  mistake, never counted against the score. Falsely correcting a perfect recitation is
  the one thing this system must not do.
- **`error`** — a confident, real mistake; `errors[]` says what and why.

Independent of status: **`trimmed: true`** means the word sat on a chunk boundary and
was NOT scored at all — render it neutrally, never as a checkmark.

**5. Reposition mid-session** (the reciter jumped elsewhere):

```json
{"type": "seek", "sura": 2, "aya": 255, "word_idx": 0}
```

**6. End the stream:**

```json
{"type": "end"}
```

The server flushes any in-progress utterance (one last `feedback` event, maybe), then:

```json
{"type": "done"}
```

and closes the socket.

**Try it from a terminal**, no frontend needed, with any WebSocket client, e.g.
[websocat](https://github.com/vi/websocat):

```bash
websocat ws://localhost:8100/ws/session
> {"type":"start","sura":1,"aya":1,"word_idx":0}
< {"type":"session","session_id":"...","engine":"mock","sample_rate":16000}
> {"type":"end"}
< {"type":"done"}
```

(Send real 16 kHz mono PCM16-LE audio bytes instead of ending immediately to see
`feedback` events; `TAJWID_ASR_ENGINE=mock` needs no GPU or model download.)
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.rest import router as rest_router
from .api.ws import router as ws_router
from .asr.engine import build_engines
from .config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build every eagerly-available engine once, in a single thread, before
    # serving — same reasoning as before (avoid a concurrent-first-call race
    # triggering two simultaneous GPU model loads), just now producing a
    # dict instead of one engine so a session can pick per-connection (see
    # api/ws.py). app.state.default_engine_name is what an omitted/unknown
    # per-session choice falls back to.
    settings = get_settings()
    app.state.engines = await asyncio.to_thread(build_engines, settings)
    app.state.default_engine_name = settings.resolved_asr_engine
    yield


_OPENAPI_TAGS = [
    {
        "name": "health",
        "description": "Is the service up, and which ASR engine did it load.",
    },
    {
        "name": "recitation",
        "description": (
            "The tajwid-style (moshaf) schema for the settings panel, the rules a "
            "session can be graded on (leniency), and offline whole-file "
            "transcription. **The live recitation path is `WS /ws/session` "
            "— it cannot be listed here (OpenAPI has no WebSocket operation type); see "
            "this page's top-level description for its full protocol and examples.**"
        ),
    },
    {
        "name": "search",
        "description": (
            "Find ayat by wording (`keyword`), meaning (`vector`), or both "
            "(`hybrid`, default), optionally with LLM query expansion (`hyde`)."
        ),
    },
]


def create_app() -> FastAPI:
    app = FastAPI(
        title="Tajwid recitation feedback service",
        description=__doc__,
        version="1.0.0",
        lifespan=lifespan,
        openapi_tags=_OPENAPI_TAGS,
    )
    # The Java Spring backend / dev frontend call us from another origin (VPC-internal
    # in production). Tighten with an env-based allowlist when the topology is known.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(rest_router)
    app.include_router(ws_router)

    @app.get(
        "/",
        tags=["health"],
        summary="Service + docs pointer",
        responses={
            200: {
                "content": {
                    "application/json": {
                        "example": {
                            "service": "Tajwid recitation feedback",
                            "engine": "mock",
                            "docs": "/docs",
                            "endpoints": {
                                "GET /health": "status + loaded engine",
                                "GET /moshaf-schema": "the tajwid-style fields for the "
                                "settings panel",
                                "GET /tajweed-rules": "the rules a session can be graded "
                                "on; send a subset as `rules` in the start message to be "
                                "corrected on those only",
                                "GET /search": "find āyāt by wording (keyword), meaning "
                                "(vector), or both (hybrid)",
                                "POST /transcribe-file": "offline: upload a recitation, "
                                "get chunk transcripts",
                                "WS /ws/session": "live: JSON start config, then 16 kHz "
                                "mono PCM16-LE frames; per-waqf-chunk word feedback "
                                "comes back — see /docs's top-level description for the "
                                "full protocol (OpenAPI can't list WebSocket routes)",
                            },
                        }
                    }
                }
            }
        },
    )
    def index() -> dict:
        s = get_settings()
        return {
            "service": "Tajwid recitation feedback",
            "engine": s.resolved_asr_engine,
            "docs": "/docs",
            "endpoints": {
                "GET /health": "status + loaded engine",
                "GET /moshaf-schema": "the tajwid-style fields for the settings panel",
                "GET /tajweed-rules": "the rules a session can be graded on; send a "
                "subset as `rules` in the start message to be corrected on those only",
                "GET /search": "find āyāt by wording (keyword), meaning (vector), or both (hybrid)",
                "POST /transcribe-file": "offline: upload a recitation, get chunk transcripts",
                "WS /ws/session": "live: JSON start config, then 16 kHz mono PCM16-LE "
                "frames; per-waqf-chunk word feedback comes back — see /docs's top-level "
                "description for the full protocol (OpenAPI can't list WebSocket routes)",
            },
        }

    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8100)


if __name__ == "__main__":
    main()
