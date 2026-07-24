# API reference

The complete client-facing contract for the Al-Mahir AI service: the live recitation
WebSocket, the two ASR models a session can choose between, and āyah search.

Written for the Android and iOS teams. Nothing here assumes you have read the Python.

For getting a server running, see [SETUP.md](SETUP.md).

Contents:

1. [Base URL and transport](#1-base-url-and-transport)
2. [Endpoint summary](#2-endpoint-summary)
3. [GET /health, and discovering which models exist](#3-get-health-and-discovering-which-models-exist)
4. [The two models: Muaalem and zipformer](#4-the-two-models-muaalem-and-zipformer)
5. [WS /ws/session: live recitation](#5-ws-wssession-live-recitation)
   - [5.1 Handshake](#51-handshake)
   - [5.2 The start message](#52-the-start-message)
   - [5.3 Audio format](#53-audio-format)
   - [5.4 The feedback event](#54-the-feedback-event)
   - [5.5 Word status, and the three rules you must follow](#55-word-status-and-the-three-rules-you-must-follow)
   - [5.6 The error object](#56-the-error-object)
   - [5.7 Seeking](#57-seeking)
   - [5.8 Ending](#58-ending)
   - [5.9 Failure and edge cases](#59-failure-and-edge-cases)
6. [GET /moshaf-schema: the recitation settings panel](#6-get-moshaf-schema-the-recitation-settings-panel)
7. [GET /tajweed-rules: leniency](#7-get-tajweed-rules-leniency)
8. [GET /search: semantic and keyword āyah search](#8-get-search-semantic-and-keyword-āyah-search)
9. [POST /transcribe-file: offline transcription](#9-post-transcribe-file-offline-transcription)
10. [Client implementation notes](#10-client-implementation-notes)

---

## 1. Base URL and transport

```
http://<host>:8100
ws://<host>:8100
```

Mobile clients connect directly to the Python service. Make the base URL configurable in
the app rather than compiling it in; a gateway may later sit in front and add a path
prefix.

There is no authentication today. CORS is wide open (`allow_origins=["*"]`), which is fine
inside a VPC and must be tightened before any public exposure.

All request and response bodies are JSON, UTF-8. Arabic is returned unescaped.

The service holds no user data and does not log search queries or audio.

---

## 2. Endpoint summary

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Service identity and an endpoint list |
| GET | `/health` | Status, the default engine, and which engines exist |
| GET | `/moshaf-schema` | Fields for the recitation settings panel |
| GET | `/tajweed-rules` | Every rule a session can be graded on |
| GET | `/search` | Find āyāt by wording, meaning, or both |
| POST | `/transcribe-file` | Offline whole-file transcription (needs a GPU) |
| WS | `/ws/session` | Live recitation feedback |

Interactive docs at `/docs`. The WebSocket does not appear as a route there, because
OpenAPI has no operation type for WebSockets. It is documented in that page's top-level
description, and in full below.

---

## 3. GET /health, and discovering which models exist

Call this at app start. It is how you know which engines the client may offer.

**Request**

```http
GET /health
```

**Response, from a CPU dev server (real capture)**

```json
{
  "status": "healthy",
  "engine": "mock",
  "available_engines": ["mock"],
  "device": "cpu",
  "dtype": "bfloat16",
  "muaalem_model": "obadx/muaalem-model-v3_2",
  "segmenter_model": "obadx/recitation-segmenter-v2"
}
```

**Response, from a GPU server with zipformer staged**

```json
{
  "status": "healthy",
  "engine": "real",
  "available_engines": ["real", "zipformer"],
  "device": "cuda",
  "dtype": "bfloat16",
  "muaalem_model": "obadx/muaalem-model-v3_2",
  "segmenter_model": "obadx/recitation-segmenter-v2",
  "muaalem_device": "cuda:0",
  "segmenter_device": "cuda:0"
}
```

| Field | Meaning |
|---|---|
| `engine` | The server **default**, used when a session does not request one |
| `available_engines` | Every engine this server built. Only these can be requested per session |
| `device`, `dtype` | Where the models run |
| `muaalem_device`, `segmenter_device` | Present only when `engine` is `real` |

Grey out any engine missing from `available_engines`. Requesting one that is absent does
not error; the server silently uses its default instead, and the session ack tells you what
actually ran.

---

## 4. The two models: Muaalem and zipformer

A session picks its ASR engine with the `engine` field of the start message. The choice
changes what the feedback contains, so the UI has to adapt.

| | `real` (Muaalem) | `zipformer` | `mock` | `remote` |
|---|---|---|---|---|
| Hardware | NVIDIA GPU | CPU | Anything | A GPU elsewhere |
| Phonemes | Yes | Yes | Fabricated from the cursor | Yes |
| Word-level correct / error | Yes | Yes | Yes | Yes |
| Per-character confidence | Yes | **No**, `char_probs` is empty | Fake, flat 0.97 | Yes |
| The 10 ṣifāt | Yes | **No**, every attribute is `null` | Copied from the reference | Yes |
| `almost` softening | Yes | **Never fires** | Rare | Yes |
| Tajwīd rule findings | Full | Limited, no ṣifāt channel | Full | Full |
| Speed | Slowest | Fast | Instant | Model time plus a round trip |

`remote` is `real` with the model on another machine, typically a free Kaggle or Colab GPU
reached through a tunnel. It runs the same decode function and returns the same fields, so
the client cannot tell the difference beyond latency. It exists for demonstrations, and it
appears in `available_engines` only when the server was started with a remote URL
configured. See [REMOTE_GPU.md](REMOTE_GPU.md).

### What this means in the UI

**On `real`,** everything in this document applies. Words come back `correct`, `almost` or
`error`, tajwīd and ṣifāt findings both arrive, and `strictness` controls how confident the
model must be before it accuses.

**On `zipformer`,** the model has no confidence head and no ṣifāt head. The service reports
that honestly rather than inventing numbers: `confidence` is `null` on every finding, and
every ṣifā attribute is `null`. Because a finding with unknown confidence grades to
`almost` at every strictness level, a zipformer session shows hints where a Muaalem session
would show hard errors. Sessions get word-level ḥifẓ marking and no articulation feedback.

Do not paper over this. If the user picks zipformer, say the session gives word tracking
without tajwīd grading. The frontend does this with a toast, and mobile should do the
equivalent.

**On `mock`,** the acoustic model is faked and nothing else is. VAD chunking, muṣḥaf
tracking, the reference diff, ṣifāt comparison, scoring and word aggregation are all the
production code paths, so the protocol you build against is the real one. Two caveats:
`mock` derives its phonemes from the session cursor, so a session that sends no
`sura`/`aya` emits nothing at all; and its confidences are invented, so never screenshot
`mock` output as evidence of accuracy.

---

## 5. WS /ws/session: live recitation

```
ws://<host>:8100/ws/session
```

One connection is one recitation session. The shape of the exchange:

```
client                                server
  |-- connect ----------------------->|
  |-- {"type":"start", ...} --------->|   JSON, must be first
  |<------------- {"type":"session"} -|   the ack
  |-- <binary PCM16 frame> ---------->|
  |-- <binary PCM16 frame> ---------->|
  |<----------- {"type":"feedback"} --|   pushed when the reciter pauses
  |-- <binary PCM16 frame> ---------->|
  |<----------- {"type":"feedback"} --|
  |-- {"type":"seek", ...} ---------->|   optional, any time
  |-- {"type":"end"} ---------------->|
  |<----------- {"type":"feedback"} --|   the flush, if audio was pending
  |<--------------- {"type":"done"} --|
  |<------------------ close ---------|
```

Feedback is **pushed**, not polled. One event per finalized waqf chunk, which in practice
means every time the reciter pauses for about 300 ms.

### 5.1 Handshake

Send the start message before any audio. It must be valid JSON. If the first frame the
server receives is not JSON, it closes the socket with code **1002** and you get nothing.

### 5.2 The start message

```json
{
  "type": "start",
  "sura": 1,
  "aya": 1,
  "word_idx": 0,
  "strictness": "normal",
  "engine": "real",
  "rules": ["aared_madd", "ghonna"],
  "moshaf": {"madd_monfasel_len": 4, "madd_aared_len": 2},
  "include_units": false
}
```

**Every field is optional, including the position.** Here is what each one does.

#### `sura`, `aya`, `word_idx`

The starting position. `sura` and `aya` are 1-based; `word_idx` is 0-based within the āyah.
It seeds the cursor.

Send it whenever you know it, which for a muṣḥaf reading view is always. With a position,
each chunk is matched by a windowed search around where the reciter already is: cheap, and
structurally immune to mutashābihāt (repeated passages).

Without a position, the first chunk is matched by a fuzzy search over the whole Qur'an. For
a distinctive passage that returns `"status": "ok"` and the cursor seeds itself, after which
every later chunk tracks normally. For a passage that genuinely occurs in several places it
returns `"status": "ambiguous"` with a candidate list.

The case that matters is the basmalah, which matches both Al-Fātiḥa 1:1 and An-Naml 27:30
and is the single most likely thing a reciter opens with. A "just start reciting, find me"
mode is legitimate; it just has to show candidates and let the next chunk disambiguate.

On the `mock` engine, omitting the position produces no output at all.

#### `strictness`

One of `"lenient"`, `"normal"` (default), `"strict"`.

It does not change what counts as a mistake. The diff finds identical errors at every
level. It sets the model-confidence threshold at which a finding is reported as a hard
`error` instead of being softened to `almost`. **A lower threshold is a harsher teacher.**

| Level | Phoneme threshold | Ṣifā threshold |
|---|---|---|
| `lenient` | 0.90 | 0.95 |
| `normal` | 0.70 | 0.85 |
| `strict` | 0.50 | 0.65 |

A finding the model is 0.75 sure of grades `almost` on `lenient` and `error` on the other
two. At 0.55 it is `error` only on `strict`.

A finding with **unknown** confidence (`"confidence": null`, for example a pure deletion
where the reciter said nothing and there is no probability to read) grades `almost` at every
level, `strict` included. No setting turns a guess into an accusation.

An unrecognised value, including a mis-cased `"Normal"`, falls back to the default rather
than failing the session.

> These thresholds are hand-picked placeholders awaiting calibration against a labelled
> set. Treat the ordering as meaningful and the absolute numbers as provisional. Do not
> build a user-facing accuracy claim on them.

#### `engine`

Picks an ASR engine for this session only. Use a value from `/health`'s
`available_engines`. An unknown or unbuilt name falls back to the server default without an
error, so **read the `engine` field of the session ack** to learn what actually ran, and
tell the user if it differs from what they picked.

#### `rules`

Leniency. Grade only these tajwīd rules and ṣifāt, and stay silent about the rest, so a
learner drilling madd al-ʿāriḍ is not also corrected on qalqalah.

| Value | Meaning |
|---|---|
| omitted, or `null` | Grade everything. The default |
| `["aared_madd", "ghonna"]` | Grade these, stay silent on every other tajwīd rule |
| `[]` | A real choice, not a missing one: no tajwīd rule at all, ḥifẓ and tashkīl only |

Keys come from `GET /tajweed-rules` (section 7). Filtered findings are dropped before they
can mark a word, so `status` and `errors[]` always agree. You will never receive a word
marked `error` with an empty `errors[]`.

**Ḥifẓ and tashkīl are never filtered.** A wrong or missing word (`error_type: "normal"`)
and a wrong ḥaraka (`"tashkeel"`) are reported whatever the selection. A `tajweed` finding
carrying an empty `tajweed_rules[]` (the reciter skipped a rule-bearing letter outright) is
also never filtered, since there is no rule to match it against.

Unknown keys are not an error. They simply match nothing, so sending only unknown keys
grades no tajwīd rule at all, exactly as `[]` does. That means a stale client degrades to a
narrower selection instead of a dropped connection.

#### `moshaf`

The reciter's tajwīd style. Send back whichever subset of `/moshaf-schema`'s fields the user
changed. The server layers your object over its own defaults, so you do not need to send
fields you did not touch, and you do not need to reconstruct the ones the schema hides.

An invalid combination (madd al-leen longer than madd al-ʿāriḍ, an out-of-range value) falls
back to the full default rather than dropping the session. A bad setting costs you the
setting, not the recitation.

#### `include_units`

Default `false`. Set `true` to add a `units` array to every feedback event: one entry per
phoneme group with its predicted ṣifāt and probability. This is debug and research data,
not something to render. It makes events considerably larger.

### The session ack

```json
{"type": "session", "session_id": "8fd196bc-7f74-4362-88b2-f1455f54d4b3",
 "engine": "mock", "sample_rate": 16000}
```

`engine` is the source of truth for what actually got used. `sample_rate` is the rate the
server expects your audio in, and it is always 16000.

### 5.3 Audio format

After the ack, send **binary** WebSocket frames.

| Property | Value |
|---|---|
| Sample rate | 16000 Hz |
| Channels | 1 (mono) |
| Encoding | PCM signed 16-bit integer |
| Byte order | Little-endian |
| Container | None. Raw samples only |
| Frame size | Your choice. 100 ms (3200 bytes) works well |

Common mistakes, in the order they actually happen: sending WAV bytes with the 44-byte
header still attached; sending float32; sending 44100 Hz or 48000 Hz; sending stereo;
sending base64 text instead of a binary frame.

The server buffers audio and runs silero VAD over it. When it sees a run of silence of at
least 300 ms after speech, it finalizes a chunk and pushes one feedback event. Speech
shorter than 200 ms is discarded as a breath or a click. A chunk is force-cut at 19 seconds
if the reciter never pauses, and that event carries `"forced_cut": true`.

**Silence produces nothing.** If the reciter does not speak, no events arrive. Do not build
a timeout that assumes a response per frame sent.

### 5.4 The feedback event

One per finalized chunk. This is a real capture from a Fātiḥa recitation:

```json
{
  "type": "feedback",
  "chunk_seq": 0,
  "audio_span_sec": [0.168, 19.296],
  "forced_cut": true,
  "phonemes": "بِسمِللَااهِررَحمَاانِررَحِۦۦم",
  "feedback": {
    "status": "ok",
    "span": {"sura": 1, "aya": 1, "word_idx": 0},
    "end": {"sura": 1, "aya": 1, "word_idx": 3},
    "uthmani_text": "بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ",
    "predicted_phonemes": "...",
    "reference_phonemes": "...",
    "words": [
      {"sura": 1, "aya": 1, "word_idx": 0, "uthmani": "بِسْمِ",
       "status": "correct", "errors": [], "trimmed": false},
      {"sura": 1, "aya": 1, "word_idx": 1, "uthmani": "ٱللَّهِ",
       "status": "correct", "errors": [], "trimmed": false},
      {"sura": 1, "aya": 1, "word_idx": 2, "uthmani": "ٱلرَّحْمَٰنِ",
       "status": "correct", "errors": [], "trimmed": false},
      {"sura": 1, "aya": 1, "word_idx": 3, "uthmani": "ٱلرَّحِيمِ",
       "status": "correct", "errors": [], "trimmed": false}
    ],
    "candidates": [],
    "non_verse": []
  },
  "cursor": {"sura": 1, "aya": 1, "word_idx": 3}
}
```

#### Top level

| Field | Type | Meaning |
|---|---|---|
| `type` | string | Always `"feedback"` |
| `chunk_seq` | int | 0-based, increments per event within the session |
| `audio_span_sec` | [float, float] | Start and end of this chunk in session time |
| `forced_cut` | bool | `true` if the 19 s cap ended the chunk rather than a pause |
| `phonemes` | string | What the model heard, as phonemes |
| `feedback` | object | The graded result, below |
| `cursor` | object or null | Where the reciter now is. Feed this into a `seek` if you need to resync |
| `units` | array | Present only when `include_units` was `true` |

#### The `feedback` object

| Field | Type | Meaning |
|---|---|---|
| `status` | `"ok"` \| `"ambiguous"` \| `"no_match"` | Whether the passage was identified |
| `span` | object or null | First word of the matched passage |
| `end` | object or null | Last word of it |
| `uthmani_text` | string or null | The matched passage as text |
| `predicted_phonemes` | string | What was heard |
| `reference_phonemes` | string | What was expected |
| `words` | array | Per-word feedback, in recitation order |
| `candidates` | array | Populated only when `status` is `"ambiguous"` |
| `non_verse` | array | Recognised non-verse text that was excluded from scoring |

`non_verse` contains any of `"istiaatha"`, `"basmalah"`, `"sadaka"`. It is reported rather
than silently dropped, because the learner knows they recited it and swallowing it reads as
the system not listening. Acknowledge it in the UI; do not score it.

#### A word

| Field | Type | Meaning |
|---|---|---|
| `sura`, `aya` | int | 1-based |
| `word_idx` | int | 0-based within the āyah |
| `uthmani` | string | The word as written |
| `status` | `"correct"` \| `"almost"` \| `"error"` | |
| `errors` | array | Findings for this word. Empty when correct |
| `trimmed` | bool | `true` means the word was **not scored** |

### 5.4a The progress event (live word-fill)

On the `real` and `remote` engines the server also pushes a lightweight `progress` event
while the reciter is still speaking, roughly every 300 ms, to fill words in before the
pause. It is **provisional and forward-only**:

```json
{"type": "progress",
 "confirmed": [{"sura": 1, "aya": 5, "word_idx": 0}],
 "skipped": [],
 "cursor": {"sura": 1, "aya": 5, "word_idx": 2}}
```

- `confirmed` — words the reciter has provisionally reached. Render them as *reached*, not
  as *correct* (no tick, no green): the authoritative `feedback` event grades them at the
  pause and may override.
- `skipped` — words the reciter passed over, asserted only once a later word is confirmed.
  A gentle positional hint, never a pronunciation mistake.
- `cursor` — the furthest confirmed word, for advancing the highlight and following the page.

A `progress` event carries **no errors, no phonemes, no ṣifāt**. Never render a mistake from
it; all criticism comes from `feedback`. `mock` and `zipformer` sessions never send it. A
client that ignores `progress` degrades to today's pause-only behavior.

The word-fill is produced by a CPU streaming model (zipformer) running locally on the
server, independent of the grading engine, so it stays responsive even when Muaalem grades
on a remote GPU. It is re-anchored to the authoritative cursor at every waqf, which is why
it carries coordinates only and is always reconciled by the next `feedback`.

### 5.5 Word status, and the three rules you must follow

These are contract requirements, not styling suggestions. The engine deliberately declines
to assert things it cannot support, and a client that renders those declines as assertions
undoes the safety property.

**`almost` is not a soft error.** It means the model was not confident enough to accuse.
Render it as a hint. Never as a mistake, never in the mistake list, never counted against a
score. Falsely correcting a perfect recitation is the one thing this system must not do.

**`trimmed: true` means unverified, not correct.** The word sat on a chunk boundary and was
cut in half by the chunker, not by the reciter, so it was not scored and its errors were
dropped. It is returned so you can still draw the word. Render it neutrally. Never with a
tick or a green highlight. The three statuses cannot express "not checked", so this flag
carries it, and you must read it before trusting `status`.

Real capture, from the second chunk of a Fātiḥa session, showing exactly this:

```json
{"sura": 1, "aya": 1, "word_idx": 3, "uthmani": "ٱلرَّحِيمِ",
 "status": "correct", "errors": [], "trimmed": true}
```

That word is `"correct"` and must not be shown as correct.

**`ambiguous` and `no_match` assert nothing.** No words, no errors, no candidates painted as
answers. Scoring someone against a verse they were not reciting is the worst failure
available to this service, so it declines instead of guessing.

### 5.6 The error object

Real capture. The reciter held a two-count madd for three counts on the second word of
ar-Raḥmān:

```json
{
  "sura": 1, "aya": 1, "word_idx": 2, "uthmani": "ٱلرَّحْمَـٰنِ",
  "status": "error",
  "trimmed": false,
  "errors": [
    {
      "error_type": "tajweed",
      "speech_error_type": "replace",
      "uthmani_pos": [25, 26],
      "ph_pos": [18, 20],
      "pred_ph_pos": [18, 21],
      "expected_ph": "اا",
      "predicted_ph": "ااا",
      "expected_len": 2,
      "predicted_len": 3,
      "tajweed_rules": [
        {"name_ar": "المد الطبيعي", "name_en": "Normal Madd",
         "golden_len": 2, "correctness_type": "count", "tag": "alif"}
      ],
      "confidence": 0.97
    }
  ]
}
```

| Field | Type | Meaning |
|---|---|---|
| `error_type` | `"tajweed"` \| `"normal"` \| `"tashkeel"` \| `"sifa"` | Which channel. `normal` is ḥifẓ: a wrong or missing word |
| `speech_error_type` | `"insert"` \| `"delete"` \| `"replace"` | What the reciter did |
| `uthmani_pos` | [int, int] | Character span in the Uthmani text, for highlighting |
| `ph_pos` | [int, int] | Character span in the **reference** phoneme string |
| `pred_ph_pos` | [int, int] or null | Span in the **predicted** string. `null` for a pure deletion |
| `expected_ph` | string | What should have been said |
| `predicted_ph` | string | What was said |
| `expected_len` | int or null | For count-based rules, the correct number of ḥarakāt |
| `predicted_len` | int or null | How many were actually held |
| `tajweed_rules` | array | Which rules this finding touches. May be empty |
| `confidence` | float or null | Model confidence. `null` means **unscored, not certain** |

`uthmani_pos` is the field to use for inline character highlighting. The two phoneme spans
are diagnostic; they use different coordinate systems and diverge on every insert, delete
and wrong-length madd, which is to say on every error worth scoring.

A `tajweed_rules` entry has `name_ar`, `name_en`, `golden_len`, `correctness_type`
(`"match"` or `"count"`), and an optional `tag`. Use these to explain the mistake:
"المد الطبيعي: expected 2, you held 3."

Note that the same finding would not arrive at all in a session started with
`"rules": ["aared_madd"]`, since Normal Madd is not in that selection. The word would come
back `correct`, not `error` with an empty list.

**On `confidence: null`.** Absence of confidence is not high confidence. It grades `almost`
regardless of strictness. On the zipformer engine every finding looks like this.

### 5.7 Seeking

The reciter jumped somewhere else. Reset the cursor:

```json
{"type": "seek", "sura": 2, "aya": 255, "word_idx": 0}
```

No reply is sent. `sura` and `aya` are required; `word_idx` defaults to 0. A message
missing `sura` or `aya` is ignored silently.

Send this whenever the user taps a different āyah or turns the page. Without it, the tracker
keeps searching near the old position and will start reporting mismatches.

### 5.8 Ending

```json
{"type": "end"}
```

The bare string `end` is accepted too. The server flushes any in-progress utterance, which
may produce one final `feedback` event, then sends:

```json
{"type": "done"}
```

and closes. Wait for `done` before tearing down, or you will lose the last chunk of the
recitation.

### 5.9 Failure and edge cases

#### `status: "ambiguous"`

The passage occurs in more than one place and the session had no cursor to disambiguate it.

```json
{"status": "ambiguous",
 "words": [],
 "candidates": [
   {"sura": 1, "aya": 1, "word_idx": 0, "uthmani_text": "بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ",
    "end": {"sura": 1, "aya": 1, "word_idx": 3}},
   {"sura": 27, "aya": 30, "word_idx": 3, "uthmani_text": "..."}
 ]}
```

Each candidate carries its **text**, not just coordinates, because `(2, 147)` is a lookup
the user then has to perform themselves. Show the verses and let them pick, or wait for the
next chunk to resolve it. The list is capped; a match with too many candidates comes back as
`no_match` instead, since a list of 1,599 is not a shortlist.

#### `status: "no_match"`

Nothing in the muṣḥaf matched. `span` is `null`, `words` and `candidates` are empty. Causes
include background noise, a non-Qur'anic utterance, and a passage too short to place. Show
a neutral "did not catch that" state. Do not mark anything wrong.

#### No event at all

A chunk that transcribes to an empty string produces no event. Silence, noise and very short
sounds are all normal reasons. Never treat "no event" as an error condition.

#### Engine fallback

Requesting an engine the server did not build is not an error. You get the default. The ack
is the only place this is visible, so compare it:

```
requested "zipformer" -> ack says "engine":"mock"  ->  tell the user
```

#### Close codes

| Code | Meaning | What to do |
|---|---|---|
| 1000 | Normal close after `done` | Nothing |
| 1002 | The first message was not valid JSON | Fix the client. This is a protocol error, not a transient one |
| 1006 | Abnormal close, usually the network | Reconnect and resume with the last `cursor` you saw |

#### Values the server repairs instead of rejecting

The session boundary is deliberately forgiving. Each of these costs you the setting, not the
recitation:

| You send | You get |
|---|---|
| `"strictness": "Normal"` (bad case) | The server default |
| `"engine": "banana"` | The server default |
| `"moshaf": {invalid combination}` | The full default moshaf |
| `"rules": ["not_a_rule"]` | Matches nothing, so no tajwīd rule is graded |
| A `seek` missing `sura`/`aya` | Ignored |
| A non-JSON text frame mid-session | Ignored |

---

## 6. GET /moshaf-schema: the recitation settings panel

Returns the recitation attributes the reciter can adjust, introspected from the model that
validates them, so the panel cannot drift from what the grader accepts.

```http
GET /moshaf-schema
```

```json
{"fields": [
  {"key": "recitation_speed",
   "name_ar": "سرعة التلاوة",
   "description": "The recitation speed sorted from slowest to the fastest ...",
   "default": "murattal",
   "options": [{"value": "mujawad", "label": "مجود"},
               {"value": "above_murattal", "label": "فويق المرتل"},
               {"value": "murattal", "label": "مرتل"},
               {"value": "hadr", "label": "حدر"}]},
  {"key": "madd_monfasel_len",
   "name_ar": "مد المنفصل",
   "description": "The length of Mad Al Monfasel \"مد النفصل\" for Hafs Rewaya.",
   "default": 2,
   "options": [{"value": 2, "label": "2"}, {"value": 3, "label": "3"},
               {"value": 4, "label": "4"}, {"value": 5, "label": "5"}]}
]}
```

The live response has **37 fields**, covering madd lengths, sakt positions, and the specific
disputed words (`مصر`, `ضعف`, `سلاسلا`, and so on). Build the panel from the response; do not
hard-code the list.

`value` may be a string or an integer. `label` is Arabic and display-ready.

**Fields with only one possible value are deliberately omitted.** There is nothing to choose,
`rewaya` being the obvious case (always `hafs` here). You do not need to reconstruct them.
Send back only the fields the user changed, and the server fills in the rest.

> **One thing to watch.** `default` is a sensible starting value for the picker, and it is
> not always what the server grades with when you omit `moshaf` entirely. The clearest case
> is `madd_monfasel_len`: the schema reports `2` while the server's own default is `4`. If
> the displayed setting must match the graded setting, send the field explicitly rather than
> relying on the two defaults agreeing.

---

## 7. GET /tajweed-rules: leniency

Every rule a session can ask to be graded on. Send a subset as `rules` in the start message.

```http
GET /tajweed-rules
```

```json
{"rules": [
  {"key": "aared_madd", "name_ar": "المد العارض للسكون", "name_en": "Aared Madd", "kind": "tajweed"},
  {"key": "ghonna", "name_ar": "الغنة", "name_en": "Ghonna", "kind": "sifa"}
]}
```

The complete list, verified against a live server. 8 tajwīd rules and 10 ṣifāt:

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

The table is here so a settings screen can be designed without a running server. Fetch the
endpoint in the app rather than pasting it in.

Three things worth reading twice:

**`ghonna` is a ṣifā, not a tajwīd rule.** Ghunnah is the rule learners ask for by name, so
it is the key most likely to be looked for in the wrong half of the list.

**`qalqalah` and `qalqla` are different keys for the same phenomenon** on the two channels,
the tajwīd rule and the ṣifā. Offer both or neither. The `(صفة)` suffix exists only so a UI
showing both does not render two identical chips.

**`name_ar` for `qalqalah` reads قلقة, not قلقلة.** That is the upstream spelling, passed
through unaltered so the service and the grader always agree on what a rule is called.
Override it in your UI if you prefer, but key off `key`.

Reminder: on the `zipformer` engine, no ṣifā findings are produced at all, so a ṣifā-only
selection grades nothing. Consider hiding the ṣifā half of the picker on that engine.

---

## 8. GET /search: semantic and keyword āyah search

Find āyāt by wording, by meaning, or both.

```http
GET /search?q=<text>&mode=keyword|vector|hybrid&hyde=true|false&lang=ar|en&alpha=<float>&limit=20
```

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `q` | string | required | Arabic or English. Blank or whitespace returns 422 |
| `mode` | enum | `hybrid` | `keyword`, `vector`, `hybrid`. Anything else returns 422 |
| `hyde` | bool | `false` | LLM query expansion before embedding. Ignored for `keyword` |
| `lang` | enum | auto | `ar` or `en`. Omit to detect from the script |
| `alpha` | float | `0.20` | Lexical weight in hybrid. Range 0 to 2 |
| `limit` | int | `20` | Range 1 to 50 |

### The two knobs

`mode` says where the score comes from. `hyde` says whether the query is rewritten before
embedding. They are separate on purpose, and every combination is legal.

| `mode` | Score | Needs |
|---|---|---|
| `keyword` | BM25 over the Uthmani words **and** their gold Qur'anic roots, so a query noun (الغيبة) matches an āyah's verb form (يغتب). Arabic only | Nothing. About 10 ms |
| `vector` | Cosine over BGE-M3 embeddings of the āyah plus its Muyassar tafsīr. The tafsīr is embed-only and never displayed | The embedding model |
| `hybrid` | `cosine + alpha * (bm25 / max)`. The vector finds meaning, BM25 pins wording | The embedding model |

`hyde=true` expands the query into a short hypothetical passage via an LLM and embeds that,
which disambiguates short polysemous queries (الغيبة "backbiting" against الغيب "the
unseen"). The passage is never shown. It is embedded and discarded, so every result on
screen is a real āyah. Without a server-side API key it silently falls back to the raw
query.

Only the embedding is expanded. The lexical half of `hybrid` always scores the raw query,
because BM25 over an LLM's paraphrase throws away the exact wording the user typed.

### Response

Real capture from `?q=الرحمن الرحيم&mode=keyword&limit=2`:

```json
{
  "hits": [
    {"sura": 1, "aya": 3,
     "text_uthmani": "ٱلرَّحْمَٰنِ ٱلرَّحِيمِ",
     "translation": "The Entirely Merciful, the Especially Merciful,",
     "score": 5.762516498565674},
    {"sura": 41, "aya": 2,
     "text_uthmani": "تَنزِيلٌ مِّنَ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ",
     "translation": "[This is] a revelation from the Entirely Merciful, the Especially Merciful -",
     "score": 5.418358325958252}
  ],
  "matched_lang": "ar",
  "mode": "keyword",
  "hyde_used": false
}
```

The Arabic āyah is always the primary result whatever the query language. The translation
rides along as secondary and may be `null`.

### The three fields that report what actually ran

`matched_lang`, `mode` and `hyde_used` describe the search that happened, which is not
always the one you asked for. A silent degradation is a bug you debug twice, so they are
reported explicitly:

- `hybrid` on an English query comes back as `"mode": "vector"`, because there is no Arabic
  lexical bag to fuse on the English side.
- `hyde` on `mode=keyword` comes back `"hyde_used": false`, because nothing gets embedded.
- No LLM key, or a provider failure, also comes back `"hyde_used": false`, with results
  from the raw query.

Surface at least the last one. The web UI tells the reciter when expansion was unavailable.

### Scores

Returned **raw**, with no relevance threshold and no normalisation. BM25 scores and cosine
similarities are not on the same scale, so do not compare across modes and do not render a
percentage. There is no calibrated cutoff, and inventing one would be false confidence. If
you want to hide weak hits, pick a threshold per mode from your own testing.

`keyword` mode drops zero-scoring hits, since BM25 zero means the āyah shares no token with
the query. `vector` and `hybrid` always return `limit` results, however weak.

### Latency, and what to show the user

| Case | Time |
|---|---|
| `keyword` | About 10 ms |
| `vector` / `hybrid`, model already loaded | A few hundred ms |
| `vector` / `hybrid`, **first query on a fresh server** | Minutes. BGE-M3 downloads (~2.3 GB), then about 10 s to load |
| `hyde=true` | Add one LLM round trip, typically under a second |

The first-query cost is the one that will look like a hang. Either warm the server before a
demo (see SETUP.md section 6) or show an explicit "loading the search model" state.

### Errors

**422, blank query:**

```json
{"detail": "q must not be empty or whitespace"}
```

**422, unrecognised mode:**

```json
{"detail": [{"type": "string_pattern_mismatch", "loc": ["query", "mode"],
             "msg": "String should match pattern '^(keyword|vector|hybrid)$'",
             "input": "bogus"}]}
```

Note the two shapes differ: the first is a plain string, the second is FastAPI's validation
array. Handle both.

**500** with an index error means the server's search artifacts are missing. That is a
deployment problem, not a client one.

### Which modes to expose

The web UI offers `keyword` and `hybrid`, plus a HyDE switch on the meaning tab. It does not
offer `vector`, because on Arabic it is strictly worse than hybrid and on English it **is**
hybrid, which makes it a choice with no right answer. The API keeps it for evaluation.
Mobile should do the same.

---

## 9. POST /transcribe-file: offline transcription

Upload a complete recitation, get per-chunk transcripts.

```http
POST /transcribe-file
Content-Type: multipart/form-data
file=<audio>
```

Any format ffmpeg can decode.

```json
{"chunks": [
  {"session_id": "my_recitation", "chunk_seq": 0, "is_final": false,
   "audio_span_sec": [0.0, 3.42],
   "predicted_phonemes": "بِسمِللَااهِررَحمَاانِررَحِۦۦم",
   "units": [
     {"phonemes_group": "بِ", "prob": 0.98,
      "sifat": {"hams_or_jahr": {"text": "jahr", "prob": 0.97},
                "shidda_or_rakhawa": {"text": "shadeed", "prob": 0.95},
                "qalqla": {"text": "qalqla", "prob": 0.91},
                "ghonna": null}}]}
]}
```

Two limits before you build on this.

**It is real-engine only.** The W2V-BERT segmenter is the chunker, and this endpoint does
not fall back to `mock` or `zipformer` regardless of the server's configured engine. It
needs a GPU with both models loaded.

**It returns raw model output**, not feedback. There is no tracking, no diff against a
reference, and no word-level grading. If you want per-word correct or error marking, use the
WebSocket.

---

## 10. Client implementation notes

### Recommended flow

1. `GET /health` at app start. Cache `available_engines`.
2. `GET /moshaf-schema` and `GET /tajweed-rules` once, cache them, build the settings screens.
3. When the user starts reciting, open the WebSocket, send `start` with the position from the
   muṣḥaf view, plus their saved settings.
4. Compare the ack's `engine` to what you requested. Warn on a mismatch.
5. Stream audio. Render each `feedback` event as it arrives.
6. On page turn or āyah tap, send `seek`.
7. On stop, send `end` and wait for `done`.

### Android

Capture with `AudioRecord`:

```kotlin
AudioRecord(
    MediaRecorder.AudioSource.VOICE_RECOGNITION,
    16000,
    AudioFormat.CHANNEL_IN_MONO,
    AudioFormat.ENCODING_PCM_16BIT,
    bufferSize
)
```

`ENCODING_PCM_16BIT` gives you exactly what the server wants. When converting a `ShortArray`
to bytes, set the buffer order explicitly rather than trusting the default:

```kotlin
ByteBuffer.allocate(shorts.size * 2).order(ByteOrder.LITTLE_ENDIAN)
```

Use OkHttp's `WebSocket` and send with `send(ByteString)`, not the `String` overload. The
string overload sends a text frame, and the server will treat it as a control message and
ignore it.

`VOICE_RECOGNITION` as the source matters. It disables the aggressive noise suppression and
AGC that `MIC` and `VOICE_COMMUNICATION` apply, which distort the sustained vowels the madd
grading measures.

### iOS

`AVAudioEngine`'s input tap gives Float32 at the hardware rate, usually 48 kHz. Convert with
`AVAudioConverter` to a 16 kHz mono Int16 format before sending:

```swift
let target = AVAudioFormat(commonFormat: .pcmFormatInt16,
                           sampleRate: 16000,
                           channels: 1,
                           interleaved: true)!
```

Int16 is little-endian on all Apple silicon and Intel devices, so `audioBufferList` bytes go
out as-is.

Use `URLSessionWebSocketTask` and send `.data(...)`, not `.string(...)`.

Set the audio session category to `.record` or `.playAndRecord` with mode `.measurement`.
`.measurement` disables system-applied signal processing, for the same reason
`VOICE_RECOGNITION` does on Android.

### Reconnection

The `cursor` in every feedback event is the resume point. Keep the last one. On an abnormal
close, reconnect and send a `start` carrying that cursor as `sura`/`aya`/`word_idx`. The
session resumes where the reciter actually is instead of restarting the passage.

### Backpressure

The server processes each chunk on a worker thread and pushes results when they are ready.
It does not block your sends. A slow GPU means feedback arrives later, not that audio is
dropped, so keep streaming rather than pausing capture while you wait.

### Rendering checklist

Before shipping, confirm each of these against a live session:

- `almost` renders as a hint, is excluded from the mistakes list, and does not affect a score.
- `trimmed: true` renders neutrally, with no tick and no green.
- `ambiguous` shows candidate verses without marking any word.
- `no_match` shows a neutral state and marks nothing.
- `non_verse` entries are acknowledged, not scored.
- An engine mismatch between requested and acked is surfaced.
- The Tanzil Project is credited with a link to <https://tanzil.net>, as its CC BY 3.0
  license requires of any app shipping the text.
