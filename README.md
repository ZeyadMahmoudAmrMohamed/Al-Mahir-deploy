# Tajwid — Al-Mahir

Real-time Qur'an recitation feedback. A learner recites into the microphone; the app
follows along on the muṣḥaf, filling in words as they are read and marking mistakes
inline — a tashkīl slip, a changed or missed word, a tajwīd rule — the way Tarteel does
on mobile, on the web.

This directory is the merge of the two halves of the pipeline into one project:

```
 microphone
    │  16 kHz PCM
    ▼
 ┌──────────────────────────────  backend (FastAPI, one WebSocket)  ──────────────────────────────┐
 │  silero VAD endpointing  →  Muaalem CTC model  →  phonemes + 10 ṣifāt + per-char confidence     │
 │  (waqf chunking)            (reference-free)        │                                            │
 │                                                     ▼                                            │
 │                          locate / track the passage in the Quran (cursor, no ayah skipped)       │
 │                                                     │                                            │
 │                          diff vs reference · ṣifāt · score · aggregate onto WORDS                │
 └──────────────────────────────────────────────────  │  ────────────────────────────────────────┘
                                                       ▼  per-word feedback JSON
 ┌──────────────────────────────  frontend (React, KFGQPC muṣḥaf)  ────────────────────────────────┐
 │  the real 15-line muṣḥaf page (QPC per-page fonts) · words fill in live · inline colour marks     │
 └───────────────────────────────────────────────────────────────────────────────────────────────┘
```

The two halves used to be separate repos with separate Streamlit demos. The seam between
them — the model's output crossing into the feedback engine **in-process, whole** — is
`backend/src/tajwid/session.py`. The old HTTP handoff dropped the per-character
confidences and the ṣifāt; carrying the object across a function call instead is what
keeps confidence grading and articulation feedback alive.

## Prerequisites (new machine)

- Python **3.10+**, Node **18+**, git.
- No GPU required to try the whole app: `TAJWID_ASR_ENGINE=mock` (default fallback when
  CUDA isn't present, see below) exercises every real code path except the acoustic
  model itself, and search's `keyword` mode needs no model at all.
- An NVIDIA GPU + CUDA-capable PyTorch build, only if you want the `real` engine
  (production-quality tajwīd/ṣifāt feedback). See **GPU / CUDA** below.
- Disk + network, on first use of each feature (all one-time, cached after):
  - `real` engine: Muaalem + W2V-BERT segmenter download from Hugging Face on first run.
  - `vector`/`hybrid` search: BGE-M3 downloads on the first such query (~2 GB, ~10 s to
    load). `keyword` search and the `mock`/`zipformer` engines need none of this.
  - `zipformer` engine: its files are **not** in the repo and are not fetched
    automatically — see the `zipformer` bullet under **Backend** below. Skip it if you
    don't need it; the service boots fine without it.

## Running it

Two processes: the Python **backend** (the AI service) and the React **frontend**.

### Backend

```bash
cd backend
pip install -e .            # or: uv pip install -e .
cp .env.example .env        # optional: only needed for HyDE search (see Āyah search below)
tajwid-serve               # http://localhost:8100
```

#### GPU / CUDA (for the `real` engine)

`pip install -e .` pulls whatever `torch>=2.7.0` wheel PyPI resolves for your platform,
which is not guaranteed to be CUDA-enabled. Check:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If that's `False` on a machine with an NVIDIA GPU, reinstall torch from the CUDA index
matching your driver (pick your CUDA version at <https://pytorch.org/get-started/locally/>):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121   # example: CUDA 12.1
```

`TAJWID_ASR_ENGINE=auto` (the default) then picks `real` on its own; force it explicitly
with `TAJWID_ASR_ENGINE=real` once `torch.cuda.is_available()` is `True`. `TAJWID_DEVICE`
and `TAJWID_SEGMENTER_DEVICE` (see `config.py`) let you place components on different
devices (e.g. Muaalem on a small GPU, segmenter on CPU).

The engine is chosen at startup from `TAJWID_ASR_ENGINE`:

- `auto` (default) — the real GPU model if CUDA is present, otherwise the mock.
- `real` — force the Muaalem model + silero VAD + W2V-BERT segmenter. **Needs a GPU**;
  the models download from Hugging Face on first run. This is the production path.
- `mock` — no acoustic model. It fabricates, from the phonetizer, what a perfect reciter
  continuing from the session cursor would have produced, so the **entire backend +
  frontend loop runs on a machine with no GPU**. Only the acoustic model is faked — the
  VAD endpointing, the tracking, the diff, the ṣifāt comparison, the scoring and the word
  aggregation are all the real code paths. `TAJWID_MOCK_ERROR_RATE=0.3` injects the odd
  mistake so the feedback colours can be demonstrated.
- `zipformer` — a CPU-only ONNX phoneme model (`sherpa-onnx`), no GPU needed. Faster and
  lighter than `real`, but it does not produce ṣifāt or per-character confidence —
  `char_probs` comes back `[]` and every ṣifā attribute is `None` rather than a faked
  number (see `test_char_probs_are_honestly_unscored_not_fabricated`), so sessions on it
  get word-level mistake marking with no tajwīd/articulation feedback and no `almost`
  grading.

  **Model files are not in the repo and are not downloaded automatically.** They come
  from the `Muno459/zipformer_p-quran` Hugging Face repo — get the ONNX export and its
  `tokens.txt` from there (the tokens file specifically **must** be the one shipped in
  that repo; one rebuilt from `phoneme_units.json` has blank at the wrong id, see
  `ZipformerAsrEngine`'s docstring). Put them under `models/asr_zipformer/`, or point
  `TAJWID_ZIPFORMER_MODEL_PATH` / `TAJWID_ZIPFORMER_TOKENS_PATH` elsewhere.

  **Without the files, the service still boots fine.** `build_engines()` checks for them
  at startup; if missing, it logs a warning and skips zipformer rather than crashing —
  it simply won't appear in `/health`'s `available_engines` or be selectable per session
  until the files are supplied. (Explicitly forcing `TAJWID_ASR_ENGINE=zipformer` with no
  files is treated as a real misconfiguration and still raises, rather than silently
  falling back to a different engine.)

On the GPU machine, `tajwid-serve` with no env vars picks `real` on its own. Nothing about
the CUDA code changes between here and there; it simply was not exercised on this CPU box.

**Per-session override.** The startup engine is only the *default* — whichever engines
actually got built are each addressable per session: the WS `start` message can carry
`"engine": "zipformer"` to use a different one for just that session. `real`/`mock`
(whichever `resolved_asr_engine` picks) is always one of them; `zipformer` joins them
too, but ONLY if its model files were found at startup (see above) — it's CPU-cheap
so it's attempted eagerly, just not required. An unknown or not-built name silently
falls back to the server default rather than erroring (see `api/ws.py`); the `session`
ack's `engine` field is the source of truth for what actually ran. The frontend surfaces
this as a picker in the topbar (sliders icon) — it reads `/health`'s `available_engines`
to grey out whichever engine this server didn't build, and shows a toast if the ack's
engine differs from what was requested.

Health check: `GET /health` reports which engine loaded by default and which engines
(`available_engines`) this server built.

### Frontend

```bash
cd frontend
npm install
npm run dev                # http://localhost:5173, proxies /api -> :8100
```

Point it at a backend on another host with `TAJWID_API=http://gpu-box:8100 npm run dev`.
For production, `npm run build` emits a static `dist/` the Java (Spring) backend or the
ingress can serve, routing `/api` to the AI service inside the VPC.

### The muṣḥaf data (already built)

`frontend/public/mushaf/*.json` (the 604 pages + a sūra index) and
`frontend/public/fonts/qpc/*.woff2` (604 per-page QPC V1 fonts) ship in the repo. To
rebuild the page data:

```bash
cd backend
python scripts/build_mushaf_data.py       # fetches from quran.com, verifies, writes
```

The script refuses to write unless **every Tanzil word of the Quran is covered exactly
once** and the derived line structure agrees, page for page, with the QUL KFGQPC layout.
See its module docstring for why that check is not optional (the layout and the glyph
data come from two prints that disagree in four āyāt and paginate juz 30 differently).

### Āyah search (`GET /search`)

Find āyāt by **wording**, by **meaning**, or both — from the search sheet's three tabs
(سورة / كلمة / معنى; the first is the local sūra index and hits no network).

```
GET /api/search?q=<text>&mode=keyword|vector|hybrid&hyde=true|false&lang=ar|en&alpha=&limit=20
```

**Two orthogonal knobs, not one enum.** `mode` says where the score comes from; `hyde`
says whether the query is rewritten before embedding. Every combination is legal, so
adding a third retrieval mode or a second expander costs no new API surface:

| `mode` | score | needs |
|---|---|---|
| `keyword` | BM25 over the Uthmani words **and** their gold Quranic roots, so a query noun (الغيبة) matches an āyah's verb form (يغتب). Arabic only. | nothing — ~10 ms |
| `vector` | Cosine over BGE-M3 embeddings of **āyah + its Muyassar tafsīr** (Recall@10 0.393 with tafsīr vs 0.264 without — a bare āyah is too short to embed a topic). The tafsīr is embed-only and never displayed. | the model |
| `hybrid` *(default)* | `cosine + alpha·(bm25/max)` — the vector finds meaning, BM25 pins wording. | the model |

`hyde=true` expands the query into a short hypothetical passage via an LLM and embeds
**that**, which disambiguates short polysemous queries (الغيبة "backbiting" vs الغيب "the
unseen"). The passage is never shown — it is embedded and discarded, so every result on
screen is still a real āyah. Needs `TAJWID_LLM_API_KEY` (or `GROQ_API_KEY`); see
`backend/.env.example`. Without a key it silently falls back to the raw query.

Only the *embedding* is expanded — the lexical half of `hybrid` always scores the raw
query, since BM25 over an LLM's paraphrase throws away the exact wording the reciter typed.

The response reports what **actually ran** — `mode`, `matched_lang`, `hyde_used`. They
differ from the request when a knob doesn't apply: `hybrid` on English is `vector` (no
Arabic lexical bag), `hyde` on `keyword` is ignored (nothing gets embedded), and an
unavailable LLM comes back `hyde_used=false`. The UI surfaces the last one to the reciter.

The artifacts (`backend/data/search/`: `corpus.sqlite`, `ar.faiss`, `en.faiss`,
`ids.json`, ~71 MB) ship in the repo. The **first** vector/hybrid query downloads and
loads BGE-M3 (~2 GB, ~10 s) — the UI says so while it waits; keyword search needs none of it.

```bash
python backend/tests/test_search.py                          # keyword + the HyDE fallback
TAJWID_TEST_SEMANTIC=1 python backend/tests/test_search.py   # + vector/hybrid (downloads BGE-M3)
```

The UI offers `keyword` and `hybrid` (plus a HyDE switch on the meaning tab), not `vector`:
on Arabic it is strictly worse than hybrid and on English it *is* hybrid, so it would be a
choice with no right answer. The API keeps it for evaluation.

## Layout

```
backend/
  src/tajwid/
    config.py          all tunables; TAJWID_* env overrides
    session.py         THE SEAM: finalized chunk → transcript → feedback, in-process
    asr/               the audio half
      stream.py          silero VAD endpointing (per-session, own VAD instance)
      transcribe.py      reference-free Muaalem decode → phonemes + per-CHAR probs + ṣifāt
      segment.py         W2V-BERT waqf segmenter (offline / batch path)
      engine.py          RealMuaalemEngine (GPU) | MockEngine (CPU) behind one interface
      models.py, vad.py  model loading
      batch.py           whole-file transcription; stream_file() for CPU e2e
    feedback/          the alignment + grading half (ported from muaalem_feedback)
      pipeline.py        analyse() and analyse_session()
      locate.py track.py diff.py sifat.py confidence.py words.py nonverse.py …
    search/            āyah search — see GET /search above
      service.py         mode (keyword|vector|hybrid) × hyde, and what actually ran
      lexical.py         BM25 over surface forms + gold Quranic roots
      hyde.py llm.py     LLM query expansion; degrades to the raw query without a key
      corpus.py embeddings.py
    api/
      ws.py              WS /ws/session — the live protocol
      rest.py            /health, /transcribe-file, /search
    main.py            the FastAPI app
  src/quran_transcript/         vendored — the phonetizer + search index (untouched)
  src/quran_muaalem/            vendored — the CTC model + decode
  src/recitations_segmenter/    vendored — the W2V-BERT segmenter + silero VAD jit
  scripts/build_mushaf_data.py  builds frontend/public/mushaf from quran.com + QUL layout
  tests/                        runs CPU-only by default; some tests self-skip without
                                 real GPU model weights (RUN_MODEL_TESTS=1) or zipformer
                                 files under models/asr_zipformer/ — see Prerequisites

frontend/
  src/
    lib/mic.ts         AudioWorklet mic capture → 16 kHz PCM16 (off the main thread)
    lib/session.ts     one live recitation: the WebSocket + the mic
    lib/marks.ts       feedback → marks; the almost/trimmed/ambiguous rules live here
    lib/mushaf.ts      page + per-page-font loading, word-key coordinates
    lib/engines.ts      engine picker labels/hints, /health fetch, persisted choice
    components/         MushafPage, FeedbackBar, Sheets (3-tab search + mistakes),
                         EnginePicker (model choice popover), Icons
    App.tsx            the one screen
  public/mushaf/       604 page JSONs + index.json  (built, do not hand-edit)
  public/fonts/qpc/    604 KFGQPC V1 per-page fonts
```

## The WebSocket protocol (`WS /ws/session`)

```
client → server
  {"type":"start", "sura":1, "aya":1, "word_idx":0, "strictness":"normal"?, "moshaf":{…}?,
   "engine":"zipformer"?, "rules":["aared_madd","ghonna"]?}
        first message. Must be JSON (the socket closes with 1002 otherwise), but every
        FIELD is optional — `sura`/`aya`/`word_idx` included. They SEED THE CURSOR, and
        sending them is strongly preferred: each chunk is then matched by `track`, a
        search in a window around where the reciter already is. Omit them and the first
        chunk falls back to `locate`, a fuzzy search over the whole muṣḥaf — which
        returns `ok` outright for a distinctive passage (the cursor then self-seeds and
        every later chunk tracks normally), and `ambiguous` WITH A CANDIDATE LIST for one
        that genuinely occurs in several places. The basmalah is the case that matters:
        it matches both Al-Fātiḥa 1:1 and An-Naml 27:30, and it is the most likely thing
        a reciter opens with. This app always knows which sūra the learner picked, so it
        sends the position; a "just start reciting, find me" client can legitimately omit
        it and show the candidates. Note the `mock` engine fabricates phonemes FROM the
        cursor and so emits nothing without one — a mock limitation, not the pipeline's.
        `engine` is optional and picks a different engine than the server default for
        just this session (see "Per-session override" above); omit it for the default.
        `rules` is LENIENCY: grade only these tajwīd rules and stay silent about the
        rest, so a learner drilling madd al-ʿāriḍ is not also told about qalqalah. Keys
        come from `GET /tajweed-rules` (8 tajwīd rules + the 10 ṣifāt; that endpoint's
        entry in `/docs` lists the full key → Arabic/English mapping). Omit it, or send
        null, to grade everything — the default. An empty list is a real choice, not a
        missing one: ḥifẓ and tashkīl only. **Ḥifẓ (`normal`) and tashkīl findings are
        never filtered**, whatever is selected — leniency narrows which tajwīd rules are
        enforced, not whether the recitation is checked.
  <binary>                              a frame of 16 kHz mono PCM16-LE audio
  {"type":"seek", "sura","aya","word_idx"}   the reciter repositioned; reset the cursor
  {"type":"end"}                        flush the last utterance and close

server → client
  {"type":"session", "session_id", "engine", "sample_rate"}
  {"type":"feedback", "chunk_seq", "audio_span_sec", "phonemes", "feedback":{…}, "cursor"}
  {"type":"done"}
```

`feedback` is a `FeedbackResponse` (see `tajwid/feedback/types.py`): every word of the
matched span, in order, each with its Tanzil coordinates, `uthmani`, a `status`
(`correct` | `almost` | `error`) and its errors. Three things the frontend must respect,
and does (`lib/marks.ts`):

- **`almost` is not a soft error.** The model is not confident enough to accuse. Rendered
  as a hint, never a mistake, and never counted against the score. Falsely correcting a
  perfect recitation is the one thing this system must not do.
- **`trimmed` means unverified, not correct** — the word sat on a chunk boundary and was
  not scored. Neutral mark, never a tick.
- **`ambiguous` / `no_match` assert nothing.** No words, no candidates painted as answers.

## Licensing — this lands on the app, not just the repo

The Qur'an text (via the vendored `quran_transcript`) is the **Tanzil Qur'an Text**, ©
Tanzil Project, **CC BY 3.0** — not MIT. Its terms bind any app that ships it: the text
may not be modified, and the app **must credit Tanzil Project and link to
<https://tanzil.net>**. The frontend does this in its footer; keep it there. The muṣḥaf
glyphs and layout are © King Fahd Glorious Qur'an Printing Complex (KFGQPC). See
`../VENDOR.md`.
