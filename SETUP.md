# Setup: running Al-Mahir from scratch

Everything needed to go from a fresh clone to a working backend, a working frontend, a
working āyah search, and a working live recitation session. Written for someone who has
never run this project.

Read [API.md](API.md) instead if you only need to call the service (mobile clients).

Contents:

1. [What you are installing](#1-what-you-are-installing)
2. [Prerequisites](#2-prerequisites)
3. [Backend: venv and install](#3-backend-venv-and-install)
4. [Choosing an engine, and the GPU question](#4-choosing-an-engine-and-the-gpu-question)
5. [NVIDIA CUDA and PyTorch](#5-nvidia-cuda-and-pytorch)
6. [The models: what downloads, from where, and when](#6-the-models-what-downloads-from-where-and-when)
7. [Hugging Face login and the gated zipformer model](#7-hugging-face-login-and-the-gated-zipformer-model)
8. [The Groq API key (optional, search only)](#8-the-groq-api-key-optional-search-only)
9. [Verifying search / RAG works](#9-verifying-search--rag-works)
10. [Frontend](#10-frontend)
11. [End-to-end smoke test](#11-end-to-end-smoke-test)
12. [Running the tests](#12-running-the-tests)
13. [Environment variable reference](#13-environment-variable-reference)
14. [Troubleshooting](#14-troubleshooting)

---

## 1. What you are installing

Two processes.

The **backend** is a Python FastAPI service on port 8100. It holds the ASR models, the
recitation feedback pipeline, and the āyah search index. Mobile and web clients talk only
to this.

The **frontend** is a React + Vite dev server on port 5173, used for demos and for testing
the backend by hand. Mobile teams do not need it, but running it once is the fastest way
to confirm the whole loop works.

```
 microphone (16 kHz PCM16)
    |
    v
 backend :8100  ──  silero VAD  ->  ASR engine  ->  locate/track in the muṣḥaf
                                                  ->  diff vs reference  ->  per-word feedback
    |
    v
 frontend :5173  (or your mobile app)
```

You do **not** need a GPU to run all of this. See section 4.

---

## 2. Prerequisites

| Thing | Version | Notes |
|---|---|---|
| Python | 3.10 or newer | Developed and verified on 3.11.9. 3.12 should work; 3.13 is untested against the torch pin. |
| Node.js | 18 or newer | Only for the frontend. Verified on Node 24. |
| git | any | |
| ffmpeg | optional | Only for `POST /transcribe-file` with mp3/m4a input. WAV works without it. |
| NVIDIA GPU + driver | optional | Only for the `real` engine. See sections 4 and 5. |

Disk: about 3 GB for Python packages (torch dominates), plus model downloads (section 6).
The repo itself already carries the search index (~71 MB) and the muṣḥaf fonts.

Clone and enter the repo:

```bash
git clone <repo-url>
cd Al-Mahir-Mobile-Ready
```

---

## 3. Backend: venv and install

Always use a virtual environment. The install pulls torch, transformers, faiss and
sentence-transformers, and you do not want those in your system Python.

### Windows (PowerShell)

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

If PowerShell refuses to run the activation script, allow it for the current session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### macOS / Linux

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

`uv` works too and is much faster, if you have it:

```bash
uv venv && uv pip install -e .
```

### The `.env` file

```bash
cp .env.example .env      # PowerShell: Copy-Item .env.example .env
```

Every value in it is optional. The service starts, recites, and searches without any of
them. The only thing `.env` unlocks is HyDE query expansion in search, covered in
section 8.

### Start the service

```bash
tajwid-serve
```

It listens on `http://0.0.0.0:8100`. Confirm:

```bash
curl http://localhost:8100/health
```

On a machine with no GPU and no zipformer files, this is the real response:

```json
{"status":"healthy","engine":"mock","available_engines":["mock"],"device":"cpu",
 "dtype":"bfloat16","muaalem_model":"obadx/muaalem-model-v3_2",
 "segmenter_model":"obadx/recitation-segmenter-v2"}
```

Interactive API docs live at `http://localhost:8100/docs`. The WebSocket protocol is
documented in that page's top-level description, because OpenAPI has no operation type for
WebSocket routes.

> **Windows note.** `tajwid-serve` logs Arabic text. If you run Python one-liners that
> print Arabic to a `cmd`/PowerShell console you may hit
> `UnicodeEncodeError: 'charmap' codec can't encode...`. Set `PYTHONUTF8=1` in the
> environment. The server itself is unaffected, since it writes JSON over HTTP.

---

## 4. Choosing an engine, and the GPU question

The backend can run three different ASR engines. This is the single most important choice
during setup, so here it is in full.

| Engine | Hardware | Model download | Gives you | Use it for |
|---|---|---|---|---|
| `real` | NVIDIA GPU (CUDA) | Muaalem + W2V-BERT segmenter, automatic on first run | Phonemes, per-character confidence, all 10 ṣifāt, full tajwīd grading, `almost` softening | Production |
| `zipformer` | CPU only | Manual, gated (section 7) | Phonemes, word-level correct/error marking. No ṣifāt, no confidence, no `almost` | A cheap CPU deployment, or a laptop demo with real audio |
| `mock` | Anything | None | Fabricated phonemes from the cursor, so every code path except the acoustic model runs | Frontend and mobile development with no GPU |

Selected by `TAJWID_ASR_ENGINE`:

- `auto` (the default) picks `real` if `torch.cuda.is_available()` is true, otherwise `mock`.
- `real`, `mock`, `zipformer` force that engine as the server default.

The startup value is only the **default**. Any engine the server actually built can be
picked per session by the client, via the `engine` field of the WebSocket start message.
`GET /health`'s `available_engines` lists which ones exist on that server.

### What `mock` really does

It is not a stub server. It fakes the acoustic model and nothing else. VAD endpointing,
waqf chunking, tracking through the muṣḥaf, the reference diff, the ṣifāt comparison,
scoring, and word aggregation all run the production code. A client wired against `mock`
is wired against the real protocol.

Two limits worth knowing:

- `mock` builds its phonemes **from the session cursor**, so a session that sends no
  `sura`/`aya` produces no output at all. That is the mock's limitation, not the
  pipeline's. Use `real` or `zipformer` to exercise the cold-start `locate` path.
- Its confidences are invented (0.97 flat). Set `TAJWID_MOCK_ERROR_RATE=0.3` to make it
  inject occasional real mistakes so error rendering can be tested.

```bash
TAJWID_ASR_ENGINE=mock TAJWID_MOCK_ERROR_RATE=0.3 tajwid-serve
```

PowerShell:

```powershell
$env:TAJWID_ASR_ENGINE="mock"; $env:TAJWID_MOCK_ERROR_RATE="0.3"; tajwid-serve
```

---

## 5. NVIDIA CUDA and PyTorch

Skip this whole section if you are running `mock` or `zipformer`.

`pip install -e .` installs whatever `torch>=2.7.0` wheel PyPI resolves for your platform.
On Windows and Linux that is frequently the **CPU-only** wheel, even on a machine with a
perfectly good NVIDIA card. Check:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If it prints `False` and you have an NVIDIA GPU, find your driver's CUDA version:

```bash
nvidia-smi
```

The top-right of that output shows a CUDA version, for example `CUDA Version: 12.4`. Now
reinstall torch from the matching index. Pick your combination at
<https://pytorch.org/get-started/locally/>; these are the common ones:

```bash
# CUDA 12.1
pip install --force-reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 12.4
pip install --force-reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# CUDA 12.8
pip install --force-reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Re-check until `torch.cuda.is_available()` is `True`. Then `TAJWID_ASR_ENGINE=auto` picks
`real` by itself, and `/health` reports:

```json
{"status":"healthy","engine":"real","available_engines":["real"],"device":"cuda",
 "dtype":"bfloat16","muaalem_device":"cuda:0","segmenter_device":"cuda:0", "...": "..."}
```

### Splitting components across devices

On a small-VRAM GPU you can keep the per-chunk Muaalem model on the GPU and push the
segmenter to the CPU:

```bash
TAJWID_DEVICE=cuda TAJWID_SEGMENTER_DEVICE=cpu tajwid-serve
```

silero VAD stays on the CPU by default. It is tiny, runs in real time there, and consumes
the CPU-resident audio buffer without a transfer.

`TAJWID_DTYPE_STR` defaults to `bfloat16`, which is what Muaalem was trained in. On CPU
the code overrides it to float32 automatically, because bfloat16 matmul on CPU is slow and
only partially supported.

---

## 6. The models: what downloads, from where, and when

Four model artifacts, with very different handling. Read the "when" column carefully,
because three of the four download themselves and one does not.

| Model | Hugging Face repo | Gated? | Size | When it downloads |
|---|---|---|---|---|
| Muaalem CTC (phonemes + ṣifāt) | `obadx/muaalem-model-v3_2` | No, MIT | ~2.4 GB | Automatically, first time the `real` engine transcribes |
| Recitation segmenter (W2V-BERT) | `obadx/recitation-segmenter-v2` | No, MIT | ~2.4 GB | Automatically, with the `real` engine (and for `POST /transcribe-file`) |
| BGE-M3 embeddings (search) | `BAAI/bge-m3` | No, MIT | ~2.3 GB | Automatically, on the **first** `vector` or `hybrid` search query. Roughly 10 s to load after that |
| Zipformer phoneme CTC | `Muno459/zipformer_p-quran` | **Yes** | ~260 MB | **Never automatically.** You download it by hand. See section 7 |

silero VAD is not downloaded at all. It ships inside the repo at
`backend/src/recitations_segmenter/data/silero_vad_v4.0.jit`.

The Qur'an text, the muṣḥaf page layout, the 604 per-page fonts, and the whole search
index are already committed to the repo. Nothing to fetch for those.

### Where downloads land

Hugging Face caches into `~/.cache/huggingface/hub` (on Windows,
`C:\Users\<you>\.cache\huggingface\hub`). Move it if your C: drive is tight:

```bash
export HF_HOME=/mnt/data/hf          # bash
```

```powershell
$env:HF_HOME = "D:\hf-cache"         # PowerShell, current session
[Environment]::SetEnvironmentVariable("HF_HOME", "D:\hf-cache", "User")   # persistent
```

### Pre-downloading, so first use is not a surprise

The automatic downloads happen inside the first request, which makes that request take
minutes. On a demo machine, warm them first:

```bash
# Muaalem + segmenter (only if you are running the `real` engine)
hf download obadx/muaalem-model-v3_2
hf download obadx/recitation-segmenter-v2

# BGE-M3, for vector/hybrid search
hf download BAAI/bge-m3
```

If your `huggingface_hub` predates the `hf` CLI, the old name works the same way:

```bash
huggingface-cli download BAAI/bge-m3
```

Or warm the embedder in-process, which also exercises the exact code path search uses:

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"
```

---

## 7. Hugging Face login and the gated zipformer model

This is the only model that needs an account. `obadx/muaalem-model-v3_2`,
`obadx/recitation-segmenter-v2` and `BAAI/bge-m3` are public and need no token at all.

`Muno459/zipformer_p-quran` is gated with automatic approval and carries a
`free-non-commercial` license: you must agree that the app using it is free to end users.
Read the terms on the model page before you ship anything with it.

### Step 1: accept the terms

Open <https://huggingface.co/Muno459/zipformer_p-quran> while signed in and accept the
license. Approval is automatic, but you have to click it. There is no way to do this from
the CLI.

### Step 2: log in

```bash
pip install -U "huggingface_hub[cli]"
hf auth login
```

Paste a token from <https://huggingface.co/settings/tokens>. A read token is enough. On
older installs the command is `huggingface-cli login`.

Non-interactive (CI, Docker):

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
```

### Step 3: download the two files you need

The repo has many files. The service needs exactly two: the ONNX export and the
`tokens.txt` that ships beside it.

```bash
cd backend
hf download Muno459/zipformer_p-quran \
  quran_phoneme_zipformer.onnx tokens.txt \
  --local-dir models/asr_zipformer
```

PowerShell, same thing on one line:

```powershell
cd backend
hf download Muno459/zipformer_p-quran quran_phoneme_zipformer.onnx tokens.txt --local-dir models/asr_zipformer
```

With curl, if you would rather not install the CLI (substitute your token):

```bash
cd backend/models/asr_zipformer
curl -L -H "Authorization: Bearer $HF_TOKEN" \
  -o quran_phoneme_zipformer.onnx \
  https://huggingface.co/Muno459/zipformer_p-quran/resolve/main/quran_phoneme_zipformer.onnx
curl -L -H "Authorization: Bearer $HF_TOKEN" \
  -o tokens.txt \
  https://huggingface.co/Muno459/zipformer_p-quran/resolve/main/tokens.txt
```

PowerShell without curl:

```powershell
$h = @{ Authorization = "Bearer $env:HF_TOKEN" }
$base = "https://huggingface.co/Muno459/zipformer_p-quran/resolve/main"
New-Item -ItemType Directory -Force backend\models\asr_zipformer | Out-Null
Invoke-WebRequest -Headers $h -Uri "$base/quran_phoneme_zipformer.onnx" -OutFile backend\models\asr_zipformer\quran_phoneme_zipformer.onnx
Invoke-WebRequest -Headers $h -Uri "$base/tokens.txt" -OutFile backend\models\asr_zipformer\tokens.txt
```

The result must be:

```
backend/models/asr_zipformer/
  quran_phoneme_zipformer.onnx
  tokens.txt
```

Point elsewhere with `TAJWID_ZIPFORMER_MODEL_PATH` and `TAJWID_ZIPFORMER_TOKENS_PATH` if
you launch the service from a different working directory. Those paths are resolved
relative to the process CWD, so launching from outside `backend/` without setting them
will not find the files.

### Two warnings that will cost you an afternoon

**Use the repo's `tokens.txt`.** Do not rebuild it from `phoneme_units.json`. That
mapping puts the CTC blank symbol at the wrong id, and the result is a model that decodes
confidently into nonsense rather than failing loudly.

**Do not use `quran_phoneme_zipformer.int8.onnx`.** The service is configured for the
float ONNX export. The int8 file has the same name shape and is easy to grab by accident.

### If the files are missing

The service still starts. `build_engines()` checks for them, logs a warning, and skips
zipformer:

```
Skipping the zipformer engine: no model at models/asr_zipformer/quran_phoneme_zipformer.onnx ...
```

zipformer then does not appear in `/health`'s `available_engines` and cannot be selected
per session. Everything else works normally.

The one exception: `TAJWID_ASR_ENGINE=zipformer` with no files is treated as a real
misconfiguration and raises at startup, rather than silently running a different engine
than you asked for.

---

## 8. The Groq API key (optional, search only)

The key powers one feature: HyDE query expansion on `GET /search?hyde=true`. Recitation
feedback never calls an LLM. Keyword search never calls an LLM. Plain vector and hybrid
search never call an LLM.

Without a key, `hyde=true` falls back to the raw query and the response says
`"hyde_used": false`. Nothing errors.

Get a free key at <https://console.groq.com/keys>, then put it in `backend/.env`:

```ini
TAJWID_LLM_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
TAJWID_LLM_BASE_URL=https://api.groq.com/openai/v1
TAJWID_LLM_MODEL_SMALL=openai/gpt-oss-20b
```

`GROQ_API_KEY` and `LLM_API_KEY` are accepted as aliases for the same setting, so an
existing `GROQ_API_KEY` in your shell already works.

`backend/.env` is git-ignored. Keep it that way; do not commit a key.

Any OpenAI-compatible provider works. Switching to Together, a local vLLM, or a self-hosted
Fanar is a change to `TAJWID_LLM_BASE_URL` and the model name, with no code change.

---

## 9. Verifying search / RAG works

The search artifacts ship in the repo at `backend/data/search/`:

```
corpus.sqlite    the āyāt, translations, and the Muyassar tafsīr used for embedding
ar.faiss         Arabic index (BGE-M3 over āyah + tafsīr, diacritics stripped)
en.faiss         English index
ids.json         row -> (sura, aya)
```

They total about 71 MB and are committed directly (the repo does not use Git LFS). Confirm
they are all present before anything else; a missing `corpus.sqlite` means an incomplete
clone.

### Check 1: keyword search (no model, ~10 ms)

```bash
curl "http://localhost:8100/search?q=%D8%A7%D9%84%D8%B1%D8%AD%D9%85%D9%86%20%D8%A7%D9%84%D8%B1%D8%AD%D9%8A%D9%85&mode=keyword&limit=2"
```

Real response:

```json
{"hits":[{"sura":1,"aya":3,"text_uthmani":"ٱلرَّحْمَٰنِ ٱلرَّحِيمِ",
          "translation":"The Entirely Merciful, the Especially Merciful,","score":5.762516498565674},
         {"sura":41,"aya":2,"text_uthmani":"تَنزِيلٌ مِّنَ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ",
          "translation":"[This is] a revelation from the Entirely Merciful, the Especially Merciful -",
          "score":5.418358325958252}],
 "matched_lang":"ar","mode":"keyword","hyde_used":false}
```

If this works, the corpus and the BM25 index are fine, and no model was involved.

### Check 2: vector / hybrid search (downloads BGE-M3 on first call)

```bash
curl "http://localhost:8100/search?q=backbiting&mode=hybrid&limit=5"
```

The first call takes minutes while ~2.3 GB downloads, then about 10 s to load the model
into memory. Later calls take a few hundred milliseconds. An English query returns
`"mode":"vector"` rather than `"hybrid"`, because there is no Arabic lexical bag to fuse
on the English side. That is the response telling you what actually ran, not a bug.

### Check 3: HyDE (needs the Groq key)

```bash
curl "http://localhost:8100/search?q=%D8%A7%D9%84%D8%BA%D9%8A%D8%A8%D8%A9&mode=hybrid&hyde=true&limit=5"
```

Look at `hyde_used` in the response. `true` means the key worked and the query was expanded
before embedding. `false` means no key, or the provider failed, and the raw query was used.
Either way you get results.

The LLM's generated passage is embedded and discarded. It is never shown to a user, so a
bad expansion can only change which real āyāt come back, never put invented scripture on
screen.

### Check 4: the search test suite

```bash
cd backend
python tests/test_search.py                          # keyword + the HyDE fallback path
TAJWID_TEST_SEMANTIC=1 python tests/test_search.py   # adds vector/hybrid, downloads BGE-M3
```

---

## 10. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173>. Vite proxies `/api` to `http://localhost:8100`, so the
browser talks to one origin and the WebSocket needs no CORS handling in dev.

Point it at a backend on another machine:

```bash
TAJWID_API=http://gpu-box:8100 npm run dev
```

PowerShell:

```powershell
$env:TAJWID_API="http://gpu-box:8100"; npm run dev
```

For production, `npm run build` emits a static `dist/`, which the Java/Spring backend or
the ingress can serve while routing `/api` to this service inside the VPC.

The muṣḥaf page data (`public/mushaf/*.json`) and the 604 KFGQPC per-page fonts
(`public/fonts/qpc/*.woff2`) are committed. You should not need to rebuild them. If you do:

```bash
cd backend
python scripts/build_mushaf_data.py
```

That script refuses to write unless every Tanzil word of the Qur'an is covered exactly once
and the derived line structure agrees page for page with the QUL KFGQPC layout. The check
is not optional: the glyph data and the layout come from two prints that disagree in four
āyāt and paginate juz 30 differently.

---

## 11. End-to-end smoke test

Backend running, no GPU needed.

**1. Service is up and reports an engine:**

```bash
curl http://localhost:8100/health
```

**2. The WebSocket handshake works.** With [websocat](https://github.com/vi/websocat):

```
websocat ws://localhost:8100/ws/session
> {"type":"start","sura":1,"aya":1,"word_idx":0}
< {"type":"session","session_id":"...","engine":"mock","sample_rate":16000}
> {"type":"end"}
< {"type":"done"}
```

**3. Real audio produces real feedback.** This streams a Fātiḥa recording from the test
assets through the live socket, exactly as a mobile client would:

```bash
cd backend
PYTHONUTF8=1 TAJWID_ASR_ENGINE=mock python -c "
import json, numpy as np
from fastapi.testclient import TestClient
from tajwid.asr.batch import load_audio
from tajwid.main import create_app

wave = load_audio('tests/assets/fatiha_long_track.wav', 16000).numpy()[:16000*25]
pcm = (np.clip(wave,-1,1)*32767).astype('<i2').tobytes()
frame = int(0.1*16000)*2

with TestClient(create_app()) as c:
    with c.websocket_connect('/ws/session') as ws:
        ws.send_json({'type':'start','sura':1,'aya':1,'word_idx':0})
        print(ws.receive_json())
        for i in range(0, len(pcm), frame):
            ws.send_bytes(pcm[i:i+frame])
        ws.send_json({'type':'end'})
        while True:
            m = ws.receive_json()
            if m['type'] == 'done':
                break
            fb = m['feedback']
            print(m['chunk_seq'], fb['status'], len(fb['words']), 'words', m['cursor'])
"
```

Expected shape of the output (this is a real run):

```
{'type': 'session', 'session_id': '...', 'engine': 'mock', 'sample_rate': 16000}
0 ok 4 words {'sura': 1, 'aya': 1, 'word_idx': 3}
1 ok 11 words {'sura': 1, 'aya': 5, 'word_idx': 0}
```

The cursor advancing between chunks is the thing to look for. It means VAD chunking,
transcription, and muṣḥaf tracking all ran.

**4. Search answers** (section 9, check 1).

**5. The frontend loads a muṣḥaf page** and the mic button starts a session.

---

## 12. Running the tests

```bash
cd backend
pip install -e ".[test]"
pytest
```

The suite is CPU-only by default and takes a couple of minutes. Some tests skip
themselves rather than fail:

- Tests needing the real GPU weights run only with `RUN_MODEL_TESTS=1`.
- `tests/test_zipformer_engine.py` skips unless `models/asr_zipformer/` is populated.
- Semantic search tests need `TAJWID_TEST_SEMANTIC=1` and will download BGE-M3.

A skip is not a pass. If you are validating a GPU box, set the flags and read the summary
line rather than trusting a green run that skipped the model.

---

## 13. Environment variable reference

All settings are read from `backend/.env` or from the process environment. The prefix is
`TAJWID_`; the names below are the full environment variable names.

### Engine

| Variable | Default | Meaning |
|---|---|---|
| `TAJWID_ASR_ENGINE` | `auto` | `auto`, `real`, `mock`, `zipformer`, `remote` |
| `TAJWID_REMOTE_URL` | none | WebSocket endpoint of a remote GPU. Setting it adds a `remote` engine. See [REMOTE_GPU.md](REMOTE_GPU.md) |
| `TAJWID_REMOTE_TIMEOUT_S` | `120.0` | Generous: a cold notebook loads weights on the first chunk |
| `TAJWID_MOCK_ERROR_RATE` | `0.0` | Chance the mock injects a mistake per chunk. `0.3` is good for demos |
| `TAJWID_ZIPFORMER_MODEL_PATH` | `models/asr_zipformer/quran_phoneme_zipformer.onnx` | Relative to the process CWD |
| `TAJWID_ZIPFORMER_TOKENS_PATH` | `models/asr_zipformer/tokens.txt` | Must be the repo's own tokens file |

### Devices

| Variable | Default | Meaning |
|---|---|---|
| `TAJWID_DEVICE` | `cuda` if available, else `cpu` | Main device |
| `TAJWID_DTYPE_STR` | `bfloat16` | Forced to float32 on CPU |
| `TAJWID_MUAALEM_DEVICE` | falls back to `TAJWID_DEVICE` | |
| `TAJWID_SEGMENTER_DEVICE` | falls back to `TAJWID_DEVICE` | Set to `cpu` on a small GPU |
| `TAJWID_VAD_DEVICE` | `cpu` | |

### Model ids

| Variable | Default |
|---|---|
| `TAJWID_MUAALEM_MODEL_ID` | `obadx/muaalem-model-v3_2` |
| `TAJWID_SEGMENTER_MODEL_ID` | `obadx/recitation-segmenter-v2` |

### Audio and endpointing

| Variable | Default | Meaning |
|---|---|---|
| `TAJWID_SAMPLE_RATE` | `16000` | Fixed by every model in the stack. Do not change |
| `TAJWID_VAD_THRESHOLD` | `0.6` | Speech probability gate. High so shallow dips at a waqf register as silence |
| `TAJWID_MIN_SILENCE_ENDPOINT_MS` | `300` | Silence after speech that finalizes a chunk |
| `TAJWID_MIN_SPEECH_MS` | `200` | Shorter finalized speech is discarded as a breath or click |
| `TAJWID_MAX_CHUNK_S` | `19.0` | Hard cap. Muaalem was trained on segments up to 20 s |
| `TAJWID_CHUNK_LEAD_PAD_MS` | `120` | Padding before a finalized region |
| `TAJWID_CHUNK_TRAIL_PAD_MS` | `240` | Padding after it |

### Feedback defaults

| Variable | Default | Meaning |
|---|---|---|
| `TAJWID_STRICTNESS` | `normal` | `lenient`, `normal`, `strict`. Overridable per session |
| `TAJWID_MADD_MONFASEL_LEN` | `4` | Overridable per session via `moshaf` |
| `TAJWID_MADD_MOTTASEL_LEN` | `4` | |
| `TAJWID_MADD_MOTTASEL_WAQF` | `4` | |
| `TAJWID_MADD_AARED_LEN` | `4` | |

### Search

| Variable | Default | Meaning |
|---|---|---|
| `TAJWID_SEARCH_MODE` | `hybrid` | Default when a request omits `mode` |
| `TAJWID_SEARCH_HYDE` | `false` | Default when a request omits `hyde` |
| `TAJWID_SEARCH_HYBRID_ALPHA` | `0.20` | Lexical weight in hybrid. Small on purpose; the vector stays dominant |
| `TAJWID_LLM_API_KEY` | none | Aliases: `GROQ_API_KEY`, `LLM_API_KEY` |
| `TAJWID_LLM_BASE_URL` | `https://api.groq.com/openai/v1` | Any OpenAI-compatible provider |
| `TAJWID_LLM_MODEL_SMALL` | `openai/gpt-oss-20b` | HyDE is a rewrite, so a small model is correct here |

---

## 14. Troubleshooting

**`torch.cuda.is_available()` is False on a machine with an NVIDIA GPU.** You have the
CPU-only torch wheel. Section 5.

**`/health` shows `"engine":"mock"` but you wanted `real`.** `auto` fell back because CUDA
was not available. Fix CUDA first, then optionally set `TAJWID_ASR_ENGINE=real` to make a
future failure loud instead of silent.

**`available_engines` does not list `zipformer`.** The model files are not where the
service looked. The startup log names the exact paths it checked. Remember they are
relative to the working directory you launched from.

**Zipformer decodes garbage.** Almost always the wrong `tokens.txt`, or the int8 ONNX
instead of the float one. Section 7.

**A WebSocket session produces no `feedback` events.** Check three things in order. Are you
sending raw 16 kHz mono PCM16 little-endian, not WAV headers, not float32, not 44.1 kHz?
Did you send the JSON start message first (a non-JSON first message closes the socket with
code 1002)? Are you on the `mock` engine without a `sura`/`aya` in the start message, which
produces silence by design?

**The socket closes immediately with 1002.** The first message was not valid JSON.

**The first search request hangs for minutes.** BGE-M3 is downloading. Section 6 covers
pre-downloading it.

**`IndexMissingError` from search.** `backend/data/search/` is missing or incomplete.

**`hyde_used` is always false.** No key was loaded. Confirm `backend/.env` exists, that you
launched the service from the `backend/` directory (the `.env` is read relative to the
process CWD), and that the key is valid.

**401 or 403 downloading zipformer.** You have not accepted the license on the model page,
or your token is missing. Section 7, step 1.

**`UnicodeEncodeError: 'charmap' codec` on Windows.** Console encoding, not a service bug.
Set `PYTHONUTF8=1`.

**`pip install -e .` fails building a package on Windows.** Usually a missing C++ build
toolchain for a source-only dependency. Install the Visual Studio Build Tools with the
"Desktop development with C++" workload, or use a Python version with prebuilt wheels for
your platform (3.11 is the verified one).

---

## Licensing, which follows the app and not just the repo

The Qur'an text comes from the Tanzil Project under CC BY 3.0, not MIT. Those terms bind
any app that ships it: the text may not be modified, and the app must credit the Tanzil
Project with a link to <https://tanzil.net>. The web frontend does this in its footer, and
mobile apps must do the equivalent.

The muṣḥaf glyphs and page layout are © King Fahd Glorious Qur'an Printing Complex.

`Muno459/zipformer_p-quran` is licensed for free-to-end-user applications only. If the app
is monetized, do not ship that engine.

See the licensing section at the end of `README.md` for more.
