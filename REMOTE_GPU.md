# Remote GPU inference (demo)

Run the Muaalem model on a free Kaggle or Colab GPU while everything else stays on your
machine. Local inference is untouched and still selectable; this adds a `remote` engine
alongside it.

**For demonstrations only.** The tunnel has no authentication, free GPU sessions expire
after a few hours, and every waqf pays a network round trip.

---

## Every session, in five cells

Kaggle wipes the disk between sessions, so the clone and install repeat each time. Set
**Accelerator: GPU T4 x2** and **Internet: On** in the sidebar first, and add `GH_TOKEN`
and `NGROK_TOKEN` under Add-ons → Secrets.

The rest of this file explains what these do and how to recover when one misbehaves.

```python
# 1. Code + deps  (~2 min)
from kaggle_secrets import UserSecretsClient
token = UserSecretsClient().get_secret("GH_TOKEN")
!git clone --depth 1 https://{token}@github.com/3omdawy11/Al-Mahir-Mobile-Ready.git /kaggle/working/almahir
!rm -rf /kaggle/working/almahir/frontend /kaggle/working/almahir/backend/data
%cd /kaggle/working/almahir/backend
!pip install -q -e . 2>&1 | tail -2
```

```python
# 2. Server  (~10 s)
import subprocess, time
subprocess.run(["pkill", "-f", "remote_server.py"]); time.sleep(3)
log = open("/kaggle/working/server.log", "w")
server = subprocess.Popen(["python", "remote_server.py", "--port", "8200"],
                          stdout=log, stderr=subprocess.STDOUT, text=True)
time.sleep(10)
print(subprocess.run(["curl","-s","localhost:8200/health"], capture_output=True, text=True).stdout)
```

```python
# 3. Tunnel — `domain=` pins the URL so backend/.env never changes
!pip install -q pyngrok
from pyngrok import ngrok
ngrok.set_auth_token(UserSecretsClient().get_secret("NGROK_TOKEN"))
url = ngrok.connect(8200, "http", domain="qualm-mountable-cultivate.ngrok-free.dev").public_url
print("TAJWID_REMOTE_URL=" + url.replace("https://","wss://") + "/infer")
```

```python
# 4. Warm the model  (~2-4 min — the slow step, and the only one worth waiting on)
!curl -s -X POST localhost:8200/warmup-async
```

```python
# 5. Keep alive — leave running for the whole demo
import time
while True: time.sleep(60)
```

Check progress from a scratch cell whenever you like:

```python
!tail -5 /kaggle/working/server.log
!curl -s localhost:8200/health
```

Ready when the log says `Muaalem ready on cuda:0` and health says `"loaded": true`.

Locally: nothing to do. `backend/.env` already holds the pinned URL, so start
`tajwid-serve`, open the frontend, and pick **المُعلِّم (سحابي)** from the engine picker.

---

## Where the seam is

```
mobile / frontend
      |  16 kHz PCM16 over WS
      v
local backend :8100
      |  VAD, waqf chunking            <- stays local
      |
      |--> engine.transcribe_chunk() ---------------------+
      |                                                   |
      |    LOCAL:  real (your CUDA) | zipformer | mock     |
      |    REMOTE: --(PCM16 over WSS)--> Kaggle/Colab GPU  |
      |                                                   |
      |<-- ChunkTranscript -------------------------------+
      |
      |  tracking, diff, ṣifāt, scoring  <- stays local
      v
   per-word feedback
```

Only the acoustic model moves. `ws.py`, `session.py` and everything in `feedback/` are
unchanged, so the application behaves identically either way.

Two consequences worth knowing. Only **speech** crosses the wire, because VAD has already
dropped the silence, so a 20-minute session sends only what was actually recited. And the
remote holds no session state, so a dropped tunnel costs one chunk rather than the session.

---

## Part 1: the GPU side

### Kaggle

Kaggle gives longer GPU sessions than Colab and is the better choice if it works for you.

1. New Notebook, then in the sidebar set **Accelerator** to `GPU T4 x2` or `P100`, and
   turn **Internet** on. Without internet the tunnel cannot open and nothing downloads.
2. Add your ngrok token under **Add-ons → Secrets** as `NGROK_TOKEN` (skip if using
   Cloudflare).
3. Run these cells.

**Cell 1, get the code and its dependencies.** The repo is private, so this needs a GitHub
personal access token. Put it in Kaggle's **Add-ons → Secrets** as `GH_TOKEN` rather than
pasting it into a cell, since notebooks get shared and shell history is saved:

```python
from kaggle_secrets import UserSecretsClient
token = UserSecretsClient().get_secret("GH_TOKEN")

!git clone --depth 1 https://{token}@github.com/3omdawy11/Al-Mahir-Mobile-Ready.git /kaggle/working/almahir

# The remote only runs the acoustic model. The frontend and the search index are ~250 MB
# it will never touch, and Kaggle's working disk is not generous.
!rm -rf /kaggle/working/almahir/frontend /kaggle/working/almahir/backend/data

%cd /kaggle/working/almahir/backend
!pip install -q -e . 2>&1 | tail -2
```

`--depth 1` skips the history and cuts the clone to well under a minute.

On Colab, swap the first two lines for `from google.colab import userdata` and
`token = userdata.get("GH_TOKEN")`, and clone into `/content` instead.

**Cell 2, confirm the GPU is real:**

```python
import torch
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
```

Kaggle's preinstalled torch is already CUDA-enabled. If this prints `False`, the
accelerator is not on.

**Cell 3, start the server in the background.** This cell returns in about ten seconds; if
it does not, the server failed to start and the log will say why.

```python
import subprocess, time

# Logs go to a FILE, not subprocess.PIPE. A pipe nobody reads fills its OS buffer at
# around 64 kB and then blocks the server on its next print, which looks like a hang
# with no error anywhere. Model loading logs more than that.
log = open("/kaggle/working/server.log", "w")
server = subprocess.Popen(
    ["python", "remote_server.py", "--port", "8200"],
    stdout=log, stderr=subprocess.STDOUT, text=True,
)
time.sleep(10)
print(subprocess.run(["curl", "-s", "localhost:8200/health"], capture_output=True, text=True).stdout)
```

Watch the server from any cell, any time:

```python
!tail -20 /kaggle/working/server.log
```

**Cell 4a, expose it with Cloudflare** (no account, no token):

```python
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /usr/local/bin/cloudflared
!chmod +x /usr/local/bin/cloudflared

import subprocess, re, time
tunnel = subprocess.Popen(
    ["cloudflared", "tunnel", "--url", "http://localhost:8200", "--no-autoupdate"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
)
url = None
while url is None:
    line = tunnel.stdout.readline()
    m = re.search(r"https://[-\w]+\.trycloudflare\.com", line)
    if m:
        url = m.group(0)
print("TAJWID_REMOTE_URL=" + url.replace("https://", "wss://") + "/infer")
```

**Cell 4b, or ngrok** (needs a free account, gives a stabler tunnel):

```python
!pip install -q pyngrok
from pyngrok import ngrok
from kaggle_secrets import UserSecretsClient

ngrok.set_auth_token(UserSecretsClient().get_secret("NGROK_TOKEN"))
url = ngrok.connect(8200, "http").public_url
print("TAJWID_REMOTE_URL=" + url.replace("https://", "wss://") + "/infer")
```

On Colab, replace the `kaggle_secrets` lines with your token directly, or use
`google.colab.userdata`.

**Cell 5, warm the model before the demo.** This loads ~2.4 GB of weights, which takes
minutes on a fresh notebook. Do it now so the first reciter's first waqf does not pay for
it.

```python
!curl -s -X POST localhost:8200/warmup-async     # returns immediately
```

Then watch the log rather than a blocking request:

```python
!tail -5 /kaggle/working/server.log
```

You are waiting for the second of these two lines:

```
Loading Muaalem (obadx/muaalem-model-v3_2) onto cuda…
Muaalem ready on cuda:0
```

`GET /health` flips to `"loaded": true` at the same moment.

The blocking form, `curl -s -X POST localhost:8200/warmup`, works too and returns
`{"status":"warm"}` when finished. It just sits there silently for minutes first, which is
hard to tell apart from a hang. Either way, run warmup from **inside** the notebook: a
tunnel cuts off a request this slow and reports a gateway error while the load quietly
continues.

**Cell 6, keep the notebook alive.** Kaggle stops an idle notebook, which kills the
tunnel:

```python
import time
while True:
    time.sleep(60)
```

### Colab

Identical, with three differences: clone into `/content` instead of `/kaggle/working`,
`Runtime → Change runtime type → T4 GPU`, and read secrets with
`from google.colab import userdata`.

### A note on the token

Use a **fine-grained** GitHub token scoped to this one repository with read-only Contents
permission, and give it a short expiry. A classic `repo`-scoped token grants write access
to everything you own, which is more than a notebook should hold. Revoke it at
<https://github.com/settings/tokens> when the demo is over.

---

## Part 2: the local side

Copy the printed URL into `backend/.env`:

```ini
TAJWID_REMOTE_URL=wss://sudden-mountain-1234.trycloudflare.com/infer
```

Or pass it at launch:

```bash
TAJWID_REMOTE_URL=wss://....trycloudflare.com/infer tajwid-serve
```

```powershell
$env:TAJWID_REMOTE_URL="wss://....trycloudflare.com/infer"; tajwid-serve
```

Confirm the engine registered:

```bash
curl http://localhost:8100/health
```

```json
{"status":"healthy","engine":"mock","available_engines":["mock","remote"], "...":"..."}
```

`remote` now appears alongside whatever runs locally.

### Choosing LOCAL or REMOTE

Two ways, and the second is the one that makes this a runtime choice rather than a
deployment.

**Server default**, for every session:

```bash
TAJWID_ASR_ENGINE=remote TAJWID_REMOTE_URL=wss://... tajwid-serve
```

**Per session**, the better option, since one server offers both and the client picks:

```json
{"type": "start", "sura": 1, "aya": 1, "engine": "remote"}
```

Send `"engine": "real"` on the same server to go local for that recitation. The session
ack reports which one actually ran, exactly as it does for every other engine:

```json
{"type":"session","session_id":"...","engine":"remote","sample_rate":16000}
```

The web frontend's engine picker (the sliders icon) reads `/health` and will list `remote`
with no change. Mobile gets it the same way.

Nothing else in the client changes. Same protocol, same feedback shape, same word statuses.

---

## What you get, and what it costs

`remote` runs the same `transcribe_reference_free` the local `real` engine runs, imported
from the same package, so the output is identical for identical audio by construction.
Per-character confidences and all 10 ṣifāt cross the wire intact, which means confidence
grading, `almost` softening and articulation feedback all work. That is the one property
worth guarding, and `tests/test_remote_engine.py` asserts it rather than assuming it: the
pre-merge HTTP handoff dropped exactly those fields and silently disabled both features.

| | Local `real` | `remote` |
|---|---|---|
| Needs your GPU | Yes | No |
| Ṣifāt, confidence, `almost` | Yes | Yes |
| Per-chunk latency | Model time | Model time + a round trip, typically 0.3 to 1.5 s |
| Bandwidth up | None | 32 kB per second **of speech** (silence never sent) |
| Fails when | Never | The tunnel drops or the notebook expires |

### Failure behaviour

A failed chunk is retried once, on the assumption that the common failure is a tunnel
dropping an idle connection between waqfs. A second failure logs at ERROR and returns an
empty transcript, which the pipeline already treats as "nothing was said" and drops
without emitting an event.

The session survives. The tradeoff is that a persistently broken remote looks to the
reciter like a microphone that stopped working, so watch the local logs during a demo:

```
ERROR Remote inference at wss://... failed twice: ... Returning an empty transcript
```

---

## Troubleshooting

**`remote` missing from `available_engines`.** `TAJWID_REMOTE_URL` was not set when the
server started. It is read at startup, so restart after editing `.env`.

**Every chunk logs a failure.** Check the scheme is `wss://` (not `https://`) and that the
path ends in `/infer`. Confirm the tunnel from your machine:
`curl https://<subdomain>.trycloudflare.com/health`.

**The first chunk times out, later ones work.** The model was loading. Call `/warmup`.

**It worked, then stopped.** The free GPU session expired or the notebook went idle. Both
kill the tunnel and both give a new URL on restart.

**Feedback is slower than local.** Expected. The round trip is per waqf, so it is felt at
every pause.

**`ModuleNotFoundError: tajwid` in the notebook.** `pip install -e .` was run from the
wrong directory. It must run inside `backend/`.

---

## Cleanup

Stop the tunnel and the notebook when you are done. An open tunnel is an unauthenticated
endpoint spending your GPU quota, and the URL is guessable enough to matter if it leaks.
Unset `TAJWID_REMOTE_URL` locally to drop back to local-only.
