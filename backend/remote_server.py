"""The GPU side of remote inference. Runs on Kaggle or Colab, not on your machine.

    local backend  --(16 kHz PCM16 over WSS)-->  this  -->  Muaalem on a Kaggle GPU
                   <--(ChunkTranscript as JSON)--

This is the acoustic model and nothing else. It holds no session state, knows nothing
about the muṣḥaf, and never sees a cursor: it is handed one finalized waqf chunk at a
time and answers with phonemes, per-character confidences and the 10 ṣifāt. Everything
that makes those into feedback stays in the local process.

It imports ``transcribe_reference_free`` from the same package the local `real` engine
uses, so remote and local produce identical output for identical audio by construction
rather than by careful maintenance.

Run it (see REMOTE_GPU.md for the full notebook):

    python remote_server.py --port 8200

For demonstrations. There is no authentication: anything that can reach the tunnel can
spend your GPU quota. Keep the URL private and stop the tunnel when you are done.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import threading

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("remote_server")

app = FastAPI(title="Tajwid remote GPU inference", version="1.0.0")

_muaalem = None
# Loading is minutes long and can be requested concurrently (a /warmup racing the first
# chunk). The lock makes the second caller wait for the first's model rather than start a
# second multi-GB load onto the same GPU.
_load_lock = threading.Lock()


def _models():
    """Load Muaalem once, on first use. MUAALEM ONLY — not the full bundle.

    `asr.models.get_models()` loads all three models the local pipeline needs: silero VAD,
    the W2V-BERT segmenter (2.32 GB) and Muaalem (2.42 GB). This server uses NONE of the
    first two. VAD endpointing and waqf chunking happen on the local side, which is what
    decides where one chunk ends; by the time audio reaches here it is already one
    finalized utterance. Calling get_models() here downloads ~4.8 GB to use 2.4 GB of it,
    which on a fresh notebook is several extra minutes of what looks like a hang.

    Lazily rather than at import so the server binds its port (and the tunnel comes up,
    and you can read the URL) while the weights are still downloading.

    BLOCKING, for minutes, on that first call. Every caller must reach this off the event
    loop (see `_models_async`), or the whole server — /health included — goes dark for the
    duration, which reads from outside as a crashed notebook.
    """
    global _muaalem
    with _load_lock:
        if _muaalem is None:
            from quran_muaalem.inference import Muaalem

            from tajwid.config import get_settings

            s = get_settings()
            device = s.resolved_muaalem_device
            logger.info("Loading Muaalem (%s) onto %s…", s.muaalem_model_id, device)
            _muaalem = Muaalem(
                model_name_or_path=s.muaalem_model_id,
                device=device,
                dtype=s.dtype_for(device),
            )
            logger.info("Muaalem ready on %s", _muaalem.device)
    return _muaalem


async def _models_async():
    return await asyncio.to_thread(_models)


@app.get("/health")
def health() -> dict:
    import torch

    return {
        "status": "healthy",
        "role": "remote-inference",
        "cuda": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "loaded": _muaalem is not None,
    }


@app.post("/warmup")
def warmup() -> dict:
    """Load the model now rather than on the first reciter's first waqf.

    Call this from INSIDE the notebook (`curl localhost:8200/warmup`), not through the
    tunnel. ngrok and Cloudflare both cut off a request that takes too long to answer, and
    loading 2.4 GB reliably takes longer than that: over the tunnel this returns a gateway
    error while the load continues, which looks like a failure and is not one.

    A plain `def` (not `async def`) so FastAPI runs it in a worker thread and /health keeps
    answering while the weights load.
    """
    import torch

    from tajwid.asr.transcribe import transcribe_reference_free

    muaalem = _models()
    transcribe_reference_free(muaalem, [torch.zeros(16000)], 16000)
    return {"status": "warm"}


@app.post("/warmup-async")
async def warmup_async() -> dict:
    """Start loading and return immediately, so a tunnel timeout cannot misreport it.

    Poll `GET /health` for `"loaded": true`. This is the safe way to warm the model from
    outside the notebook.
    """
    asyncio.get_running_loop().run_in_executor(None, _models)
    return {"status": "loading", "poll": "GET /health until loaded is true"}


@app.websocket("/infer")
async def infer(websocket: WebSocket) -> None:
    """One binary frame of 16 kHz PCM16 in, one ChunkTranscript as JSON out.

    Strictly one reply per message, in order, so the client can treat the socket as a
    synchronous request/response channel and does not need to correlate ids.
    """
    await websocket.accept()
    logger.info("Client connected")

    from tajwid.asr.remote import pcm16_to_wave, transcript_to_wire
    from tajwid.asr.transcribe import transcribe_reference_free

    try:
        while True:
            data = await websocket.receive_bytes()
            wave = pcm16_to_wave(data)
            if wave.numel() == 0:
                await websocket.send_json(
                    {
                        "phonemes_text": "",
                        "char_probs": [],
                        "groups": [],
                        "group_probs": [],
                        "sifat": [],
                    }
                )
                continue

            # Both of these block for a long time — the first for MINUTES while weights
            # load. Running them on the event loop would stall this coroutine, every other
            # connection, and /health, for the duration. From outside that is
            # indistinguishable from a dead notebook.
            muaalem = await _models_async()
            transcript = (
                await asyncio.to_thread(
                    transcribe_reference_free, muaalem, [wave], 16000
                )
            )[0]
            logger.info(
                "%.2fs audio -> %d phonemes", wave.numel() / 16000, len(transcript.phonemes_text)
            )
            await websocket.send_json(transcript_to_wire(transcript))
    except WebSocketDisconnect:
        logger.info("Client disconnected")


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8200)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
