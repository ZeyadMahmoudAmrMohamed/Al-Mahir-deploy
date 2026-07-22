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
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("remote_server")

app = FastAPI(title="Tajwid remote GPU inference", version="1.0.0")

_bundle = None


def _models():
    """Load the Muaalem bundle once, on first use.

    Lazily rather than at import so the server binds its port (and the tunnel comes up,
    and you can read the URL) while ~2.4 GB of weights are still downloading. The first
    chunk therefore pays the load; every one after is warm.
    """
    global _bundle
    if _bundle is None:
        from tajwid.asr.models import get_models

        logger.info("Loading Muaalem…")
        _bundle = get_models()
        logger.info("Muaalem ready on %s", _bundle.muaalem.device)
    return _bundle


@app.get("/health")
def health() -> dict:
    import torch

    return {
        "status": "healthy",
        "role": "remote-inference",
        "cuda": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "loaded": _bundle is not None,
    }


@app.post("/warmup")
def warmup() -> dict:
    """Load the model now rather than on the first reciter's first waqf.

    Worth calling from the notebook right after the tunnel is up: it moves a two-minute
    wait out of the demo and into setup.
    """
    import torch

    from tajwid.asr.transcribe import transcribe_reference_free

    bundle = _models()
    transcribe_reference_free(bundle.muaalem, [torch.zeros(16000)], 16000)
    return {"status": "warm"}


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

            bundle = _models()
            transcript = transcribe_reference_free(bundle.muaalem, [wave], 16000)[0]
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
