/**
 * Microphone capture -> 16 kHz mono PCM16 frames, over an AudioWorklet.
 *
 * An AudioWorklet, not a ScriptProcessorNode: the deprecated processor runs on the main
 * thread, so every React render competes with audio capture and drops frames — which
 * looks exactly like the model mishearing. The worklet runs on the audio thread and
 * cannot be starved by rendering.
 *
 * Resampling to 16 kHz is done by asking the AudioContext for a 16 kHz rate directly
 * where the browser allows it, and otherwise by decimating with a simple average. The
 * whole model stack is fixed at 16 kHz.
 */

const FRAME_MS = 100;
export const SAMPLE_RATE = 16000;

// Inlined so there is no second file to keep in sync and no separate fetch. It buffers
// to whole frames because the worklet is handed 128 samples at a time — 8 ms — and one
// WebSocket message per 8 ms is 125 messages a second of pure overhead.
const WORKLET = `
class PcmFrames extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.frame = options.processorOptions.frameSamples;
    this.buf = new Float32Array(this.frame);
    this.n = 0;
  }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    for (let i = 0; i < ch.length; i++) {
      this.buf[this.n++] = ch[i];
      if (this.n === this.frame) {
        // Copy: the buffer keeps filling while this one is in flight.
        this.port.postMessage(this.buf.slice(0));
        this.n = 0;
      }
    }
    return true;
  }
}
registerProcessor('pcm-frames', PcmFrames);
`;

export type MicHandlers = {
  onFrame: (pcm: ArrayBuffer) => void;
  /** 0..1 loudness of each frame, for the live meter. */
  onLevel?: (level: number) => void;
};

export class Mic {
  private ctx?: AudioContext;
  private stream?: MediaStream;
  private node?: AudioWorkletNode;
  private source?: MediaStreamAudioSourceNode;

  async start(handlers: MicHandlers): Promise<void> {
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });

    // Ask for 16 kHz up front so the browser resamples in native code. Firefox and
    // some Safari builds ignore the hint, so never assume it was honoured.
    this.ctx = new AudioContext({ sampleRate: SAMPLE_RATE });
    const blob = new Blob([WORKLET], { type: "application/javascript" });
    const url = URL.createObjectURL(blob);
    try {
      await this.ctx.audioWorklet.addModule(url);
    } finally {
      URL.revokeObjectURL(url);
    }

    const ratio = this.ctx.sampleRate / SAMPLE_RATE;
    const frameSamples = Math.round((FRAME_MS / 1000) * this.ctx.sampleRate);

    this.node = new AudioWorkletNode(this.ctx, "pcm-frames", {
      numberOfInputs: 1,
      numberOfOutputs: 0,
      processorOptions: { frameSamples },
    });

    this.node.port.onmessage = (e: MessageEvent<Float32Array>) => {
      const samples = ratio === 1 ? e.data : decimate(e.data, ratio);
      handlers.onLevel?.(rms(samples));
      handlers.onFrame(toPcm16(samples));
    };

    this.source = this.ctx.createMediaStreamSource(this.stream);
    this.source.connect(this.node);
    if (this.ctx.state === "suspended") await this.ctx.resume();
  }

  async stop(): Promise<void> {
    this.node?.port.close();
    this.source?.disconnect();
    this.node?.disconnect();
    this.stream?.getTracks().forEach((t) => t.stop());
    await this.ctx?.close();
    this.ctx = this.stream = this.node = this.source = undefined;
  }
}

/** Box-average decimation. Averaging (not picking every Nth sample) is what keeps
 *  high frequencies from folding back down into the speech band as alias noise. */
function decimate(input: Float32Array, ratio: number): Float32Array {
  const out = new Float32Array(Math.floor(input.length / ratio));
  for (let i = 0; i < out.length; i++) {
    const start = Math.floor(i * ratio);
    const end = Math.min(input.length, Math.floor((i + 1) * ratio));
    let sum = 0;
    for (let j = start; j < end; j++) sum += input[j];
    out[i] = end > start ? sum / (end - start) : 0;
  }
  return out;
}

function toPcm16(samples: Float32Array): ArrayBuffer {
  const out = new Int16Array(samples.length);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out.buffer;
}

function rms(samples: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < samples.length; i++) sum += samples[i] * samples[i];
  return Math.min(1, Math.sqrt(sum / samples.length) * 4);
}
