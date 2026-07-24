import { Mic } from "./mic";
import type {
  EngineChoice,
  FeedbackEvent,
  MoshafConfig,
  ProgressEvent,
  RuleSelection,
  SessionEvent,
  Span,
  Strictness,
} from "./types";

const WS_URL = () => {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${location.host}/api/ws/session`;
};

export type SessionHandlers = {
  onFeedback: (e: FeedbackEvent) => void;
  onProgress?: (e: ProgressEvent) => void;
  onLevel: (level: number) => void;
  onState: (state: SessionStatus) => void;
  onError: (message: string) => void;
  /**
   * The engine the server actually loaded for this session, from the "session" ack.
   * The source of truth for what ran — an unavailable/unknown request silently falls
   * back server-side (see api/ws.py), so this can differ from what `start()` asked for.
   */
  onEngine?: (engine: string) => void;
};

export type SessionStatus = "idle" | "connecting" | "listening" | "closing";

/**
 * One live recitation: the mic, the socket, and nothing else. Feedback state lives in
 * React; this object owns only the transport, so a dropped connection cannot take the
 * session's mistake log with it.
 */
export class RecitationSession {
  private ws?: WebSocket;
  private mic = new Mic();
  private status: SessionStatus = "idle";

  constructor(
    private handlers: SessionHandlers,
    /** The reciter's moshaf attributes; omitted lets the backend use its own default. */
    private moshaf?: MoshafConfig | null,
    /**
     * Which tajwid rules to be graded on. `null`/undefined grades everything; an empty
     * array is a real choice (hifz and tashkeel only) and IS sent.
     */
    private rules?: RuleSelection,
    /**
     * How sure the grader must be before it accuses. Omitted uses the server default.
     */
    private strictness?: Strictness | null,
    /**
     * Whether to run the provisional live word-fill. Omitted defers to the server's
     * TAJWID_LIVE_FEEDBACK. This can only turn the tier OFF -- the server still requires
     * a zipformer build and a Muaalem grader, so `true` where neither holds is a no-op.
     */
    private live?: boolean | null,
  ) {}

  async start(from: Span, engine?: EngineChoice): Promise<void> {
    this.set("connecting");
    const ws = new WebSocket(WS_URL());
    ws.binaryType = "arraybuffer";
    this.ws = ws;

    const opened = new Promise<void>((resolve, reject) => {
      ws.onopen = () => resolve();
      ws.onerror = () => reject(new Error("Cannot reach the recitation service."));
    });

    ws.onmessage = (e) => {
      const msg: SessionEvent = JSON.parse(e.data);
      if (msg.type === "feedback") this.handlers.onFeedback(msg);
      else if (msg.type === "progress") this.handlers.onProgress?.(msg);
      else if (msg.type === "session") this.handlers.onEngine?.(msg.engine);
    };
    ws.onclose = () => {
      if (this.status !== "idle") void this.stop();
    };

    try {
      await opened;
    } catch (err) {
      this.set("idle");
      this.handlers.onError((err as Error).message);
      return;
    }

    // The start position SEEDS THE CURSOR. The protocol allows omitting it — the server
    // then runs a cold whole-Quran search — but this app never should: the most common
    // opening, the basmalah, is ambiguous with 27:30, so a position-less start would
    // open by telling the reciter we cannot tell what they are reciting. We always know
    // which sura they picked, so there is nothing to gain by making the server guess.
    // `engine` and `moshaf` are omitted rather than sent empty when unset, so the
    // server's own defaults pick (the resolved engine, and its default moshaf) rather
    // than an empty value matching nothing — see api/ws.py's fallbacks for each.
    // `rules` follows the same "omit when unset" rule but CANNOT use a truthiness test:
    // `[]` is a deliberate selection (grade no tajwid rule) and must reach the server,
    // where it means something different from an absent key.
    ws.send(
      JSON.stringify({
        type: "start",
        ...from,
        ...(engine ? { engine } : {}),
        ...(this.moshaf ? { moshaf: this.moshaf } : {}),
        ...(this.rules != null ? { rules: this.rules } : {}),
        ...(this.strictness ? { strictness: this.strictness } : {}),
        ...(this.live != null ? { live: this.live } : {}),
      }),
    );

    try {
      await this.mic.start({
        onFrame: (pcm) => {
          if (ws.readyState === WebSocket.OPEN) ws.send(pcm);
        },
        onLevel: this.handlers.onLevel,
      });
    } catch {
      this.handlers.onError("Microphone access was denied. Allow it to recite.");
      await this.stop();
      return;
    }
    this.set("listening");
  }

  /** The reciter moved: tell the tracker rather than letting it hunt. */
  seek(to: Span): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: "seek", ...to }));
    }
  }

  async stop(): Promise<void> {
    if (this.status === "idle") return;
    this.set("closing");
    await this.mic.stop();
    if (this.ws?.readyState === WebSocket.OPEN) {
      // Flush: the last utterance is still in the endpointer's buffer, and it is the
      // one the reciter just finished — the most likely to be waited on.
      this.ws.send(JSON.stringify({ type: "end" }));
      await new Promise((r) => setTimeout(r, 400));
      this.ws.close();
    }
    this.ws = undefined;
    this.set("idle");
  }

  private set(s: SessionStatus) {
    this.status = s;
    this.handlers.onState(s);
  }
}
