import type { EngineChoice, HealthInfo } from "./types";

/**
 * Static metadata for the models a reciter can pick between. Kept separate from the
 * backend's engine names so the UI copy lives in one place — "real" over the wire is
 * المُعلِّم (Muaalem) here, never shown to the user as "real".
 *
 * Whichever of these the server did not build is greyed out by EnginePicker from
 * /health's available_engines, so listing one costs nothing on a server without it.
 */
export const ENGINES: { id: EngineChoice; label: string; hint: string }[] = [
  {
    id: "real",
    label: "المُعلِّم",
    hint: "تحليل تجويد كامل، مع درجة ثقة لكل حرف",
  },
  {
    id: "zipformer",
    label: "Zipformer",
    hint: "أخفّ وأسرع، بدون تقييم تجويد",
  },
  {
    // Same model and same output as المُعلِّم, running on a GPU elsewhere. Labelled by
    // WHERE it runs rather than what it does, because to the reciter the only
    // difference is that each waqf waits on a network round trip.
    id: "remote",
    label: "المُعلِّم (سحابي)",
    hint: "نفس تحليل المُعلِّم، على معالج رسوميات بعيد — أبطأ قليلًا",
  },
];

export function labelFor(engineName: string): string {
  return ENGINES.find((e) => e.id === engineName)?.label ?? engineName;
}

const STORAGE_KEY = "tajwid.engine";

/** The reciter's last pick, if the browser remembers one. */
export function loadStoredEngineChoice(): EngineChoice | null {
  try {
    const v = localStorage.getItem(STORAGE_KEY);
    return ENGINES.some((e) => e.id === v) ? (v as EngineChoice) : null;
  } catch {
    return null; // private browsing / storage disabled — just don't persist
  }
}

export function storeEngineChoice(choice: EngineChoice): void {
  try {
    localStorage.setItem(STORAGE_KEY, choice);
  } catch {
    // ignore — not remembering the choice is not worth surfacing an error for
  }
}

// --- Live (responsive) mode ---------------------------------------------------
//
// The provisional word-fill tier. It is a COMPANION to the grade, never a replacement:
// the server runs it only when المُعلِّم is grading AND a zipformer build is present, so
// this flag can turn it off but can never force it on (see api/ws.py's `live`).

const LIVE_KEY = "tajwid.live";

/** Whether the live tier can run at all here: it needs a local zipformer build, and it
 *  only accompanies a المُعلِّم grade — picking Zipformer AS the grader leaves nothing
 *  for it to accompany. `available === null` means /health hasn't answered yet, so
 *  nothing is judged unavailable on a guess. */
export function liveAvailable(
  engine: EngineChoice,
  available: Set<string> | null,
): boolean {
  if (engine === "zipformer") return false;
  return available === null || available.has("zipformer");
}

/** The reciter's saved live-mode pick, or null to defer to the server's default. */
export function loadStoredLive(): boolean | null {
  try {
    const v = localStorage.getItem(LIVE_KEY);
    return v === null ? null : v === "true";
  } catch {
    return null;
  }
}

export function storeLive(on: boolean): void {
  try {
    localStorage.setItem(LIVE_KEY, String(on));
  } catch {
    /* private mode — the setting just won't persist */
  }
}

let healthPromise: Promise<HealthInfo> | null = null;

/** Which engines this server actually built (see /health's available_engines). */
export function loadHealth(): Promise<HealthInfo> {
  if (!healthPromise) {
    healthPromise = fetch("/api/health").then((r) => {
      if (!r.ok) throw new Error(`health: ${r.status}`);
      return r.json();
    });
  }
  return healthPromise;
}
