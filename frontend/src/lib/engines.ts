import type { EngineChoice, HealthInfo } from "./types";

/**
 * Static metadata for the two models a reciter can pick between. Kept separate from
 * the backend's engine names so the UI copy lives in one place — "real" over the wire
 * is المُعلِّم (Muaalem) here, never shown to the user as "real".
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
];

export function labelFor(engineName: string): string {
  return ENGINES.find((e) => e.id === engineName)?.label ?? engineName;
}

const STORAGE_KEY = "tajwid.engine";

/** The reciter's last pick, if the browser remembers one. */
export function loadStoredEngineChoice(): EngineChoice | null {
  try {
    const v = localStorage.getItem(STORAGE_KEY);
    return v === "real" || v === "zipformer" ? v : null;
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
