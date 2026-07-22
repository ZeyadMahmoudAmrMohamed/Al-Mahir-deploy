import type { AyahHit, SearchMode, SearchResult } from "./types";

/**
 * Āyah search against the backend.
 *
 * The API exposes the full matrix (mode × hyde × alpha); this app deliberately uses a
 * slice of it. `keyword` and `hybrid` are the two a reciter has a reason to choose
 * between — "I remember the words" vs "I remember the meaning" — and `hyde` is offered as
 * a toggle on the meaning search. Pure `vector` is not surfaced: it is strictly worse than
 * hybrid on Arabic and identical to it on English, so it exists for evaluation, not for a
 * menu the reciter has to reason about.
 *
 * The FIRST vector-mode call on a cold server loads a 2 GB model, so it can take ~10 s;
 * HyDE adds an LLM round trip on top. The caller shows a spinner and passes an
 * AbortSignal so a superseded query is dropped.
 */
export async function searchAyahs(
  q: string,
  mode: SearchMode,
  opts: { hyde?: boolean; signal?: AbortSignal } = {},
): Promise<SearchResult> {
  const params = new URLSearchParams({ q, mode, limit: "20" });
  if (opts.hyde) params.set("hyde", "true");
  const r = await fetch(`/api/search?${params}`, { signal: opts.signal });
  if (!r.ok) throw new Error(`search: ${r.status}`);
  const d = (await r.json()) as { hits: AyahHit[]; mode: string; hyde_used: boolean };
  return { hits: d.hits, mode: d.mode, hydeUsed: d.hyde_used };
}
