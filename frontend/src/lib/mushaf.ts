import type { MushafPage, SuraInfo, Span } from "./types";

const pageCache = new Map<number, Promise<MushafPage>>();
const fontsLoaded = new Set<number>();

export const PAGES = 604;
export const pad3 = (n: number) => String(n).padStart(3, "0");
export const fontFamilyFor = (page: number) => `QCF_P${pad3(page)}`;

/**
 * Each muṣḥaf page has its OWN font, in which every word of that page is a single
 * glyph. So the page cannot render until its font is in — with the font missing the
 * glyphs fall back to unrelated Arabic letters and the page looks like nonsense
 * rather than looking broken. Hence `document.fonts.load`, awaited.
 */
export async function loadPageFont(page: number): Promise<void> {
  if (fontsLoaded.has(page)) return;
  const family = fontFamilyFor(page);
  const face = new FontFace(family, `url(/fonts/qpc/QCF_P${pad3(page)}.woff2)`, {
    display: "block",
  });
  await face.load();
  document.fonts.add(face);
  fontsLoaded.add(page);
}

export function loadPage(page: number): Promise<MushafPage> {
  let p = pageCache.get(page);
  if (!p) {
    p = fetch(`/mushaf/${page}.json`).then((r) => {
      if (!r.ok) throw new Error(`page ${page}: ${r.status}`);
      return r.json();
    });
    pageCache.set(page, p);
  }
  return p;
}

/** Page data + font together: what "the page is ready" actually means. */
export async function loadPageReady(page: number): Promise<MushafPage> {
  const [data] = await Promise.all([loadPage(page), loadPageFont(page)]);
  return data;
}

let suraIndex: Promise<SuraInfo[]> | null = null;
export function loadSuraIndex(): Promise<SuraInfo[]> {
  if (!suraIndex) suraIndex = fetch("/mushaf/index.json").then((r) => r.json());
  return suraIndex;
}

/**
 * Which page a word is on. Used to follow the reciter across a page break.
 *
 * The word matters, not just the āyah: a long āyah spans two pages, and matching only
 * `(sura, aya)` returns the FIRST such page and pins the muṣḥaf there for the whole
 * second half — the "sometimes it follows, sometimes it doesn't". Given `wordIdx` we
 * find the page holding that exact word; without it we fall back to the āyah's opening
 * page.
 */
export async function pageOf(
  sura: number,
  aya: number,
  wordIdx?: number,
): Promise<number | null> {
  const has = (data: MushafPage) =>
    data.lines.some((l) =>
      l.words.some(
        (w) =>
          w.sura === sura &&
          w.aya === aya &&
          (wordIdx == null || w.word_idxs.includes(wordIdx)),
      ),
    );

  const cached = [...pageCache.keys()].sort((a, b) => a - b);
  for (const p of cached) {
    if (has(await pageCache.get(p)!)) return p;
  }
  // Not among the pages already in hand: fall back to a scan from the sūra's first
  // page. Sūras are contiguous, so this walks forward a few pages at most.
  const index = await loadSuraIndex();
  const info = index.find((s) => s.sura === sura);
  if (!info) return null;
  for (let p = info.page; p < Math.min(PAGES, info.page + 60); p++) {
    if (has(await loadPage(p))) return p;
  }
  return null;
}

/** A stable key for one Tanzil word — the unit the backend grades. */
export const wordKey = (s: number, a: number, i: number) => `${s}:${a}:${i}`;

export const spanKey = (s: Span) => wordKey(s.sura, s.aya, s.word_idx);

/** Every Tanzil word a glyph covers, as keys. */
export function keysOf(w: { sura: number; aya: number; word_idxs: number[] }): string[] {
  return w.word_idxs.map((i) => wordKey(w.sura, w.aya, i));
}
