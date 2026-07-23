/**
 * Turning the backend's feedback into marks on the page.
 *
 * This is where the contract's three warnings are honoured, or quietly broken:
 *
 *   `almost`     is NOT a soft error. It means the model is not confident enough to
 *                accuse. It renders as a hint, never as a mistake, and never counts
 *                against the score. A learner falsely corrected on a verse they recited
 *                perfectly concludes the app cannot hear them, and does not come back.
 *   `trimmed`    means UNVERIFIED, not correct: the word sat on a chunk boundary and was
 *                never scored. It gets its own neutral mark — never a tick.
 *   `ambiguous` / `no_match` assert NOTHING. `words` is empty. Nothing is painted, and
 *                the candidates are never treated as an answer.
 */

import { wordKey } from "./mushaf";
import type {
  FeedbackEvent,
  MistakeLog,
  ProgressEvent,
  WordFeedback,
  WordMark,
} from "./types";

export type MarkState = {
  /** wordKey -> how it is painted. */
  marks: Map<string, WordMark>;
  /** wordKey -> why, for the tooltip. */
  detail: Map<string, WordFeedback>;
  /** Every mistake this session, in order. */
  log: MistakeLog[];
  /** Words the reciter has reached, for the reveal in hidden mode. */
  reached: Set<string>;
};

export const emptyMarks = (): MarkState => ({
  marks: new Map(),
  detail: new Map(),
  log: [],
  reached: new Set(),
});

function markOf(w: WordFeedback): WordMark {
  if (w.trimmed) return "unverified"; // read the flag BEFORE the status
  if (w.status === "error") return "error";
  if (w.status === "almost") return "almost";
  return "recited";
}

/** Committed (authoritative) verdicts. Provisional marks never overwrite these. */
const COMMITTED = new Set<WordMark>(["error", "almost", "recited", "unverified"]);

/**
 * Fold one Tier 1 `progress` event into the page state. Provisional: sets `heard` on
 * confirmed words and `skipped` on skipped ones, never touching a committed verdict, and
 * carries no errors and no log entries. Pure: returns a new state.
 */
export function applyProgress(prev: MarkState, event: ProgressEvent): MarkState {
  if (event.confirmed.length === 0 && event.skipped.length === 0) return prev;

  const marks = new Map(prev.marks);
  const reached = new Set(prev.reached);

  for (const w of event.confirmed) {
    const key = wordKey(w.sura, w.aya, w.word_idx);
    reached.add(key); // live reveal in hidden mode
    const cur = marks.get(key);
    if (cur && COMMITTED.has(cur)) continue; // never downgrade an authoritative verdict
    marks.set(key, "heard");
  }
  for (const w of event.skipped) {
    const key = wordKey(w.sura, w.aya, w.word_idx);
    if (marks.get(key)) continue; // don't touch a verdict OR an already-heard word
    marks.set(key, "skipped");
  }

  return { ...prev, marks, reached };
}

/** Fold one chunk's feedback into the page state. Pure: returns a new state. */
export function applyFeedback(prev: MarkState, event: FeedbackEvent): MarkState {
  const fb = event.feedback;
  // `ambiguous` and `no_match` carry no words and assert nothing. Painting the
  // candidates would be scoring the learner against a verse we did not identify.
  if (fb.status !== "ok" || fb.words.length === 0) return prev;

  const marks = new Map(prev.marks);
  const detail = new Map(prev.detail);
  const reached = new Set(prev.reached);
  const log = [...prev.log];

  for (const w of fb.words) {
    const key = wordKey(w.sura, w.aya, w.word_idx);
    const mark = markOf(w);
    reached.add(key);

    // A word cut by OUR chunk boundary comes back `unverified`. But `overlap_words`
    // means the neighbouring chunk reaches back over that same word and scores it for
    // real. Whichever order the two chunks arrive, the real verdict must win: never let
    // an `unverified` re-emission grey out a word an overlapping chunk already judged.
    const prevMark = marks.get(key);
    // Block only a real verdict from being greyed by a later `unverified`. A provisional
    // `heard`/`skipped` carries no verdict, so let `unverified` (and anything else) win.
    if (
      mark === "unverified" &&
      prevMark &&
      COMMITTED.has(prevMark) &&
      prevMark !== "unverified"
    )
      continue;

    marks.set(key, mark);
    detail.set(key, w);

    if (mark === "error" || mark === "almost") {
      log.push({
        at: event.audio_span_sec[0],
        sura: w.sura,
        aya: w.aya,
        word_idx: w.word_idx,
        uthmani: w.uthmani,
        status: mark,
        kinds: [...new Set(w.errors.map((e) => e.error_type))],
        rules: [...new Set(w.errors.flatMap((e) => e.tajweed_rules.map((r) => r.name_ar)))],
      });
    }
  }

  return { marks, detail, log, reached };
}

/**
 * Session accuracy: the share of SCORED words that drew no confident error.
 *
 * `almost` and `unverified` words are excluded from the denominator rather than counted
 * as correct or as mistakes — we did not check them, so they are not evidence either
 * way. Counting them as correct would inflate the score; as mistakes, it would punish
 * the reciter for our uncertainty.
 */
export function accuracy(state: MarkState): number | null {
  let scored = 0;
  let clean = 0;
  for (const mark of state.marks.values()) {
    if (mark === "recited") {
      scored++;
      clean++;
    } else if (mark === "error") {
      scored++;
    }
  }
  return scored === 0 ? null : Math.round((clean / scored) * 100);
}

/** Confident mistakes on the words of one page. `almost` is not a mistake. */
export function mistakesOnPage(state: MarkState, pageKeys: Set<string>): number {
  let n = 0;
  for (const [key, mark] of state.marks) {
    if (mark === "error" && pageKeys.has(key)) n++;
  }
  return n;
}
