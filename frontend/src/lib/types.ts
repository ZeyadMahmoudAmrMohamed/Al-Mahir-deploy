// The contracts this app speaks. Backend types mirror tajwid.feedback.types and
// tajwid.session; muṣḥaf types mirror scripts/build_mushaf_data.py's output.

// --- The muṣḥaf page data -----------------------------------------------------

export type MushafWord = {
  sura: number;
  aya: number;
  /**
   * The backend's coordinates for this glyph: 0-based indices into the āyah's Tanzil
   * Uthmani words. A LIST, not a number — in 4 āyāt one glyph spells two words
   * (بَعْدَ مَا, إِلْ يَاسِينَ). Empty for an āyah-end marker.
   */
  word_idxs: number[];
  glyph: string;
  uthmani: string;
};

export type MushafLine = {
  line: number;
  type: "ayah" | "surah_name" | "basmallah";
  centered: boolean;
  words: MushafWord[];
  surah?: number;
  /** The header shares its line with the basmalah (sūras 82, 86, 91). */
  with_basmalah?: boolean;
};

export type MushafPage = {
  page: number;
  lines: MushafLine[];
  suras: number[];
};

export type SuraInfo = {
  sura: number;
  name_ar: string;
  name_en: string;
  ayat: number;
  page: number;
  revelation: string;
};

/** Retrieval mode. The API also accepts "vector"; the UI doesn't offer it (see search.ts). */
export type SearchMode = "keyword" | "hybrid";

export type SearchResult = {
  hits: AyahHit[];
  // What the backend actually ran — "hybrid" degrades to "vector" on English, and
  // hydeUsed is false if the LLM was unavailable. Shown to the reciter, not swallowed.
  mode: string;
  hydeUsed: boolean;
};

/** One āyah returned by GET /api/search. */
export type AyahHit = {
  sura: number;
  aya: number;
  text_uthmani: string;
  translation: string | null;
  // Cosine (semantic) or BM25 (keyword) — raw, uncalibrated, so we rank by it and
  // deliberately do not show it as a percentage.
  score: number;
};

// --- The backend's feedback ---------------------------------------------------

export type Span = { sura: number; aya: number; word_idx: number };

export type TajweedRule = {
  name_ar: string;
  name_en: string;
  golden_len: number;
  correctness_type: "match" | "count";
  tag: string | null;
};

export type FeedbackError = {
  error_type: "tajweed" | "normal" | "tashkeel" | "sifa";
  speech_error_type: "insert" | "delete" | "replace";
  expected_ph: string;
  predicted_ph: string;
  expected_len: number | null;
  predicted_len: number | null;
  tajweed_rules: TajweedRule[];
  /** null means UNSCORED, not certain. */
  confidence: number | null;
};

export type WordFeedback = {
  sura: number;
  aya: number;
  word_idx: number;
  uthmani: string;
  status: "correct" | "almost" | "error";
  errors: FeedbackError[];
  /**
   * The word sat on a chunk boundary and was NOT scored. Unverified, not correct —
   * render it neutrally, never with a tick.
   */
  trimmed: boolean;
};

export type FeedbackResponse = {
  status: "ok" | "ambiguous" | "no_match";
  span: Span | null;
  end: Span | null;
  uthmani_text: string | null;
  predicted_phonemes: string;
  reference_phonemes: string;
  words: WordFeedback[];
  candidates: { sura: number; aya: number; word_idx: number; uthmani_text: string }[];
  non_verse: string[];
};

export type FeedbackEvent = {
  type: "feedback";
  chunk_seq: number;
  audio_span_sec: [number, number];
  forced_cut: boolean;
  phonemes: string;
  feedback: FeedbackResponse;
  cursor: Span | null;
};

/**
 * Tier 1 live word-fill. Provisional and forward-only: coordinates only, never a
 * verdict. `confirmed` = words heard so far; `skipped` = words the reciter passed over
 * (asserted only once a later word is confirmed). Reconciled by the next `feedback`.
 */
export type ProgressEvent = {
  type: "progress";
  confirmed: Span[];
  skipped: Span[];
  cursor: Span | null;
};

export type SessionEvent =
  | {
      type: "session";
      session_id: string;
      engine: string;
      sample_rate: number;
      /** Whether this session is ACTUALLY being recorded (see backend capture.py).
       *  False when the reciter asked but the server has no capture directory — so
       *  the app can say so rather than believe it has a recording. */
      capture: boolean;
    }
  | FeedbackEvent
  | ProgressEvent
  | { type: "done" };

// --- ASR engine choice ----------------------------------------------------

/**
 * The two acoustic models a session can pick between (mirrors tajwid.asr.engine —
 * "mock" exists on the backend too, but it's a no-GPU dev fallback, not a user choice).
 */
export type EngineChoice = "real" | "zipformer" | "remote";

/** How confident the model must be before it accuses. Mirrors the backend's STRICTNESS
 *  table (feedback/confidence.py): lenient accuses least, strict accuses most. */
export type Strictness = "lenient" | "normal" | "strict";

export type HealthInfo = {
  status: string;
  engine: string;
  available_engines: string[];
  /** Whether this server was started with TAJWID_CAPTURE_DIR. */
  capture_available: boolean;
  device: string;
  dtype: string;
  muaalem_model: string;
  segmenter_model: string;
};

// --- What the UI derives ------------------------------------------------------

/**
 * How a word is painted. `pending` = not yet reached; `recited` = read, no fault.
 * `heard`/`skipped` are PROVISIONAL (Tier 1), always overridden by a real verdict.
 */
export type WordMark =
  | "pending"
  | "recited"
  | "almost"
  | "error"
  | "unverified"
  | "heard"
  | "skipped";

// --- Moshaf (recitation) attributes: the settings the reciter can tune ---------

export type MoshafOption = { value: string | number; label: string };

export type MoshafField = {
  key: string;
  name_ar: string;
  description: string | null;
  default: string | number;
  options: MoshafOption[];
};

/** The chosen value per attribute key — sent to the backend in the start message. */
export type MoshafConfig = Record<string, string | number>;

// --- Leniency: which tajwid rules this reciter wants graded ---------------------

/** One gradeable rule, from GET /api/tajweed-rules. */
export type TajweedRuleDef = {
  key: string;
  name_ar: string;
  name_en: string;
  /** Which channel it arrives on. The UI groups by it; the backend does the filtering. */
  kind: "tajweed" | "sifa";
};

/**
 * The rules to be graded on, sent as `rules` in the start message.
 *
 * `null` and `[]` are DIFFERENT and both are meaningful: `null` grades everything (the
 * default), `[]` grades no tajwid rule at all — hifz and tashkeel only. Anything that
 * touches this value must not collapse the two with a truthiness test.
 */
export type RuleSelection = string[] | null;

export type MistakeLog = {
  at: number;
  sura: number;
  aya: number;
  word_idx: number;
  uthmani: string;
  status: "almost" | "error";
  kinds: FeedbackError["error_type"][];
  rules: string[];
};
