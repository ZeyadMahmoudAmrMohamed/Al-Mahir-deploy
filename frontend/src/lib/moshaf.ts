import type {
  MoshafConfig,
  MoshafField,
  RuleSelection,
  Strictness,
  TajweedRuleDef,
} from "./types";

// The reciter's moshaf attributes: which madd lengths they hold, which rules they read.
// The schema is the backend's — introspected from MoshafAttributes — so this file never
// hard-codes the fields; it only fetches them and remembers the reciter's choices.

const KEY = "tajwid.moshaf";
const RULES_KEY = "tajwid.rules";

let schema: Promise<MoshafField[]> | null = null;
let rules: Promise<TajweedRuleDef[]> | null = null;

/** The attribute schema (fields, options, defaults), fetched once from the backend. */
export function loadMoshafSchema(): Promise<MoshafField[]> {
  if (!schema) {
    schema = fetch("/api/moshaf-schema")
      .then((r) => {
        if (!r.ok) throw new Error(`moshaf-schema: ${r.status}`);
        return r.json();
      })
      .then((d) => d.fields as MoshafField[])
      .catch((e) => {
        schema = null; // let a later open retry
        throw e;
      });
  }
  return schema;
}

/** Every attribute at its starting value — a COMPLETE config, so required fields hold. */
export function defaultConfig(fields: MoshafField[]): MoshafConfig {
  return Object.fromEntries(fields.map((f) => [f.key, f.default]));
}

export function loadMoshafConfig(): MoshafConfig | null {
  try {
    const s = localStorage.getItem(KEY);
    return s ? (JSON.parse(s) as MoshafConfig) : null;
  } catch {
    return null;
  }
}

export function saveMoshafConfig(cfg: MoshafConfig | null): void {
  try {
    if (cfg) localStorage.setItem(KEY, JSON.stringify(cfg));
    else localStorage.removeItem(KEY);
  } catch {
    /* private mode / disabled storage — the setting just won't persist */
  }
}

// --- Strictness ---------------------------------------------------------------
//
// How confident the model must be before an error is stated as one. Distinct from
// leniency below: leniency picks WHICH rules are graded, strictness picks how sure the
// grader must be to accuse on the rules it does grade. Sent as a top-level start field,
// NOT inside `moshaf` — it is not a moshaf attribute and the backend would reject it there.

const STRICTNESS_KEY = "tajwid.strictness";

export const STRICTNESS_LEVELS: { id: Strictness; label: string; hint: string }[] = [
  { id: "lenient", label: "متساهل", hint: "لا يُنبّه إلا على الخطأ الواضح" },
  { id: "normal", label: "متوسط", hint: "الموازنة الافتراضية" },
  { id: "strict", label: "صارم", hint: "ينبّه على أدقّ الملاحظات" },
];

export function loadStrictness(): Strictness | null {
  try {
    const v = localStorage.getItem(STRICTNESS_KEY);
    return STRICTNESS_LEVELS.some((l) => l.id === v) ? (v as Strictness) : null;
  } catch {
    return null;
  }
}

export function saveStrictness(s: Strictness | null): void {
  try {
    if (s) localStorage.setItem(STRICTNESS_KEY, s);
    else localStorage.removeItem(STRICTNESS_KEY);
  } catch {
    /* private mode / disabled storage — the setting just won't persist */
  }
}

// --- Leniency -----------------------------------------------------------------
//
// Which tajwid rules the reciter wants to be graded on. Like the moshaf schema, the
// catalogue is the backend's: it derives the keys from the rule classes the grader
// actually recognises, so a chip here can never name a rule nothing enforces.

/** The gradeable rules (tajwid + sifat), fetched once from the backend. */
export function loadRuleCatalogue(): Promise<TajweedRuleDef[]> {
  if (!rules) {
    rules = fetch("/api/tajweed-rules")
      .then((r) => {
        if (!r.ok) throw new Error(`tajweed-rules: ${r.status}`);
        return r.json();
      })
      .then((d) => d.rules as TajweedRuleDef[])
      .catch((e) => {
        rules = null; // let a later open retry
        throw e;
      });
  }
  return rules;
}

/** The saved selection, or null for "grade everything" — including on a parse failure. */
export function loadRuleSelection(): RuleSelection {
  try {
    const s = localStorage.getItem(RULES_KEY);
    if (!s) return null;
    const parsed = JSON.parse(s);
    // Guard the shape rather than trusting it: a corrupted entry that deserialised to
    // a non-array would otherwise be sent as `rules` and match nothing, silently
    // muting every tajwid rule the reciter had picked.
    return Array.isArray(parsed) ? (parsed as string[]) : null;
  } catch {
    return null;
  }
}

export function saveRuleSelection(sel: RuleSelection): void {
  try {
    // `[]` is a real selection and must be written, so this tests for null explicitly.
    if (sel === null) localStorage.removeItem(RULES_KEY);
    else localStorage.setItem(RULES_KEY, JSON.stringify(sel));
  } catch {
    /* private mode / disabled storage — the setting just won't persist */
  }
}
