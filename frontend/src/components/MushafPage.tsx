import { memo, useState } from "react";
import { fontFamilyFor, wordKey } from "../lib/mushaf";
import type { MarkState } from "../lib/marks";
import type {
  FeedbackError,
  MushafLine,
  MushafPage as PageData,
  MushafWord,
  Span,
  SuraInfo,
  WordFeedback,
} from "../lib/types";

type Props = {
  page: PageData;
  suras: SuraInfo[];
  marks: MarkState;
  /** Where the reciter is. The word here is about to be recited, not yet judged. */
  cursor: Span | null;
  /** Recite from memory: words are veiled until reached or revealed. */
  hidden: boolean;
  /** Words revealed by hand via the arrow controls. */
  revealed: Set<string>;
  showMarks: boolean;
  onWord: (w: MushafWord) => void;
  /**
   * The engine that produced `marks` (the session ack's engine, not just what was
   * requested). Zipformer never has sifat/tajweed-rule data to name, so its tooltip
   * shows a bare recited-vs-expected phonetic comparison instead — see `tooltip()`.
   */
  engine?: string;
};

const nameOf = (suras: SuraInfo[], n?: number) =>
  suras.find((s) => s.sura === n)?.name_ar ?? "";

export const MushafPage = memo(function MushafPage({
  page,
  suras,
  marks,
  cursor,
  hidden,
  revealed,
  showMarks,
  onWord,
  engine,
}: Props) {
  const framed = page.page === 1 || page.page === 2;
  const family = fontFamilyFor(page.page);

  return (
    <div
      className={`page${framed ? " page--framed" : ""}${showMarks ? "" : " marks-off"}`}
    >
      {page.lines.map((line) => (
        <Line
          key={line.line}
          line={line}
          family={family}
          suras={suras}
          marks={marks}
          cursor={cursor}
          hidden={hidden}
          revealed={revealed}
          onWord={onWord}
          engine={engine}
        />
      ))}
    </div>
  );
});

function Line({
  line,
  family,
  suras,
  marks,
  cursor,
  hidden,
  revealed,
  onWord,
  engine,
}: Omit<Props, "page" | "showMarks"> & { line: MushafLine; family: string }) {
  if (line.type === "surah_name") {
    return (
      <>
        <div className="surah-head">
          <span className="surah-head__rule" />
          <span>سورة {nameOf(suras, line.surah)}</span>
          <span className="surah-head__rule" style={{ ["--dir" as string]: "right" }} />
        </div>
        {/* Sūras 82, 86 and 91 get one line for both ornaments: this edition ran the
            previous sūra's text into the line their header would have used. */}
        {line.with_basmalah && <Basmalah />}
      </>
    );
  }

  if (line.type === "basmallah") return <Basmalah />;

  return (
    <div className={`line${line.centered ? " line--centered" : ""}`}>
      {line.words.map((w, i) => (
        <Word
          key={`${w.sura}:${w.aya}:${w.word_idxs.join(",") || `m${i}`}`}
          word={w}
          family={family}
          marks={marks}
          cursor={cursor}
          hidden={hidden}
          revealed={revealed}
          onWord={onWord}
          engine={engine}
        />
      ))}
    </div>
  );
}

/** Rendered in the page's own font, so it matches the muṣḥaf rather than the UI. */
const Basmalah = () => (
  <div className="basmalah" style={{ fontFamily: "Thmanyah" }}>
    بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ
  </div>
);

function Word({
  word,
  family,
  marks,
  cursor,
  hidden,
  revealed,
  onWord,
  engine,
}: Omit<Props, "page" | "suras" | "showMarks"> & { word: MushafWord; family: string }) {
  const isMarker = word.word_idxs.length === 0;

  // A glyph can spell more than one Tanzil word (بَعْدَ مَا), so its state is the WORST
  // of the words it holds — a mistake inside a joined spelling must still show.
  const keys = word.word_idxs.map((i) => wordKey(word.sura, word.aya, i));
  const mark = keys
    .map((k) => marks.marks.get(k))
    .reduce<string | undefined>(
      (worst, m) => (severity(m) > severity(worst) ? m : worst),
      undefined,
    );
  const detail = keys.map((k) => marks.detail.get(k)).find(Boolean);

  const atCursor =
    !!cursor &&
    cursor.sura === word.sura &&
    cursor.aya === word.aya &&
    word.word_idxs.includes(cursor.word_idx);

  const reached = keys.some((k) => marks.reached.has(k));
  const shown = keys.some((k) => revealed.has(k));
  const veiled = hidden && !isMarker && !reached && !shown;

  const kinds = new Set(detail?.errors.map((e) => e.error_type));

  const cls = [
    "word",
    isMarker && "word--marker",
    veiled && "word--veiled",
    atCursor && "word--cursor",
    mark && mark !== "pending" && mark !== "recited" && `word--${mark}`,
    kinds.has("tajweed") && "word--tajweed",
    kinds.has("tashkeel") && "word--tashkeel",
  ]
    .filter(Boolean)
    .join(" ");

  // The rich detail panel is worth showing only when we have something to say about the
  // word: a verdict, a boundary note, or at least a fault we can stand behind. Zipformer
  // has no sifat and no backed tajweed, so its only showable faults are phoneme-level
  // (normal/tashkeel) — mirror that here so we never open an empty panel for it.
  const [open, setOpen] = useState(false);
  const showable =
    engine === "zipformer"
      ? detail?.errors.filter((e) => e.error_type === "normal" || e.error_type === "tashkeel")
      : detail?.errors;
  const hasDetail = !veiled && !!detail && (detail.trimmed || (showable?.length ?? 0) > 0);

  return (
    <button
      type="button"
      className={cls}
      style={{ fontFamily: family }}
      onClick={() => onWord(word)}
      onMouseEnter={hasDetail ? () => setOpen(true) : undefined}
      onMouseLeave={hasDetail ? () => setOpen(false) : undefined}
      onFocus={hasDetail ? () => setOpen(true) : undefined}
      onBlur={hasDetail ? () => setOpen(false) : undefined}
      // A short plain-text summary stays as the native tooltip: the touch fallback and
      // the accessible name of the mistake. Engine-aware, like the rich panel below.
      title={veiled ? undefined : tooltip(word, detail, engine)}
      aria-label={veiled ? "كلمة مخفية" : word.uthmani || "نهاية الآية"}
    >
      {word.glyph}
      {open && detail && <WordDetail word={word} detail={detail} engine={engine} />}
    </button>
  );
}

const severity = (m?: string) =>
  m === "error"
    ? 3
    : m === "almost"
      ? 2
      : m === "unverified"
        ? 1
        : m === "recited"
          ? 0.5
          : m === "skipped"
            ? 0.45
            : m === "heard"
              ? 0.4
              : 0;

function tooltip(
  word: MushafWord,
  detail: import("../lib/types").WordFeedback | undefined,
  engine?: string,
): string {
  if (!detail) return word.uthmani;
  if (detail.trimmed) return `${word.uthmani} — لم تُقيَّم (على حدّ المقطع)`;
  if (!detail.errors.length) return word.uthmani;

  const hedge = detail.status === "almost" ? "ربما — " : "";

  if (engine === "zipformer") {
    // Zipformer never gets sifat (every attribute comes back None) and its
    // phoneme-level "tajweed" classification has nothing backing it either — so
    // unlike Muaalem below, this shows only the bare recited-vs-expected phonetics,
    // never a rule name it cannot actually support.
    const parts = detail.errors
      .filter((e) => e.error_type === "normal" || e.error_type === "tashkeel")
      .map(plainDiff);
    if (!parts.length) return word.uthmani;
    return `${word.uthmani} — ${hedge}${parts.join(" · ")}`;
  }

  const parts = detail.errors.map((e) => {
    if (e.error_type === "sifa") return describeSifa(e);
    const rules = e.tajweed_rules.map((r) => r.name_ar).join("، ");
    if (e.expected_len != null && e.predicted_len != null) {
      return `${rules || "مدّ"}: المتوقَّع ${e.expected_len}، قرأت ${e.predicted_len}`;
    }
    return rules || errorLabel(e.error_type);
  });
  return `${word.uthmani} — ${hedge}${parts.join(" · ")}`;
}

/** Recited vs. expected, in the phonetic script (which reuses Uthmani letters —
 *  see quran_transcript's alphabet — so it reads as Arabic, not IPA). Used by the
 *  Zipformer tooltip, which has only phonetics to show. */
function plainDiff(e: FeedbackError): string {
  const label = e.error_type === "tashkeel" ? "التشكيل" : "النطق";
  if (!e.predicted_ph) return `${label}: سقطت «${e.expected_ph}»`;
  if (!e.expected_ph) return `${label}: زيادة «${e.predicted_ph}»`;
  return `${label}: قرأت «${e.predicted_ph}» بدل «${e.expected_ph}»`;
}

const errorLabel = (t: string) =>
  t === "tajweed" ? "تجويد" : t === "tashkeel" ? "تشكيل" : t === "sifa" ? "صفة" : t;

/**
 * A sifa (articulation) error arrives as `attr=value` in `expected_ph`/`predicted_ph`
 * with no tajweed rule, so the old tooltip could only say "sifa". Decode it into the
 * صفة by name and say which way it was read: "القلقلة: المتوقَّع مُقلقَل، نُطقت غير مُقلقَل".
 */
function describeSifa(e: import("../lib/types").FeedbackError): string {
  const [attr, exp] = (e.expected_ph ?? "").split("=");
  const pred = (e.predicted_ph ?? "").split("=")[1];
  const info = SIFA[attr];
  const name = info?.name ?? attr ?? "صفة";
  const expL = info?.values[exp] ?? exp;
  const predL = info?.values[pred] ?? pred;
  if (!expL || !predL) return name;
  return `${name}: المتوقَّع ${expL}، نُطقت ${predL}`;
}

// The ten articulation attributes the model emits, in Arabic. Names and value labels
// mirror quran_transcript/phonetics/sifa.py (SifaOutput) — the source of these strings.
const SIFA: Record<string, { name: string; values: Record<string, string> }> = {
  hams_or_jahr: { name: "الهمس والجهر", values: { hams: "همس", jahr: "جهر" } },
  shidda_or_rakhawa: {
    name: "الشدّة والرخاوة",
    values: { shadeed: "شدّة", between: "بَينيّة", rikhw: "رخاوة" },
  },
  tafkheem_or_taqeeq: {
    name: "التفخيم والترقيق",
    values: { mofakham: "تفخيم", moraqaq: "ترقيق", low_mofakham: "تفخيم أدنى" },
  },
  itbaq: { name: "الإطباق", values: { motbaq: "إطباق", monfateh: "انفتاح" } },
  safeer: { name: "الصفير", values: { safeer: "صفير", no_safeer: "بلا صفير" } },
  qalqla: { name: "القلقلة", values: { moqalqal: "قلقلة", not_moqalqal: "بلا قلقلة" } },
  tikraar: { name: "التكرار", values: { mokarar: "تكرار", not_mokarar: "بلا تكرار" } },
  tafashie: { name: "التفشّي", values: { motafashie: "تفشٍّ", not_motafashie: "بلا تفشٍّ" } },
  istitala: {
    name: "الاستطالة",
    values: { mostateel: "استطالة", not_mostateel: "بلا استطالة" },
  },
  ghonna: { name: "الغنّة", values: { maghnoon: "غنّة", not_maghnoon: "بلا غنّة" } },
};

// --- The rich hover panel ----------------------------------------------------

const TYPE_AR: Record<string, string> = {
  tajweed: "تجويد",
  tashkeel: "تشكيل",
  sifa: "صفة",
  normal: "نُطق",
};

// What the reciter DID to the letter, not what rule it broke.
const SPEECH_AR: Record<string, string> = {
  insert: "زيادة",
  delete: "نقص",
  replace: "إبدال",
};

/** One error, decoded into a title line and labelled detail rows for the panel. */
function describeError(e: FeedbackError) {
  const rows: { k: string; v: string }[] = [];
  let title: string;

  if (e.error_type === "sifa") {
    title = describeSifa(e);
  } else if (e.expected_len != null && e.predicted_len != null) {
    const rule = e.tajweed_rules.map((r) => r.name_ar).join("، ");
    title = rule || "المدّ";
    rows.push({ k: "الحركات", v: `المتوقَّع ${e.expected_len} · قرأت ${e.predicted_len}` });
  } else {
    title = e.tajweed_rules.map((r) => r.name_ar).join("، ") || errorLabel(e.error_type);
  }

  if (SPEECH_AR[e.speech_error_type]) rows.push({ k: "النوع", v: SPEECH_AR[e.speech_error_type] });
  if (e.expected_ph && e.error_type !== "sifa") {
    rows.push({ k: "المتوقَّع", v: e.expected_ph });
    if (e.predicted_ph) rows.push({ k: "قرأت", v: e.predicted_ph });
  }
  if (e.tajweed_rules.length && e.error_type !== "tajweed") {
    rows.push({ k: "قاعدة", v: e.tajweed_rules.map((r) => r.name_ar).join("، ") });
  }
  return { title, rows, type: e.error_type, confidence: e.confidence };
}

/** Confidence 0–1 → a percent + a word. `null` means we never scored it, not "sure".*/
function confidenceLabel(c: number | null): { text: string; pct: number | null } {
  if (c == null) return { text: "غير مُقيَّم", pct: null };
  return { text: `${Math.round(c * 100)}٪`, pct: Math.round(c * 100) };
}

/**
 * The rich box shown on hover: every fault on the word, stacked, each with its kind,
 * how it was misread, expected vs read, and the model's confidence. Anchored to the
 * word; the muṣḥaf is RTL so it reads right-to-left like the text under it.
 */
function WordDetail({
  word,
  detail,
  engine,
}: {
  word: MushafWord;
  detail: WordFeedback;
  engine?: string;
}) {
  const statusCls =
    detail.status === "error" ? "worddetail--error" : detail.status === "almost" ? "worddetail--almost" : "";

  // Zipformer has no sifat and its tajweed/madd classification is unbacked, so keep the
  // panel to the phoneme-level faults it can actually stand behind — the same restraint
  // as its plain tooltip. Muaalem shows everything.
  const errors =
    engine === "zipformer"
      ? detail.errors.filter((e) => e.error_type === "normal" || e.error_type === "tashkeel")
      : detail.errors;

  return (
    <div className={`worddetail ${statusCls}`} role="tooltip" dir="rtl">
      <div className="worddetail__head">
        <span className="worddetail__word" style={{ fontFamily: "Thmanyah" }}>
          {word.uthmani}
        </span>
        <span className="worddetail__status">
          {detail.trimmed
            ? "لم تُقيَّم"
            : detail.status === "error"
              ? "خطأ"
              : detail.status === "almost"
                ? "غير مؤكَّد"
                : "صحيح"}
        </span>
      </div>

      {detail.trimmed ? (
        <p className="worddetail__note">لم تُقيَّم هذه الكلمة — وقعت على حدّ المقطع الصوتيّ.</p>
      ) : (
        <ul className="worddetail__list">
          {errors.map((e, i) => {
            const d = describeError(e);
            const conf = confidenceLabel(d.confidence);
            return (
              <li key={i} className="worddetail__item">
                <div className="worddetail__row">
                  <span className={`worddetail__badge worddetail__badge--${d.type}`}>
                    {TYPE_AR[d.type] ?? d.type}
                  </span>
                  <span className="worddetail__title">{d.title}</span>
                </div>
                {d.rows.map((r, j) => (
                  <div key={j} className="worddetail__kv">
                    <span className="worddetail__k">{r.k}</span>
                    <span className="worddetail__v">{r.v}</span>
                  </div>
                ))}
                <div className="worddetail__conf">
                  <span className="worddetail__k">الثقة</span>
                  {conf.pct == null ? (
                    <span className="worddetail__v">{conf.text}</span>
                  ) : (
                    <span className="worddetail__bar" aria-label={conf.text}>
                      <span style={{ width: `${conf.pct}%` }} />
                      <em>{conf.text}</em>
                    </span>
                  )}
                </div>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
