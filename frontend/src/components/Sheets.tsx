import { useEffect, useMemo, useRef, useState } from "react";
import { CloseIcon } from "./Icons";
import { defaultConfig, loadMoshafSchema, loadRuleCatalogue } from "../lib/moshaf";
import { searchAyahs } from "../lib/search";
import type {
  MistakeLog,
  MoshafConfig,
  MoshafField,
  RuleSelection,
  SearchResult,
  SuraInfo,
  TajweedRuleDef,
} from "../lib/types";

type Tab = "sura" | "keyword" | "meaning";

const TABS: { id: Tab; label: string; hint: string }[] = [
  { id: "sura", label: "سورة", hint: "ابحث عن سورة بالاسم أو الرقم…" },
  { id: "keyword", label: "كلمة", hint: "اكتب كلمات من الآية…" },
  { id: "meaning", label: "معنى", hint: "اكتب معنى أو موضوعًا…" },
];

/**
 * The way around the muṣḥaf: three searches behind one box.
 *
 *   سورة  — the local sūra index (name, transliteration, or number). No network.
 *   كلمة  — backend BM25 over the Uthmani words AND their roots, so a query noun finds
 *           the āyah's verb form. Debounced, ~10 ms, searches as you type.
 *   معنى  — backend hybrid search: embeddings for the meaning, fused with the same BM25
 *           for the wording. Runs on submit, NOT per keystroke — the first query on a cold
 *           server pays a ~10 s model load, and every one after is far dearer than BM25.
 *           Its «توسيع بالذكاء الاصطناعي» switch turns on HyDE, which rewrites the query
 *           through an LLM before embedding it (helps short, ambiguous words like الغيبة).
 *
 * The backend also exposes a pure `vector` mode. It is not a tab: on Arabic it is strictly
 * worse than hybrid and on English it IS hybrid, so offering it would be a choice with no
 * right answer. The API keeps it for evaluation.
 */
export function SearchSheet({
  suras,
  onPick,
  onPickAyah,
  onClose,
}: {
  suras: SuraInfo[];
  onPick: (sura: SuraInfo) => void;
  onPickAyah: (sura: number, aya: number) => void;
  onClose: () => void;
}) {
  const [mode, setMode] = useState<Tab>("sura");
  const [q, setQ] = useState("");
  const [hyde, setHyde] = useState(false);
  const [result, setResult] = useState<SearchResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(false);
  // Set when the reciter hits enter on معنى; ignored by the other two modes.
  const [submitted, setSubmitted] = useState("");
  const abort = useRef<AbortController | null>(null);

  const suraHits = useMemo(() => {
    const needle = q.trim().toLowerCase();
    if (!needle) return suras;
    // Match the Arabic name, the transliteration, or the number — a reciter reaches
    // for whichever they know.
    const bare = stripHarakat(needle);
    return suras.filter(
      (s) =>
        stripHarakat(s.name_ar).includes(bare) ||
        s.name_en.toLowerCase().includes(needle) ||
        String(s.sura) === needle,
    );
  }, [q, suras]);

  // The one network effect for both āyah modes. `keyword` keys off the live query (with a
  // debounce), `meaning` off the submitted one — so switching modes or typing cancels
  // whatever request is in flight and the last one to be asked for is the one displayed.
  // `hyde` is a dependency too: toggling it re-runs the query it applies to.
  const term = mode === "keyword" ? q.trim() : mode === "meaning" ? submitted : "";
  useEffect(() => {
    abort.current?.abort();
    if (!term) {
      setResult(null);
      setBusy(false);
      setError(false);
      return;
    }
    const ctl = new AbortController();
    abort.current = ctl;
    setBusy(true);
    setError(false);
    const go = () =>
      searchAyahs(term, mode === "keyword" ? "keyword" : "hybrid", {
        hyde: mode === "meaning" && hyde,
        signal: ctl.signal,
      })
        .then((r) => {
          setResult(r);
          setBusy(false);
        })
        .catch((e) => {
          if (ctl.signal.aborted) return; // superseded, not failed
          setError(true);
          setBusy(false);
          console.error(e);
        });
    // Debounce typing; a submitted meaning query fires at once.
    const t = setTimeout(go, mode === "keyword" ? 300 : 0);
    return () => {
      clearTimeout(t);
      ctl.abort();
    };
  }, [term, mode, hyde]);

  const tab = TABS.find((t) => t.id === mode)!;

  return (
    <div className="sheet" role="dialog" aria-label="البحث">
      <div className="sheet__bar">
        <button className="iconbtn" onClick={onClose} title="إغلاق">
          <CloseIcon />
        </button>
        <form
          style={{ display: "contents" }}
          onSubmit={(e) => {
            e.preventDefault();
            if (mode === "meaning") setSubmitted(q.trim());
          }}
        >
          <input
            className="search"
            autoFocus
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder={tab.hint}
            aria-label={tab.hint}
            enterKeyHint="search"
          />
        </form>
      </div>

      <div className="searchtabs" role="tablist" aria-label="نوع البحث">
        {TABS.map((t) => (
          <button
            key={t.id}
            role="tab"
            aria-selected={mode === t.id}
            className={`moshaf__opt${mode === t.id ? " moshaf__opt--on" : ""}`}
            onClick={() => setMode(t.id)}
          >
            {t.label}
          </button>
        ))}
        {/* HyDE belongs to the meaning search only — it steers what gets embedded, and
            keyword search embeds nothing. Shown where it applies, not in a settings panel
            three taps away. */}
        {mode === "meaning" && (
          <label className="searchtabs__toggle" title="يعيد صياغة سؤالك عبر نموذج لغوي قبل البحث — أدقّ مع الكلمات القصيرة الملتبسة، وأبطأ قليلًا">
            <input type="checkbox" checked={hyde} onChange={(e) => setHyde(e.target.checked)} />
            توسيع بالذكاء الاصطناعي
          </label>
        )}
      </div>

      <div className="sheet__body">
        {mode === "sura" ? (
          <>
            {suraHits.map((s) => (
              <button key={s.sura} className="surarow" onClick={() => onPick(s)}>
                <span className="surarow__no">{s.sura}</span>
                <span>
                  <div className="surarow__name">{s.name_ar}</div>
                  <div className="surarow__meta">
                    {s.name_en} · {s.ayat} آية ·{" "}
                    {s.revelation === "makkah" ? "مكية" : "مدنية"}
                  </div>
                </span>
                <span className="surarow__spacer" />
                <span className="surarow__meta">ص {s.page}</span>
              </button>
            ))}
            {!suraHits.length && <p className="empty">لا سورة بهذا الاسم.</p>}
          </>
        ) : error ? (
          <p className="empty">تعذّر البحث — تأكّد من تشغيل الخدمة.</p>
        ) : busy ? (
          <p className="empty">
            يبحث…
            {mode === "meaning" && (
              <>
                <br />
                قد يستغرق البحث الأول لحظات لتحميل النموذج.
              </>
            )}
          </p>
        ) : !term ? (
          <p className="empty">
            {mode === "keyword"
              ? "اكتب كلمة أو أكثر من الآية — يبحث في رسم الكلمات وجذورها."
              : "اكتب معنى أو موضوعًا بالعربية أو الإنجليزية، ثم اضغط Enter."}
          </p>
        ) : !result?.hits.length ? (
          <p className="empty">لا نتائج.</p>
        ) : (
          <>
            {/* The reciter asked for the expansion and did not get it (no key configured,
                or the provider was down). Search still answered — say which one answered
                rather than letting a silently different result look like the same one. */}
            {hyde && mode === "meaning" && !result.hydeUsed && (
              <p className="searchnote">تعذّر التوسيع بالذكاء الاصطناعي — نتائج البحث المعتاد.</p>
            )}
            {result.hits.map((h) => (
            <button
              key={`${h.sura}:${h.aya}`}
              className="ayahrow"
              onClick={() => onPickAyah(h.sura, h.aya)}
            >
              <div className="ayahrow__text">{h.text_uthmani}</div>
              <div className="ayahrow__meta">
                <span>
                  {suras.find((s) => s.sura === h.sura)?.name_ar ?? h.sura} · {h.sura}:{h.aya}
                </span>
                {h.translation && <span className="ayahrow__tr">{h.translation}</span>}
              </div>
            </button>
            ))}
          </>
        )}
      </div>
    </div>
  );
}

const stripHarakat = (s: string) => s.replace(/[ً-ْـٰ]/g, "");

/** The session's mistake log — the "past mistakes" the brain icon opens. */
export function MistakesSheet({
  log,
  suras,
  onClose,
}: {
  log: MistakeLog[];
  suras: SuraInfo[];
  onClose: () => void;
}) {
  const name = (n: number) => suras.find((s) => s.sura === n)?.name_ar ?? n;
  const errors = log.filter((m) => m.status === "error");

  return (
    <div className="sheet" role="dialog" aria-label="الأخطاء">
      <div className="sheet__bar">
        <button className="iconbtn" onClick={onClose} title="إغلاق">
          <CloseIcon />
        </button>
        <span className="sheet__title">أخطاء الجلسة</span>
        <span className="surarow__spacer" />
        <span className="surarow__meta">
          {errors.length} خطأ · {log.length - errors.length} تنبيه
        </span>
      </div>
      <div className="sheet__body">
        {log.length === 0 ? (
          <p className="empty">
            لا أخطاء بعد.
            <br />
            ابدأ التلاوة وسيظهر هنا كل ما تُنبَّه عليه.
          </p>
        ) : (
          [...log].reverse().map((m, i) => (
            <div className="mistake" key={`${m.sura}:${m.aya}:${m.word_idx}:${i}`}>
              <span className="mistake__word">{m.uthmani}</span>
              <span className="mistake__where">
                {name(m.sura)} {m.aya}:{m.word_idx + 1}
              </span>
              <span className={`mistake__kind mistake__kind--${m.status}`}>
                {m.status === "error" ? m.rules[0] ?? kindLabel(m.kinds[0]) : "غير مؤكَّد"}
              </span>
            </div>
          ))
        )}
        {/* `almost` is a hint, not a mistake — say so where it is being counted. */}
        {log.some((m) => m.status === "almost") && (
          <p className="empty" style={{ padding: "1.25rem 0.5rem 0", textAlign: "start" }}>
            «غير مؤكَّد» ليس خطأً: لم يسمعك النموذج بوضوح كافٍ ليحكم.
          </p>
        )}
      </div>
    </div>
  );
}

const kindLabel = (k?: string) =>
  k === "tajweed" ? "تجويد" : k === "tashkeel" ? "تشكيل" : k === "sifa" ? "صفة" : "نطق";

/**
 * The reciter's moshaf attributes — madd holding lengths and the rule choices that
 * change what a correct recitation sounds like, so scoring matches how THIS reciter
 * reads. Generated from the backend schema; a change applies to the next session.
 */
export function MoshafSheet({
  value,
  rules,
  onSave,
  onClose,
}: {
  value: MoshafConfig | null;
  /** The leniency selection; null = every rule graded. */
  rules: RuleSelection;
  onSave: (cfg: MoshafConfig | null, rules: RuleSelection) => void;
  onClose: () => void;
}) {
  const [fields, setFields] = useState<MoshafField[] | null>(null);
  const [error, setError] = useState(false);
  const [cfg, setCfg] = useState<MoshafConfig>({});
  const [catalogue, setCatalogue] = useState<TajweedRuleDef[] | null>(null);
  const [sel, setSel] = useState<RuleSelection>(rules);

  useEffect(() => {
    let live = true;
    loadMoshafSchema()
      .then((fs) => {
        if (!live) return;
        setFields(fs);
        setCfg(value ?? defaultConfig(fs));
      })
      .catch(() => live && setError(true));
    // The rule catalogue is a SEPARATE concern from the schema: if it fails to load the
    // panel still works, the reciter just can't narrow the rules. Failing the whole
    // sheet over the optional half would be a worse trade than losing the new feature.
    loadRuleCatalogue()
      .then((rs) => live && setCatalogue(rs))
      .catch(() => {});
    return () => {
      live = false;
    };
  }, [value]);

  useEffect(() => setSel(rules), [rules]);

  const set = (key: string, v: string | number) => setCfg((c) => ({ ...c, [key]: v }));

  const toggleRule = (key: string) =>
    setSel((s) => {
      const cur = s ?? [];
      return cur.includes(key) ? cur.filter((k) => k !== key) : [...cur, key];
    });

  const apply = () => {
    onSave(cfg, sel);
    onClose();
  };
  const useModelDefault = () => {
    onSave(null, null); // no moshaf, no narrowing — the backend's own defaults
    onClose();
  };
  const resetShown = () => {
    if (fields) setCfg(defaultConfig(fields));
    setSel(null);
  };

  return (
    <div className="sheet" role="dialog" aria-label="إعدادات التلاوة">
      <div className="sheet__bar">
        <button className="iconbtn" onClick={onClose} title="إغلاق">
          <CloseIcon />
        </button>
        <span className="sheet__title">خصائص المصحف والتلاوة</span>
        <span className="surarow__spacer" />
        {fields && (
          <button className="moshaf__link" onClick={resetShown}>
            القيم الافتراضية
          </button>
        )}
      </div>

      <div className="sheet__body">
        {error ? (
          <p className="empty">تعذّر تحميل الإعدادات — تأكّد من تشغيل خدمة التلاوة.</p>
        ) : !fields ? (
          <p className="empty">يُحمّل…</p>
        ) : (
          <>
            <p className="moshaf__hint">
              تُطبَّق هذه الخصائص على الجلسة التالية، وتُضبط بها مطابقة التجويد لطريقة قراءتك.
            </p>
            {fields.map((f) => (
              <div className="moshaf__field" key={f.key}>
                <label className="moshaf__label" title={f.description ?? undefined}>
                  {f.name_ar}
                </label>
                <div className="moshaf__opts" role="group" aria-label={f.name_ar}>
                  {f.options.map((o) => (
                    <button
                      key={String(o.value)}
                      className={`moshaf__opt${cfg[f.key] === o.value ? " moshaf__opt--on" : ""}`}
                      aria-pressed={cfg[f.key] === o.value}
                      onClick={() => set(f.key, o.value)}
                    >
                      {o.label}
                    </button>
                  ))}
                </div>
              </div>
            ))}

            {/* Leniency. A learner drilling madd al-aared does not want to be told
                about qalqalah; narrowing the rules keeps the session about the thing
                they came to practise. Hifz and tashkeel stay graded whatever is
                chosen here — that is said on screen, not just in the API docs, because
                a reciter who thinks they turned off ALL checking would trust a green
                word they should not. */}
            {catalogue && (
              <div className="moshaf__field">
                <label className="moshaf__label">أحكام التجويد المطلوبة</label>
                <div className="moshaf__opts" role="group" aria-label="أحكام التجويد المطلوبة">
                  <button
                    className={`moshaf__opt${sel === null ? " moshaf__opt--on" : ""}`}
                    aria-pressed={sel === null}
                    onClick={() => setSel(sel === null ? [] : null)}
                  >
                    كل الأحكام
                  </button>
                </div>
                {sel !== null && (
                  <>
                    <div className="moshaf__opts" role="group" aria-label="اختر الأحكام">
                      {catalogue.map((r) => (
                        <button
                          key={r.key}
                          className={`moshaf__opt${sel.includes(r.key) ? " moshaf__opt--on" : ""}`}
                          aria-pressed={sel.includes(r.key)}
                          title={r.name_en}
                          onClick={() => toggleRule(r.key)}
                        >
                          {r.name_ar}
                        </button>
                      ))}
                    </div>
                    <p className="moshaf__hint">
                      {sel.length === 0
                        ? "لم تختر حكمًا — لن يُنبَّه على التجويد، ويبقى الحفظ والتشكيل مُقيَّمين."
                        : "يُنبَّه على ما اخترته فقط. الحفظ والتشكيل مُقيَّمان دائمًا."}
                    </p>
                  </>
                )}
              </div>
            )}
          </>
        )}
      </div>

      {fields && !error && (
        <div className="sheet__foot">
          <button className="moshaf__ghost" onClick={useModelDefault}>
            الوضع الافتراضي
          </button>
          <button className="moshaf__save" onClick={apply}>
            حفظ
          </button>
        </div>
      )}
    </div>
  );
}
