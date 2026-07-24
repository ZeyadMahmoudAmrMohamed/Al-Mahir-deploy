import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { EnginePicker } from "./components/EnginePicker";
import { FeedbackBar } from "./components/FeedbackBar";
import {
  ChevronIcon,
  ChipIcon,
  MaddStroke,
  MicIcon,
  SearchIcon,
  SlidersIcon,
  SoundIcon,
  SoundOffIcon,
  StopIcon,
} from "./components/Icons";
import { MushafPage } from "./components/MushafPage";
import { MistakesSheet, MoshafSheet, SearchSheet } from "./components/Sheets";
import {
  loadMoshafConfig,
  loadRuleSelection,
  loadStrictness,
  saveMoshafConfig,
  saveRuleSelection,
  saveStrictness,
} from "./lib/moshaf";
import { cueMistake } from "./lib/cue";
import {
  labelFor,
  loadHealth,
  loadStoredEngineChoice,
  loadStoredLive,
  storeEngineChoice,
  storeLive,
} from "./lib/engines";
import { accuracy, applyFeedback, applyProgress, emptyMarks, mistakesOnPage } from "./lib/marks";
import { PAGES, loadPageReady, loadSuraIndex, pageOf, spanKey, wordKey } from "./lib/mushaf";
import { RecitationSession, type SessionStatus } from "./lib/session";
import type {
  EngineChoice,
  MushafPage as PageData,
  MushafWord,
  Span,
  SuraInfo,
} from "./lib/types";

export default function App() {
  const [suras, setSuras] = useState<SuraInfo[]>([]);
  const [page, setPage] = useState(1);
  const [data, setData] = useState<PageData | null>(null);

  const [marks, setMarks] = useState(emptyMarks);
  const [cursor, setCursor] = useState<Span | null>({ sura: 1, aya: 1, word_idx: 0 });
  const [status, setStatus] = useState<SessionStatus>("idle");
  const [level, setLevel] = useState(0);

  const [hidden, setHidden] = useState(false);
  const [showMarks, setShowMarks] = useState(true);
  const [sound, setSound] = useState(true);
  const [revealed, setRevealed] = useState<Set<string>>(new Set());
  const [sheet, setSheet] = useState<"index" | "mistakes" | "moshaf" | null>(null);
  const [toast, setToast] = useState<string | null>(null);
  const [moshaf, setMoshaf] = useState(loadMoshafConfig);
  // null = every rule graded (the default). See lib/types RuleSelection: `[]` differs.
  const [rules, setRules] = useState(loadRuleSelection);

  const [engineChoice, setEngineChoice] = useState<EngineChoice>(() => loadStoredEngineChoice() ?? "real");
  const [availableEngines, setAvailableEngines] = useState<Set<string> | null>(null);
  const [activeEngine, setActiveEngine] = useState<string | null>(null);
  const [enginePickerOpen, setEnginePickerOpen] = useState(false);

  // null = defer to the server's TAJWID_LIVE_FEEDBACK until the reciter expresses a
  // preference. Only ever turns the tier OFF; the server still gates it (see ws.py).
  const [liveMode, setLiveMode] = useState<boolean | null>(loadStoredLive);
  const [strictness, setStrictness] = useState(loadStrictness);

  const chooseEngine = useCallback((choice: EngineChoice) => {
    setEngineChoice(choice);
    storeEngineChoice(choice);
  }, []);

  const chooseLive = useCallback((on: boolean) => {
    setLiveMode(on);
    storeLive(on);
  }, []);

  const session = useRef<RecitationSession | null>(null);
  const soundRef = useRef(sound);
  soundRef.current = sound;

  useEffect(() => {
    loadSuraIndex().then(setSuras).catch(() => setToast("تعذّر تحميل فهرس السور."));
  }, []);

  useEffect(() => {
    loadHealth()
      .then((h) => {
        const avail = new Set(h.available_engines);
        setAvailableEngines(avail);
        // The stored/default pick might not exist on THIS server (no GPU, say) —
        // Zipformer is always built (see main.py's build_engines), so it's the one
        // safe fallback to land on rather than silently keep an unusable choice.
        setEngineChoice((current) => (avail.has(current) ? current : avail.has("zipformer") ? "zipformer" : current));
      })
      .catch(() => {}); // a nicety, not a requirement — start() still works unverified
  }, []);

  useEffect(() => {
    let live = true;
    loadPageReady(page)
      .then((d) => live && setData(d))
      .catch(() => live && setToast(`تعذّر تحميل الصفحة ${page}.`));
    return () => {
      live = false;
    };
  }, [page]);

  // The next page's font and data, fetched while the reciter is still on this one.
  // A page that arrives late is a page the reciter is already reciting from memory.
  useEffect(() => {
    if (page < PAGES) void loadPageReady(page + 1).catch(() => {});
  }, [page]);

  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(() => setToast(null), 4000);
    return () => clearTimeout(t);
  }, [toast]);

  // --- The live session ---------------------------------------------------

  const onFeedback = useCallback((event: Parameters<typeof applyFeedback>[1]) => {
    setMarks((prev) => {
      const next = applyFeedback(prev, event);
      // Sound only on a CONFIDENT new mistake — see lib/cue.
      if (soundRef.current && next.log.length > prev.log.length) {
        if (next.log.slice(prev.log.length).some((m) => m.status === "error")) {
          cueMistake();
        }
      }
      return next;
    });
    if (event.cursor) setCursor(event.cursor);
    if (event.feedback.status === "no_match") {
      setToast("لم نتعرّف على موضعك — تابع التلاوة أو اختر السورة.");
    }
  }, []);

  const onProgress = useCallback((event: import("./lib/types").ProgressEvent) => {
    setMarks((prev) => applyProgress(prev, event));
    // The live cursor is what makes the page follow the reciter and the current word
    // highlight move before the pause. Reconciled by the authoritative cursor at the waqf.
    if (event.cursor) setCursor(event.cursor);
  }, []);

  const start = useCallback(async () => {
    if (!cursor) return;
    const s = new RecitationSession(
      {
        onFeedback,
        onProgress,
        onLevel: setLevel,
        onState: setStatus,
        onError: setToast,
        onEngine: (engine) => {
          setActiveEngine(engine);
          // engine !== requested means api/ws.py couldn't honour the pick (not built
          // on this server, or unreachable) and silently fell back — tell the reciter,
          // since it changes what kind of feedback they're about to get.
          if (engine !== engineChoice) {
            setToast(`تعذّر تشغيل «${labelFor(engineChoice)}»؛ يعمل الآن بمحرك ${labelFor(engine)}.`);
          }
        },
      },
      moshaf,
      rules,
      strictness,
      liveMode,
    );
    session.current = s;
    await s.start(cursor, engineChoice);
  }, [cursor, onFeedback, onProgress, engineChoice, moshaf, rules, strictness, liveMode]);

  const stop = useCallback(async () => {
    await session.current?.stop();
    session.current = null;
    setLevel(0);
  }, []);

  useEffect(() => () => void session.current?.stop(), []);

  // Follow the reciter across page breaks. This is the whole point of tracking: they
  // recite on, and the muṣḥaf turns itself.
  useEffect(() => {
    if (!cursor || status !== "listening") return;
    let live = true;
    pageOf(cursor.sura, cursor.aya, cursor.word_idx).then((p) => {
      if (live && p && p !== page) setPage(p);
    });
    return () => {
      live = false;
    };
  }, [cursor, status, page]);

  // Turn the page. `dir` is +1 forward through the muṣḥaf, -1 back — a direction in the
  // Quran, not on the screen, so callers never have to reason about RTL.
  const turnPage = useCallback((dir: 1 | -1) => {
    setPage((p) => Math.min(PAGES, Math.max(1, p + dir)));
  }, []);

  // Keyboard: in an RTL muṣḥaf, forward is to the LEFT. PageUp/Down and Space work too,
  // so the page turns by whatever key the reader reaches for.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement) return; // typing in the search box
      if (e.key === "ArrowLeft" || e.key === "PageDown") turnPage(1);
      else if (e.key === "ArrowRight" || e.key === "PageUp") turnPage(-1);
      else return;
      e.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [turnPage]);

  // Swipe, like Tarteel: drag right-to-left to go forward. A horizontal drag that
  // outruns its vertical component is a page turn; anything more vertical is a scroll,
  // so a reader flicking down the page never turns it by accident.
  const swipe = useRef<{ x: number; y: number } | null>(null);
  const onPointerDown = useCallback((e: React.PointerEvent) => {
    if (e.pointerType === "mouse") return; // mouse users have the arrows + keyboard
    swipe.current = { x: e.clientX, y: e.clientY };
  }, []);
  const onPointerUp = useCallback(
    (e: React.PointerEvent) => {
      const s = swipe.current;
      swipe.current = null;
      if (!s) return;
      const dx = e.clientX - s.x;
      const dy = e.clientY - s.y;
      if (Math.abs(dx) < 60 || Math.abs(dx) < Math.abs(dy)) return;
      turnPage(dx < 0 ? 1 : -1); // finger moved right→left = forward
    },
    [turnPage],
  );

  // --- Deriving what the bar shows ----------------------------------------

  const pageKeys = useMemo(() => {
    const keys = new Set<string>();
    for (const line of data?.lines ?? []) {
      for (const w of line.words) {
        for (const i of w.word_idxs) keys.add(wordKey(w.sura, w.aya, i));
      }
    }
    return keys;
  }, [data]);

  const pageMistakes = useMemo(() => mistakesOnPage(marks, pageKeys), [marks, pageKeys]);
  const score = useMemo(() => accuracy(marks), [marks]);

  // The sūra label names the page you are LOOKING AT, not where the cursor was seeded.
  // While reciting the page follows the cursor, so page-derived is right in both cases;
  // a page holding two sūras shows the one it opens on.
  const here = useMemo(() => {
    const first = data?.lines.flatMap((l) => l.words).find((w) => w.word_idxs.length);
    const sura = suras.find((s) => s.sura === first?.sura);
    return sura?.name_ar ?? "";
  }, [data, suras]);

  // --- Reveal controls (hidden mode) --------------------------------------

  /** The words of the current āyah, in order, as keys. */
  const ayahKeys = useMemo(() => {
    if (!cursor || !data) return [];
    return data.lines
      .flatMap((l) => l.words)
      .filter((w) => w.sura === cursor.sura && w.aya === cursor.aya)
      .flatMap((w) => w.word_idxs.map((i) => ({ i, key: wordKey(w.sura, w.aya, i) })))
      .sort((a, b) => a.i - b.i);
  }, [cursor, data]);

  // Step the cursor onto the next āyah and reveal it. The reveal arrows do not stop at
  // the end of an āyah: finishing one moves you to the next, turning the page if it
  // begins on the following one. `whole` reveals the entire next āyah (double arrow)
  // when it is on the page we already hold; otherwise just its opening word.
  const advanceAyah = useCallback(
    (whole: boolean) => {
      if (!cursor) return;
      const info = suras.find((s) => s.sura === cursor.sura);
      const next =
        info && cursor.aya < info.ayat
          ? { sura: cursor.sura, aya: cursor.aya + 1, word_idx: 0 }
          : cursor.sura < 114
            ? { sura: cursor.sura + 1, aya: 1, word_idx: 0 }
            : null;
      if (!next) return;
      setCursor(next);
      session.current?.seek(next);
      setRevealed((r) => {
        const n = new Set(r).add(spanKey(next));
        if (whole && data) {
          for (const w of data.lines.flatMap((l) => l.words)) {
            if (w.sura === next.sura && w.aya === next.aya) {
              for (const i of w.word_idxs) n.add(wordKey(next.sura, next.aya, i));
            }
          }
        }
        return n;
      });
      void pageOf(next.sura, next.aya, 0).then((p) => p && setPage(p));
    },
    [cursor, suras, data],
  );

  const nextWord = useCallback(() => {
    const next = ayahKeys.find((w) => !revealed.has(w.key) && !marks.reached.has(w.key));
    if (next) setRevealed((r) => new Set(r).add(next.key));
    else advanceAyah(false); // āyah done — carry the reader onto the next one
  }, [ayahKeys, revealed, marks.reached, advanceAyah]);

  const wholeAyah = useCallback(() => {
    const allShown =
      ayahKeys.length > 0 &&
      ayahKeys.every((w) => revealed.has(w.key) || marks.reached.has(w.key));
    if (allShown) {
      advanceAyah(true);
      return;
    }
    setRevealed((r) => {
      const next = new Set(r);
      for (const w of ayahKeys) next.add(w.key);
      return next;
    });
  }, [ayahKeys, revealed, marks.reached, advanceAyah]);

  const pickSura = useCallback(
    async (s: SuraInfo) => {
      setSheet(null);
      setPage(s.page);
      const at = { sura: s.sura, aya: 1, word_idx: 0 };
      setCursor(at);
      // Tell the tracker outright rather than letting it hunt: it is why the app knows
      // where the reciter is before they make a sound.
      session.current?.seek(at);
    },
    [],
  );

  /** A search hit: open the page that holds this āyah and seed the tracker on its first
   * word, so the reciter can start reciting from the result they just found. */
  const pickAyah = useCallback(async (sura: number, aya: number) => {
    setSheet(null);
    const at = { sura, aya, word_idx: 0 };
    setCursor(at);
    session.current?.seek(at);
    const p = await pageOf(sura, aya, 0);
    if (p) setPage(p);
  }, []);

  /** Tapping a word repositions the session there. */
  const onWord = useCallback((w: MushafWord) => {
    if (!w.word_idxs.length) return;
    const at = { sura: w.sura, aya: w.aya, word_idx: w.word_idxs[0] };
    setCursor(at);
    session.current?.seek(at);
  }, []);

  const live = status === "listening";

  return (
    <div className="app">
      <header className="topbar">
        <button className="iconbtn" onClick={() => setSheet("index")} title="الفهرس والبحث">
          <SearchIcon />
        </button>

        <div className="topbar__where">
          <span className="topbar__sura">{here}</span>
          <span className="topbar__page">صفحة {page} / {PAGES}</span>
        </div>

        <div className="topbar__actions">
          <button
            className="iconbtn"
            onClick={() => setEnginePickerOpen((o) => !o)}
            aria-pressed={enginePickerOpen}
            aria-haspopup="menu"
            title={`محرك التعرّف الصوتي: ${labelFor(activeEngine ?? engineChoice)}`}
          >
            <ChipIcon />
          </button>
          <button
            className={`iconbtn${moshaf || rules ? " iconbtn--set" : ""}`}
            onClick={() => setSheet("moshaf")}
            title={
              rules
                ? `خصائص المصحف والتلاوة — التقييم مقصور على ${rules.length} حكمًا`
                : "خصائص المصحف والتلاوة"
            }
          >
            <SlidersIcon />
          </button>
          <button
            className="iconbtn"
            onClick={() => setSound((s) => !s)}
            aria-pressed={sound}
            title={sound ? "كتم الصوت" : "تشغيل الصوت"}
          >
            {sound ? <SoundIcon /> : <SoundOffIcon />}
          </button>
        </div>
      </header>

      {enginePickerOpen && (
        <EnginePicker
          choice={engineChoice}
          onChoose={chooseEngine}
          available={availableEngines}
          live={liveMode ?? true}
          onLiveChange={chooseLive}
          locked={status !== "idle"}
          onClose={() => setEnginePickerOpen(false)}
        />
      )}

      <main className="pagewrap" onPointerDown={onPointerDown} onPointerUp={onPointerUp}>
        {data && suras.length ? (
          <>
            <MushafPage
              page={data}
              suras={suras}
              marks={marks}
              cursor={cursor}
              hidden={hidden}
              revealed={revealed}
              showMarks={showMarks}
              onWord={onWord}
              engine={activeEngine ?? undefined}
            />
            <Legend />
            <Credit />
          </>
        ) : (
          <div className="loading">
            <MaddStroke level={0.6} />
            <span>يُفتح المصحف…</span>
          </div>
        )}
      </main>

      {/* Page turning stays manual as well as automatic: a reciter reviewing a page
          they just finished should not have to out-wait the tracker. Arrows, plus the
          keyboard and swipe handlers above. */}
      <PageNav page={page} onTurn={turnPage} disabled={live} />

      <button
        className={`fab${live ? " fab--live" : ""}${status === "connecting" ? " fab--connecting" : ""}`}
        style={{ ["--level" as string]: level }}
        onClick={() => (live ? void stop() : void start())}
        disabled={status === "closing"}
        title={live ? "إيقاف التلاوة" : "ابدأ التلاوة"}
        aria-label={live ? "إيقاف التلاوة" : "ابدأ التلاوة"}
      >
        {live ? <StopIcon size={22} /> : <MicIcon size={22} />}
      </button>

      <FeedbackBar
        mistakesOnPage={pageMistakes}
        accuracy={score}
        hidden={hidden}
        showMarks={showMarks}
        canReveal={hidden}
        onHistory={() => setSheet("mistakes")}
        onToggleHidden={() => setHidden((h) => !h)}
        onToggleMarks={() => setShowMarks((s) => !s)}
        onNextWord={nextWord}
        onWholeAyah={wholeAyah}
      />

      {sheet === "index" && (
        <SearchSheet
          suras={suras}
          onPick={pickSura}
          onPickAyah={pickAyah}
          onClose={() => setSheet(null)}
        />
      )}
      {sheet === "mistakes" && (
        <MistakesSheet log={marks.log} suras={suras} onClose={() => setSheet(null)} />
      )}
      {sheet === "moshaf" && (
        <MoshafSheet
          value={moshaf}
          rules={rules}
          strictness={strictness}
          locked={status !== "idle"}
          onSave={(cfg, sel, str) => {
            setMoshaf(cfg);
            saveMoshafConfig(cfg);
            setRules(sel);
            saveRuleSelection(sel);
            setStrictness(str);
            saveStrictness(str);
            setToast(
              cfg || sel
                ? "حُفظت خصائص التلاوة — تُطبَّق على الجلسة التالية."
                : "أُعيد الوضع الافتراضي.",
            );
          }}
          onClose={() => setSheet(null)}
        />
      )}

      {toast && <div className="toast" role="status">{toast}</div>}
    </div>
  );
}

function PageNav({
  page,
  onTurn,
  disabled,
}: {
  page: number;
  onTurn: (dir: 1 | -1) => void;
  disabled: boolean;
}) {
  // Forward is to the LEFT in an RTL muṣḥaf, so the "next" arrow sits on the inline-end
  // (left) edge and the chevron points that way. The buttons hug the page column rather
  // than the far screen edge, so on any width they land where the eye already is.
  return (
    <>
      <button
        className="pagenav pagenav--next"
        onClick={() => onTurn(1)}
        disabled={disabled || page >= PAGES}
        aria-label="الصفحة التالية"
        title="الصفحة التالية"
      >
        <ChevronIcon size={26} />
      </button>
      <button
        className="pagenav pagenav--prev"
        onClick={() => onTurn(-1)}
        disabled={disabled || page <= 1}
        aria-label="الصفحة السابقة"
        title="الصفحة السابقة"
      >
        <ChevronIcon size={26} />
      </button>
    </>
  );
}

const Legend = () => (
  <div className="legend" aria-hidden="true">
    <span style={{ color: "var(--on-dark-error)" }}>خطأ في الكلمة</span>
    <span style={{ color: "var(--on-dark-almost)" }}>غير مؤكَّد</span>
    <span style={{ color: "var(--mark-tajweed)" }}>تجويد</span>
    <span style={{ color: "var(--mark-unverified)" }}>لم تُقيَّم</span>
  </div>
);

/**
 * Not optional. The Quran text ships under CC BY 3.0 from the Tanzil Project, whose
 * terms bind any application that uses it: credit the source and link to it. That
 * obligation lands on the frontend, which is this file.
 */
const Credit = () => (
  <p className="credit">
    نصّ القرآن الكريم من <a href="https://tanzil.net" target="_blank" rel="noreferrer">مشروع تنزيل</a>{" "}
    (CC BY 3.0) · رسم وخطوط مجمع الملك فهد لطباعة المصحف الشريف (KFGQPC)
    <br />
    الماهر — تدريب التلاوة والتجويد
  </p>
);
