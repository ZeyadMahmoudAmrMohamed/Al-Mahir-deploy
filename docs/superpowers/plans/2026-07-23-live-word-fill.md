# Live word-fill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fill words in on the muṣḥaf *as they are recited*, before the reciter pauses, without changing the accuracy of the authoritative grade that already lands at each waqf.

**Architecture:** A second, provisional "Tier 1" runs alongside the unchanged waqf pipeline. Every ~300 ms while the reciter is still speaking, `LiveSession` re-runs the *same* Muaalem model on the growing utterance buffer, aligns the result to the cursor with the *same* `track()` the grader uses, and emits a lightweight `progress` event listing confirmed and skipped words. The frontend paints these provisionally; the authoritative `feedback` event overrides them at the pause via the same key-based reconciliation already used for chunk-overlap dedup.

**Tech Stack:** Python 3.10+, FastAPI, PyTorch, pydantic-settings (backend); React 18 + TypeScript + Vite (frontend); pytest (backend tests). No new dependencies.

## Global Constraints

- **The human commits, not the agent.** Every task ends at a green check; the *human partner* reviews and runs the commit. Never run `git commit`. Commit commands shown in checkpoints are suggestions for the human.
- **The authoritative grade must stay bit-for-bit identical to today's.** Tier 1 only adds a new event; it must not alter `analyse_session`, the waqf pipeline, `SessionState` advancement, or any `feedback` field.
- **Tier 1 never accuses.** A `progress` event carries only word coordinates — no errors, no phoneme diffs, no ṣifāt. It may confirm (word heard) and flag a clean skip (position only). It must never emit a pronunciation verdict. All criticism originates from the `feedback` event.
- **Tier 1 is real/remote only.** Gate on `engine.name in ("real", "remote")`. `mock` (fabricated phonemes) and `zipformer` stay off — no `progress` events, no regression.
- **Reconciliation rule:** a provisional mark (`heard`/`skipped`) must never overwrite a committed verdict (`error`/`almost`/`recited`); a committed verdict always overwrites a provisional mark.
- Audio is 16 kHz mono float32 in `[-1, 1]`; `sample_rate` is fixed at 16000.
- Backend tests run CPU-only (no GPU, no model download). Use injected fake engines and the verified phoneme fixtures.
- Frontend has **no test runner** (see `frontend/package.json` — only `dev`/`build`/`typecheck`). Verify frontend work with `npm run typecheck` and one documented manual session check. Do not add a test framework.

---

### Task 1: `confirmed_and_skipped()` — the alignment used by the live tier

**Files:**
- Modify: `backend/src/tajwid/feedback/track.py` (append a new function after `track`)
- Test: `backend/tests/test_track.py`

**Interfaces:**
- Consumes: `track()`, `_ordinal_of_word()`, `_word_of_ordinal()`, `Span` (all already in `track.py`).
- Produces: `confirmed_and_skipped(phonemes: str, cursor: Span, moshaf: MoshafAttributes, lookahead_words: int = 1, penalty: int = 0) -> tuple[list[Span], list[Span]]` — `(confirmed, skipped)`, both in mushaf order, forward of the cursor. Empty on a failed/blank match.

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_track.py`:

```python
# --- Live word-fill alignment (confirmed_and_skipped) ------------------------
# Verified phonetizer output; conftest.AL_FATIHA_1_5 is the source of truth and
# test_golden guards it. Inlined here so this test is self-contained.
_AL_FATIHA_1_5 = "ءِييَااكَنَعبُدُوَءِييَااكَنَستَعِۦۦۦۦن"


def test_confirmed_and_skipped_fills_forward_with_no_false_skip(moshaf):
    from tajwid.feedback.track import confirmed_and_skipped
    from tajwid.feedback.types import Span

    confirmed, skipped = confirmed_and_skipped(
        _AL_FATIHA_1_5, Span(sura=1, aya=5, word_idx=0), moshaf, lookahead_words=1
    )
    assert confirmed, "a correct recitation from the cursor should confirm words"
    assert skipped == [], "reciting from the cursor forward is never a skip"
    assert Span(sura=1, aya=5, word_idx=0) in confirmed, "the cursor word is confirmed"
    assert all((c.sura, c.aya) == (1, 5) for c in confirmed)


def test_confirmed_and_skipped_holds_back_the_last_word(moshaf):
    from tajwid.feedback.track import confirmed_and_skipped
    from tajwid.feedback.types import Span

    conf1, _ = confirmed_and_skipped(
        _AL_FATIHA_1_5, Span(sura=1, aya=5, word_idx=0), moshaf, lookahead_words=1
    )
    conf0, _ = confirmed_and_skipped(
        _AL_FATIHA_1_5, Span(sura=1, aya=5, word_idx=0), moshaf, lookahead_words=0
    )
    # Holding back one word commits strictly fewer words than holding back none.
    assert len(conf1) < len(conf0), "lookahead must hold back the trailing word(s)"


def test_confirmed_and_skipped_flags_a_real_skip(moshaf):
    from tajwid.feedback.track import confirmed_and_skipped
    from tajwid.feedback.types import Span

    # The last two words of 1:5 only (وإياك نستعين): a slice of the verified string, so
    # no transcription risk. Cursor still sits at the verse's first word.
    tail = _AL_FATIHA_1_5[len("ءِييَااكَنَعبُدُ"):]
    confirmed, skipped = confirmed_and_skipped(
        tail, Span(sura=1, aya=5, word_idx=0), moshaf, lookahead_words=1
    )
    assert confirmed, "a later word is confirmed, which is what licenses a skip claim"
    assert skipped, "the leading words the reciter passed over are flagged"
    assert Span(sura=1, aya=5, word_idx=0) in skipped, "the cursor word was skipped"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd backend && python -m pytest tests/test_track.py -k confirmed_and_skipped -v`
Expected: FAIL with `ImportError: cannot import name 'confirmed_and_skipped'`.

- [ ] **Step 3: Implement `confirmed_and_skipped`**

Append to `backend/src/tajwid/feedback/track.py`:

```python
def confirmed_and_skipped(
    phonemes: str,
    cursor: Span,
    moshaf: MoshafAttributes,
    lookahead_words: int = 1,
    penalty: int = 0,
) -> tuple[list[Span], list[Span]]:
    """Live word-fill: which words the reciter has provisionally reached, and which
    they appear to have skipped — from the SAME windowed ``track`` the authoritative
    grade uses, so the live cursor cannot drift from the graded one.

    ``lookahead_words`` is held back from the END of the match: the final word(s) of a
    decode-so-far are still in flight (the reciter may be mid-madd), so they are not
    committed until later audio stabilises them. A skip is asserted ONLY when a later
    word was confirmed (``confirmed`` is non-empty) — that keeps skip-detection a fact
    about audio already gone by, never a guess about audio still arriving.

    Returns ``(confirmed, skipped)`` in mushaf order. Empty on a failed/blank match; the
    caller emits nothing rather than guess.
    """
    found = track(phonemes, cursor, moshaf, penalty=penalty)
    if found.status != "ok" or found.span is None or found.end is None:
        return [], []

    ord_of = _ordinal_of_word()
    words_by_ord = _word_of_ordinal()
    cur_ord = ord_of.get((cursor.sura, cursor.aya, cursor.word_idx))
    span_ord = ord_of.get((found.span.sura, found.span.aya, found.span.word_idx))
    end_ord = ord_of.get((found.end.sura, found.end.aya, found.end.word_idx))
    if cur_ord is None or span_ord is None or end_ord is None:
        return [], []

    last_confirm_ord = end_ord - max(0, lookahead_words)
    confirmed = (
        [
            Span(sura=s, aya=a, word_idx=w)
            for (s, a, w) in (words_by_ord[o] for o in range(span_ord, last_confirm_ord + 1))
        ]
        if last_confirm_ord >= span_ord
        else []
    )

    # Only flag a skip once a later word is confirmed. The window reaches backwards too
    # (reciters repeat), so a match BEHIND the cursor gives span_ord < cur_ord and an
    # empty range here — no false skip on a legitimate rewind.
    skipped = (
        [
            Span(sura=s, aya=a, word_idx=w)
            for (s, a, w) in (words_by_ord[o] for o in range(cur_ord, span_ord))
        ]
        if confirmed
        else []
    )

    return confirmed, skipped
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd backend && python -m pytest tests/test_track.py -k confirmed_and_skipped -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Checkpoint (human commits)**

Suggested for the human:
```bash
git add backend/src/tajwid/feedback/track.py backend/tests/test_track.py
git commit -m "feat(feedback): confirmed_and_skipped alignment for live word-fill"
```

---

### Task 2: `LiveSession` live-decode loop + settings + engine gating

**Files:**
- Modify: `backend/src/tajwid/config.py` (add settings near line 104 and a property near line 212)
- Modify: `backend/src/tajwid/session.py` (imports; `LiveSession.__init__`; `feed`; new `_live_progress`)
- Test: `backend/tests/test_live_session.py`

**Interfaces:**
- Consumes: `confirmed_and_skipped` (Task 1); `strip_non_verse`; `ChunkContext`; `Settings.live_interval_samples`, `Settings.live_lookahead_words`, `Settings.live_feedback`.
- Produces: `LiveSession.feed` now returns `feedback` events (unchanged) **and** may interleave `{"type": "progress", "confirmed": [...], "skipped": [...], "cursor": {...}|null}` dicts.

- [ ] **Step 1: Add the settings**

In `backend/src/tajwid/config.py`, after the `chunk_overlap_ms` field (line 104), add:

```python
    # --- Live word-fill (Tier 1: provisional per-word feedback before the waqf) ---
    # Re-decode the growing utterance this often (ms) to fill words in live. Gated to
    # the real/remote engines in LiveSession — mock fabricates phonemes and zipformer
    # has no acoustic confidence, so neither drives live-fill.
    live_feedback: bool = True
    live_interval_ms: int = 300
    # Words held back from the END of each live match: the last word(s) of a
    # decode-so-far are still in flight (the reciter may be mid-madd), committed only
    # once later audio stabilises them.
    live_lookahead_words: int = 1
```

In the same file, after the `chunk_overlap_samples` property (line 212), add:

```python
    @property
    def live_interval_samples(self) -> int:
        return int(self.live_interval_ms * self.sample_rate / 1000)
```

- [ ] **Step 2: Write the failing tests**

Append to `backend/tests/test_live_session.py`:

```python
# --- Tier 1 live word-fill ---------------------------------------------------

_AL_FATIHA_1_5 = "ءِييَااكَنَعبُدُوَءِييَااكَنَستَعِۦۦۦۦن"


class _FakeLiveEngine:
    """A real-named engine that ignores the audio and returns a fixed transcript, so the
    live-decode wiring can be exercised on CPU with no model."""

    name = "real"

    def __init__(self, phonemes: str):
        self._ph = phonemes

    def transcribe_chunk(self, wave, sample_rate, ctx=None):
        from quran_transcript import chunck_phonemes
        from tajwid.asr.transcribe import ChunkTranscript

        groups = chunck_phonemes(self._ph)
        return ChunkTranscript(
            phonemes_text=self._ph,
            char_probs=[1.0] * len(self._ph),
            groups=groups,
            group_probs=[1.0] * len(groups),
            sifat=[],
        )


def test_live_progress_fills_words_before_a_pause():
    s = Settings(asr_engine="real")
    sess = LiveSession(
        _FakeLiveEngine(_AL_FATIHA_1_5),
        session_id="t",
        start=Span(sura=1, aya=5, word_idx=0),
        settings=s,
    )
    # Silence: silero finds no speech, so nothing finalizes and the live path fires on
    # the interval counter. Length must clear both the interval and min_speech gates.
    frame = np.zeros(s.live_interval_samples + s.min_speech_samples, dtype=np.float32)
    events = sess.feed(frame)

    progress = [e for e in events if e["type"] == "progress"]
    assert progress, "a progress event should fire once past the live interval"
    confirmed = progress[-1]["confirmed"]
    assert confirmed, "confirmed words expected"
    assert {"sura": 1, "aya": 5, "word_idx": 0} in confirmed
    # Provisional events assert nothing negative about pronunciation.
    assert "words" not in progress[-1]


def test_mock_engine_emits_no_live_progress(mock_engine):
    s = Settings(asr_engine="mock")
    sess = LiveSession(mock_engine, session_id="t", start=Span(sura=1, aya=1, word_idx=0), settings=s)
    frame = np.zeros(s.live_interval_samples + s.min_speech_samples, dtype=np.float32)
    events = sess.feed(frame)
    assert not [e for e in events if e.get("type") == "progress"], "mock must not drive live-fill"
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `cd backend && python -m pytest tests/test_live_session.py -k "live_progress or no_live_progress" -v`
Expected: FAIL — `KeyError: 'type'` / no progress events (the live path does not exist yet).

- [ ] **Step 4: Wire the live decode into `LiveSession`**

In `backend/src/tajwid/session.py`, add these imports next to the existing feedback imports (after the `from .feedback.pipeline import analyse_session` line):

```python
from .feedback.nonverse import strip_non_verse
from .feedback.track import confirmed_and_skipped
```

In `LiveSession.__init__`, after `self._prev_forced = False` (line 181), add:

```python
        # Tier 1 (live word-fill): a running buffer of the current, not-yet-finalized
        # utterance, re-decoded every `live_interval_samples` to fill words in before the
        # waqf. Gated to engines that actually hear (real/remote); mock/zipformer stay off.
        self._live_buffer = np.zeros(0, dtype=np.float32)
        self._samples_since_live = 0
        self._live_enabled = bool(self.s.live_feedback) and getattr(
            engine, "name", ""
        ) in ("real", "remote")
```

Replace the existing `feed` method (lines 184-186) with:

```python
    def feed(self, samples: np.ndarray) -> list[dict]:
        """Append audio; return one feedback event per finalized waqf chunk, plus any
        interleaved live `progress` events (Tier 1)."""
        events: list[dict] = []
        fins = self.stream.feed(samples)
        self._live_buffer = np.concatenate(
            [self._live_buffer, np.asarray(samples, dtype=np.float32).reshape(-1)]
        )
        for fin in fins:
            if e := self._process(fin):
                events.append(e)
        if fins:
            # A waqf finalized: Tier 2 authoritatively graded this audio. Drop the live
            # buffer so the next utterance's provisional decode starts clean.
            # ponytail: clears the WHOLE buffer, so a few ms of the next utterance already
            # captured past the endpoint is dropped and simply rebuilt by the next frames —
            # a provisional view Tier 2 re-grades from real audio regardless.
            self._live_buffer = np.zeros(0, dtype=np.float32)
            self._samples_since_live = 0
        elif self._live_enabled:
            self._samples_since_live += int(np.asarray(samples).size)
            if self._samples_since_live >= self.s.live_interval_samples:
                self._samples_since_live = 0
                if pe := self._live_progress():
                    events.append(pe)
        return events
```

Add this method to `LiveSession` (e.g. after `feed`):

```python
    def _live_progress(self) -> dict | None:
        """Re-decode the current utterance so far and report the words provisionally
        reached (and any clean skip). Provisional: coordinates only, never a verdict."""
        import torch

        if self.state.cursor is None or self._live_buffer.size < self.s.min_speech_samples:
            return None
        # ponytail: O(n^2) — re-decodes from utterance start each tick, ≤19 s of audio,
        # fine on a real GPU. Ceiling: bound to the last few seconds / cache encoder
        # states if a long single-breath recitation ever measures too slow.
        sr = self.s.sample_rate
        ctx = ChunkContext(
            duration_s=self._live_buffer.size / sr,
            cursor=self.state.cursor,
            moshaf=self.state.moshaf,
        )
        transcript = self.engine.transcribe_chunk(
            torch.from_numpy(self._live_buffer), sr, ctx
        )
        if not transcript.phonemes_text:
            return None
        verse_ph, _nv, _s0, _e0 = strip_non_verse(
            transcript.phonemes_text, self.state.moshaf
        )
        if not verse_ph:
            return None
        confirmed, skipped = confirmed_and_skipped(
            verse_ph,
            self.state.cursor,
            self.state.moshaf,
            lookahead_words=self.s.live_lookahead_words,
            penalty=self.state.penalty,
        )
        if not confirmed and not skipped:
            return None
        tip = confirmed[-1] if confirmed else self.state.cursor
        return {
            "type": "progress",
            "confirmed": [w.model_dump() for w in confirmed],
            "skipped": [w.model_dump() for w in skipped],
            "cursor": tip.model_dump() if tip else None,
        }
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd backend && python -m pytest tests/test_live_session.py -v`
Expected: PASS (the two new tests plus all existing `test_live_session.py` tests still green).

- [ ] **Step 6: Run the whole backend suite to confirm no regression**

Run: `cd backend && python -m pytest -q`
Expected: PASS / same skips as before (GPU-gated tests self-skip). No new failures.

- [ ] **Step 7: Checkpoint (human commits)**

Suggested for the human:
```bash
git add backend/src/tajwid/config.py backend/src/tajwid/session.py backend/tests/test_live_session.py
git commit -m "feat(session): Tier 1 live word-fill decode loop (real/remote only)"
```

---

### Task 3: Frontend types + transport + reconciliation

**Files:**
- Modify: `frontend/src/lib/types.ts`
- Modify: `frontend/src/lib/session.ts`
- Modify: `frontend/src/lib/marks.ts`

**Interfaces:**
- Consumes: the `progress` event from Task 2.
- Produces: `ProgressEvent` type; `SessionHandlers.onProgress?`; `applyProgress(prev: MarkState, event: ProgressEvent): MarkState`; two new `WordMark`s `"heard"` and `"skipped"`.

- [ ] **Step 1: Extend the types**

In `frontend/src/lib/types.ts`, add after the `FeedbackEvent` type (line 124):

```typescript
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
```

Add `ProgressEvent` to the `SessionEvent` union (line 126-129):

```typescript
export type SessionEvent =
  | { type: "session"; session_id: string; engine: string; sample_rate: number }
  | FeedbackEvent
  | ProgressEvent
  | { type: "done" };
```

Extend `WordMark` (line 152):

```typescript
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
```

- [ ] **Step 2: Route the event in the transport**

In `frontend/src/lib/session.ts`, add `ProgressEvent` to the type import (line 2-9) and add an `onProgress` handler to `SessionHandlers` (after `onFeedback`, line 17):

```typescript
  onProgress?: (e: ProgressEvent) => void;
```

In the `ws.onmessage` handler (line 63-67), add the `progress` branch:

```typescript
    ws.onmessage = (e) => {
      const msg: SessionEvent = JSON.parse(e.data);
      if (msg.type === "feedback") this.handlers.onFeedback(msg);
      else if (msg.type === "progress") this.handlers.onProgress?.(msg);
      else if (msg.type === "session") this.handlers.onEngine?.(msg.engine);
    };
```

- [ ] **Step 3: Add `applyProgress` and fix the reconciliation guard**

In `frontend/src/lib/marks.ts`, add `ProgressEvent` to the type import (line 17), then add near the top (after `markOf`):

```typescript
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
```

Then fix the overlap guard inside `applyFeedback` (line 65-66) so an `unverified` re-emission is still blocked by a real verdict, but a *provisional* mark is freely overwritten. Replace:

```typescript
    const prev = marks.get(key);
    if (mark === "unverified" && prev && prev !== "unverified") continue;
```

with:

```typescript
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
```

(`accuracy` and `mistakesOnPage` need no change — they read only `recited` and `error`, so provisional marks are already excluded from the score.)

- [ ] **Step 4: Typecheck**

Run: `cd frontend && npm run typecheck`
Expected: PASS, no type errors.

- [ ] **Step 5: Checkpoint (human commits)**

Suggested for the human:
```bash
git add frontend/src/lib/types.ts frontend/src/lib/session.ts frontend/src/lib/marks.ts
git commit -m "feat(frontend): progress event type, transport routing, provisional reconciliation"
```

---

### Task 4: Frontend rendering — paint provisional marks and wire the handler

**Files:**
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/components/MushafPage.tsx`
- Modify: the global stylesheet (find with the grep in Step 1)

**Interfaces:**
- Consumes: `applyProgress` (Task 3), `ProgressEvent`, `WordMark` `"heard"`/`"skipped"`.
- Produces: the live cursor advance + provisional word styling in the UI.

- [ ] **Step 1: Locate the stylesheet that defines the mark colors**

Run: `cd frontend && grep -rn "mark-unverified" src`
Expected: one match in a `.css` file (the global stylesheet). Note that path — call it `<STYLES>` in Step 4.

- [ ] **Step 2: Wire the handler in `App.tsx`**

In `frontend/src/App.tsx`, add `applyProgress` to the marks import (line 25):

```typescript
import { accuracy, applyFeedback, applyProgress, emptyMarks, mistakesOnPage } from "./lib/marks";
```

Add an `onProgress` callback next to `onFeedback` (after the `onFeedback` `useCallback`, line 126):

```typescript
  const onProgress = useCallback((event: import("./lib/types").ProgressEvent) => {
    setMarks((prev) => applyProgress(prev, event));
    // The live cursor is what makes the page follow the reciter and the current word
    // highlight move before the pause. Reconciled by the authoritative cursor at the waqf.
    if (event.cursor) setCursor(event.cursor);
  }, []);
```

Pass it into the `RecitationSession` handlers object inside `start` (line 131-145), alongside `onFeedback`:

```typescript
      {
        onFeedback,
        onProgress,
        onLevel: setLevel,
        onState: setStatus,
        onError: setToast,
        onEngine: (engine) => {
```

Add `onProgress` to the `start` callback's dependency array (line 151):

```typescript
  }, [cursor, onFeedback, onProgress, engineChoice, moshaf, rules]);
```

- [ ] **Step 3: Rank the provisional marks in `MushafPage`**

In `frontend/src/components/MushafPage.tsx`, replace the `severity` function (line 206-207) so provisional marks rank *below* every real verdict (a real verdict always wins on a multi-word glyph) but still render when alone:

```typescript
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
```

The existing class line already emits `word--heard` / `word--skipped` (it renders `word--${mark}` for any mark that is not `pending`/`recited`), so no change is needed there.

- [ ] **Step 4: Style the provisional marks**

Append to `<STYLES>` (the file found in Step 1), near the other `.word--*` rules:

```css
/* Tier 1 provisional marks. `heard` is the correct-ink one hair brighter — imperceptible
   unless looked for; it settles to full-black `recited` the instant Tier 2 confirms.
   `skipped` is a gentle positional hint, never a mistake color.
   ponytail: the exact shades are calibration knobs — tune --ink-heard/--mark-skipped to
   the muṣḥaf's ink; the inline fallbacks keep it working without them. */
.word--heard {
  color: var(--ink-heard, #1a1a1a);
}
.word--skipped {
  text-decoration: underline dotted var(--mark-skipped, #c9a227);
  text-underline-offset: 0.3em;
}
```

- [ ] **Step 5: Typecheck and build**

Run: `cd frontend && npm run typecheck && npm run build`
Expected: PASS, clean build.

- [ ] **Step 6: Manual verification (documented — no runner exists)**

With a `real`/`remote` backend running (`TAJWID_API=...` per README), start the frontend, pick a sūra, and recite a long āyah **without pausing**. Confirm:
- Words fill in / brighten (`heard`) as you speak, before you pause.
- At the pause, words settle to normal ink or flip to a mistake mark — no green→red reversal.
- Skipping a word shows the gentle dotted hint, and the authoritative `feedback` then marks it.
- On a `mock`/`zipformer` session, behavior is unchanged (no live fill).

- [ ] **Step 7: Checkpoint (human commits)**

Suggested for the human:
```bash
git add frontend/src/App.tsx frontend/src/components/MushafPage.tsx frontend/src/<the styles file>
git commit -m "feat(frontend): render live word-fill and advance the cursor provisionally"
```

---

### Task 5: Document the protocol

**Files:**
- Modify: `backend/src/tajwid/api/ws.py` (docstring only)
- Modify: `API.md`
- Modify: `README.md`

**Interfaces:** none — documentation of the `progress` event shipped in Tasks 2-4.

- [ ] **Step 1: Update the WS handler docstring**

In `backend/src/tajwid/api/ws.py`, in the module docstring's `server -> client` list (after the `feedback` line, ~line 35), add:

```
    {"type":"progress", "confirmed":[...], "skipped":[...], "cursor":{...}|null}
                                                       provisional live word-fill (Tier 1),
                                                       real/remote engines only. Coordinates
                                                       only — never a verdict. Reconciled by
                                                       the next "feedback".
```

- [ ] **Step 2: Add a `progress` section to API.md**

In `API.md` section 5 (the WS protocol), after the feedback event subsection (§5.4), add a short subsection:

```markdown
### 5.4a The progress event (live word-fill)

On the `real` and `remote` engines the server also pushes a lightweight `progress` event
while the reciter is still speaking, roughly every 300 ms, to fill words in before the
pause. It is **provisional and forward-only**:

```json
{"type": "progress",
 "confirmed": [{"sura": 1, "aya": 5, "word_idx": 0}],
 "skipped": [],
 "cursor": {"sura": 1, "aya": 5, "word_idx": 2}}
```

- `confirmed` — words the reciter has provisionally reached. Render them as *reached*, not
  as *correct* (no tick, no green): the authoritative `feedback` event grades them at the
  pause and may override.
- `skipped` — words the reciter passed over, asserted only once a later word is confirmed.
  A gentle positional hint, never a pronunciation mistake.
- `cursor` — the furthest confirmed word, for advancing the highlight and following the page.

A `progress` event carries **no errors, no phonemes, no ṣifāt**. Never render a mistake from
it; all criticism comes from `feedback`. `mock` and `zipformer` sessions never send it. A
client that ignores `progress` degrades to today's pause-only behavior.
```

- [ ] **Step 3: Note the two tiers in README.md**

In `README.md`, under the pipeline description, add one line noting that on `real`/`remote`
a provisional live word-fill (`progress` events) runs alongside the waqf pipeline, with the
waqf `feedback` remaining authoritative. Keep it to a sentence.

- [ ] **Step 4: Checkpoint (human commits)**

Suggested for the human:
```bash
git add backend/src/tajwid/api/ws.py API.md README.md
git commit -m "docs: document the progress (live word-fill) event"
```

---

## Notes carried from the design

- **Pausal tajwīd cannot stream.** مد العارض للسكون and other pause-defined rules are only
  finalized at the waqf, by `feedback`. That is physics, not a limitation of this plan;
  Tier 1 deliberately never touches them.
- **GPU cost.** Muaalem now runs ~3×/s per active session. Within one session there is no
  overlap (each `feed` runs to completion on the worker thread). Across many concurrent
  sessions this multiplies — a deployment concern; `live_interval_ms` is the throttle and
  `live_feedback=False` the kill switch.
