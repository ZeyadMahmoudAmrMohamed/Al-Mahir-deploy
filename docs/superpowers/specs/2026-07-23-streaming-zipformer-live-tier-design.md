# Streaming-zipformer live tier — design

**Date:** 2026-07-23
**Status:** approved design, pre-implementation
**Supersedes:** the Muaalem re-decode live tier from
`2026-07-23-live-word-fill-design.md`. That approach re-ran the *grading* model on the
whole growing utterance every 300 ms — O(n²), and over a remote tunnel, unusable. This
replaces the *source* of the live word-fill while keeping the same `progress` event and
the entire frontend contract untouched.

---

## 1. The core idea

Two tiers, two models, each used the way it's actually good at:

- **Live tier — streaming zipformer (CPU, local).** A single persistent sherpa-onnx
  `OnlineRecognizer` stream per session, fed audio incrementally. It emits a cumulative
  phoneme partial that grows in O(new audio) with **no re-decode**. A cheap, forward-only
  monotonic matcher aligns that partial against the *known* expected words from the cursor
  and reports which words are confirmed / skipped. This drives the live word-fill.
- **Authoritative tier — Muaalem at the waqf (unchanged).** The heavy
  `locate → build_reference → diff → ṣifāt → score` pipeline. It produces the real grade
  **and re-anchors the live tier** at each pause. Muaalem has the final call.

Verified empirically on the real Fātiḥa asset: one persistent stream, fed 0.5 s at a time,
never `input_finished()` — the partial grows monotonically (0→220 chars over 27 s) at
~50 ms/frame decode (max 97 ms), ~10× faster than real-time on CPU, in the same phoneme
alphabet the muṣḥaf reference uses.

### Why this shape

- The waqf/batch pipeline was built around Muaalem's per-character confidence and 10 ṣifāt.
  Zipformer has neither, so running it through that pipeline pays the full cost for word
  marks only. Its strength is streaming; the live tier is where that pays off.
- The live tier runs **locally on CPU regardless of where Muaalem runs.** With Muaalem on a
  remote tunnel, live-fill is still instant and local; only the pause-grade round-trips.
  This is what makes the feature usable on a GPU-less dev box.

## 2. Scope and non-goals

- **Companion only.** The live tier is active only when the authoritative engine is Muaalem
  (`real` or `remote`). On `mock` (fabricated phonemes) and `zipformer`-as-grader it stays
  off. Rationale: grading is Muaalem's job; a GPU-free *grading* path is a separate future
  effort (a streaming-native zipformer grader, never the borrowed batch pipeline).
- **The existing `ZipformerAsrEngine` (waqf grader) is left untouched** — still selectable
  per session. We stop building on it; we don't remove it.
- **Divergence = stall.** When the partial stops matching the expected sequence (reciter
  jumped, mutashābih repeat, off-book), the live tier stops confirming and waits for the
  next waqf grade to re-anchor. It does **no** fuzzy locate of its own. Stricter recovery is
  explicitly a future phase.
- **This replaces the Muaalem re-decode live path.** `_live_buffer` / `_live_progress` and
  their two tests are removed in favor of the aligner below.

## 3. Components

### 3.1 `LiveAligner` (new)

`backend/src/tajwid/asr/live_aligner.py`. Owns the streaming state for one session.

State:
- `recognizer` — a shared sherpa-onnx `OnlineRecognizer` (see 3.3).
- `stream` — this session's persistent `OnlineStream`.
- `anchor: Span` — the authoritative cursor at the last re-anchor.
- `expected_norm: str` — normalized phonemes for the next `live_window_words` words from
  the anchor (recomputed on re-anchor).
- `word_char_ends: list[int]` — cumulative character length of `expected_norm` at each word
  boundary, so a matched character extent maps to a confirmed word count.

Methods:
- `feed(samples: np.ndarray) -> None` — `stream.accept_waveform(...)` then drain
  `while recognizer.is_ready(stream): recognizer.decode_stream(stream)`. Keeps the partial
  current; emits nothing.
- `progress(anchor: Span) -> tuple[list[Span], list[Span]]` — read
  `recognizer.get_result(stream)`, normalize it, run the monotonic matcher (3.2) against
  `expected_norm`, return `(confirmed, skipped)` as `Span`s. Empty when the match doesn't
  advance (the stall).
- `reanchor(cursor: Span) -> None` — `recognizer.reset(stream)`, clear the partial state,
  set `anchor = cursor`, recompute `expected_norm` and `word_char_ends` from the cursor.

Reusable helpers already in `feedback/track.py`: `normalized_phonemes_for_span`,
`_ordinal_of_word`, `_word_of_ordinal`, `_slice_by_ordinal`, and `_search_engine()._normalize_query`.

### 3.2 The monotonic matcher

Given `expected_norm` (from the anchor) and the normalized cumulative `partial`:

1. Align `partial` (source) against `expected_norm` (target) with a single
   `Levenshtein.opcodes(partial, expected_norm)` — one forward alignment, **not** a grid
   search over candidate spans. Because the partial is the current utterance from the
   anchor, it aligns as an approximate prefix of `expected_norm` (CTC noise shows up as
   `replace`; the unread remainder as a trailing `insert`).
2. Walk the opcodes to find:
   - `matched_extent` — the index in `expected_norm` reached by the end of `partial`
     (i.e. how far the reciter has read).
   - `skipped word boundaries` — a word whose `expected_norm` characters fall entirely in an
     `insert` op (present in expected, absent from partial) that precedes a later matched
     word. That word was skipped.
3. Confirm every word whose `word_char_ends[k] <= matched_extent` minus a **1-word
   lookahead** (`live_lookahead_words`) — the trailing word is still in flight and the
   streaming CTC tail can still revise, so it is not committed yet.
4. **Stall guard:** if the alignment quality is poor (edit ratio over `partial` beyond a
   threshold, or `matched_extent` did not advance past the anchor), return `([], [])`. This
   is the "defer to Muaalem" behavior.

Cost is bounded: `expected_norm` is capped at `live_window_words` and the partial is one
utterance (a waqf reset arrives first). Caching the DP row for true O(new-chars) increments
is the documented "make it stricter" next phase, not v1.

### 3.3 Shared recognizer

The zipformer `OnlineRecognizer` loads a 263 MB ONNX model. Build it **once** (it already
exists as `ZipformerAsrEngine._recognizer` from `build_engines`) and let each session's
`LiveAligner` create its own `OnlineStream` from it — sherpa-onnx supports many streams per
recognizer. Concurrent `decode_stream` calls from different session threads are serialized
with one module-level lock.

> ponytail: global decode lock, ~50 ms per 0.5 s of audio per tick. Fine for a handful of
> concurrent sessions. Ceiling: per-session recognizers or sherpa batched decode if
> throughput ever demands it.

## 4. Wiring into `LiveSession`

`LiveSession.__init__` gains an optional `zipformer_engine` (passed from `ws.py`, which
reads `engines.get("zipformer")`). The live tier is enabled iff:

```
settings.live_feedback
  and zipformer_engine is not None            # files present at startup
  and getattr(engine, "name", "") in ("real", "remote")   # companion to Muaalem
```

When enabled, `__init__` builds a `LiveAligner` from the zipformer engine's recognizer and
re-anchors it to the start cursor.

`feed(samples)`:
1. `fins = self.stream.feed(samples)` — silero VAD (unchanged).
2. If live enabled: `self.live.feed(samples)` — keep the zipformer stream current.
3. For each `fin`: `self._process(fin)` — Muaalem grade (unchanged, still crash-guarded).
4. If any `fin` finalized (a waqf): after grading advances `self.state.cursor`, call
   `self.live.reanchor(self.state.cursor)`. This resets the zipformer stream and re-syncs
   the live tier to the authoritative position.
5. Else, on the `live_interval_ms` cadence: `confirmed, skipped = self.live.progress(...)`;
   if non-empty, append a `progress` event (same shape as today).

`seek(span)`: after resetting the cursor, `self.live.reanchor(span)`.

The `progress` event, its frontend routing, `applyProgress`, the `heard`/`skipped` marks
and their rendering are **all unchanged** — only the backend producer changes.

## 5. Config

Reuse `live_feedback`, `live_interval_ms`, `live_lookahead_words`. Add:

```python
# Words of expected context the live matcher aligns against, from the anchor forward.
# A waqf re-anchor arrives well before an utterance could exhaust this.
live_window_words: int = 40
```

Remove nothing from config; `live_interval_samples` stays (still the cadence clock).

## 6. Failure and edge cases

- **No zipformer files** → live tier simply off (same as `live_feedback=false`). No crash.
- **Authoritative engine is `mock`/`zipformer`** → live tier off (companion-only gate).
- **Aligner throws** → caught by the existing live-tier guard in `feed`; the tick is dropped,
  the session and the authoritative grade are unaffected.
- **Long single-breath utterance** → bounded by `live_window_words`; if the reciter somehow
  runs past it before any waqf, the matcher confirms up to the window and stalls — the next
  waqf re-anchors.
- **Reciter jumps** → matcher stalls (no confident forward extent); next waqf re-anchors.

## 7. Tests

- **Matcher, pure (CPU, no model):** feed canned normalized partials + a known anchor to the
  monotonic matcher directly; assert (a) forward recitation confirms words minus the
  lookahead, (b) a partial that skips a leading word flags it as skipped, (c) a garbage /
  non-matching partial stalls to `([], [])`.
- **Streaming integration (needs zipformer files; self-skip if absent, like the other
  zipformer tests):** feed `tests/assets/fatiha_long_track.wav` through a `LiveAligner`
  anchored at 1:1:0 in ~0.5 s frames; assert confirmed words advance monotonically and never
  exceed the audio actually heard.
- **Re-anchor:** after `reanchor` to a new cursor, the stream result resets and the matcher
  aligns against the new expected window.
- **Gating:** a `mock`-authoritative session builds no live tier and emits no `progress`;
  a `real`/`remote`-authoritative session with zipformer files present does.

Frontend: unchanged, so `npm run typecheck` remains the only check.

## 8. Success criteria

- With Muaalem on `remote`, live word-fill is **local and instant** (no tunnel in the live
  path); only the waqf grade round-trips.
- The live cursor advances word-by-word during continuous recitation and stalls, rather than
  mis-confirms, when the reciter diverges.
- The authoritative grade is byte-for-byte what it is today — the Muaalem path is untouched
  except for the added `reanchor` call after it advances the cursor.
- On a box without zipformer files, behavior is exactly today's (live tier off), no crash.

## 9. Out of scope (named future phases)

- A streaming-native **zipformer grader** for fully GPU-free grading (its own endpointing +
  word-correctness from the stream, not the batch pipeline).
- True O(new-chars) incremental matching via cached DP state, and a stricter matcher
  (confidence-like gating, tighter divergence detection).
- Removing the legacy `ZipformerAsrEngine` waqf grader.
