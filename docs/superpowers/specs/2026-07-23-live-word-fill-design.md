# Live word-fill — design

**Date:** 2026-07-23
**Status:** approved design, pre-implementation
**Goal:** kill the "recite → pause → wait → feedback" lag. Words should fill in *as they are
spoken*, throughout, without changing the accuracy of the final grade.

---

## 1. The problem

Today, feedback fires **only on a waqf** (pause). Two independent causes, in the code:

- **Endpointing** (`asr/stream.py`, `StreamSession._process`): a chunk is emitted only on a
  ≥300 ms silence run (`min_silence_endpoint_samples`) or a 19 s forced cut. Nothing comes
  out mid-flow. A long āyah recited without a full stop is *structurally silent* until the
  reciter pauses.
- **Batch grading** (`session.py`, `LiveSession._process`): `transcribe_chunk` →
  `analyse_session` run on the **whole finalized wave at once**.

These are separable. This design attacks the first (responsiveness) without touching the
second (accuracy).

## 2. Key facts this design rests on

- **Muaalem is a CTC model — frame-synchronous.** It emits phoneme posteriors every
  ~20–40 ms and does **not** need a waqf to run. The per-waqf cadence is a *choice*, not a
  model requirement.
- **This app almost always knows the reference.** The reciter picked the sūra/āyah and
  `SessionState.cursor` tracks position. That turns live feedback from open-vocabulary
  *transcription* into *verification against a known target* — far easier and genuinely
  incremental.
- **One causal limit we cannot beat:** a word cannot be graded before it is fully uttered,
  and the **pausal tajwīd rules** (مد العارض للسكون, pausal sukūn, pausal word-final forms)
  are *undefined* until a pause exists. Those must defer to the waqf. Everything else —
  ḥifẓ, tashkīl, most tajwīd — does not depend on the pause and can stream.

## 3. Architecture — two tiers, one model, one WebSocket

### Tier 2 — authoritative grade (UNCHANGED)

Today's waqf pipeline, untouched. On each finalized chunk: `transcribe_chunk` →
`analyse_session` → the real grade (ḥifẓ, tashkīl, tajwīd, ṣifāt, **including the pausal
rules**). The final grade is **bit-for-bit identical to today's**. This tier is the source
of truth and is the only thing allowed to assert a mistake.

### Tier 1 — live word-fill (NEW)

A loop that, every ~300 ms **while the reciter is still speaking**:

1. Re-runs Muaalem on the **growing buffer** (decode-so-far), not on a finalized chunk.
2. Aligns the decoded phonemes against the **expected** phoneme sequence from
   `SessionState.cursor`.
3. **Commits** words that are (a) complete and (b) stable behind a short look-ahead window
   (a word is only committed once the audio for its final phoneme is behind the
   look-ahead — the "only commit what won't change" rule).

Tier 1 is **forward-only**. It advances the cursor and confirms words heard. It never emits
a pronunciation error and never paints anything red.

## 4. Marking policy — what Tier 1 is allowed to say

**Confirm + skip-detection. Forward-only. Never a pronunciation judgment.**

- **Confirm (word heard):** commit "reciter reached word N" once N is complete and stable.
- **Skip-detection:** flag "word N skipped" **only after a *later* word is confidently
  committed** — at that point N is definitively in the past and unmatched, which is a fact
  about audio already gone by, not a guess about in-progress audio. Skip-detection is free
  (it reads the alignment Tier 1 already computed) and adds no latency.

Two hard limits keep skip-detection confident:

- It flags **clean skips only** ("you moved past without saying it"), never
  substitutions/mispronunciations ("you said something here but it was wrong") — those are
  pronunciation judgments on possibly-incomplete audio and stay with Tier 2.
- It goes **silent when tracking itself is uncertain** (mutashābihāt, a reposition). A skip
  fires only inside a confidently-tracked passage.

This does not violate the system's one hard rule (*never falsely correct a perfect
recitation*): a skip is a **position** signal, not a critique of how a word was said, and
Tier 2 confirms it regardless (the waqf pipeline already reports missing words as ḥifẓ
`normal` errors). The live flag is an early preview of a verdict that was coming anyway.

## 5. What streams vs. what waits

| Signal | Tier | When |
|---|---|---|
| Cursor advance, word-heard | Tier 1 | live, ~every 300 ms |
| Clean skip of an expected word | Tier 1 | live, retroactively once a later word commits |
| Pronunciation errors, tashkīl, non-pausal tajwīd | Tier 2 | at the pause |
| **Pausal tajwīd** (مد العارض, pausal sukūn, pausal forms) | Tier 2 | at the pause (physically cannot be earlier) |
| ṣifāt | Tier 2 | at the pause |

## 6. Reconciliation

When Tier 2's `feedback` event lands, it **overrides** the provisional live marks for that
span. This is the *exact* override-by-key rule the frontend already runs for chunk-overlap
dedup (`lib/marks.ts`: keep the scored verdict over a later unverified re-emission).

Because Tier 1 only ever shows neutral "heard" (never a verdict), provisional → graded is
**neutral → verdict**, never the reversal of a prior verdict. There is no green→red flip.

## 7. Visual

- **Confirmed-live word:** the **same ink** as final "correct (zero mistakes)" text, just a
  hair brighter — imperceptible unless looked for. The word **settles** from near-black to
  full black the instant Tier 2 confirms `correct`, and flips to red only on a Tier 2
  `error`. No jarring palette switch.
- **Skip:** a gentle positional hint on the skipped word, not a mistake mark.
- This honors the existing `trimmed`-means-unverified philosophy: a provisional word is
  drawn, but never shown with a tick or a "correct" treatment until graded.

## 8. Protocol

Tier 1 emits a new lightweight server→client event, distinct from `feedback`:

```json
{"type": "progress", "confirmed": [{"sura":1,"aya":1,"word_idx":2}],
 "skipped": [{"sura":1,"aya":1,"word_idx":1}], "cursor": {"sura":1,"aya":1,"word_idx":2}}
```

- Carries coordinates only — no errors, no phoneme diffs, no ṣifāt.
- `feedback` (Tier 2) is unchanged and remains the authoritative event.
- A client that ignores `progress` degrades exactly to today's behavior.

## 9. Honest costs to design around

- **GPU load.** Muaalem now runs ~3×/s per active session instead of once per waqf. Fine on
  a real GPU for a few seconds of audio, but it multiplies with concurrent sessions.
  Mitigations required in the plan: a per-session cadence cap; skip a live pass if the
  previous one has not returned; bound the decode to the un-finalized tail of the buffer.
- **Overlap near a pause.** A live pass and the waqf pass can race on the same span. Dedup so
  no word is graded twice and the Tier 2 verdict always wins.
- **Engine scope.** Live tier is **Muaalem-only**. `mock` fabricates phonemes from the
  cursor; `zipformer` has no confidence head. On those engines Tier 1 is simply **off** — no
  regression, just no live-fill. `/health` and the session ack already tell the client which
  engine ran, so the client can decide whether to expect `progress` events.

## 10. Explicitly out of scope

- Shortening endpointing to force smaller chunks (the "cheap knob"): rejected — it slices
  words mid-utterance and misjudges pausal madd, degrading the exact accuracy this design
  protects.
- A separate on-device streaming aligner: deferred. Reusing Muaalem gets responsiveness
  everywhere now with zero new model. On-device live-fill can revisit this later if
  phone-side, network-free snappiness becomes a goal.

## 11. Success criteria

- Words visibly fill in during continuous recitation, without waiting for a pause.
- The final graded result for any session is identical to what today's pipeline produces.
- No word is ever shown as a hard error by the live tier; every criticism originates from
  Tier 2.
- On `mock`/`zipformer`, behavior is unchanged from today.
