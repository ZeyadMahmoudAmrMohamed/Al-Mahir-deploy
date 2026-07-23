# Quality investigation — findings

Root causes for the two quality complaints, with the evidence. Every number here came
from `backend/scripts/measure_chunks.py`; raw runs accumulate in
`backend/experiments/runs.jsonl`.

Reproduce any row:

```bash
cd backend
# VAD only, no GPU, ~2 s
python scripts/measure_chunks.py tests/assets/dussary_002282.mp3 --label repro
# full pipeline against the Kaggle GPU
TAJWID_ASR_ENGINE=remote TAJWID_REMOTE_URL=wss://<host>/infer \
  python scripts/measure_chunks.py scripts/A_processed.wav --engine --start 2:282:0 --label repro
```

`A_processed.wav` / `B_raw.wav` are one 186 s recitation of 2:282 captured through two
simultaneous `getUserMedia` streams (`backend/scripts/record_ab.html`) — A with
Chrome's echoCancellation/noiseSuppression/autoGainControl on, B with all three off.
Same performance, same pauses; only the browser processing differs.

---

## Complaint 1 — "roughly half the words render grey"

**Root cause: the trim policy costs a fixed 2 words per chunk, while the words a chunk
yields depends on recitation speed — and a learner recites ~3× slower than the studio
recitations the arithmetic was calibrated against.**

`grey_fraction ≈ 2 / words_per_chunk`, and
`words_per_chunk = chunk_duration × words_per_second`. The chunk duration was the only
term anyone examined. It was never the problem.

| | A_processed | B_raw | dussary (studio) |
|---|---|---|---|
| chunks | 23 | 20 | 16 |
| chunk duration median | 6.41 s | 7.90 s | 7.37 s |
| **words/chunk** min/med/p95 | 1 / **5** / 11.8 | 0 / **7.5** / 15.1 | 7 / **14.5** / 26.5 |
| trimmed fraction | **0.314** | **0.250** | 0.120 |
| predicted `2/median` | 0.400 | 0.267 | 0.138 |
| scored chunk fraction | 1.00 | 1.00 | — |

Chunk durations are healthy and near-identical across all three. Words per chunk is not:
**5 vs 14.5**. 137 words over 186 s is **0.74 words/second**; Dussary runs ~2.

Worst case is chunks that hold fewer than 3 words, where trimming blanks *everything*.
8 of A's 23 chunks:

```
seq  2: 0.94 s, 1 word,  1 trimmed → 100% grey
seq 14: 1.42 s, 2 words, 2 trimmed → 100% grey
seq  9: 2.38 s, 2 words, 2 trimmed → 100% grey
```

Observed 0.314 vs predicted 0.400 confirms `trim_edges` is the entire mechanism. No VAD
parameter can fix a chunk that legitimately contains one word — this is the structural
case for chunk overlap (Task 3).

## Complaint 2 — "words marked error on correct recitation"

**Root cause: the ṣifāt comparison is unreliable, and it produces ~95% of all findings
at a confidence that makes every one of them a hard `error`.**

| | ṣifāt findings | all findings | ṣifāt share | conf median | word error rate |
|---|---|---|---|---|---|
| A_processed | 42 | 44 | 95% | 1.000 | 9.5% |
| B_raw | 57 | 60 | 95% | 1.000 | 16.9% |
| fatiha (professional) | 152 | 161 | 94% | 1.000 | **80%** |

`fatiha_long_track.wav` is a professional recitation scoring **80% error**. That is not a
grading result; it is proof of false positives. It also reproduces Known Fact 6
(152 findings vs the 146 measured previously).

### Why `almost` never fires

The ṣifāt threshold at `normal` strictness is **0.85**, and ṣifāt confidences have
**median 1.000** (min 0.56 across all three files). The bucket that exists to prevent
false accusations is unreachable for the finding type responsible for 95% of findings.
Every ṣifāt disagreement is reported as a confident mistake.

### Which attribute

```
A_processed   shidda_or_rakhawa  38/42  (90%)   tikraar 4
B_raw         shidda_or_rakhawa  54/57  (95%)   tafashie 3
fatiha        shidda_or_rakhawa  51/152 (34%)   ghonna 32, hams_or_jahr 25,
                                                tafkheem_or_taqeeq 21, tikraar 9,
                                                safeer 8, itbaq 6
```

**The directions are symmetric**, which is the diagnostic detail:

```
ghonna=not_maghnoon -> maghnoon        16      ghonna=maghnoon -> not_maghnoon   16
hams_or_jahr=jahr -> hams              13      hams_or_jahr=hams -> jahr         12
shidda_or_rakhawa=rikhw -> between     14      shidda=between -> rikhw           13
tafkheem=mofakham -> moraqaq           11      tafkheem=moraqaq -> mofakham       8
```

A miscalibrated model head biased toward one label would be **asymmetric**. Symmetric
scatter in both directions, at confidence 1.000, is the signature of comparing correct
predictions against the **wrong reference groups** — an alignment or reference-derivation
fault, not a model fault. Known Fact 6 verified the *model* side of the wire
(`len(sifat) == len(groups)`, labels matching their groups); the untested side is
`feedback/reference.py`'s derivation of the expected ṣifāt.

### Counterfactual — words flagged if attributes are excluded

| | words | flagged now | excl. `shidda_or_rakhawa` | excl. all ṣifāt |
|---|---|---|---|---|
| A_processed | 137 | 13 (9%) | 4 (3%) | 2 (1%) |
| B_raw | 136 | 22 (16%) | 4 (3%) | 2 (1%) |
| fatiha | 25 | 20 (80%) | 20 (80%) | 9 (36%) |

Excluding `shidda_or_rakhawa` fixes the live recordings but does nothing for fatiha,
where ghonna/hams/tafkheem carry the errors. **This is not one bad attribute — the ṣifāt
comparison as a whole is untrustworthy.**

### Decision

**Stop grading ṣifāt until the reference derivation is verified.** No new code is needed:
the WS `start` message already takes `rules`, and the 10 ṣifāt are keys in it
(`GET /tajweed-rules`). Sending the 8 tajwīd rules without the ṣifāt keys turns ṣifāt
grading off through the existing leniency mechanism.

Rejected alternatives:
- *Raise the ṣifāt threshold* — confidences are 1.000; no threshold below 1.0 filters
  them, and one at 1.0 disables the feature anyway with more moving parts.
- *Exclude only `shidda_or_rakhawa`* — fixes 2 of 3 files, leaves fatiha at 80%.
- *Re-calibrate* — premature. If the reference side is wrong, calibration fits noise.

---

## Ruled out — do not re-run these

1. **Browser audio processing (`mic.ts`).** The leading hypothesis, dead. Paired A/B:
   raw audio scored *worse* (16.9% error vs 9.5%) and owned the only `no_match`. Chunk
   length differed by 1.5 s median, nowhere near enough to matter. Leave the constraints
   alone.
2. **Short chunks.** Median 6.4 s live, 7.4 s studio. Half-grey would need ~2 s chunks;
   the shortest median measured anywhere is 6.4 s.
3. **Chunks failing to score.** `scored_chunk_fraction` is 1.00 — every chunk matched
   `ok` except one 0.65 s fragment in B. Empty transcripts and `ambiguous`/`no_match`
   are not greying words.
4. **The 19 s forced cut.** `forced_cut_fraction` is 0.00 on every live recording. Only
   `fatiha_long_track.wav` (0.50) ever hits the cap, and that is a file artifact.
5. **Frontend frame size (100 ms).** Established previously; `asr/stream.py` re-windows
   to 1536 samples regardless.

## Chunk overlap — IMPLEMENTED, off by default

`TAJWID_CHUNK_OVERLAP_MS=2700` enables it; `0` (the default) is the historical
behaviour, byte for byte.

### Measured, one variable, 2700 ms vs 0

Per **unique word** after the frontend's merge rule — not per emission. With overlap a
word is emitted twice, trimmed by one chunk and scored by its neighbour, and
`lib/marks.ts` keeps the scored verdict. Counting emissions credits overlap with almost
nothing (31%→25%); counting what the reciter actually sees is the real number.

| | grey before | **grey after** | correct before | correct after | error before | error after |
|---|---|---|---|---|---|---|
| A_processed | 27.6% | **3.9%** | 70.9% | 85.2% | 1.6% | 10.2% |
| B_raw | 21.9% | **2.3%** | 76.6% | 88.3% | 1.6% | 7.0% |

**Grey is solved** — 27.6% → 3.9%, matching the 3-5% predicted. The residue is the
genuine first/last edge of the session, which nothing can verify.

**Read the error column honestly.** It rose because ~24% of words moved from *unverified*
to *verified*, and unverified was never a claim of correctness. Of the words overlap
newly scored, ~85% came back `correct`. Whether the remaining ~10% are true boundary
mistakes or new false positives is **not established** — chunk-edge words are exactly
where a reciter's articulation is weakest, so a raised error rate there is plausible on
the merits, but it has not been verified against a labelled set.

### Waqf artefact — the pausal-form bug overlap introduced, now fixed

The first overlap build raised *false* errors on correct recitation. Mechanism:

When a reciter stops at a word (waqf), they recite it in **pausal form** — the final
haraka drops to sukūn, and there is no cross-word ghunnah/madd, because they did not
connect. The phonetizer renders a **span-final** word pausally too (`رَيْبَ` → `...بڇ`,
verified), so in the chunk where that word is last, audio and reference agree.

Overlap re-sends that word as the *interior* head of the next chunk — where the phonetizer
renders it **connected** (`...بَ`). The audio is still pausal. The diff then reports a
tashkeel error (`expected fatha, got sukūn`) the reciter never made — the exact false
accusation Constitution VI forbids.

Two-part fix (`feedback/words.py`, `feedback/pipeline.py`, `session.py`):

1. **`trim_edges` no longer greys the last word at a waqf** — only on a *forced* cut. At
   a waqf the word is complete and its reference is pausal, so it is scored (correctly) in
   its own chunk. Bonus: this alone drops grey ~28% → ~13% with overlap *off*.
2. **The overlap chunk does not re-grade that word.** The entry cursor already equals the
   previous chunk's span-final word (`advance` sets `cursor = end`), so `blank_word_at`
   skips it. Suppressed only when the previous chunk was a waqf; a forced cut still lets
   overlap rescue its chopped word.

Measured, per unique word, overlap 1300 ms, sifat off:

| | error BEFORE waqf fix | error AFTER | grey after | almost after |
|---|---|---|---|---|
| A_processed | 10.2% | **3.9%** | 2.3% | 6.2% |
| B_raw | 7.0% | **4.7%** | 2.3% | 1.6% |

Residual ~4% error is not proven all-real — some genuine mistakes, some boundary residue —
but it is in range, cushioned by `almost`, and the systematic pausal false-positive is
gone. Ghunnah-on-non-connection is a `ghonna` *sifa*, already suppressed by sifat-off.

### Why 2 words, not 1

`trim_edges` blanks `words[0]` whenever `start.word_idx > 0`. One word of overlap does
not help: the trimmed last word of chunk N simply becomes `words[0]` of chunk N+1 and is
trimmed again. **Two words is the minimum that works** — the previously-trimmed word
lands at `words[1]`, interior to the span, and gets scored.

At 0.74 words/s (measured) two words is ~2.7 s; at studio pace ~1.0 s. A fixed
`chunk_overlap_ms` of ~1200 covers the fast case and under-covers a slow reciter, so this
should be **words-driven, not time-driven** where possible — or set generously, since the
cost is linear and small.

### Mechanism

In `StreamSession._reset_after_finalize`, stop dropping the buffer all the way to
`cut_abs`: retain `overlap_samples` so the next chunk's `_extract` naturally begins
inside the previous utterance. No new code path, no second inference queue — the existing
lead-pad lookback in `_trim` already does exactly this at smaller scale.

### Deduplication

Words carry absolute `(sura, aya, word_idx)`, so dedup is a key collision, not a
heuristic: **for a repeated word, the non-trimmed instance wins; if both are scored, the
later one wins** (it saw more right-context). The frontend's `marks.ts` already merges by
word key, so overlapping feedback events converge rather than double-paint. Nothing is
shown or counted twice.

### Does the tracker need changes? No.

`track(overlap_words=6)` already searches 6 words *backwards* from the cursor, because
"reciters legitimately repeat and back up while memorising". A 2-word rewind is well
inside that window. This is the single biggest reason to prefer overlap over VAD tuning:
the expensive part is already built.

### Does this contradict `asr/stream.py:145`?

It **extends** it, and the distinction matters. That line skips the trailing pad on a
forced cut because it "would duplicate the next chunk's onset" — written when duplicated
audio meant duplicated *words*, with nothing to reconcile them. Overlap makes duplication
deliberate and adds the dedup rule that makes it safe. Once words are merged by key, the
original objection is answered; without that rule it still stands. **Implement dedup
first, then overlap** — in the other order, the duplication is a regression.

### The cap interaction — a bug this surfaced

The overlap is prepended *after* the `max_chunk_s` force-cut check, so the first
implementation emitted a capped chunk at **20.34 s** — past the 19 s cap and past the
≤20 s Muaalem was trained on. Fixed by charging the overlap and the lead pad to the
budget before cutting (`stream.py`), with a regression test. Note a waqf-finalized chunk
can still reach ~19.05 s (cap + trail pad); that is within the model's limit, but it is
the number to watch if `chunk_overlap_ms` is raised much further.

### Cost

~2.7 s of re-sent audio on a median 6.4 s chunk ≈ **40% more inference** at the measured
recitation speed (~19% at studio pace). Live inference is batch-1 on a T4 and is not
currently latency-bound, so this is affordable — but it is the number to watch, and Task 8
(per-stage benchmark) should run before and after.

### Expected effect

Grey drops from `2 / words_per_chunk` toward `~0` for interior chunks; only the very first
and very last chunk of a session retain a genuine unverified edge. On A_processed that is
**31% → an estimated 3-5%**. Verify with `--label overlap-on` against the runs already in
`runs.jsonl`.

## Task 8 — per-stage benchmark and the bottleneck

Diagnosis only, no optimisation (as scoped). Measured with `measure_chunks.py`'s
`--profile` wrapping, which times the three stages without touching production code:
inference (`engine.transcribe_chunk`), feedback (`analyse_session` = track + diff + score),
and VAD (summed inside the silero shim). Real engine over the Kaggle tunnel (Tesla T4),
overlap 1300 ms, sifat off. Per emitted chunk:

| stage | where it runs | median | p95 | max |
|---|---|---|---|---|
| **inference** (Muaalem) | remote T4 + tunnel | **0.46–1.02 s** | 1.6–1.9 s | 2.5 s |
| feedback (track+diff+score) | local CPU | 0.27 s | 0.45 s | 0.55 s |
| VAD (silero) | local CPU | 0.18 s/chunk | — | — |

**Inference is the bottleneck** — the largest stage by 2–4×. It warms down across repeated
passes (median 0.94 s → 0.46 s), so steady state is ~0.5 s; the first passes pay a
connection/cache cost. Feedback and VAD are stable and small.

**Not a latency problem.** A median 7.7 s chunk costs ~0.5–1.0 s to score → ~8–15×
realtime headroom. Nothing here threatens real-time feedback; this is a profile, not a
regression.

**A correction worth recording.** An earlier estimate derived feedback from a *mock* run
at 0.63 s and called it co-equal with inference. The real profiled feedback is **0.27 s** —
the mock's phonetizer path inflated it. Measure the stage in isolation; do not subtract.

**Inside feedback**, the suspects if it ever needs attention: `diff_recitation`
re-phonetizes the *predicted* string every chunk (uncached — unlike `build_reference`,
which is `lru_cache`d), and `track` scans offset×n_words. Neither is worth touching at
0.27 s.

**GPU utilisation / VRAM — unresolved.** `nvidia-smi` reported `0% / 3 MiB`, i.e. an empty
GPU: `remote_server.py`'s `uvicorn.run` blocks its Kaggle cell, so the query ran in a
second notebook on a *different* GPU. So the split of that ~0.5 s between T4 compute and
tunnel round-trip is not established. To get it: launch the server non-blocking
(`subprocess.Popen`) and sample `nvidia-smi` in the same kernel under load, or add
`torch.cuda.memory_allocated()` to `/health` and query it over the tunnel. Deferred — the
bottleneck (inference) is identified regardless, and Task 8 forbids acting on it.

## Next

1. **Turn ṣifāt grading off** via the existing `rules` mechanism. One-line default change,
   drops the live error rate from 9–16% to ~1–3% and stops accusing correct recitation.
2. **Audit `feedback/reference.py`'s ṣifāt derivation** — hand-check expected attributes
   against tajwīd rules for a few letters (ب should be jahr + shadeed, س hams + rikhw).
   The symmetric confusion says the reference, not the model, is wrong. This is what
   decides whether ṣifāt can ever be turned back on.
3. **Chunk overlap** — the real fix for complaint 1. Re-send the tail of chunk N as the
   head of N+1 so boundary words become scoreable. Design pending (Task 3).
