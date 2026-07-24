# Streaming flow investigation — findings

Continues `FINDINGS.md`. Where that round asked *why are words grey* and *why is correct
recitation flagged*, this one asks **what actually happens to the audio** between the
microphone and the grade: preprocessing, VAD/chunking, and alignment.

Every number here came from `backend/experiments/streaming_flow.py` (open it as a
notebook) over `backend/scripts/probe_stream.py` and `backend/scripts/resample_compare.py`.

Reproduce:

```bash
cd backend
pip install -e ".[probe]"

# the whole investigation, as a notebook
jupytext --to notebook experiments/streaming_flow.py

# or one measurement directly
TAJWID_ASR_ENGINE=mock python -c "
import sys; sys.path.insert(0,'scripts')
from probe_stream import probe, tail_report
from tajwid.feedback.types import Span
r = probe('scripts/A_processed.wav', start=Span(sura=2,aya=282,word_idx=0))
print(tail_report(r))"
```

**Status: complete.** Acoustic numbers are engine-independent; model numbers came from
Muaalem on a Tesla T4 over the remote tunnel.

**The headline:** the boundary complaint is real, but it is **not** the chunker cutting
audio. It is the model being uncertain about its *last phoneme group* because a chunk end
has no right-context. The two witnesses disagree with correlation **−0.24**, which is what
localises the fault. See [§2](#2-vad-and-chunking--the-boundary-complaint).

---

## New capability — diagnose mode

A **Diagnose** toggle in the engine menu records a real session's raw input for offline
replay. Double-gated: the server must be started with `TAJWID_CAPTURE_DIR` *and* the
reciter must turn it on. Neither key alone records audio.

```bash
TAJWID_CAPTURE_DIR=./captures tajwid-serve
```

Four files land in `<capture_dir>/<session_id>/`: `input.wav`, `frames.jsonl`,
`events.jsonl`, `start.json` — about 6 MB for three minutes.

**It records the input, not the intermediates**, because the pipeline is deterministic
given identical frames: silero is a TorchScript module in `eval()` on fixed 1536-sample
windows, `track`/`diff_recitation`/`trim_edges`/`match_forward` are pure, and the live
tier's cadence is *sample*-driven (`session.py:238`), not wall-clock. Verified — replay
reproduces the live session's chunk boundaries exactly
(`test_replay_reproduces_the_captured_chunk_boundaries`).

That inversion is the whole value. Saved intermediates are frozen at one parameter set
forever; a saved input can be re-run at any `vad_threshold`, `chunk_trail_pad_ms` or
`chunk_overlap_ms` on **your own recitation**:

```python
replay(SESSION, settings=settings_from_capture(SESSION, vad_threshold=0.45))
```

---

## §1 Preprocessing — the `AudioContext` question

**Asked:** would replacing `new AudioContext({ sampleRate: 16000 })` with
`new AudioContext()` change anything?

**The question is sharper than it looks.** `mic.ts:88` runs `decimate()` only when
`ratio !== 1`. Chrome honours the 16 kHz hint, so **that branch is dead code today** — it
is the Firefox/Safari fallback. Dropping the hint does not tweak the resampler. It moves
**every browser onto the fallback**.

That matters because `decimate()` is a box average, and a rectangular window is a `sinc`
in frequency with roughly −13 dB first sidelobes: energy above the destination Nyquist is
attenuated but not *removed*, so it folds back into the speech band.

### Measured, on the only two native-rate sources plus one low-rate

| file | source | ratio | broadband corr | **4–8 kHz box/poly** | effect |
|---|---|---|---|---|---|
| `fatiha_long_track.wav` | 48000 | 3.00 | 0.9986 | **0.744** | loses 26% of fricative energy |
| `dussary_002282.mp3` | 44100 | 2.76 | 0.9988 | **0.879** | loses 12% |
| `hussary_053001.mp3` | 22050 | 1.38 | 0.9918 | **2.719** | *adds* 172% aliased energy |

**The damage is not monotonic in the sample rate, and it goes in both directions.** At
ratio 1.38 the `Math.floor` boundaries produce 1- and 2-sample windows — effectively no
anti-alias filter at all — so content from 8–11 kHz folds down into 5–8 kHz. At ratio 3
the passband droop eats a quarter of the same band instead.

Unvoiced fricatives (س ش ص ث) live in exactly that band.

**Broadband correlation stays ≥ 0.99 throughout.** That is the trap: speech energy is
dominated by low frequencies, so a correlation check declares this harmless. It is not
harmless, it is just invisible to the wrong statistic.

### The mechanism, isolated

A 10 kHz tone at 48 kHz is above the 8 kHz destination Nyquist and must be removed. The
3-tap box average passes it at `sin(3πf)/(3·sin(πf))` with `f = 0.2083`, i.e. **0.506** —
half survives, and folds to `|10000 − 16000| = 6000 Hz`, landing in the middle of the
fricative band as energy the reciter never produced. `scipy.signal.resample_poly` leaves
under 1%. Tested at the fold frequency, not merely as a level difference
(`test_box_average_aliases_that_energy_into_the_fricative_band`).

The box average is **not** uniformly bad: at 1 kHz the two paths correlate > 0.99
(`test_the_two_filters_agree_well_below_nyquist`). The defect is confined to the top of
the band — which happens to be the part that distinguishes س from ص.

### Verdict

**Keep `new AudioContext({ sampleRate: 16000 })`.** The proposed change would move every
Chrome user off a native resampler onto a filter whose fricative-band behaviour depends
on the sample rate of whatever hardware they own. There is no measured upside to trade
against that.

### Limitation, stated rather than worked around

Neither native-rate file is learner-paced, and **capture cannot close this gap** — it
records what the browser sent, which is already 16 kHz and already noise-suppressed. A
native-rate learner recording via `scripts/record_ab.html` is what would settle it on a
real reciter's voice. This answers "does `decimate()` degrade fricatives" soundly; it does
not answer "does it degrade *your* recitation".

`noiseSuppression` was **not** re-tested — `FINDINGS.md:137` settled it on paired
simultaneous captures, where raw scored *worse* (16.9% word error vs 9.5%).

---

## §2 VAD and chunking — the boundary complaint

**Asked:** the last letter of the last word at a chunk boundary is missing.

**Mechanism under suspicion:** `stream.py:107` ends a chunk at `_last_speech_end_abs`, the
last window scoring above `vad_threshold` — which is **0.6, raised from silero's 0.3
default** (`config.py:84`) so shallow waqf dips register as silence. A raised threshold
truncates trailing low-energy phonemes earliest, and an unvoiced fricative or a
sukūn-final consonant carries little energy. `chunk_trail_pad_ms` (240) exists to cover
the difference.

**`tail_gap_ms`** measures whether it does: milliseconds of sound still above the noise
floor *after* the chunk ends. Reported at two thresholds — a chunk cut short at 25 dB is
losing a phoneme; one that qualifies only at 15 dB is trailing breath.

### Measured — every reference file

| file | pace | chunks | cut @15 dB | **cut @25 dB** | gap median | gap p95 |
|---|---|---|---|---|---|---|
| `A_processed.wav` | learner | 23 | 7 | **1** | 0 ms | 1968 ms |
| `B_raw.wav` | learner | 20 | 4 | **0** | 0 ms | 1017 ms |
| `fatiha_long_track.wav` | professional | 2 | 1 | **0** | 335 ms | 636 ms |
| `dussary_002282.mp3` | studio | 16 | 0 | **0** | 0 ms | 0 ms |
| `hussary_053001.mp3` | studio | 1 | 0 | **0** | 0 ms | 0 ms |
| **total** | | **62** | 12 | **1** | | |

**The complaint does not reproduce as acoustic truncation. One chunk in 62.** The median
gap is 0 ms — for most chunks the sound has already stopped when the chunk ends, with the
240 ms trail pad to spare. Both studio recitations are clean at both thresholds.

The single strict failure is `A_processed` chunk 2 (14.28–15.22 s, a waqf, gap 1260 ms).
Worth listening to; not a pattern.

This measurement is **engine-independent** — VAD plus audio, no model — so the GPU outage
does not qualify it.

### Two measurement bugs found by running against real audio

Recorded because both produced dramatic, hypothesis-*confirming* results, and neither was
catchable by the synthetic tests, which pass under both semantics.

1. **`sound_end_sample` took the last frame above the floor** within a 2 s window. On
   continuous recitation the next utterance begins a few hundred ms later, so it landed
   inside the *following* word: **22 of 23 chunks "cut short", every gap pinned at exactly
   the search limit.** The tail ends at the *first* frame below the floor.

2. **A 6 dB margin answers "is there any sound", not "is a phoneme still being
   articulated".** On `A_processed` the noise floor (p10) is 0.00065 and speech (p50) is
   0.064 — a 40 dB separation — so an inter-word breath clears 6 dB easily. Default raised
   to 15 dB, and the strict 25 dB figure is the one quoted above.

**A measurement that loudly agrees with the hypothesis it was built to test is the one to
distrust most.** Both of these did.

### The second witness — and it points somewhere else

Muaalem on the T4, `prob_drop` = chunk-median group confidence minus the **final** group's:

| | chunks | final_group_prob median | min | prob_drop median | max | chunks dropping >0.1 |
|---|---|---|---|---|---|---|
| `A_processed` | 23 | 0.959 | **0.000** | 0.000 | **0.796** | **8 (35%)** |
| `B_raw` | 20 | 0.944 | 0.279 | 0.021 | 0.718 | 5 (25%) |

Interior groups sit at a median **0.986**. So on roughly a third of chunks the model is
markedly unsure of the last thing it emitted — one chunk's final group scored **0.000**.

**But it does not line up with the audio.** Correlation between `tail_gap_ms` (strict) and
`prob_drop` is **−0.237** — if anything, *negative*. Mean `prob_drop` is 0.104 on chunks
whose audio is fully contained and *lower* on the handful that overrun.

The worst case makes it concrete:

```
seq forced gap15 gap25 final medn drop  final_group
  8 False   1390    10 0.189 0.986 0.796  'ج'
  1 False   2000     0 0.645 0.991 0.347  'ممم'
  9 False   1430     0 0.666 0.986 0.320  'مم'
 19 False   2000     0 0.736 0.993 0.257  'بُ'
 11 False      0     0 0.793 0.993 0.200  'ك'
```

Chunk 8's audio is acoustically complete — 10 ms of overrun against a 240 ms pad — and the
model still assigns 0.189 to its final ج.

### Verdict — right-context starvation, not truncation

**The reciter's complaint is real. The cause is not the chunker.** A CTC acoustic model
needs *future* frames to be confident about a phoneme. At a chunk boundary there are none:
the last group is decoded with the least context of anything in the chunk, so its
posterior is diffuse regardless of how much silence was padded after it. The failing final
groups are exactly what that predicts — consonants and nasals (ج، م، ب، ك، ل، ي), where the
distinguishing cue lies in the *following* transition.

Padding more audio cannot fix this: 240 ms of trailing silence adds no phonetic
right-context, only silence. This is the problem `whisper_streaming` and `SimulStreaming`
solve with **LocalAgreement** — never commit the last token(s) until a later, longer window
confirms them.

Two mechanisms in this codebase already implement that shape and are worth pointing at:

- **`chunk_overlap_ms`** (`config.py:104`, currently **0**) — re-sends the tail so the
  boundary word is *interior* next time, decoded with full right-context.
- **`live_lookahead_words`** (`config.py:115`) — the live tier already holds back its last
  word for exactly this reason (`live_aligner.py:44`). The authoritative tier has no
  equivalent.

`trim_edges` greying the boundary word (`words.py:138`) remains a *separate* contributor to
the same felt experience — a greyed word looks missing too — and `FINDINGS.md` measured it
at 27.6% before overlap.

---

## §3 Alignment — instrumented, not yet concluded

Both tiers are traced by `probe_stream.track_grid` and `probe_stream.live_trace`, verified
to agree with production `track()` (`test_track_grid_argmax_agrees_with_track`).

| | authoritative | live |
|---|---|---|
| where | `feedback/track.py:113` | `asr/live_aligner.py:32` |
| model | Muaalem, per waqf chunk | streaming zipformer, every 300 ms |
| method | grid search, argmax Levenshtein | single forward prefix scan |
| search space | offsets `[−6,+30)` × lengths `[len/9, len/2+1]` | expected prefix from the anchor |
| moves backwards | yes — reciters repeat | no, forward-only |
| on failure | `no_match`, widen by `penalty` | stall, wait to re-anchor |

**One design note worth recording.** `track_grid`'s `margin` is computed against the best
candidate at a *different start offset*, not the second-highest grid cell. The cell at
`(best_offset, best_n + 1)` is the same starting position with one more word and always
scores nearly the same — that is the length search being smooth, not ambiguity. Taking it
as the runner-up would report a near-zero margin for every chunk and make the metric
worthless.

### Measured, real engine

| file | chunks traced | status | best_ratio median / min | margin median / min | won by < 0.05 |
|---|---|---|---|---|---|
| `A_processed.wav` | 23/23 | **100% `ok`** | 0.977 / 0.276 | 0.130 / 0.018 | 2 |
| `dussary_002282.mp3` | 16/16 | **100% `ok`** | 1.000 / 0.848 | 0.103 / 0.017 | 1 |

**Alignment is not a problem.** Every chunk located, no `ambiguous`, no `no_match`, and the
studio recitation matches its reference *exactly* (ratio 1.000) more often than not. The
learner's 0.977 median is the ASR's error rate showing through, not a tracking failure.

Margins are comfortable: the winning span beats the best candidate at a different start by
0.13 typically. Three chunks across both files won by < 0.05 and are the only ones where
the window is genuinely ambiguous — worth a look if a mislocation is ever reported, but
they all still resolved `ok`.

---

## Replay fidelity

Not yet measured against a **remote-GPU** capture — it needs a session recorded through
Diagnose while the tunnel is up. Under the mock engine replay reproduces the live chunk
boundaries exactly (`test_replay_reproduces_the_captured_chunk_boundaries`). The open
question is only whether T4 float nondeterminism moves a boundary; if it does, the
parameter sweeps are qualified and that is itself a finding. The notebook's last cell
checks it automatically once `SESSION` points at a real capture.

---

## Ruled out — do not re-run these

Carrying forward `FINDINGS.md`'s list, plus this round's.

1. **Browser audio processing (`noiseSuppression` etc.).** Paired A/B; raw scored *worse*
   (16.9% vs 9.5%). `FINDINGS.md:137`.
2. **Short chunks.** Median 6.4 s live, 7.4 s studio.
3. **Chunks failing to score.** `scored_chunk_fraction` is 1.00.
4. **The 19 s forced cut.** `forced_cut_fraction` is 0.00 on every live recording.
5. **Frontend frame size (100 ms).** `asr/stream.py` re-windows to 1536 samples regardless.
6. **NEW — acoustic tail truncation.** 1 chunk in 62 at a defensible threshold; median gap
   0 ms. The 240 ms trail pad is adequate. §2 above.
7. **NEW — broadband correlation as a resampler test.** ≥ 0.99 for every path including
   one that adds 172% aliased fricative-band energy. Measure the band, not the waveform.

---

## Next

1. **Turn on `chunk_overlap_ms=2700`.** Measured on `A_processed`, real engine, one
   variable:

   | | overlap 0 | overlap 2700 |
   |---|---|---|
   | chunks with `prob_drop` > 0.1 | 8/23 | 6/23 |
   | max `prob_drop` | 0.796 | 0.477 |
   | median `final_group_prob` | 0.959 | 0.891 |
   | **grey words** | 13.4% | **0.0%** |
   | **correct** | 78.0% | **89.1%** |
   | error | 4.7% | 6.2% |

   **A prediction this document made, and got wrong.** The earlier draft predicted
   `prob_drop` would fall sharply because the affected groups "stop being final". It
   barely moved (8 → 6), and median final-group confidence got *worse*. Of course it
   did: overlap does not stop a chunk's last group from being last. Chunk N still ends
   with no right-context, it merely ends somewhere else.

   **The fix works through a different mechanism than the one predicted.** The boundary
   word is re-decoded as an *interior* word of chunk N+1, with full right-context, and
   `lib/marks.ts` keeps the scored verdict over the unverified one. That is a word-level
   win a per-chunk metric cannot see: grey 13.4% → 0.0%, correct 78% → 89%. Error rising
   is the expected accounting — ~13% of words moved from unverified to verified, and
   unverified was never a claim of correctness (`FINDINGS.md` makes the same point).

   The diagnosis stands; the causal chain runs through *re-decoding the word*, not
   through *improving the chunk's tail*.
2. **Do not change `mic.ts`.** §1 — the proposed `new AudioContext()` is a regression on
   every Chrome client, with rate-dependent damage in both directions.
3. **Consider a lookahead hold on the authoritative tier.** The live tier already refuses
   to commit its last word (`live_lookahead_words`, `live_aligner.py:44`); the Muaalem
   tier has no equivalent and confidently emits a group it scored 0.189. This is the
   `whisper_streaming` / `SimulStreaming` LocalAgreement idea, and overlap is the cheap
   version of it. Try overlap first.
4. **Record with Diagnose while the GPU is up** to close replay fidelity, and run the
   sweeps on a real reciter's own session.
5. **Record a native-rate learner sample** via `scripts/record_ab.html` if §1 needs to be
   settled on a reciter's voice rather than on studio audio.
