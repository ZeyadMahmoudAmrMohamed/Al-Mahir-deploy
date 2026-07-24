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

**Status: the acoustic half is complete; the model half is not.** The Kaggle GPU tunnel
went offline mid-investigation, so every number below is either engine-independent (VAD,
audio, resampling) or was produced under the mock engine and is labelled as such. See
[Pending](#pending--needs-the-real-engine).

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

### Where to look instead

The complaint is real as an *experience*; it is not acoustic truncation. The remaining
candidates, in order:

1. **The model's tail confidence** — needs the real engine (see Pending). If the final
   phoneme group's CTC probability collapses while the audio is demonstrably present,
   the loss is in inference, not chunking.
2. **`trim_edges` greying the boundary word** (`words.py:138`) — a greyed word *looks*
   missing. `FINDINGS.md` measured this at 27.6% of words before overlap. This is the
   likeliest explanation for the reported experience and it is already understood.
3. **The alignment tier dropping the word**, not the chunker cutting it.

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

**No margin numbers are reported here.** Under the mock engine the predicted phonemes are
fabricated *from the reference at the cursor*, so `track()` matches perfectly by
construction and every margin is optimistic. Publishing those would be worse than
publishing nothing.

---

## Pending — needs the real engine

The Kaggle tunnel returned `ERR_NGROK_3200` (endpoint offline) partway through. These
three items are instrumented and ready; they need a live GPU and nothing else.

1. **Tail confidence.** `tail_report`'s `prob_drop` — the final phoneme group's CTC
   confidence against the chunk median (`contract.py:45` already carries it). Under mock
   every group reports 0.97, so the column is meaningless. This is the *second witness*
   for §2 and the most likely place the boundary complaint actually lives.
2. **Alignment margins.** §3's distribution, and the count of chunks won by < 0.05.
3. **Replay fidelity.** Whether a remote-GPU replay reproduces `events.jsonl` bit for bit,
   or whether float nondeterminism moves boundaries. If it diverges, the sweep methodology
   is qualified and that is itself a finding.

To run: restart the Kaggle server, update `TAJWID_REMOTE_URL` in `backend/.env`, then
`TAJWID_ASR_ENGINE=remote jupytext --to notebook experiments/streaming_flow.py` and execute.

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

1. **Restore the GPU and close the three Pending items.** Nothing else is blocked.
2. **Do not change `mic.ts`.** §1 — the proposed `new AudioContext()` is a regression on
   every Chrome client, with rate-dependent damage.
3. **Re-examine the boundary complaint as a greying problem, not a cutting one**
   (`feedback/words.py:138`). `FINDINGS.md`'s chunk-overlap work already addresses it and
   is implemented but off by default (`chunk_overlap_ms=0`). Turning it on is a
   one-variable experiment the replay harness can now run on a real captured session.
4. **Record a native-rate learner sample** via `scripts/record_ab.html` if §1 needs to be
   settled on a real reciter's voice rather than on studio audio.
