# %% [markdown]
# # Streaming flow investigation
#
# Three blocks: **preprocessing**, **VAD/chunking**, **alignment**. Run from `backend/`.
#
# Set `SESSION` to a captured session directory (produced by the Diagnose toggle) to run
# everything against a real recitation. Leave it `None` to use the reference files.
#
# ```bash
# cd backend
# jupytext --to notebook experiments/streaming_flow.py
# ```

# %%
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Audio, display

# Work from `backend/` whether this was launched there or from `experiments/` (nbconvert
# runs a notebook in its own directory). Every path below is then relative to one place.
ROOT = Path.cwd()
if not (ROOT / "scripts").is_dir():
    ROOT = ROOT.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "scripts"))

from probe_stream import (  # noqa: E402
    band_energy,
    live_trace,
    noise_floor,
    probe,
    replay,
    settings_from_capture,
    tail_report,
    track_grid,
)
from resample_compare import compare, decimate, filter_response  # noqa: E402
from tajwid.feedback.types import Span  # noqa: E402

# A captured session directory, produced by the Diagnose toggle. Set to None to fall
# back to the reference files. Pick the newest capture automatically so this does not
# need editing after every recitation.
_caps = sorted(Path("captures").glob("*/input.wav"), key=lambda p: p.stat().st_mtime)
SESSION = _caps[-1].parent if _caps else None

ASSETS = Path("tests/assets")
# (path, where the reciter starts). The start span seeds the cursor; the MOCK engine
# additionally needs it, since it fabricates its output from the phonetizer at the
# cursor and has nothing to say without one.
REFERENCE = [
    (Path("scripts/A_processed.wav"), Span(sura=2, aya=282, word_idx=0)),
    (Path("scripts/B_raw.wav"), Span(sura=2, aya=282, word_idx=0)),
    (ASSETS / "fatiha_long_track.wav", Span(sura=1, aya=1, word_idx=0)),
    (ASSETS / "dussary_002282.mp3", Span(sura=2, aya=282, word_idx=0)),
    (ASSETS / "hussary_053001.mp3", Span(sura=53, aya=1, word_idx=0)),
]

# %%
if SESSION:
    result = replay(SESSION)
    label = Path(SESSION).name
else:
    path, start = REFERENCE[0]
    result = probe(path, start=start)
    label = path.name

s = result.settings
print(f"{label}: {len(result.chunks)} chunks over "
      f"{result.audio.size / result.sample_rate:.1f} s")
print(f"vad_threshold={s.vad_threshold}  lead_pad={s.chunk_lead_pad_ms}ms  "
      f"trail_pad={s.chunk_trail_pad_ms}ms  overlap={s.chunk_overlap_ms}ms  "
      f"min_silence={s.min_silence_endpoint_ms}ms")

# %% [markdown]
# ---
# ## §1 Preprocessing — does dropping the 16 kHz hint cost anything?
#
# `mic.ts:88` runs `decimate()` only when `ratio !== 1`. Chrome honours the
# `sampleRate: 16000` hint, so **that branch is dead code today** — it is the
# Firefox/Safari fallback. `new AudioContext()` does not tweak the resampler; it moves
# **every** browser onto the fallback.

# %%
fig, ax = plt.subplots(figsize=(10, 3.6))
for ratio, label_ in ((3.0, "48k→16k (ratio 3)"),
                      (2.75, "44.1k→16k (ratio 2.75)"),
                      (1.38, "22.05k→16k (ratio 1.38)")):
    freqs, mag_db = filter_response(ratio)
    line, = ax.plot(freqs, mag_db, label=label_)
    # Everything surviving past the destination Nyquist folds back into the band.
    ax.axvline(0.5 / ratio, color=line.get_color(), ls=":", alpha=.6)
ax.axhline(-13.3, ls="--", c="grey", lw=1, label="first sidelobe ≈ −13 dB")
ax.set(xlabel="normalised frequency (cycles/sample of the INPUT rate)",
       ylabel="magnitude (dB)", ylim=(-60, 3),
       title="Box-average anti-alias response\n"
             "dotted = destination Nyquist; anything surviving to its right folds back in")
ax.legend(fontsize=8)
ax.grid(alpha=.3)
plt.tight_layout()

# %% [markdown]
# ### On real native-rate audio
#
# `band_ratio` is box ÷ polyphase energy in 4–8 kHz, where the unvoiced fricatives
# (س ش ص ث) live. **Below 1 means the box average lost fricative energy; above 1 means
# it added aliased energy that was never uttered.**

# %%
for p in (ASSETS / "fatiha_long_track.wav",
          ASSETS / "dussary_002282.mp3",
          ASSETS / "hussary_053001.mp3"):
    r = compare(str(p))
    print(f"{p.name:26s} src {r['src_sr']:6d}  ratio {r['ratio']:.2f}  "
          f"corr {r['correlation']:.4f}  4–8kHz box/poly {r['band_ratio']:.3f}  "
          f"rms_diff {r['rms_difference']:.5f}")

# %% [markdown]
# Broadband correlation stays ≥ 0.99 throughout, because speech energy is dominated by
# low frequencies — which is exactly why a correlation check would call this harmless.
# The fricative band tells a different story, and it is **not monotonic in the ratio**.
#
# **Limitation, stated rather than worked around.** None of these files is learner-paced,
# and a captured session cannot help: capture records what the browser sent, which is
# already 16 kHz and already noise-suppressed. Closing that gap needs a native-rate
# capture via `scripts/record_ab.html`.
#
# `noiseSuppression` is **not** re-tested. `FINDINGS.md:137` measured it on paired
# simultaneous captures and raw scored *worse* (16.9% word error vs 9.5%).

# %% [markdown]
# ---
# ## §2 VAD and chunking

# %%
def plot_vad(result, t0=0.0, t1=None):
    """Waveform, VAD probability and finalized chunks on one shared time axis."""
    sr = result.sample_rate
    t1 = t1 if t1 is not None else result.audio.size / sr
    a, b = int(t0 * sr), int(t1 * sr)
    audio = result.audio[a:b]
    t = np.arange(audio.size) / sr + t0

    probs = result.vad_probs
    pt = np.arange(probs.size) * result.settings.vad_window_samples / sr

    fig, axes = plt.subplots(2, 1, figsize=(13, 5), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(t, audio, lw=.4)
    axes[0].set(ylabel="Audio\nwaveform")
    axes[1].plot(pt, probs, lw=1.2)
    axes[1].axhline(result.settings.vad_threshold, c="orange", lw=1.4)
    axes[1].set(ylabel="VAD output\nprobability", xlabel="time (s)", ylim=(0, 1))

    for c in result.chunks:
        if c.end_s < t0 or c.start_s > t1:
            continue
        for ax_ in axes:
            ax_.axvspan(c.start_s, c.end_s, color="orange", alpha=.25, lw=0)
        axes[0].annotate(str(c.seq), (c.start_s, axes[0].get_ylim()[1] * .9),
                         fontsize=8, va="top")
    axes[1].set_xlim(t0, t1)
    plt.tight_layout()
    return fig


plot_vad(result, 0, min(30, result.audio.size / result.sample_rate));

# %% [markdown]
# ### The boundary question
#
# `stream.py:107` ends a chunk at `_last_speech_end_abs` — the last window scoring above
# `vad_threshold`, which is **0.6, raised from silero's 0.3 default** (`config.py:84`) so
# shallow waqf dips register as silence. A raised threshold truncates trailing low-energy
# phonemes earliest. `chunk_trail_pad_ms` (240) exists to cover the difference.
#
# `tail_gap_ms` is sound still above the noise floor *after* the chunk ends. It is
# reported at **two thresholds**: a chunk cut short at 25 dB is losing a phoneme, one
# that qualifies only at 15 dB is trailing breath. A single threshold hides that.
#
# `prob_drop` is the independent **model** witness — the final phoneme group's CTC
# confidence against the chunk median. Two witnesses agreeing is evidence; disagreeing
# means the symptom has another cause.

# %%
rows = tail_report(result)
g = [r["tail_gap_ms"] for r in rows]
gs = [r["tail_gap_ms_strict"] for r in rows]
drops = [r["prob_drop"] for r in rows if r["prob_drop"] is not None]

print(f"chunks                {len(rows)}")
print(f"cut short @15 dB      {sum(r['cut_short'] for r in rows)}")
print(f"cut short @25 dB      {sum(r['cut_short_strict'] for r in rows)}   <- the honest number")
print(f"tail_gap_ms  @15 dB   median {np.median(g):6.0f}  p95 {np.percentile(g, 95):6.0f}  max {max(g):6.0f}")
print(f"tail_gap_ms  @25 dB   median {np.median(gs):6.0f}  p95 {np.percentile(gs, 95):6.0f}  max {max(gs):6.0f}")
if drops:
    print(f"prob_drop             median {np.median(drops):.3f}  max {max(drops):.3f}")

# %%
fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
ax[0].hist([g, gs], bins=20, label=["15 dB", "25 dB (strict)"])
ax[0].axvline(result.settings.chunk_trail_pad_ms, c="r", ls="--", label="trail pad")
ax[0].set(xlabel="tail_gap_ms", ylabel="chunks", title="Sound left outside the chunk")
ax[0].legend(fontsize=8)

if drops:
    xs = [r["tail_gap_ms_strict"] for r in rows if r["prob_drop"] is not None]
    ax[1].scatter(xs, drops, s=20)
    ax[1].axvline(result.settings.chunk_trail_pad_ms, c="r", ls="--")
    ax[1].axhline(0, c="grey", lw=.8)
    ax[1].set(xlabel="tail_gap_ms (strict)", ylabel="median − final group prob",
              title="Acoustic cut vs model confidence at the tail")
else:
    ax[1].text(.5, .5, "no group probs\n(mock engine?)", ha="center", va="center")
    ax[1].set_axis_off()
plt.tight_layout()

# %% [markdown]
# ### Hear it
#
# Two players per chunk: the exact samples the model received, and the same span with
# 1 s either side. **The cut is what is _missing_ from the first**, so only the second
# can reveal it.

# %%
worst = sorted(rows, key=lambda r: -r["tail_gap_ms_strict"])[:5]
for r in worst:
    c = next(x for x in result.chunks if x.seq == r["seq"])
    print(f"--- chunk {r['seq']}  {r['start_s']:.2f}–{r['end_s']:.2f}s  "
          f"{'FORCED' if r['forced'] else 'waqf'}  "
          f"gap {r['tail_gap_ms']:.0f} ms (strict {r['tail_gap_ms_strict']:.0f})")
    if c.predicted_phonemes:
        print(f"    model : {c.predicted_phonemes[:90]}")
        print(f"    final group {r['final_group']!r} prob={r['final_group_prob']} "
              f"(chunk median {r['median_group_prob']})")
    print("    sent to the model:")
    display(Audio(c.wave, rate=result.sample_rate))
    print("    with 1 s of context either side:")
    display(Audio(result.context(c.seq, 1.0), rate=result.sample_rate))

# %% [markdown]
# ### Across every reference file
#
# Learner pace vs studio pace: if boundary truncation were a property of the pipeline it
# would appear everywhere; if it is a property of soft learner articulation it would not.

# %%
for path, start in REFERENCE:
    r = probe(path, start=start)
    t = r.tail = tail_report(r)
    gg = [x["tail_gap_ms"] for x in t]
    print(f"{path.name:26s} chunks {len(t):3d}  cut@15 {sum(x['cut_short'] for x in t):3d}  "
          f"cut@25 {sum(x['cut_short_strict'] for x in t):3d}  "
          f"gap median {np.median(gg):5.0f}  p95 {np.percentile(gg, 95):6.0f}")

# %% [markdown]
# ### Parameter sweeps
#
# These need a **capture**, not a file: the point is re-running *your own recitation*
# under settings the original session never used.

# %%
if SESSION:
    print("vad_threshold")
    for th in (0.3, 0.4, 0.5, 0.6):
        r = replay(SESSION, settings=settings_from_capture(SESSION, vad_threshold=th))
        t = tail_report(r)
        gg = [x["tail_gap_ms_strict"] for x in t]
        print(f"  {th}: {len(r.chunks):3d} chunks  gap median {np.median(gg):5.0f} ms  "
              f"cut@25 {sum(x['cut_short_strict'] for x in t)}")
else:
    print("Set SESSION to sweep.")

# %%
if SESSION:
    print("chunk_trail_pad_ms")
    for pad in (240, 400, 600):
        r = replay(SESSION, settings=settings_from_capture(SESSION, chunk_trail_pad_ms=pad))
        t = tail_report(r)
        print(f"  {pad:3d} ms: cut@25 {sum(x['cut_short_strict'] for x in t)} / {len(t)}")

# %%
if SESSION:
    print("chunk_overlap_ms  (grey/error per UNIQUE word, after the frontend's merge)")
    for ov in (0, 1300, 2700):
        r = replay(SESSION, settings=settings_from_capture(SESSION, chunk_overlap_ms=ov))
        words = {}
        for c in r.chunks:
            for w in c.words:
                k = (w["sura"], w["aya"], w["word_idx"])
                prev = words.get(k)
                if prev is None or (prev["trimmed"] and not w.get("trimmed")):
                    words[k] = {"trimmed": w.get("trimmed", False), "status": w["status"]}
        n = len(words) or 1
        grey = sum(1 for v in words.values() if v["trimmed"])
        err = sum(1 for v in words.values() if not v["trimmed"] and v["status"] == "error")
        print(f"  {ov:4d} ms: {n:4d} unique words  grey {grey/n:.1%}  error {err/n:.1%}")

# %% [markdown]
# ---
# ## §3 Alignment
#
# Two algorithms run concurrently and answer different questions:
#
# | | authoritative | live |
# |---|---|---|
# | where | `feedback/track.py:113` | `asr/live_aligner.py:32` |
# | model | Muaalem, per waqf chunk | streaming zipformer, every 300 ms |
# | method | grid search, argmax Levenshtein | single forward prefix scan |
# | search | offsets `[−6,+30)` × lengths `[len/9, len/2+1]` | expected prefix from the anchor |
# | backwards | yes (reciters repeat) | no, forward-only |
# | on failure | `no_match`, widen by `penalty` | stall, wait to re-anchor |

# %%
from tajwid.session import default_moshaf  # noqa: E402

moshaf = default_moshaf()
scored = [c for c in result.chunks if c.match_status and c.cursor_before]
grids = [(c, track_grid(c.predicted_phonemes, c.cursor_before, moshaf)) for c in scored]
print(f"{len(grids)} chunks traced")

margins = [gd["margin"] for _, gd in grids if gd["best"]]
if margins:
    print(f"win margin over the best DIFFERENT start: "
          f"median {np.median(margins):.3f}  min {min(margins):.3f}")
    narrow = [(c.seq, gd["margin"]) for c, gd in grids if gd["best"] and gd["margin"] < 0.05]
    print(f"{len(narrow)} chunks won by < 0.05 — the ambiguous ones: {narrow[:8]}")

# %% [markdown]
# A near-tie and a clear win both leave `track()` as a plain `ok`, but they are different
# failures: a near-tie means the window is genuinely ambiguous (the mutashābihāt case
# `track.py:128` describes), while a clear win that is still wrong means the phonemes
# were bad. Only the margin separates them.

# %%
if grids:
    c, gd = min(grids, key=lambda cg: cg[1]["margin"] if cg[1]["best"] else 1.0)
    fig, ax = plt.subplots(figsize=(11, 4))
    im = ax.imshow(gd["ratios"].T, aspect="auto", origin="lower", cmap="viridis",
                   extent=[gd["offsets"][0], gd["offsets"][-1],
                           gd["n_words"][0], gd["n_words"][-1]])
    if gd["best"]:
        ax.plot(gd["best"][0], gd["best"][1], "r*", ms=18)
    ax.set(xlabel="start offset from the cursor (words)",
           ylabel="candidate span length (words)",
           title=f"track() grid — chunk {c.seq} (narrowest win): "
                 f"best {gd['best_ratio']:.3f}, runner-up {gd['runner_up_ratio']:.3f}, "
                 f"margin {gd['margin']:.3f}")
    plt.colorbar(im, label="Levenshtein match ratio")
    plt.tight_layout()

# %%
for c, gd in grids[:8]:
    print(f"chunk {c.seq:3d}: {c.match_status:9s} best={gd['best_ratio']:.3f} "
          f"margin={gd['margin']:.3f}  words={len(c.words):2d} "
          f"trimmed={sum(1 for w in c.words if w.get('trimmed'))}")

# %% [markdown]
# ### Live tier
#
# Over-confirming is the dangerous direction (`live_aligner.py:56`): in hidden hifz mode
# a confirmed word is **revealed on the page**, so a tier running ahead hands the reciter
# the word they were trying to recall.

# %%
lt = live_trace(result)
if lt:
    ahead = [r["ahead_of_grade"] for r in lt if r["ahead_of_grade"] is not None]
    print(f"{len(lt)} live ticks; {sum(1 for a in ahead if a > 0)} ran ahead of the grade")
    if ahead:
        print(f"ahead_of_grade: median {np.median(ahead):.1f}  max {max(ahead)}")
        plt.figure(figsize=(9, 3))
        plt.plot(ahead, lw=1)
        plt.axhline(0, c="grey", ls="--")
        plt.ylabel("live tip − graded cursor (words)")
        plt.xlabel("tick")
        plt.title("Live tier vs authoritative cursor — positive means running ahead")
        plt.tight_layout()
else:
    print("No live ticks. The tier is gated to a zipformer build AND a Muaalem grader "
          "(session.py:210), so a mock/zipformer session correctly produces none.")

# %% [markdown]
# ---
# ## Replay fidelity
#
# Muaalem on a remote GPU is the one component that may not be bit-exact, so the replay
# is **checked** against what the live session actually emitted rather than trusted.
# A divergence here is a finding, not an embarrassment — it would invalidate the sweeps.

# %%
if SESSION and result.recorded_events:
    live_starts = sorted(round(e["audio_span_sec"][0], 2)
                         for e in result.recorded_events if e["type"] == "feedback")
    replay_starts = sorted(round(c.start_s, 2) for c in result.chunks if c.match_status)
    print("identical" if live_starts == replay_starts else "DIVERGED — this is a finding")
    print(f"  live   {live_starts[:12]}")
    print(f"  replay {replay_starts[:12]}")
else:
    print("Set SESSION to check replay fidelity.")
