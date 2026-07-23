"""Task 1 harness: stream an audio file through the live path and report chunk stats.

Two modes:
  --vad-only  (default)  silero endpointing only. No model, no GPU, runs anywhere.
                         Answers "why are the chunks short" — duration/forced/rate.
  --engine    full LiveSession (mock / real / remote). Adds words-per-chunk,
              trimmed fraction and the status split, which need the feedback half.

For the Kaggle tunnel:
    TAJWID_ASR_ENGINE=remote TAJWID_REMOTE_URL=wss://<sub>.ngrok.app/infer \
      python scripts/measure_chunks.py tests/assets/fatiha_long_track.wav \
        --engine --start 1:1:0 --label kaggle-baseline

Every run appends one JSON line to backend/experiments/runs.jsonl (Task 6) and dumps
the per-window silero probabilities to a .npy beside it (Task 7's plot input).
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from tajwid.asr.batch import load_audio
from tajwid.asr.stream import StreamSession
from tajwid.asr.vad import load_vad
from tajwid.config import get_settings
from tajwid.feedback.types import Span

RUNS = Path(__file__).resolve().parent.parent / "experiments" / "runs.jsonl"

# The tunables whose effect we are measuring; recorded with every run so two lines of
# runs.jsonl can be compared later instead of remembered.
TRACKED = (
    "vad_threshold",
    "min_silence_endpoint_ms",
    "min_speech_ms",
    "max_chunk_s",
    "chunk_lead_pad_ms",
    "chunk_trail_pad_ms",
    "vad_window_samples",
)


class ProbeVad:
    """Records every silero speech probability. The VAD is called as vad(window, sr)
    and reset via reset_states(), so a two-method proxy is the whole shim."""

    def __init__(self, vad):
        self._vad = vad
        self.probs: list[float] = []
        self.total_s = 0.0  # cumulative silero time, so VAD gets its own stage figure

    def __call__(self, window, sr):
        t = time.perf_counter()
        p = self._vad(window, sr)
        self.total_s += time.perf_counter() - t
        self.probs.append(float(p))
        return p

    def reset_states(self):
        self._vad.reset_states()


def pct(xs, q):
    return float(np.percentile(xs, q)) if xs else None


def summary(xs):
    if not xs:
        return None
    return {
        "min": round(min(xs), 3),
        "median": round(st.median(xs), 3),
        "p95": round(pct(xs, 95), 3),
        "max": round(max(xs), 3),
    }


def run_vad_only(wave, sr, frame):
    probe = ProbeVad(load_vad())
    stream = StreamSession(probe, get_settings())
    stream.vad = probe
    chunks = []
    t0 = time.perf_counter()
    for i in range(0, len(wave), frame):
        for fin in stream.feed(wave[i : i + frame]):
            chunks.append(fin)
    chunks.extend(stream.flush())
    wall = time.perf_counter() - t0
    rows = [
        {
            "seq": i,
            "start_s": round(c.start_sample / sr, 3),
            "duration_s": round((c.end_sample - c.start_sample) / sr, 3),
            "forced": c.forced,
        }
        for i, c in enumerate(chunks)
    ]
    return rows, probe.probs, wall


def run_engine(path, wave, sr, frame, start: Span | None):
    from tajwid.asr.engine import make_engine
    from tajwid.session import LiveSession

    session = LiveSession(make_engine(), session_id=Path(path).stem, start=start)
    probe = ProbeVad(session.stream.vad)
    session.stream.vad = probe

    # Per-stage timing (Task 8). The three stages that dominate a chunk, wrapped without
    # touching production code: inference (engine.transcribe_chunk, GPU/tunnel) and
    # feedback (analyse_session = track + diff + sifat + score, CPU-local). VAD is
    # summed inside ProbeVad. Preprocessing (float cast, PCM pack) lives inside the
    # engine call; it is not separable from the wire here and is negligible beside it.
    stage: dict[str, list[float]] = {"inference": [], "feedback": []}

    import tajwid.session as _sess

    _real_transcribe = session.engine.transcribe_chunk
    def _timed_transcribe(*a, **kw):
        t = time.perf_counter()
        r = _real_transcribe(*a, **kw)
        stage["inference"].append(time.perf_counter() - t)
        return r
    session.engine.transcribe_chunk = _timed_transcribe

    _real_analyse = _sess.analyse_session
    def _timed_analyse(*a, **kw):
        t = time.perf_counter()
        r = _real_analyse(*a, **kw)
        stage["feedback"].append(time.perf_counter() - t)
        return r
    _sess.analyse_session = _timed_analyse

    # Record what ENDPOINTING produced, before the engine can swallow it. A dead remote
    # GPU returns an empty transcript, LiveSession._process then returns None, and the
    # chunk would vanish from an events-derived count — losing VAD numbers that never
    # needed a GPU in the first place.
    endpointed: list = []
    for name in ("feed", "flush"):
        orig = getattr(session.stream, name)
        def wrapped(*a, _orig=orig, **kw):
            got = _orig(*a, **kw)
            endpointed.extend(got)
            return got
        setattr(session.stream, name, wrapped)

    events, lat = [], []
    t0 = time.perf_counter()
    for i in range(0, len(wave), frame):
        t = time.perf_counter()
        got = session.feed(wave[i : i + frame])
        if got:
            lat.append(time.perf_counter() - t)
        events.extend(got)
    got = session.flush()
    if got:
        lat.append(time.perf_counter() - t0)
    events.extend(got)
    wall = time.perf_counter() - t0

    # Endpointed chunks are the spine; feedback is attached where the engine produced any.
    findings: list[dict] = []
    seen: dict[str, dict] = {}
    by_span = {}
    for e in events:
        by_span[round(e["audio_span_sec"][0], 3)] = e
    rows = []
    for i, c in enumerate(endpointed):
        row = {
            "seq": i,
            "start_s": round(c.start_sample / sr, 3),
            "duration_s": round((c.end_sample - c.start_sample) / sr, 3),
            "forced": c.forced,
            "scored": False,
        }
        e = by_span.get(row["start_s"])
        if e is None:
            rows.append(row)
            continue
        fb = e["feedback"]
        words = fb.get("words") or []
        # What the RECITER sees, not what the wire carried. With overlap a word is
        # emitted twice — trimmed by one chunk, scored by its neighbour — and
        # frontend lib/marks.ts keeps the scored verdict. Counting emissions would
        # credit overlap with nothing; counting unique words is the real metric.
        for w in words:
            key = f"{w['sura']}:{w['aya']}:{w['word_idx']}"
            prev = seen.get(key)
            if prev is None or (prev["trimmed"] and not w.get("trimmed")):
                seen[key] = {"trimmed": w.get("trimmed", False), "status": w["status"]}
        for w in words:
            for err in w.get("errors") or []:
                findings.append(
                    {
                        "chunk": row["seq"],
                        "word": f"{w['sura']}:{w['aya']}:{w['word_idx']}",
                        "word_status": w["status"],
                        "trimmed": w.get("trimmed", False),
                        "error_type": err["error_type"],
                        "speech_error_type": err["speech_error_type"],
                        "confidence": err.get("confidence"),
                        "expected_ph": err.get("expected_ph"),
                        "predicted_ph": err.get("predicted_ph"),
                        "rules": [r["name_en"] for r in err.get("tajweed_rules") or []],
                    }
                )
        row |= {
            "scored": True,
            "match": fb["status"],
            "n_words": len(words),
            "n_trimmed": sum(1 for w in words if w.get("trimmed")),
            "status_counts": {
                s: sum(1 for w in words if w["status"] == s)
                for s in ("correct", "almost", "error")
            },
        }
        rows.append(row)

    # Restore the module-level patch so processing a second file doesn't double-wrap it.
    _sess.analyse_session = _real_analyse
    stage["vad"] = [probe.total_s]  # summed across all windows, one figure for the run
    return rows, probe.probs, wall, lat, findings, seen, stage


def metrics(rows, audio_s, seen=None):
    durs = [r["duration_s"] for r in rows]
    out = {
        "audio_s": round(audio_s, 2),
        "n_chunks": len(rows),
        "chunks_per_min": round(len(rows) / (audio_s / 60), 2) if audio_s else None,
        "chunk_duration_s": summary(durs),
        "forced_cut_fraction": round(
            sum(1 for r in rows if r["forced"]) / len(rows), 3
        )
        if rows
        else None,
    }
    scored = [r for r in rows if r.get("scored")]
    if rows and any("scored" in r for r in rows):
        out["scored_chunk_fraction"] = round(len(scored) / len(rows), 3)
    if scored:
        rows = scored
        wc = [r["n_words"] for r in rows]
        total_w = sum(wc)
        total_t = sum(r["n_trimmed"] for r in rows)
        agg = {s: sum(r["status_counts"][s] for r in rows) for s in ("correct", "almost", "error")}
        out |= {
            "words_per_chunk": summary(wc),
            "total_words": total_w,
            "trimmed_fraction": round(total_t / total_w, 3) if total_w else None,
            "status_fraction": {
                s: round(n / total_w, 3) for s, n in agg.items()
            }
            if total_w
            else None,
            "match_status_counts": {
                s: sum(1 for r in rows if r["match"] == s)
                for s in {r["match"] for r in rows}
            },
        }
        # Known Fact 1's arithmetic: grey_fraction should be ~ 2 / median_words_per_chunk.
        med = st.median(wc) if wc else 0
        out["predicted_trimmed_fraction"] = round(2 / med, 3) if med else None
    if seen:
        # THE user-visible numbers: unique words after the frontend's merge rule.
        n = len(seen)
        grey = sum(1 for v in seen.values() if v["trimmed"])
        out["unique_words"] = n
        out["grey_fraction_deduped"] = round(grey / n, 3)
        out["status_fraction_deduped"] = {
            s: round(
                sum(1 for v in seen.values() if not v["trimmed"] and v["status"] == s) / n, 3
            )
            for s in ("correct", "almost", "error")
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", nargs="+")
    ap.add_argument("--engine", action="store_true", help="run the full pipeline, not just VAD")
    ap.add_argument("--start", help="cursor seed, e.g. 1:1:0")
    ap.add_argument("--frame-ms", type=int, default=100)
    ap.add_argument("--label", default="baseline")
    ap.add_argument("--note", default="")
    ap.add_argument("--per-chunk", action="store_true", help="print every chunk row")
    args = ap.parse_args()

    s = get_settings()
    sr = s.sample_rate
    frame = int(args.frame_ms * sr / 1000)
    start = None
    if args.start:
        a, b, c = (int(x) for x in args.start.split(":"))
        start = Span(sura=a, aya=b, word_idx=c)

    RUNS.parent.mkdir(exist_ok=True)
    for path in args.audio:
        wave = load_audio(path, sr).numpy()
        if args.engine:
            rows, probs, wall, lat, findings, seen, stage = run_engine(
                path, wave, sr, frame, start
            )
        else:
            rows, probs, wall = run_vad_only(wave, sr, frame)
            lat, findings, seen, stage = [], [], None, None

        m = metrics(rows, len(wave) / sr, seen)
        n_chunks = m["n_chunks"] or 1
        run = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "label": args.label,
            "note": args.note,
            "audio": Path(path).name,
            "mode": "engine" if args.engine else "vad-only",
            "engine": s.resolved_asr_engine if args.engine else None,
            "params": {k: getattr(s, k) for k in TRACKED},
            "wall_s": round(wall, 2),
            "chunk_latency_s": summary(lat),
            "metrics": m,
        }
        if stage:
            # Per-stage: inference and feedback are per-chunk (mean/p95); VAD is one sum
            # for the whole stream, so also express it per chunk for comparison.
            run["stage_timing_s"] = {
                "inference": summary(stage["inference"]),
                "feedback": summary(stage["feedback"]),
                "vad_total": round(stage["vad"][0], 3),
                "vad_per_chunk": round(stage["vad"][0] / n_chunks, 4),
            }
        stem = f"{args.label}_{Path(path).stem}"
        np.save(RUNS.parent / f"probs_{stem}.npy", np.asarray(probs, dtype=np.float32))
        run["probs_file"] = f"probs_{stem}.npy"
        run["chunks"] = rows
        if findings:
            fp = RUNS.parent / f"findings_{stem}.jsonl"
            with fp.open("w", encoding="utf-8") as f:
                for x in findings:
                    f.write(json.dumps(x, ensure_ascii=False) + "\n")
            run["findings_file"] = fp.name
            by_type: dict[str, dict] = {}
            for x in findings:
                k = f"{x['error_type']} {x['speech_error_type']}"
                d = by_type.setdefault(k, {"n": 0, "unscored": 0, "conf": []})
                d["n"] += 1
                if x["confidence"] is None:
                    d["unscored"] += 1
                else:
                    d["conf"].append(x["confidence"])
            run["findings_breakdown"] = {
                k: {
                    "n": d["n"],
                    "unscored": d["unscored"],
                    "conf_min": round(min(d["conf"]), 3) if d["conf"] else None,
                    "conf_median": round(st.median(d["conf"]), 3) if d["conf"] else None,
                    "conf_max": round(max(d["conf"]), 3) if d["conf"] else None,
                }
                for k, d in sorted(by_type.items(), key=lambda kv: -kv[1]["n"])
            }
        with RUNS.open("a", encoding="utf-8") as f:
            f.write(json.dumps(run, ensure_ascii=False) + "\n")

        print(f"\n=== {Path(path).name}  [{run['mode']}]  {json.dumps(run['params'])}")
        print(json.dumps(m, ensure_ascii=False, indent=2))
        if "stage_timing_s" in run:
            print("stage timing (s):")
            print(json.dumps(run["stage_timing_s"], ensure_ascii=False, indent=2))
        if "findings_breakdown" in run:
            print("findings by type:")
            print(json.dumps(run["findings_breakdown"], ensure_ascii=False, indent=2))
        if args.per_chunk:
            for r in rows:
                print("   ", json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
