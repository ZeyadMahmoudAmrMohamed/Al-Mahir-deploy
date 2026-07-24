"""Analyse a Diagnose capture: VAD plots, a summary, and (optionally) the notebook.

    python scripts/analyze_capture.py                 # newest capture
    python scripts/analyze_capture.py --notebook      # also execute the notebook
    python scripts/analyze_capture.py --watch         # every new recitation, forever
    python scripts/analyze_capture.py captures/<id>   # a specific one

``--watch`` polls for new capture directories and analyses each once its audio stops
growing. Polling, not a filesystem watcher: a capture is only "done" when the WebSocket
closed, and nothing writes a marker for that (`{"type":"done"}` goes straight to the
socket in ws.py, not through the recorder). Size-stability is the honest test and needs
no dependency.

Deliberately a separate process, not something the server does on session end: analysis
re-runs the whole recitation through the model, and a reciter finishing a session should
not wait on that -- nor should a GPU outage during analysis be able to affect recording.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_stream import (  # noqa: E402
    compare_to_live,
    replay,
    save_vad_plots,
    tail_report,
)

ROOT = Path(__file__).resolve().parent.parent
CAPTURES = ROOT / "captures"


def newest_capture() -> Path | None:
    caps = sorted(CAPTURES.glob("*/input.wav"), key=lambda p: p.stat().st_mtime)
    return caps[-1].parent if caps else None


def is_settled(session_dir: Path, quiet_s: float = 4.0) -> bool:
    """True once the capture has stopped growing -- i.e. the session ended."""
    wav = session_dir / "input.wav"
    if not wav.exists():
        return False
    return (time.time() - wav.stat().st_mtime) >= quiet_s


def analyse(session_dir: Path, notebook: bool = False) -> dict:
    session_dir = Path(session_dir)
    print(f"\n=== {session_dir.name}")
    result = replay(session_dir)
    sr = result.sample_rate

    plots = save_vad_plots(result, session_dir / "vad_plots")
    print(f"  {len(plots)} VAD plots -> {session_dir / 'vad_plots'}")

    rows = tail_report(result)
    gaps = [r["tail_gap_ms_strict"] for r in rows]
    print(
        f"  {len(rows)} chunks over {result.audio.size / sr:.1f}s  "
        f"cut@25dB {sum(r['cut_short_strict'] for r in rows)}  "
        f"cut@15dB {sum(r['cut_short'] for r in rows)}  "
        f"tail_gap median {np.median(gaps) if gaps else 0:.0f}ms"
    )

    drops = [r["prob_drop"] for r in rows if r["prob_drop"] is not None]
    if drops and max(drops) > 0:
        print(
            f"  final-group confidence: drop median {np.median(drops):.3f}  "
            f"max {max(drops):.3f}  chunks dropping >0.1: {sum(1 for d in drops if d > 0.1)}"
        )

    cmp = compare_to_live(result)
    if cmp["dropped_by_live"]:
        print(
            f"  !! {len(cmp['dropped_by_live'])} chunk(s) LOST BY THE LIVE SESSION at "
            f"{cmp['dropped_by_live']} s -- recited, endpointed, never graded. "
            f"The engine returned an empty transcript (session.py's phonemes_text guard)."
        )
    if cmp["missing_from_replay"]:
        print(
            f"  !! replay did not reproduce {cmp['missing_from_replay']} -- this one "
            f"would qualify the parameter sweeps."
        )
    if cmp["identical"]:
        print("  replay matches the live session exactly")

    if notebook:
        run_notebook()
    return cmp


def run_notebook() -> None:
    """Execute the notebook against whatever capture it auto-selects (the newest)."""
    src = ROOT / "experiments" / "streaming_flow.py"
    nb = ROOT / "experiments" / "streaming_flow.ipynb"
    print("  executing the notebook (this re-runs the recitation several times)...")
    subprocess.run(
        [sys.executable, "-m", "jupytext", "--to", "notebook", str(src)],
        check=True,
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
    )
    subprocess.run(
        [
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "notebook", "--execute",
            "--ExecutePreprocessor.timeout=3000",
            str(nb), "--output", "streaming_flow_run.ipynb",
        ],
        check=True,
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
    )
    print(f"  -> {ROOT / 'experiments' / 'streaming_flow_run.ipynb'}")


def watch(notebook: bool, poll_s: float = 3.0) -> None:
    print(f"watching {CAPTURES} for new recitations (Ctrl-C to stop)")
    CAPTURES.mkdir(exist_ok=True)
    done = {p.parent for p in CAPTURES.glob("*/input.wav")}
    print(f"  ignoring {len(done)} existing capture(s); analysing only new ones")
    while True:
        for wav in CAPTURES.glob("*/input.wav"):
            d = wav.parent
            if d in done or not is_settled(d):
                continue
            done.add(d)
            try:
                analyse(d, notebook=notebook)
            except Exception as err:  # noqa: BLE001 -- one bad capture must not stop
                # the watcher; the next recitation should still be analysed.
                print(f"  analysis failed for {d.name}: {err!r}")
        time.sleep(poll_s)


def main() -> None:
    # Line-buffer stdout. Python block-buffers when it is not a terminal, so
    # `--watch > log.txt` (or any redirect, which is how a watcher is usually run)
    # shows nothing for minutes and looks hung.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:  # pragma: no cover - very old Python
        pass

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("session", nargs="?", help="capture dir (default: the newest)")
    ap.add_argument("--notebook", action="store_true", help="also execute the notebook")
    ap.add_argument("--watch", action="store_true", help="analyse every new recitation")
    args = ap.parse_args()

    if args.watch:
        watch(args.notebook)
        return

    d = Path(args.session) if args.session else newest_capture()
    if d is None:
        # ASCII only: a redirected Windows console is cp1252 and would raise
        # UnicodeEncodeError on Arabic, turning "no captures yet" into a crash.
        raise SystemExit(
            f"No captures in {CAPTURES}. Start the server with TAJWID_CAPTURE_DIR set "
            f"and turn the Diagnose toggle on."
        )
    analyse(d, notebook=args.notebook)


if __name__ == "__main__":
    main()
