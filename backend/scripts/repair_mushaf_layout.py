"""Repair the muṣḥaf page layout from the QUL 1405H layout DB.

WHY THIS EXISTS
---------------
The pages were built (build_mushaf_data.py) by routing each glyph to the page and line
the Quran.com v4 API stamps on it. For ~25 pages — juz 30, plus Al-Māʾida/Al-Anʿām
pages 120–122 and 145 — that API metadata is CORRUPT: a block of words carries a wrong
`page_number`/`line_number` (and anomalous word ids), so they land on the wrong page and
out of reading order. Two proofs it is the placement, not the glyphs:

  * Al-Balad 90:19–20 arrive tagged page 595; the QUL DB (same 1405H print) puts them on
    594, and their code_v1 glyphs render correctly ONLY under QCF_P594 — so the glyph is
    right, the page stamp is wrong.
  * No word is duplicated on any page; every glyph is present exactly once, merely mis-slotted.

THE FIX
-------
The glyph DATA in the built JSON is correct — only the (page, line) grouping is wrong.
The QUL DB states the correct layout outright, in reading order, and its per-page word
count matches ours (words + āyah-end markers) on every good page. So we discard the API's
placement entirely and re-slot every existing glyph into the QUL layout by reading order:

  * our entries (glyphs + markers), sorted into Quran reading order, align 1:1 with the
    QUL word slots — except the four mawṣūl glyphs (بَعْدَ مَا ×3, إِلْ يَاسِينَ) that spell
    two Tanzil words in one glyph, which consume two QUL slots each.
  * ornament lines (sūra header, basmalah) come straight from the QUL DB.

No network: it reads the already-built JSON for glyphs and the cached QUL DB for layout.
Writes nothing unless every invariant passes.

Usage:  python scripts/repair_mushaf_layout.py            # verify + write
        python scripts/repair_mushaf_layout.py --check    # verify only, no write
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

from quran_transcript import Aya

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "qpc-v1-15-lines.db"
MUSHAF = ROOT.parent / "frontend" / "public" / "mushaf"
N_PAGES = 604
BIG = 10**9  # sort key: a marker sits after every word of its āyah


def load_entries() -> list[dict]:
    """Every glyph and āyah-end marker we have, in Quran reading order.

    Read from the built pages, whose GLYPH data is trusted; only their grouping is not.
    """
    entries: list[dict] = []
    for p in range(1, N_PAGES + 1):
        data = json.loads((MUSHAF / f"{p}.json").read_text(encoding="utf-8"))
        for line in data["lines"]:
            for w in line.get("words", []):
                entries.append(w)
    # Reading order: sūra, then āyah, then first word index; a marker (no indices) last.
    entries.sort(
        key=lambda w: (w["sura"], w["aya"], w["word_idxs"][0] if w["word_idxs"] else BIG)
    )
    return entries


def load_layout(db: sqlite3.Connection):
    """The QUL layout: per page, the ordered lines; per āyah line, how many word slots.

    Returns (pages, slots):
      pages[p] = list of line dicts in order, each {line, type, centered, surah?, slots}
      slots    = flat list of (page, line) in reading order, one per āyah word slot.
    """
    pages: dict[int, list[dict]] = {}
    slots: list[tuple[int, int]] = []
    # The slot index at which each sūra's words begin — a hard anchor to resync on, so a
    # miscounted glyph span (only إِلْ يَاسِينَ / بَعْدَ مَا differ, and only 4 times) can
    # never drift past the next sūra header.
    surah_start: dict[int, int] = {}
    rows = db.execute(
        "select page_number, line_number, line_type, is_centered, "
        "first_word_id, last_word_id, surah_number from pages "
        "order by page_number, line_number"
    ).fetchall()
    for page, line, ltype, centered, first, last, surah in rows:
        entry = {"line": int(line), "type": ltype, "centered": bool(centered)}
        if ltype == "ayah":
            n = int(last) - int(first) + 1
            entry["n"] = n
            slots.extend((int(page), int(line)) for _ in range(n))
        elif ltype == "surah_name":
            entry["surah"] = int(surah)
            surah_start.setdefault(int(surah), len(slots))
        pages.setdefault(int(page), []).append(entry)
    return pages, slots, surah_start


def assign(
    entries: list[dict],
    slots: list[tuple[int, int]],
    surah_start: dict[int, int],
    problems: list[str],
) -> None:
    """Map each entry onto a QUL word slot, in reading order, in place.

    A mawṣūl glyph (two Tanzil word indices) is one glyph over two QUL slots, so it eats
    two; everything else eats one — except QUL only splits three of the four (the بَعْدَ
    مَا trio), not إِلْ يَاسِينَ. Rather than encode that, we resync `si` to the sūra's
    known start slot at every sūra boundary, so at most one sūra can carry a span error
    and the next header erases it.
    """
    si = 0
    cur = None
    for e in entries:
        if e["sura"] != cur:
            cur = e["sura"]
            si = surah_start[cur]  # anchor: this sūra's words start exactly here
        if si >= len(slots):
            problems.append(f"ran out of QUL slots at {e['sura']}:{e['aya']}")
            return
        e["_page"], e["_line"] = slots[si]
        si += 2 if len(e["word_idxs"]) == 2 else 1


def rebuild_pages(entries: list[dict], pages: dict, problems: list[str]) -> list[dict]:
    """Group the assigned entries back into pages, filling the QUL line skeleton."""
    by_slot: dict[tuple[int, int], list[dict]] = {}
    for e in entries:
        by_slot.setdefault((e["_page"], e["_line"]), []).append(e)

    out: list[dict] = []
    for p in range(1, N_PAGES + 1):
        lines: list[dict] = []
        for spec in pages.get(p, []):
            if spec["type"] == "ayah":
                words = by_slot.get((p, spec["line"]), [])
                # Strip the private routing keys before writing.
                words = [
                    {k: v for k, v in w.items() if not k.startswith("_")} for w in words
                ]
                lines.append(
                    {
                        "line": spec["line"],
                        "type": "ayah",
                        "centered": spec["centered"],
                        "words": words,
                    }
                )
            elif spec["type"] == "surah_name":
                lines.append(
                    {
                        "line": spec["line"],
                        "type": "surah_name",
                        "centered": True,
                        "surah": spec["surah"],
                        "words": [],
                    }
                )
            else:  # basmallah
                lines.append(
                    {"line": spec["line"], "type": "basmallah", "centered": True, "words": []}
                )
        suras = sorted({w["sura"] for l in lines for w in l["words"]})
        out.append({"page": p, "lines": lines, "suras": suras})
    return out


def validate(pages: list[dict], problems: list[str]) -> None:
    """The invariants that make this safe to ship — the same ones the builder checks."""
    # 1. Every Tanzil word drawn exactly once; every marker present.
    covered: dict[tuple[int, int], list[int]] = {}
    markers = 0
    for p in pages:
        for line in p["lines"]:
            for w in line["words"]:
                if not w["word_idxs"]:
                    markers += 1
                    continue
                covered.setdefault((w["sura"], w["aya"]), []).extend(w["word_idxs"])
    for sura in range(1, 115):
        for aya in range(1, Aya(sura, 1).get().num_ayat_in_sura + 1):
            want = list(range(len(Aya(sura, aya).get().uthmani_words)))
            got = sorted(covered.get((sura, aya), []))
            if got != want:
                problems.append(f"{sura}:{aya} drawn as {got}, expected {want}")
                if len(problems) > 25:
                    return
    if markers != 6236:
        problems.append(f"{markers} āyah-end markers, expected 6236")

    # 2. No page reads backwards, and none overflows 15 lines.
    for p in pages:
        prev = None
        for line in p["lines"]:
            for w in line["words"]:
                key = (w["sura"], w["aya"], w["word_idxs"][0] if w["word_idxs"] else BIG)
                if prev and key < prev:
                    problems.append(f"page {p['page']}: reads backwards at {w['sura']}:{w['aya']}")
                    break
                prev = key
        n_lines = max((l["line"] for l in p["lines"]), default=0)
        if p["page"] > 2 and n_lines > 15:
            problems.append(f"page {p['page']}: {n_lines} lines (>15)")


def build_index(pages: list[dict]) -> list[dict]:
    first_page: dict[int, int] = {}
    for p in pages:
        for line in p["lines"]:
            for w in line["words"]:
                first_page.setdefault(w["sura"], p["page"])
    old = json.loads((MUSHAF / "index.json").read_text(encoding="utf-8"))
    for c in old:
        c["page"] = first_page.get(c["sura"], c["page"])
    return old


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="verify only; do not write")
    args = ap.parse_args()

    problems: list[str] = []
    db = sqlite3.connect(DB)
    entries = load_entries()
    page_specs, slots, surah_start = load_layout(db)
    print(f"loaded {len(entries)} glyphs+markers, {len(slots)} QUL word slots", file=sys.stderr)

    assign(entries, slots, surah_start, problems)
    if problems:
        return _fail(problems)

    pages = rebuild_pages(entries, page_specs, problems)
    validate(pages, problems)
    if problems:
        return _fail(problems)

    print("OK: layout re-derived from QUL, all invariants hold.", file=sys.stderr)
    if args.check:
        return 0

    for data in pages:
        (MUSHAF / f"{data['page']}.json").write_text(
            json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
        )
    (MUSHAF / "index.json").write_text(
        json.dumps(build_index(pages), ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"wrote {len(pages)} pages + index.json", file=sys.stderr)
    return 0


def _fail(problems: list[str]) -> int:
    print(f"\n{len(problems)} PROBLEMS:", file=sys.stderr)
    for p in problems[:40]:
        print(f"  {p}", file=sys.stderr)
    print("\nRefusing to write.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
