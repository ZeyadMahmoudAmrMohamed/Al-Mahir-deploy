"""Build the frontend's static muṣḥaf data: KFGQPC V1 layout + per-word glyph codes.

Writes one JSON per page into the frontend's public/ tree, so the app ships with the
whole muṣḥaf baked in and makes no runtime call to any third party.

ONE source of data, one source of proof:

* **The Quran.com v4 API** builds the muṣḥaf: per-word glyph codes (`code_v1`), Uthmani
  text, and each word's page and line. It is the source because it is the one that
  MATCHES OUR FONTS — the QCF_P*.woff2 files are the `code_v1` lineage, and a layout
  from a different edition would put glyphs on lines their font never drew.
* **The QUL layout DB** (`qpc-v1-15-lines.db`, KFGQPC V1 / 1405H print) is used ONLY as
  an ORACLE: it states `line_type` and `is_centered` outright, so the structure derived
  here is checked against it and the build fails on any disagreement it cannot account
  for. It is deliberately not a data source. It is a different print: it paginates juz 30
  differently (its page 586 line 15 is a sūra header; in the `code_v1` edition that line
  is āyah 80:41), and its word ids index QUL's own word table, which segments بَعْدَ مَا
  as two words where the glyph list has one — so joining on them drifts by 364 words and
  silently misplaces text. Two editions, mixed, make a muṣḥaf that is neither.

The ornament lines — a sūra's header and its basmalah — are derived from the gaps the
words leave, and the derivation is not as obvious as it looks: a header and its basmalah
can fall on DIFFERENT PAGES (An-Nisāʾ's header ends page 76; its basmalah opens page 77),
At-Tawba (9) has a header and no basmalah, and Al-Fātiḥa (1) likewise, because its
basmalah IS āyah 1 and so is real text. Hence the oracle.

THE CORRECTNESS PROBLEM THIS SCRIPT EXISTS TO SOLVE
---------------------------------------------------
The backend speaks (sura, aya, word_idx) over **Tanzil Uthmani** words — word_idx is a
0-based index into `Aya(s, a).get().uthmani_words`. The layout speaks global word ids.
If those two segmentations disagree anywhere, feedback silently paints the WRONG WORD —
the exact class of bug the whole feedback contract exists to prevent.

They mostly agree, but not everywhere, and the exceptions are real orthography rather
than data errors. In 4 āyāt the muṣḥaf writes joined (*mawṣūl*) what Tanzil separates:

    2:181, 8:6, 13:37   بَعْدَ + مَا   -> one glyph  بَعْدَمَا
    37:130              إِلْ + يَاسِينَ  -> one glyph  إِلْيَاسِينَ

Neither convention is wrong, so the data model carries the truth instead of pretending:
a rendered word owns a LIST of Tanzil word indices (`word_idxs`), normally of length 1.
Feedback for either underlying word lights up the glyph that contains it.

Rather than hard-code those four, this aligns every āyah by consuming Tanzil words until
their letters match the layout word's letters, and fails the build if any āyah cannot be
aligned that way. Every invariant it can check, it checks.

Usage:
    python scripts/build_mushaf_data.py --verify-only
    python scripts/build_mushaf_data.py
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
import zipfile
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen

from quran_transcript import Aya

API = (
    "https://api.quran.com/api/v4/verses/by_page/{page}"
    # per_page must exceed the most āyāt any single page holds — the short sūras of juz
    # 30 put well over 50 on a page, and the default paginates them away silently: the
    # page renders, just missing its last lines.
    "?words=true&per_page=300&word_fields=code_v1,line_number,page_number,text_uthmani"
)
LAYOUT_ZIP = (
    "https://raw.githubusercontent.com/blueheron786/"
    "quranic-universal-library-mushaf-layouts/main/qpc-v1-15-lines.db.zip"
)
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data"
OUT = ROOT.parent / "frontend" / "public" / "mushaf"
N_PAGES = 604
UA = {"User-Agent": "tajwid-mushaf-build/1.0"}  # the API 403s urllib's default


def _get(url: str, retries: int = 4) -> bytes:
    for attempt in range(retries):
        try:
            with urlopen(Request(url, headers=UA), timeout=30) as r:
                return r.read()
        except Exception as exc:  # noqa: BLE001 - network flakiness, retry
            if attempt == retries - 1:
                raise RuntimeError(f"{url} failed: {exc}") from exc
            time.sleep(1.5 * (attempt + 1))
    raise AssertionError("unreachable")


def layout_db() -> sqlite3.Connection:
    """The QUL KFGQPC V1 15-line layout, downloaded once and cached in data/."""
    path = CACHE / "qpc-v1-15-lines.db"
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(BytesIO(_get(LAYOUT_ZIP))) as z:
            name = next(n for n in z.namelist() if n.endswith(".db"))
            path.write_bytes(z.read(name))
    return sqlite3.connect(path)


def _skeleton(text: str) -> str:
    """A coarse consonant skeleton, for ALIGNMENT ONLY — never for display or scoring.

    The two sources spell the same word differently in ways that carry no phonetic
    weight, so the skeleton folds away everything they disagree about:

      * harakat, shadda, sukun, tanween, the small-mark family (incl. the sajda sign ۩),
        and tatweel;
      * bidi controls — the layout appends a RLM after the sajda sign (27:26);
      * the whole alef/ya family. The layout writes ٱفْتَرَاهُ with a plain alef where
        Tanzil writes ٱفْتَرَىٰهُ with alef-maqṣūra + dagger alef (11:13). Same word,
        same sound, two orthographies.

    Deliberately lossy: it only has to be sharp enough to tell one word of an āyah from
    the next, and align_words() additionally requires the whole āyah to consume exactly,
    so an over-eager fold cannot pass unnoticed.
    """
    out = []
    for ch in text:
        code = ord(ch)
        if 0x064B <= code <= 0x065F or 0x06D6 <= code <= 0x06ED:
            continue  # harakat / tanween / small marks / sajda sign
        if code in (0x0640, 0x0670, 0x06E5, 0x06E6):
            continue  # tatweel, superscript (dagger) alef, small waw/ya
        if 0x200B <= code <= 0x200F or code == 0xFEFF:
            continue  # zero-width + bidi controls
        if ch.isspace():
            continue
        if ch in "أإآٱاىیي":
            ch = "ا"
        elif ch == "ة":
            ch = "ه"
        out.append(ch)
    return "".join(out)


def align_words(layout: list[str], tanzil: list[str]) -> list[list[int]] | None:
    """Map each layout word to the Tanzil word indices it covers.

    Walks both sides, consuming Tanzil words into the current layout word until their
    skeletons match. Normally 1:1; a *mawṣūl* spelling consumes two. Returns None if the
    two cannot be reconciled — which must fail the build, never be papered over.
    """
    out: list[list[int]] = []
    t = 0
    for word in layout:
        target = _skeleton(word)
        taken: list[int] = []
        acc = ""
        while t < len(tanzil) and acc != target:
            acc += _skeleton(tanzil[t])
            taken.append(t)
            t += 1
        if acc != target or not taken:
            return None
        out.append(taken)
    return out if t == len(tanzil) else None


def fetch_all_words(pages: int, problems: list[str]) -> dict[int, dict[int, list[dict]]]:
    """page -> line -> the glyphs to draw, for the whole muṣḥaf, in one pass.

    Every word is routed by its OWN `page_number`/`line_number`, not by the page whose
    response carried it. Those differ: `verses/by_page/N` returns each verse that BEGINS
    on page N, including the words that spill onto page N+1 — and it does not repeat that
    verse in page N+1's response. Filtering each response to `page_number == N` therefore
    drops the spill from both pages (305 words of the Quran), leaving pages whose last
    line is quietly short. Routing by the word's own page cannot lose them.

    Words are deduplicated by id, so a verse arriving in more than one response is drawn
    once.

    Each entry is either a word (carrying the Tanzil coordinates the backend speaks) or
    an āyah-end marker (`word_idxs: []`).
    """
    out: dict[int, dict[int, list[dict]]] = {}
    seen: set[int] = set()
    ordered: list[tuple[int, int, int, dict]] = []  # (page, line, word id, entry)

    for page in range(1, pages + 1):
        raw = json.loads(_get(API.format(page=page)).decode())

        # Never trust the page to fit: a paginated response loses the page's last lines
        # and still renders, the quietest possible way to ship a wrong muṣḥaf.
        pagination = raw.get("pagination") or {}
        if pagination.get("next_page"):
            problems.append(
                f"page {page}: response is paginated ({pagination.get('total_records')} "
                "āyāt) — raise per_page in API"
            )

        for verse in raw["verses"]:
            sura, aya = (int(x) for x in verse["verse_key"].split(":"))
            words = [w for w in verse["words"] if w["char_type_name"] == "word"]
            tanzil = Aya(sura, aya).get().uthmani_words

            mapping = align_words([w["text_uthmani"] for w in words], tanzil)
            if mapping is None:
                problems.append(
                    f"{sura}:{aya} cannot align: layout {len(words)} words vs "
                    f"Tanzil {len(tanzil)}"
                )
                continue

            idx_of = {id(w): m for w, m in zip(words, mapping)}
            for w in verse["words"]:
                wid = int(w["id"])
                if wid in seen:
                    continue
                seen.add(wid)
                if w["char_type_name"] == "word":
                    idxs = idx_of[id(w)]
                    entry = {
                        "sura": sura,
                        "aya": aya,
                        # 0-based, Tanzil-aligned: the backend's coordinate. A list
                        # because one glyph can spell two Tanzil words (see the docstring).
                        "word_idxs": idxs,
                        "glyph": w["code_v1"],
                        "uthmani": " ".join(tanzil[i] for i in idxs),
                    }
                else:
                    entry = {
                        "sura": sura,
                        "aya": aya,
                        "word_idxs": [],
                        "glyph": w["code_v1"],
                        "uthmani": "",
                    }
                ordered.append((int(w["page_number"]), int(w["line_number"]), wid, entry))

        if page % 50 == 0:
            print(f"  … {page}/{pages} pages fetched", file=sys.stderr)

    # Word ids run in muṣḥaf order, so sorting by them restores reading order within a
    # line regardless of which response each word arrived in.
    for page, line, _wid, entry in sorted(ordered, key=lambda r: r[2]):
        out.setdefault(page, {}).setdefault(line, []).append(entry)
    return out


# Pages 1 and 2 are the decorative opening spread: Al-Fātiḥa and the start of Al-Baqara
# are set inside a frame with wide leading, not on the standard 15-line grid.
FRAMED_PAGES = (1, 2)
LINES_PER_PAGE = 15


def derive_ornaments(
    all_words: dict[int, dict[int, list[dict]]], pages: int, problems: list[str]
) -> dict[tuple[int, int], dict]:
    """(page, line) -> the sūra header / basmalah that belongs in that empty line.

    Derived from where the words are NOT. Every sūra is preceded, in reading order, by
    its header and then (except sūras 1 and 9) its basmalah, each on its own line — so
    walking backwards from a sūra's first word over the empty lines immediately before it
    identifies them, and does so across a page break, which is where the naive per-page
    reading of gaps goes wrong.
    """
    # Every line slot of the muṣḥaf, in reading order.
    slots: list[tuple[int, int]] = [
        (p, l)
        for p in range(1, pages + 1)
        for l in range(1, (max(all_words.get(p, {1: []}), default=1) if p in FRAMED_PAGES else LINES_PER_PAGE) + 1)
    ]
    slot_no = {s: i for i, s in enumerate(slots)}
    occupied = {
        (p, l) for p, lines in all_words.items() for l, words in lines.items() if words
    }

    # Where each sūra's first word sits.
    first_slot: dict[int, int] = {}
    for p, lines in all_words.items():
        for l, words in lines.items():
            for w in words:
                if w["aya"] == 1 and w["word_idxs"] == [0] and w["sura"] not in first_slot:
                    first_slot[w["sura"]] = slot_no[(p, l)]

    out: dict[tuple[int, int], dict] = {}
    for sura in range(1, 115):
        if sura not in first_slot:
            problems.append(f"sūra {sura}: never found its opening word")
            continue
        # Al-Fātiḥa's basmalah IS āyah 1; At-Tawba has none. Everyone else gets one.
        want = ["surah_name"] if sura in (1, 9) else ["surah_name", "basmallah"]

        i = first_slot[sura] - 1
        empty: list[tuple[int, int]] = []
        while i >= 0 and slots[i] not in occupied and len(empty) < len(want):
            empty.append(slots[i])
            i -= 1
        empty.reverse()  # reading order: header first, then basmalah

        if len(empty) == len(want):
            for slot, kind in zip(empty, want):
                out[slot] = {"type": kind, "surah": sura}
        elif len(empty) == 1 and len(want) == 2:
            # One line for two ornaments. This edition ran the previous sūra's text into
            # the line the 1405H print reserved for this header (sūras 82, 86, 91), so
            # the header and its basmalah share the single line that is left. Recorded
            # explicitly rather than dropping the basmalah: a sūra silently opening
            # without its basmalah is a defect a reader would notice immediately.
            out[empty[0]] = {"type": "surah_name", "surah": sura, "with_basmalah": True}
        else:
            problems.append(
                f"sūra {sura}: needs {want} but found {len(empty)} empty line(s) "
                f"before its first word"
            )

    return out


def build_page(
    page: int,
    words_by_line: dict[int, list[dict]],
    ornaments: dict[tuple[int, int], dict],
    centered: set[tuple[int, int]],
    problems: list[str],
) -> dict:
    """Assemble one page from its words and the ornaments derived around them."""
    n_lines = max(words_by_line, default=0) if page in FRAMED_PAGES else LINES_PER_PAGE
    lines: list[dict] = []

    for n in range(1, n_lines + 1):
        words = words_by_line.get(n, [])
        if words:
            lines.append(
                {
                    "line": n,
                    "type": "ayah",
                    "centered": (page, n) in centered,
                    "words": words,
                }
            )
        elif (page, n) in ornaments:
            orn = ornaments[(page, n)]
            line = {
                "line": n,
                "type": orn["type"],
                "centered": True,
                "surah": orn["surah"],
                "words": [],
            }
            if orn.get("with_basmalah"):
                line["with_basmalah"] = True
            lines.append(line)
        # An empty line that is not an ornament is a real blank in the composition
        # (it happens at the end of juz 30's short sūras); drawing nothing is correct.

    suras = sorted({w["sura"] for l in lines for w in l["words"]})
    return {"page": page, "lines": lines, "suras": suras}


def centered_lines(db: sqlite3.Connection) -> set[tuple[int, int]]:
    """The (page, line) pairs the muṣḥaf sets centred rather than justified.

    Read from the layout rather than derived, because there is no rule to derive: only
    20 āyah lines in the whole muṣḥaf are centred — the framed opening (pages 1–2) and
    seven short-sūra lines in the juz-30 tail (600, 602, 603, 604). "A sūra's last line
    is centred" sounds right and is wrong; it over-centres about 100 lines.

    Taking these from the other print is safe here, and only here: none of the 8 pages
    where the two editions paginate differently is one of the pages that has a centred
    line, so there is nothing for the difference to corrupt.
    """
    return {
        (int(p), int(l))
        for p, l in db.execute(
            "select page_number, line_number from pages where is_centered = 1"
        )
    }


def verify_against_layout(
    db: sqlite3.Connection, pages: list[dict], problems: list[str]
) -> None:
    """Check the derived structure against the QUL layout, which states it outright.

    The two are different prints, so they are not required to agree everywhere — they
    demonstrably do not paginate juz 30 the same way. What IS required is that every
    disagreement be a whole-page pagination difference, never a page where we agree on
    where the words go but disagree on what the line IS. The first kind is two editions;
    the second kind is a bug in the derivation, and it is the kind that ships a muṣḥaf
    with a header in the wrong place.
    """
    agreed = differing_pages = 0
    combined = {
        (p["page"], l["line"]): True
        for p in pages
        for l in p["lines"]
        if l.get("with_basmalah")
    }

    for page_data in pages:
        page = page_data["page"]
        rows = db.execute(
            "select line_number, line_type, is_centered from pages where page_number = ?",
            (page,),
        ).fetchall()
        if not rows:
            continue
        db_lines = {int(r[0]): (r[1], bool(r[2])) for r in rows}
        ours = {l["line"]: (l["type"], l["centered"]) for l in page_data["lines"]}

        # Do the two prints even lay this page out the same? Compare where the āyāt are.
        if {n for n, (t, _) in db_lines.items() if t == "ayah"} != {
            n for n, (t, _) in ours.items() if t == "ayah"
        }:
            differing_pages += 1
            continue

        for n, (db_type, db_centered) in db_lines.items():
            if n not in ours:
                problems.append(f"page {page} line {n}: layout has {db_type}, we draw nothing")
                continue
            our_type, _our_centered = ours[n]
            # Only line_type is checked. Centring is READ from this same layout (see
            # centered_lines), so comparing it here would be checking it against itself.
            if our_type == db_type:
                continue
            # The one accountable exception: where this edition ran text into the line
            # that print reserved for a header, the header joins its basmalah on the
            # next line. So a line that print calls `basmallah` and we call a combined
            # header is the SAME ornament, placed as this edition's pagination allows.
            if db_type == "basmallah" and combined.get((page, n)):
                continue
            problems.append(
                f"page {page} line {n}: layout says {db_type}, we derived {our_type}"
            )
        agreed += 1

    print(
        f"oracle: {agreed} pages verified line-for-line against the QUL layout; "
        f"{differing_pages} paginate differently (separate print).",
        file=sys.stderr,
    )
    if differing_pages > 15:
        problems.append(
            f"{differing_pages} pages disagree with the QUL layout — too many to be "
            "edition differences; the derivation is suspect"
        )


def _fail(problems: list[str]) -> int:
    print(f"\n{len(problems)} PROBLEMS:", file=sys.stderr)
    for p in problems[:40]:
        print(f"  {p}", file=sys.stderr)
    print(
        "\nRefusing to write: the muṣḥaf data is not provably consistent with the "
        "coordinates the backend speaks.",
        file=sys.stderr,
    )
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--pages", type=int, default=N_PAGES)
    args = parser.parse_args()

    problems: list[str] = []
    db = layout_db()
    all_words = fetch_all_words(args.pages, problems)
    ornaments = derive_ornaments(all_words, args.pages, problems)
    centered = centered_lines(db)
    pages = [
        build_page(p, all_words.get(p, {}), ornaments, centered, problems)
        for p in range(1, args.pages + 1)
    ]

    if args.pages == N_PAGES:
        verify_against_layout(db, pages, problems)

        # Whole-Quran invariants. The ornament totals are only right one way: every sūra
        # has a header; every sūra but At-Tawba (9, which has none) and Al-Fātiḥa (1,
        # whose basmalah IS āyah 1 and so is real text) has a basmalah line.
        headers = {l["surah"] for p in pages for l in p["lines"] if l["type"] == "surah_name"}
        basmalahs = {l["surah"] for p in pages for l in p["lines"] if l["type"] == "basmallah"}
        basmalahs |= {
            l["surah"] for p in pages for l in p["lines"] if l.get("with_basmalah")
        }
        if headers != set(range(1, 115)):
            problems.append(f"expected a header for all 114 sūras, got {len(headers)}")
        want_basmalah = set(range(1, 115)) - {1, 9}
        if basmalahs != want_basmalah:
            problems.append(
                f"basmalah missing for sūras {sorted(want_basmalah - basmalahs)}, "
                f"unexpected for {sorted(basmalahs - want_basmalah)}"
            )

        # The invariant that actually matters, and the one the frontend depends on:
        # EVERY Tanzil word of the Quran is drawn exactly once, and every āyah gets its
        # end marker. Counting glyphs would be circular (a glyph can hold two words);
        # counting the Tanzil coordinates they claim to cover is not.
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
                    problems.append(
                        f"{sura}:{aya} drawn as word_idxs {got}, expected {want}"
                    )
                    if len(problems) > 30:
                        return _fail(problems)
        if markers != 6236:
            problems.append(f"drew {markers} āyah-end markers, expected 6236")

    if problems:
        return _fail(problems)

    print(f"OK: {args.pages} pages, every word aligned to Tanzil.", file=sys.stderr)
    if args.verify_only:
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    for data in pages:
        (OUT / f"{data['page']}.json").write_text(
            json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
        )
    (OUT / "index.json").write_text(
        json.dumps(build_index(pages), ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"wrote {len(pages)} pages + index.json -> {OUT}", file=sys.stderr)
    return 0


def build_index(pages: list[dict]) -> list[dict]:
    """The sūra index the reader's search uses: name, āyah count, and first page.

    The first page is taken from the pages we just built rather than from chapter
    metadata, so the index cannot point somewhere this muṣḥaf does not open.
    """
    chapters = json.loads(_get("https://api.quran.com/api/v4/chapters?language=ar").decode())
    first_page: dict[int, int] = {}
    for p in pages:
        for line in p["lines"]:
            for w in line["words"]:
                first_page.setdefault(w["sura"], p["page"])

    return [
        {
            "sura": c["id"],
            "name_ar": c["name_arabic"],
            "name_en": c["name_simple"],
            "ayat": c["verses_count"],
            "page": first_page.get(c["id"], c["pages"][0]),
            "revelation": c["revelation_place"],
        }
        for c in chapters["chapters"]
    ]


if __name__ == "__main__":
    raise SystemExit(main())
