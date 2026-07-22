"""Read-only access to data/search/corpus.sqlite (āyāt, words+roots, translations).

Opened in SQLite read-only mode: this is a build artifact, never written by the service.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

DATA_DIR = Path(__file__).parents[3] / "data" / "search"
DB_PATH = DATA_DIR / "corpus.sqlite"

_conn: sqlite3.Connection | None = None


class CorpusMissingError(RuntimeError):
    pass


def connect() -> sqlite3.Connection:
    """Open read-only, once. check_same_thread=False: FastAPI serves from a threadpool and
    there is no write to serialise."""
    global _conn
    if _conn is None:
        if not DB_PATH.exists():
            raise CorpusMissingError(f"No search corpus at {DB_PATH}.")
        _conn = sqlite3.connect(
            f"file:{DB_PATH.as_posix()}?mode=ro", uri=True, check_same_thread=False
        )
        _conn.row_factory = sqlite3.Row
    return _conn


def get_ayah(sura: int, aya: int) -> dict | None:
    r = connect().execute(
        "SELECT * FROM ayahs WHERE sura=? AND aya=?", (sura, aya)
    ).fetchone()
    return dict(r) if r else None


def get_translation(sura: int, aya: int, source_id: str = "en-saheeh") -> dict | None:
    r = connect().execute(
        "SELECT * FROM translations WHERE sura=? AND aya=? AND source_id=?",
        (sura, aya, source_id),
    ).fetchone()
    return dict(r) if r else None
