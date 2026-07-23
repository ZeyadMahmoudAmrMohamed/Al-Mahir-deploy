import quran_transcript.alphabet as alph
from quran_transcript import Aya

from .confidence import STRICTNESS, grade
from .reference import word_index_of_char
from .types import FeedbackError, Span, WordFeedback


def _walk_positions(uthmani_text: str, start: Span) -> list[Span]:
    """Absolute (sura, aya, word_idx) for each word in the span, in order.

    The span can run across aya boundaries, so when the running word index passes the
    end of an aya we step to the next one and reset to word 0. Assuming a single aya
    would quietly mislabel every word after the boundary.
    """
    n_words = len(uthmani_text.split(alph.uthmani.space))

    positions: list[Span] = []
    aya = Aya(start.sura, start.aya)
    word_idx = start.word_idx

    for _ in range(n_words):
        info = aya.get()
        positions.append(
            Span(sura=info.sura_idx, aya=info.aya_idx, word_idx=word_idx)
        )
        word_idx += 1
        if word_idx >= len(info.uthmani_words):
            aya = aya.step(1)
            word_idx = 0

    return positions


def aggregate(
    uthmani_text: str,
    start: Span,
    errors: list[FeedbackError],
    thresholds: tuple[float, float] = STRICTNESS["normal"],
) -> list[WordFeedback]:
    """Attach each error to the word it falls in.

    Every word is returned, correct ones included, so the frontend can render the
    whole verse from this array alone — no fetching the text separately, no deriving
    word boundaries from character offsets (FR-023 / SC-001). That reconstruction
    burden is exactly what obad's API forces on its callers, and closing it is the
    point of this project.

    A word's status is the WORST grade among its errors: any confident error makes it
    `error`; if only low-confidence or unscored findings landed on it, it stays
    `almost`. A word is never `error` on a finding we cannot vouch for
    (Constitution VI).
    """
    words = uthmani_text.split(alph.uthmani.space)
    positions = _walk_positions(uthmani_text, start)

    feedback = [
        WordFeedback(
            sura=pos.sura,
            aya=pos.aya,
            word_idx=pos.word_idx,
            uthmani=word,
            status="correct",
            errors=[],
        )
        for word, pos in zip(words, positions)
    ]

    for err in errors:
        # uthmani_pos is a character offset into uthmani_text; counting the word
        # separators before it gives the word this error belongs to. This is the
        # journey obad never makes: phoneme -> uthmani char -> word.
        local_idx = word_index_of_char(uthmani_text, err.uthmani_pos[0])
        local_idx = min(local_idx, len(feedback) - 1)  # zero-width insert at the tail

        word = feedback[local_idx]
        word.errors.append(err)

        verdict = grade(err, thresholds)
        # Worst grade wins: a confident error overrides an existing `almost`, but an
        # `almost` never downgrades a word already marked `error`.
        if verdict == "error" or word.status == "correct":
            word.status = verdict

    return feedback


def _blank(word: WordFeedback) -> None:
    word.errors = []
    word.status = "correct"  # see WordFeedback.trimmed: read the flag first
    word.trimmed = True


def blank_word_at(words: list[WordFeedback], span: Span) -> list[WordFeedback]:
    """Un-score the word at ``span`` if it is present, leaving the rest untouched.

    Used for chunk overlap (Settings.chunk_overlap_ms): the tail of the previous chunk
    is re-sent as this chunk's head so a boundary word can be scored, but the word that
    was SPAN-FINAL last time was recited in pausal (waqf) form -- a dropped final haraka,
    no cross-word ghunnah. It was already scored correctly in that chunk, against a
    pausal reference. Here it is interior and the reference is CONNECTED, so re-diffing
    it invents a tashkeel/madd error the reciter never made. Skip it; the earlier,
    correct verdict stands (the frontend keeps a scored verdict over a trimmed one).
    """
    for w in words:
        if w.sura == span.sura and w.aya == span.aya and w.word_idx == span.word_idx:
            _blank(w)
    return words


def trim_edges(
    words: list[WordFeedback], start: Span, end: Span, forced: bool
) -> list[WordFeedback]:
    """Stop scoring words that OUR chunker cut in half (FR-010).

    When a chunk boundary falls mid-word, the ASR emits a mangled fragment, the diff
    faithfully reports a mismatch, and the learner is billed for a mistake that exists
    only because of where we cut the audio. Every component behaved correctly and the
    product still lied to its user. That is a false accusation with no learner error
    behind it at all (Constitution VI), arriving through a door nobody was watching.

    A boundary is only an ARTEFACT when the span begins mid-aya or ends mid-aya. A span
    the reciter genuinely began (at word 0) or genuinely finished (at the last word) is
    real, and silently declining to score it would be its own kind of lie.

    The trailing edge is only an artefact on a FORCED cut (the max-length cap sliced a
    word mid-articulation). At a WAQF the reciter completed the last word and paused --
    it is whole, and the phonetizer renders a span-final word in pausal form (رَيْبَ ->
    ...بڇ, the final haraka dropped), which is exactly what a paused reciter says. So a
    waqf's last word is scored, not trimmed: trimming it was greying a word we could
    have verified, and its interior re-emission under overlap is handled by blank_word_at.

    Trimmed words are still returned -- the frontend draws them -- but unscored.
    """
    if not words:
        return words

    if start.word_idx > 0:
        _blank(words[0])

    last_word_of_aya = len(Aya(end.sura, end.aya).get().uthmani_words) - 1
    if forced and end.word_idx < last_word_of_aya:
        _blank(words[-1])

    return words
