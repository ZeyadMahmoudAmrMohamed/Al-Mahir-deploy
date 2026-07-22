from typing import Literal, Optional

from pydantic import BaseModel, Field


class Span(BaseModel):
    """A position in the Quran. sura/aya are 1-based; word_idx is 0-based."""

    sura: int
    aya: int
    word_idx: int


class Candidate(BaseModel):
    """One place these phonemes could have come from.

    Carries the TEXT, not just coordinates. `(2, 147)` is not an answer a human can
    act on — it is a lookup they have to go and do themselves, which is the same
    reconstruction burden FR-023 abolishes for words. Show them the verse.
    """

    sura: int
    aya: int
    word_idx: int
    end: Optional[Span] = None
    uthmani_text: str = ""

    @property
    def span(self) -> Span:
        return Span(sura=self.sura, aya=self.aya, word_idx=self.word_idx)


class LocateResult(BaseModel):
    status: Literal["ok", "ambiguous", "no_match"]
    span: Optional[Span] = None
    end: Optional[Span] = None
    uthmani_text: Optional[str] = None
    candidates: list[Candidate] = Field(default_factory=list)


class TajweedRuleInfo(BaseModel):
    name_ar: str
    name_en: str
    golden_len: int
    correctness_type: Literal["match", "count"]
    tag: Optional[str] = None


class FeedbackError(BaseModel):
    error_type: Literal["tajweed", "normal", "tashkeel", "sifa"]
    speech_error_type: Literal["insert", "delete", "replace"]

    uthmani_pos: tuple[int, int]

    # A char span in the REFERENCE phoneme string.
    ph_pos: tuple[int, int]

    # (D3) A char span in the PREDICTED phoneme string, or None for a pure deletion
    # (the reciter said nothing, so there is nothing to point at).
    #
    # This exists because `ph_pos` cannot index `phonemes.probs`: that array is over
    # the PREDICTED string, and the two coordinate systems diverge on every insert,
    # delete and wrong-length madd — which is to say, on every error worth scoring.
    # They coincide only when the recitation was correct, i.e. when there is nothing
    # to score at all.
    pred_ph_pos: Optional[tuple[int, int]] = None

    expected_ph: str
    predicted_ph: str
    expected_len: Optional[int] = None
    predicted_len: Optional[int] = None
    tajweed_rules: list[TajweedRuleInfo] = Field(default_factory=list)

    # (D2) None means UNSCORED, not certain. Defaulting this to 1.0 (as the source
    # plan did) makes every unscored finding grade as a hard `error` — so a caller
    # who omits `probs` gets a maximally accusatory system and no warning.
    # FR-018 / Constitution VI: absence of confidence is not high confidence.
    confidence: Optional[float] = None


class WordFeedback(BaseModel):
    sura: int
    aya: int
    word_idx: int
    uthmani: str
    status: Literal["correct", "almost", "error"] = "correct"
    errors: list[FeedbackError] = Field(default_factory=list)

    # (FR-010) True when this word sat on a chunk boundary and was cut in half by OUR
    # chunker, not by the reciter. It is returned so the frontend can still draw the
    # word, but it was NOT SCORED: its errors are dropped.
    #
    # `trimmed=True` means UNVERIFIED, not verified-correct. A frontend must render it
    # neutrally — never with a tick. Claiming a word is correct when we never checked it
    # is a false reassurance, which is the mirror image of the false accusation FR-018
    # exists to prevent. The three statuses cannot express "not checked", so this flag
    # carries it, and consumers must read it before trusting `status`.
    trimmed: bool = False


class FeedbackResponse(BaseModel):
    status: Literal["ok", "ambiguous", "no_match"]
    span: Optional[Span] = None
    end: Optional[Span] = None
    uthmani_text: Optional[str] = None
    predicted_phonemes: str = ""
    reference_phonemes: str = ""
    words: list[WordFeedback] = Field(default_factory=list)

    # Every place these phonemes could have come from, WITH THEIR TEXT, when the
    # status is `ambiguous`. Capped (see locate.MAX_CANDIDATES) — a list of 1,599 is
    # not a shortlist, it is a confession, and that comes back as `no_match` instead.
    candidates: list[Candidate] = Field(default_factory=list)

    # (FR-009) Recited-but-not-verse text we recognised and excluded from scoring:
    # "istiaatha", "basmalah", "sadaka".
    #
    # Reported rather than silently dropped, because the learner KNOWS they recited it.
    # Swallowing it without a word reads as the system not listening.
    non_verse: list[str] = Field(default_factory=list)


# --- Sifat (articulation attributes) -----------------------------------------
#
# The 10 attribute names, exactly as the model emits them. Defined here rather than
# in Task 6 because the input boundary below needs the SHAPE. The behaviour that
# compares them (sifat.compare_sifat) still arrives test-first, in Task 6.

SIFA_ATTRS = (
    "hams_or_jahr",
    "shidda_or_rakhawa",
    "tafkheem_or_taqeeq",
    "itbaq",
    "safeer",
    "qalqla",
    "tikraar",
    "tafashie",
    "istitala",
    "ghonna",
)


class PredictedSifa(BaseModel):
    """One phoneme group's articulation attributes, as predicted by the model."""

    phonemes_group: str
    attrs: dict[str, str]
    probs: dict[str, float] = Field(default_factory=dict)


# --- The input boundary (FR-001 / plan D14) ----------------------------------


class Phonemes(BaseModel):
    """The model's phoneme output: the string, and how sure it was of each one."""

    text: str
    probs: Optional[list[float]] = None


class MuaalemOutput(BaseModel):
    """What the Muaalem model hands us. The boundary this component begins at.

    The contract requires THREE things — `phonemes.text`, `phonemes.probs`, `sifat` —
    not one. A caller who sends only `text` gets a pipeline that can never grade
    confidence (so every finding degrades to `almost`) and never reports sifat. Both
    degradations are safe by design, and both are SILENT — which is the worst way to
    discover an integration gap.

    Modelling the boundary as a type is what turns "you forgot to send probs" from a
    keyword argument nobody passed into a visible `None`.

    No audio, no timestamps, no model: this is plain data, which is why the whole
    component is testable without a GPU (Constitution II).
    """

    phonemes: Phonemes
    sifat: list[PredictedSifa] = Field(default_factory=list)

    @classmethod
    def from_phonemes(cls, text: str, probs: Optional[list[float]] = None):
        """Convenience for the common case and for tests: phonemes only."""
        return cls(phonemes=Phonemes(text=text, probs=probs))
