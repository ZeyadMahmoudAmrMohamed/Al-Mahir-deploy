"""Per-word recitation feedback from Muaalem model phonetic output."""

from .pipeline import analyse, analyse_session
from .session import SessionState, advance
from .types import (
    Candidate,
    FeedbackError,
    FeedbackResponse,
    MuaalemOutput,
    Phonemes,
    PredictedSifa,
    Span,
    WordFeedback,
)

__version__ = "0.1.0"
__all__ = [
    "analyse",
    "analyse_session",
    "SessionState",
    "advance",
    "MuaalemOutput",
    "Phonemes",
    "PredictedSifa",
    "Candidate",
    "FeedbackResponse",
    "WordFeedback",
    "FeedbackError",
    "Span",
]
