"""The LLM client. The only file in this service that knows a provider exists.

    "Model-agnostic" needs no abstraction layer.

Groq, Together, vLLM and a self-hosted Fanar all speak the OpenAI-compatible API, so the
abstraction IS the OpenAI SDK and switching providers is TAJWID_LLM_BASE_URL. There is no
LLMProvider ABC and no GroqAdapter: one call shape, one provider at a time — an interface
with a single implementation is a class you read twice to learn nothing.

Used by HyDE query expansion only. Nothing else in this service calls an LLM, and nothing
generated here is ever shown to a user (see hyde.py).
"""

from __future__ import annotations

import random
import time

from ..config import get_settings

_client = None

_MAX_ATTEMPTS = 3
_BASE_BACKOFF_S = 0.5


class LLMUnavailable(RuntimeError):
    """No key, or upstream failed. Callers degrade; they never surface this to a user."""


def _get_client():
    """Build the client on first use, never at import.

    Constructing at import time would mean the whole recitation service cannot START
    without an LLM key — for a feature that is off by default.
    """
    global _client
    if _client is None:
        s = get_settings()
        if not s.llm_api_key:
            raise LLMUnavailable("No LLM key (set TAJWID_LLM_API_KEY or GROQ_API_KEY).")
        from openai import OpenAI

        _client = OpenAI(base_url=s.llm_base_url, api_key=s.llm_api_key)
    return _client


def reset_client() -> None:
    """Drop the cached client. For tests that swap settings between cases."""
    global _client
    _client = None


def chat(messages: list[dict], model: str | None = None, temperature: float = 0.2) -> str:
    """One completion. Raises LLMUnavailable on anything the caller can't fix.

    temperature=0.2 by default: what this service generates is grounded religious
    instruction. Creativity is the failure mode, not the goal.
    """
    from openai import APIStatusError, RateLimitError

    client = _get_client()
    s = get_settings()
    last: Exception | None = None

    for attempt in range(_MAX_ATTEMPTS):
        try:
            r = client.chat.completions.create(
                model=model or s.llm_model_small,
                messages=messages,
                temperature=temperature,
            )
            return r.choices[0].message.content or ""
        except RateLimitError as exc:  # 429 — worth retrying
            last = exc
        except APIStatusError as exc:
            if exc.status_code < 500:
                # A 4xx that isn't rate limiting is our bug (bad model name, malformed
                # request). Retrying an own-goal just burns three round trips.
                raise LLMUnavailable(f"LLM rejected the request: {exc.status_code}") from exc
            last = exc

        if attempt < _MAX_ATTEMPTS - 1:
            # Exponential backoff WITH jitter. Without jitter a burst that 429s together
            # retries in lockstep and 429s together again.
            time.sleep(_BASE_BACKOFF_S * (2**attempt) * (1 + random.random()))

    raise LLMUnavailable("LLM upstream unavailable.") from last
