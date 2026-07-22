"""HyDE (Hypothetical Document Embeddings) query expansion.

A short or polysemous query embeds poorly: `الغيبة` (backbiting) collides with `الغيب` (the
unseen) — same letters, different concept — and vector search drowns in "unseen" āyāt.
HyDE asks an LLM to expand the query into a short passage describing the concept as the
Qur'an treats it, and embeds THAT. Measured upstream: raw `الغيبة` misses 49:12 entirely;
the expansion returns it at rank 1, score 0.715.

**The safety property:** the LLM passage is NEVER shown to the user — it is embedded and
discarded. Every displayed result is a real corpus āyah. So even a bad expansion cannot
put fabricated scripture in front of a reciter; it can only steer which real āyāt come back.

Degrades gracefully: no key, or any upstream failure, returns None and the caller falls
back to the raw query. Search must never fail because expansion was unavailable.
"""

from __future__ import annotations

from functools import lru_cache

from ..config import get_settings
from . import llm

_SYSTEM = {
    "ar": (
        "أنت تساعد محرك بحث دلالي في القرآن الكريم. للكلمة أو الموضوع المُعطى، اكتب جملتين "
        "إلى ثلاث جمل بالعربية تصف هذا المعنى كما يعالجه القرآن، مستخدمًا المفردات والمرادفات "
        "ذات الصلة، لتقريبه من الآيات المناسبة. لا تخترع أرقام آيات. اكتب الوصف فقط."
    ),
    "en": (
        "You help a Qur'an semantic search engine. For the given word or topic, write two or "
        "three sentences describing that concept as the Qur'an addresses it, using related "
        "vocabulary and synonyms, so it matches the relevant verses. Do not invent verse "
        "numbers. Output only the description."
    ),
}


@lru_cache(maxsize=1024)
def expand(query: str, lang: str) -> str | None:
    """An LLM-expanded passage to embed, or None to fall back to the raw query.

    Cached: the same query across a session costs one LLM call, not one per retry.
    """
    try:
        out = llm.chat(
            [
                {"role": "system", "content": _SYSTEM.get(lang, _SYSTEM["en"])},
                {"role": "user", "content": query},
            ],
            model=get_settings().llm_model_small,
            temperature=0.3,
        )
    except Exception:  # noqa: BLE001 — no key, upstream down, anything: use the raw query
        return None
    return (out or "").strip() or None


def reset() -> None:
    expand.cache_clear()
    llm.reset_client()
