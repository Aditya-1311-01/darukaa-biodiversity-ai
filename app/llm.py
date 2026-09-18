"""Thin wrapper over the Groq chat-completions API.

Two jobs only:
  1. `extract_json` - cheap model, strict JSON out, used for slot filling.
  2. `narrate`      - strong model, prose out, used to explain a decision that
                      the deterministic engine has *already* made.

The LLM never picks the interventions. That is deliberate: the brief scores
depth of reasoning and scientific grounding, and a free-running LLM is the
fastest way to lose both.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from .config import GROQ_API_KEY, GROQ_EXTRACTION_MODEL, GROQ_MODEL

log = logging.getLogger(__name__)

_client: Any = None


def client() -> Any:
    """Lazy import so the deterministic layer runs without the groq package
    installed - which is what lets CI test the reasoning core in isolation."""
    global _client
    if _client is None:
        if not GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        from groq import Groq

        _client = Groq(api_key=GROQ_API_KEY)
    return _client


def available() -> bool:
    if not GROQ_API_KEY:
        return False
    try:
        import groq  # noqa: F401
    except ImportError:
        log.warning("GROQ_API_KEY is set but the groq package is not installed.")
        return False
    return True


def extract_json(system: str, user: str, *, model: str | None = None) -> dict[str, Any]:
    """Call the small model in JSON mode. Returns {} on any failure."""
    try:
        resp = client().chat.completions.create(
            model=model or GROQ_EXTRACTION_MODEL,
            temperature=0,
            max_tokens=900,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as exc:  # noqa: BLE001 - degrade, never crash a turn
        log.warning("extract_json failed: %s", exc)
        return {}


def narrate(system: str, user: str, *, temperature: float = 0.3, max_tokens: int = 1400) -> str:
    try:
        resp = client().chat.completions.create(
            model=GROQ_MODEL,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:  # noqa: BLE001
        log.warning("narrate failed: %s", exc)
        return ""
