"""Turn orchestration: state -> extraction -> gate -> retrieve -> reason -> narrate.

The graph, in one place:

    user message
        |
    [1] extract slots (Groq JSON mode + regex fallback) -> merge into session profile
        |
    [2] gate: >= 3 known variables and the critical slots?
        |                       \
        no -> clarify            yes
        |                          |
    ask targeted questions     [3] retrieve evidence (Chroma) + shortlist interventions
                                   |
                               [4] rank deterministically, detect synergies
                                   |
                               [5] narrate with the strong model, grounded ONLY in
                                   the selected cards and retrieved chunks
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from . import llm
from .config import (
    CRITICAL_SLOTS,
    INTERVENTIONS_FILE,
    MAX_RECOMMENDATIONS,
    MIN_VARIABLES_FOR_RECOMMENDATION,
    TOP_K,
)
from .reasoning import build_recommendation, detect_synergies, rank
from .retriever import load_interventions, store
from .schemas import AssistantTurn, EcosystemProfile

log = logging.getLogger(__name__)

INTERVENTIONS = load_interventions(INTERVENTIONS_FILE)

SLOT_QUESTIONS = {
    "soil_organic_carbon_pct": "What is your soil organic carbon, as a percentage? A recent soil test report will have it.",
    "soil_ph": "What is the soil pH?",
    "land_use": "How is the land currently used - monoculture crop, mixed cropping, orchard, grazing, fallow, or degraded?",
    "climate_zone": "Which climate zone is the site in - arid, semi-arid, sub-humid, humid, tropical wet, temperate, or montane?",
    "rainfall_regime": "Is annual rainfall low (under 600 mm), medium (600-1100 mm), or high (over 1100 mm)?",
    "current_crop": "What crop is grown there at the moment?",
    "tree_cover_pct": "Roughly what percentage of the area has tree cover?",
    "area_hectares": "How large is the plot, in hectares?",
    "fertiliser_use": "How is fertiliser applied at present - rate and timing?",
    "pesticide_use": "How often are pesticides applied, and on what basis?",
    "grazing_pressure": "Is the land grazed, and how heavily?",
    "irrigation_source": "What is the irrigation source, if any?",
}

EXTRACTION_SYSTEM = """You extract environmental facts from a landholder's message into JSON.

Return ONLY a JSON object. Use these keys when, and only when, the message states or clearly implies the value. Omit every key you are not confident about. Never guess.

soil_organic_carbon_pct (number), soil_ph (number), soil_moisture_pct (number),
soil_texture (string), land_use (one of: monoculture_crop, mixed_cropping, orchard,
grazing_land, fallow, degraded_land, plantation_forest, natural_forest, wetland, peri_urban),
current_crop (string), area_hectares (number),
climate_zone (one of: arid, semi_arid, sub_humid, humid, tropical_wet, temperate, montane),
rainfall_regime (one of: low, medium, high), annual_rainfall_mm (number),
mean_temperature_c (number), tree_cover_pct (number),
species_richness_note (string), habitat_diversity_note (string),
irrigation_source (string), fertiliser_use (string), pesticide_use (string),
grazing_pressure (string), pollution_note (string),
latitude (number), longitude (number), region_name (string)

Map rainfall words: "scanty", "erratic", "kam baarish", "drought-prone" -> low.
Map land use: "only wheat", "sirf gehu", "single crop" -> monoculture_crop.
The user may write in English, Hindi or a mix. Handle all three."""

NARRATION_SYSTEM = """You are an environmental scientist writing up a site assessment.

You will be given: (a) a site profile, (b) interventions that a deterministic rule engine has ALREADY selected, (c) retrieved evidence passages.

Rules, without exception:
- Do not invent interventions, numbers, percentages, citations or study names. Every figure you use must appear verbatim in the supplied cards or passages.
- Do not soften or generalise. "Use sustainable practices" is a failure.
- Open with 2-4 sentences of site diagnosis that names the binding constraint and links at least three variables causally. This is the part that must show reasoning, not summary.
- Then, for each intervention, write one tight paragraph explaining why THIS site's specific numbers make it appropriate, and what the interaction with the other recommendations is.
- State trade-offs plainly. Never present an intervention as costless.
- Where a figure is marked unverified, do not overstate it; attribute it to the organisation rather than asserting it as settled fact.
- Close with a short monitoring note: which indicator should move first and roughly when.
- No headings, no bullet lists, no markdown tables. Flowing prose. The interface renders the structured cards separately, so do not duplicate them."""


# --------------------------------------------------------------------------
# Session memory
# --------------------------------------------------------------------------
@dataclass
class Session:
    session_id: str
    profile: EcosystemProfile = field(default_factory=EcosystemProfile)
    history: list[dict[str, str]] = field(default_factory=list)
    asked: set[str] = field(default_factory=set)
    turn_index: int = 0


class SessionStore:
    """In-memory sessions. Swap for Redis/Postgres behind the same interface."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    def get(self, session_id: str) -> Session:
        if session_id not in self._sessions:
            self._sessions[session_id] = Session(session_id=session_id)
        return self._sessions[session_id]

    def reset(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


SESSIONS = SessionStore()


# --------------------------------------------------------------------------
# [1] Slot extraction
# --------------------------------------------------------------------------
NUMERIC_PATTERNS = {
    "soil_organic_carbon_pct": r"(?:soc|organic carbon|carbon)\D{0,20}?(\d+(?:\.\d+)?)\s*%",
    "soil_ph": r"\bph\D{0,10}?(\d+(?:\.\d+)?)",
    "annual_rainfall_mm": r"(\d{2,4})\s*mm",
    "area_hectares": r"(\d+(?:\.\d+)?)\s*(?:ha\b|hectare)",
    "tree_cover_pct": r"tree cover\D{0,15}?(\d+(?:\.\d+)?)\s*%",
}
KEYWORD_PATTERNS = {
    "rainfall_regime": [
        (r"\b(low rainfall|scanty|erratic rainfall|drought[- ]prone|kam baarish)\b", "low"),
        (r"\b(high rainfall|heavy rainfall|zyada baarish)\b", "high"),
    ],
    "climate_zone": [
        (r"\bsemi[- ]?arid\b", "semi_arid"),
        (r"\barid\b", "arid"),
        (r"\bsub[- ]?humid\b", "sub_humid"),
        (r"\bhumid\b", "humid"),
        (r"\btemperate\b", "temperate"),
    ],
    "land_use": [
        (r"\b(monoculture|sole crop|only wheat|only rice|single crop|sirf)\b", "monoculture_crop"),
        (r"\b(degraded|barren|wasteland|banjar)\b", "degraded_land"),
        (r"\b(grazing|pasture|rangeland|charagah)\b", "grazing_land"),
        (r"\b(orchard|bagicha|plantation crop)\b", "orchard"),
        (r"\b(fallow|parti)\b", "fallow"),
        (r"\bwetland|pond ecosystem|marsh\b", "wetland"),
    ],
}


def regex_extract(text: str) -> dict[str, Any]:
    low = text.lower()
    found: dict[str, Any] = {}
    for field_name, pattern in NUMERIC_PATTERNS.items():
        m = re.search(pattern, low)
        if m:
            found[field_name] = float(m.group(1))
    for field_name, rules in KEYWORD_PATTERNS.items():
        for pattern, value in rules:
            if re.search(pattern, low):
                found[field_name] = value
                break
    if "annual_rainfall_mm" in found and "rainfall_regime" not in found:
        mm = found["annual_rainfall_mm"]
        found["rainfall_regime"] = "low" if mm < 600 else "medium" if mm <= 1100 else "high"
    return found


def coerce(raw: dict[str, Any]) -> EcosystemProfile:
    """Drop anything the schema rejects rather than failing the turn."""
    clean: dict[str, Any] = {}
    valid = set(EcosystemProfile.model_fields)
    for key, value in raw.items():
        if key in valid and value not in (None, "", "unknown", "null"):
            clean[key] = value
    try:
        return EcosystemProfile(**clean)
    except Exception:  # noqa: BLE001 - retry field by field
        safe: dict[str, Any] = {}
        for key, value in clean.items():
            try:
                EcosystemProfile(**{key: value})
                safe[key] = value
            except Exception:  # noqa: BLE001
                log.debug("dropped slot %s=%r", key, value)
        return EcosystemProfile(**safe)


def extract_slots(message: str, structured: dict[str, Any] | None) -> EcosystemProfile:
    merged: dict[str, Any] = regex_extract(message)
    if llm.available() and message.strip():
        merged.update({k: v for k, v in llm.extract_json(EXTRACTION_SYSTEM, message).items() if v is not None})
    if structured:
        merged.update(structured)  # explicit JSON input always wins
    return coerce(merged)


# --------------------------------------------------------------------------
# [2] Readiness gate
# --------------------------------------------------------------------------
def missing_critical(profile: EcosystemProfile) -> list[str]:
    data = profile.model_dump()
    return [slot for slot in CRITICAL_SLOTS if data.get(slot) is None]


def next_questions(profile: EcosystemProfile, asked: set[str], limit: int = 3) -> list[str]:
    data = profile.model_dump()
    ordered = CRITICAL_SLOTS + [k for k in SLOT_QUESTIONS if k not in CRITICAL_SLOTS]
    questions: list[str] = []
    for slot in ordered:
        if data.get(slot) is None and slot not in asked and slot in SLOT_QUESTIONS:
            questions.append(SLOT_QUESTIONS[slot])
            asked.add(slot)
        if len(questions) >= limit:
            break
    return questions


def ready(profile: EcosystemProfile) -> bool:
    known = len(profile.known_variables())
    return known >= MIN_VARIABLES_FOR_RECOMMENDATION and len(missing_critical(profile)) <= 1


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------
def handle_turn(
    session_id: str, message: str, structured: dict[str, Any] | None = None
) -> AssistantTurn:
    session = SESSIONS.get(session_id)
    session.turn_index += 1
    session.history.append({"role": "user", "content": message})

    delta = extract_slots(message, structured)
    session.profile = session.profile.merge(delta)

    if not ready(session.profile):
        questions = next_questions(session.profile, session.asked)
        known = session.profile.known_variables()
        if known:
            lead = (
                "Noted: " + ", ".join(k.replace("_", " ") for k in known[:6]) + ". "
                "That is not yet enough to reason across variables without guessing, "
                "so before recommending anything:"
            )
        else:
            lead = (
                "To give you recommendations that connect soil, water and habitat rather "
                "than generic advice, I need a few site facts first:"
            )
        turn = AssistantTurn(
            mode="clarify",
            message=lead,
            clarifying_questions=questions
            or ["Tell me anything else you know about the site - soil, water, crop, or wildlife."],
            profile=session.profile,
            missing_critical=missing_critical(session.profile),
        )
        session.history.append({"role": "assistant", "content": turn.message})
        return turn

    return _recommend(session, message)


def _recommend(session: Session, message: str) -> AssistantTurn:
    profile = session.profile
    query = _profile_query(profile, message)

    vs = store()
    evidence = vs.query_science(query, k=TOP_K)
    shortlist = vs.query_interventions(query, k=8)

    picked = rank(INTERVENTIONS, profile, limit=MAX_RECOMMENDATIONS, shortlist=shortlist)
    if not picked:
        return AssistantTurn(
            mode="discuss",
            message=(
                "Nothing in the knowledge base matches this site's conditions without a "
                "contraindication firing. Tell me more about the constraint you are most "
                "worried about and I will reason from there."
            ),
            profile=profile,
        )

    evidence_ids = [c.chunk_id for c in evidence]
    recommendations = [
        build_recommendation(item, fit, reasons, evidence_ids) for item, fit, reasons in picked
    ]
    synergies = detect_synergies([r.intervention_id for r in recommendations])

    narrative = _narrate(profile, recommendations, evidence, synergies, message)

    turn = AssistantTurn(
        mode="recommend",
        message=narrative,
        profile=profile,
        missing_critical=missing_critical(profile),
        recommendations=recommendations,
        evidence_used=[
            {
                "chunk_id": c.chunk_id,
                "source_org": c.metadata.get("source_org"),
                "source_title": c.metadata.get("source_title"),
                "score": c.score,
                "excerpt": c.text[:280],
            }
            for c in evidence
        ],
    )
    session.history.append({"role": "assistant", "content": narrative})
    return turn


def _render(value) -> str:
    """Enum members print as 'LandUse.monoculture_crop'; we want the value."""
    return str(getattr(value, "value", value))


def _profile_query(profile: EcosystemProfile, message: str) -> str:
    bits = [f"{k}={_render(getattr(profile, k))}" for k in profile.known_variables()]
    return f"{message}\nSite: " + "; ".join(bits)


def _narrate(profile, recommendations, evidence, synergies, message: str) -> str:
    if not llm.available():
        return _fallback_narrative(profile, recommendations, synergies)

    cards = [
        {
            "title": r.title,
            "what_to_do": r.what_to_do,
            "why_it_works": r.why_it_works,
            "causal_chain": r.causal_chain,
            "effects": [e.model_dump() for e in r.metric_effects],
            "trade_offs": r.trade_offs,
            "horizon": r.horizon.value,
            "confidence": r.confidence.value,
            "citations": [
                f"{c.organisation}, {c.title} ({c.year}){'' if c.verified else ' [figure unverified]'}"
                for c in r.citations
            ],
            "matched_because": r.preconditions_met,
        }
        for r in recommendations
    ]
    payload = {
        "user_message": message,
        "site_profile": {k: _render(getattr(profile, k)) for k in profile.known_variables()},
        "selected_interventions": cards,
        "synergies_detected": synergies,
        "retrieved_evidence": [
            {"source": c.metadata.get("source_org"), "text": c.text[:900]} for c in evidence
        ],
    }
    out = llm.narrate(NARRATION_SYSTEM, json.dumps(payload, indent=2))
    return out or _fallback_narrative(profile, recommendations, synergies)


def _fallback_narrative(profile, recommendations, synergies) -> str:
    """Deterministic prose if Groq is unreachable. The system never goes silent."""
    known = ", ".join(
        f"{k.replace('_', ' ')} = {_render(getattr(profile, k))}"
        for k in profile.known_variables()[:6]
    )
    lines = [
        f"Site as recorded: {known}.",
        "Selected interventions, ranked by fit against these conditions:",
    ]
    for i, r in enumerate(recommendations, 1):
        chain = " -> ".join(r.causal_chain[:3])
        lines.append(f"{i}. {r.title}. {r.what_to_do} Mechanism: {chain}.")
        if r.trade_offs:
            lines.append(f"   Trade-off: {r.trade_offs}")
    lines.extend(synergies)
    return "\n".join(lines)
