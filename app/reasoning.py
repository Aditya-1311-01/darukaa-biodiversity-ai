"""The deterministic reasoning core.

This module decides *which* interventions apply and *why*. The LLM is not
involved. That is the whole design argument of this project: an LLM asked to
pick interventions will produce fluent, plausible, unfalsifiable advice, which
is exactly what the brief rules out. Here the LLM only narrates a decision the
engine has already made and can justify field by field.

Scoring is intentionally simple and inspectable:

    fit = precondition_fit * 0.45
        + variable_coverage * 0.25      # rewards multi-variable engagement
        + need_urgency     * 0.20       # how bad the current state actually is
        + evidence_strength* 0.10
"""
from __future__ import annotations

from typing import Any

from .schemas import (
    Citation,
    Confidence,
    EcosystemProfile,
    Horizon,
    MetricEffect,
    Recommendation,
)

CONFIDENCE_WEIGHT = {"low": 0.4, "medium": 0.7, "high": 1.0}


# --------------------------------------------------------------------------
# Precondition matching
# --------------------------------------------------------------------------
def _match_numeric(value: float, rule: dict[str, float]) -> bool:
    if "min" in rule and value < rule["min"]:
        return False
    if "max" in rule and value > rule["max"]:
        return False
    return True


def _match_categorical(value: Any, allowed: list[str]) -> bool:
    text = str(getattr(value, "value", value)).lower()
    return any(option.lower() in text or text in option.lower() for option in allowed)


def evaluate_conditions(
    profile: EcosystemProfile, conditions: dict[str, Any]
) -> tuple[int, int, list[str]]:
    """Returns (matched, evaluable, human-readable reasons)."""
    data = profile.model_dump()
    matched = 0
    evaluable = 0
    reasons: list[str] = []

    for field, rule in conditions.items():
        value = data.get(field)
        if value is None:
            continue  # unknown slots neither help nor hurt
        evaluable += 1
        ok = _match_numeric(value, rule) if isinstance(rule, dict) else _match_categorical(value, rule)
        if ok:
            matched += 1
            reasons.append(f"{field} = {getattr(value, 'value', value)}")
    return matched, evaluable, reasons


# --------------------------------------------------------------------------
# Need / urgency
# --------------------------------------------------------------------------
def need_signals(profile: EcosystemProfile) -> dict[str, float]:
    """How degraded is each dimension, on 0..1. Drives urgency weighting."""
    signals: dict[str, float] = {}

    soc = profile.soil_organic_carbon_pct
    if soc is not None:
        signals["soil_carbon_depletion"] = min(max((0.75 - soc) / 0.75, 0.0), 1.0)

    if profile.rainfall_regime is not None:
        signals["water_scarcity"] = {"low": 1.0, "medium": 0.5, "high": 0.1}[
            profile.rainfall_regime.value
        ]
    if profile.climate_zone is not None:
        signals["aridity"] = {
            "arid": 1.0, "semi_arid": 0.8, "sub_humid": 0.4,
            "humid": 0.2, "tropical_wet": 0.2, "temperate": 0.3, "montane": 0.4,
        }[profile.climate_zone.value]

    if profile.land_use is not None:
        signals["structural_simplicity"] = {
            "monoculture_crop": 1.0, "fallow": 0.8, "degraded_land": 1.0,
            "grazing_land": 0.7, "plantation_forest": 0.6, "orchard": 0.5,
            "mixed_cropping": 0.4, "peri_urban": 0.7, "wetland": 0.5,
            "natural_forest": 0.1,
        }[profile.land_use.value]

    if profile.tree_cover_pct is not None:
        signals["woody_deficit"] = min(max((20 - profile.tree_cover_pct) / 20, 0.0), 1.0)

    if profile.soil_ph is not None:
        if profile.soil_ph >= 8.4:
            signals["sodicity_alkalinity"] = min((profile.soil_ph - 8.4) / 0.8, 1.0)
        elif profile.soil_ph <= 5.5:
            signals["acidity"] = min((5.5 - profile.soil_ph) / 1.0, 1.0)

    crop = (profile.current_crop or "").lower()
    if "rice" in crop or "paddy" in crop:
        signals["irrigated_paddy"] = 1.0 if profile.irrigation_source else 0.7

    for field, key in (
        ("pesticide_use", "chemical_load"),
        ("fertiliser_use", "nutrient_load"),
        ("grazing_pressure", "grazing_load"),
    ):
        raw = getattr(profile, field)
        if raw:
            text = str(raw).lower()
            if any(w in text for w in ("high", "heavy", "routine", "calendar", "continuous", "excess")):
                signals[key] = 1.0
            elif any(w in text for w in ("moderate", "medium", "some")):
                signals[key] = 0.6
            else:
                signals[key] = 0.2
    return signals


CATEGORY_NEEDS = {
    "soil_biology": ["soil_carbon_depletion", "structural_simplicity"],
    "soil_physics": ["soil_carbon_depletion", "water_scarcity", "aridity"],
    "soil_chemistry": ["sodicity_alkalinity", "acidity", "soil_carbon_depletion"],
    "agroforestry": ["aridity", "woody_deficit", "structural_simplicity", "water_scarcity"],
    "cropping_system": ["structural_simplicity", "nutrient_load"],
    "hydrology": ["water_scarcity", "aridity"],
    "habitat_structure": ["structural_simplicity", "woody_deficit"],
    "rangeland": ["grazing_load", "structural_simplicity"],
    "restoration": ["structural_simplicity", "woody_deficit"],
    "chemical_load": ["chemical_load", "nutrient_load"],
    "water_quality": ["nutrient_load", "chemical_load"],
    "urban_ecology": ["structural_simplicity"],
}


def targets_of(item: dict[str, Any]) -> list[str]:
    """Which degradation signals this intervention is meant to address."""
    return item.get("need_keys") or CATEGORY_NEEDS.get(item.get("category", ""), [])


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def score_intervention(
    item: dict[str, Any], profile: EcosystemProfile, signals: dict[str, float]
) -> tuple[float, list[str]] | None:
    pre_matched, pre_evaluable, reasons = evaluate_conditions(
        profile, item.get("preconditions", {})
    )
    if pre_evaluable and pre_matched == 0:
        return None  # every checkable precondition failed

    contra_matched, _, contra_reasons = evaluate_conditions(
        profile, item.get("contraindications", {})
    )
    if contra_matched:
        return None  # ruled out on site conditions

    precondition_fit = (pre_matched / pre_evaluable) if pre_evaluable else 0.35

    # Multi-variable coverage: how many distinct profile variables does this
    # intervention's precondition set actually engage with?
    engaged = len(reasons)
    variable_coverage = min(engaged / 3.0, 1.0)

    urgency = max((signals.get(key, 0.0) for key in targets_of(item)), default=0.3)
    evidence_strength = CONFIDENCE_WEIGHT.get(item.get("confidence", "medium"), 0.7)

    # An intervention that directly addresses a severely degraded dimension gets
    # a step up, so narrowly targeted fixes are not buried by broadly useful ones.
    targeted_bonus = 0.10 if urgency >= 0.6 else 0.0

    fit = (
        precondition_fit * 0.40
        + variable_coverage * 0.20
        + urgency * 0.25
        + evidence_strength * 0.05
        + targeted_bonus
    )
    return round(min(fit, 1.0), 4), reasons


# --------------------------------------------------------------------------
# Synergy detection — the "connect multiple variables" differentiator
# --------------------------------------------------------------------------
SYNERGY_RULES: list[tuple[set[str], str]] = [
    (
        {"legume_cover_crop", "residue_retention_no_till"},
        "Cover cropping and residue retention compound: the cover crop supplies the carbon "
        "input while zero tillage stops that carbon being oxidised again each season. Run "
        "separately, each gives roughly half of what the pair gives.",
    ),
    (
        {"alley_cropping_agroforestry", "farm_pond_water_harvesting"},
        "Hedgerows cut the evaporative demand side of the water balance while the pond "
        "raises the supply side. Together they address the dry-spell constraint from both "
        "directions, which is what determines whether the tree component survives year one.",
    ),
    (
        {"native_hedgerow_field_margin", "integrated_pest_management"},
        "The hedgerow builds the natural-enemy population and threshold spraying stops it "
        "being destroyed. Doing the hedgerow while still calendar-spraying wastes it.",
    ),
    (
        {"mycorrhizal_inoculation", "split_nitrogen_application"},
        "High soluble nutrient availability switches the mycorrhizal symbiosis off, so the "
        "inoculation only pays if fertiliser rates come down at the same time.",
    ),
    (
        {"riparian_buffer_strip", "split_nitrogen_application"},
        "Reducing nitrogen surplus at source plus intercepting what still escapes is the "
        "standard two-barrier approach to nitrate load; the buffer alone saturates over time.",
    ),
    (
        {"native_hedgerow_field_margin", "pollinator_nesting_habitat"},
        "Forage without nesting substrate produces visiting pollinators, not resident ones. "
        "The pair is what converts visitation into a persistent local population.",
    ),
    (
        {"assisted_natural_regeneration", "rotational_grazing"},
        "Regeneration fails under continuous grazing because resprouts are browsed before "
        "they escape. Rest periods are the precondition that makes regeneration possible.",
    ),
]


def detect_synergies(ids: list[str]) -> list[str]:
    chosen = set(ids)
    return [note for combo, note in SYNERGY_RULES if combo.issubset(chosen)]


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------
def build_recommendation(
    item: dict[str, Any], fit: float, reasons: list[str], evidence_ids: list[str]
) -> Recommendation:
    effects = [
        MetricEffect(
            metric=e["metric"],
            direction=e["direction"],
            magnitude=e["magnitude"],
            horizon=Horizon(e["horizon"]),
            confidence=Confidence(e["confidence"]),
        )
        for e in item.get("effects", [])
    ]
    citations = [Citation(**c) for c in item.get("citations", [])]
    return Recommendation(
        intervention_id=item["id"],
        title=item["title"],
        what_to_do=item["what_to_do"],
        why_it_works=item["why_it_works"],
        causal_chain=item.get("causal_chain", []),
        metric_effects=effects,
        horizon=Horizon(item.get("horizon", "medium")),
        confidence=Confidence(item.get("confidence", "medium")),
        trade_offs=item.get("trade_offs"),
        preconditions_met=reasons,
        citations=citations,
        retrieved_evidence=evidence_ids,
        fit_score=fit,
    )


def rank(
    interventions: list[dict[str, Any]],
    profile: EcosystemProfile,
    *,
    limit: int = 3,
    shortlist: list[str] | None = None,
) -> list[tuple[dict[str, Any], float, list[str]]]:
    signals = need_signals(profile)
    scored: list[tuple[dict[str, Any], float, list[str]]] = []

    for item in interventions:
        result = score_intervention(item, profile, signals)
        if result is None:
            continue
        fit, reasons = result
        if shortlist and item["id"] in shortlist:
            fit = round(min(fit + 0.05, 1.0), 4)  # small semantic-retrieval boost
        scored.append((item, fit, reasons))

    scored.sort(key=lambda row: row[1], reverse=True)

    # Diversity guard: at most two from any one category, so the answer is
    # never three flavours of the same idea.
    picked: list[tuple[dict[str, Any], float, list[str]]] = []
    per_category: dict[str, int] = {}
    for row in scored:
        cat = row[0].get("category", "")
        if per_category.get(cat, 0) >= 2:
            continue
        per_category[cat] = per_category.get(cat, 0) + 1
        picked.append(row)
        if len(picked) >= limit:
            break

    return _enforce_binding_constraint(picked, scored, signals)


SEVERE = 0.7
MAX_SWAPS = 2


def _enforce_binding_constraint(picked, scored, signals):
    """Every severely degraded dimension must be addressed by something.

    Without this, broadly-useful interventions crowd out the narrow fix the site
    actually needs - a strongly sodic soil getting cover-crop advice while
    nothing touches the sodium that is causing the problem. Ranking alone cannot
    fix that, because a narrow fix engages fewer variables by construction.
    """
    if not picked or not signals:
        return picked

    severe = sorted(
        ((k, v) for k, v in signals.items() if v >= SEVERE),
        key=lambda kv: kv[1],
        reverse=True,
    )
    swaps = 0

    for constraint, _ in severe:
        if swaps >= MAX_SWAPS:
            break
        chosen_ids = {item["id"] for item, _, _ in picked}
        if any(constraint in targets_of(item) for item, _, _ in picked):
            continue

        replacement = next(
            (row for row in scored if constraint in targets_of(row[0])
             and row[0]["id"] not in chosen_ids),
            None,
        )
        if replacement is None:
            continue

        # Drop the lowest-scoring pick that is not the only cover for another
        # severe constraint.
        covered_once = {
            c for c, _ in severe
            if sum(1 for item, _, _ in picked if c in targets_of(item)) == 1
        }
        droppable = [
            row for row in picked
            if not (covered_once & set(targets_of(row[0])))
        ] or picked
        victim = min(droppable, key=lambda row: row[1])

        picked = [row for row in picked if row[0]["id"] != victim[0]["id"]] + [replacement]
        swaps += 1

    picked.sort(key=lambda row: row[1], reverse=True)
    return picked
