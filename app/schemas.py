"""Typed contracts for the whole system.

Everything that crosses a boundary (HTTP, LLM, retriever, rule engine) is a
Pydantic model. That is what keeps the LLM from inventing structure.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------
# Controlled vocabularies
# --------------------------------------------------------------------------
class LandUse(str, Enum):
    monoculture_crop = "monoculture_crop"
    mixed_cropping = "mixed_cropping"
    orchard = "orchard"
    grazing_land = "grazing_land"
    fallow = "fallow"
    degraded_land = "degraded_land"
    plantation_forest = "plantation_forest"
    natural_forest = "natural_forest"
    wetland = "wetland"
    peri_urban = "peri_urban"


class Climate(str, Enum):
    arid = "arid"
    semi_arid = "semi_arid"
    sub_humid = "sub_humid"
    humid = "humid"
    tropical_wet = "tropical_wet"
    temperate = "temperate"
    montane = "montane"


class Rainfall(str, Enum):
    low = "low"          # < 600 mm/yr
    medium = "medium"    # 600-1100 mm/yr
    high = "high"        # > 1100 mm/yr


class Horizon(str, Enum):
    short = "short"      # < 1 year
    medium = "medium"    # 1-4 years
    long = "long"        # 5+ years


class Confidence(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


# --------------------------------------------------------------------------
# Conversation state — the slot-filling target
# --------------------------------------------------------------------------
class EcosystemProfile(BaseModel):
    """Accumulated knowledge about the user's land. Filled across turns."""

    # Soil
    soil_organic_carbon_pct: float | None = Field(
        None, ge=0, le=20, description="Soil organic carbon, percent by weight"
    )
    soil_ph: float | None = Field(None, ge=2, le=11)
    soil_moisture_pct: float | None = Field(None, ge=0, le=100)
    soil_texture: str | None = Field(None, description="sandy | loam | clay | silt ...")

    # Land
    land_use: LandUse | None = None
    current_crop: str | None = None
    area_hectares: float | None = Field(None, gt=0)

    # Climate
    climate_zone: Climate | None = None
    rainfall_regime: Rainfall | None = None
    annual_rainfall_mm: float | None = Field(None, ge=0)
    mean_temperature_c: float | None = None

    # Biodiversity
    species_richness_note: str | None = None
    habitat_diversity_note: str | None = None
    tree_cover_pct: float | None = Field(None, ge=0, le=100)

    # Human impact
    irrigation_source: str | None = None
    fertiliser_use: str | None = None
    pesticide_use: str | None = None
    grazing_pressure: str | None = None
    pollution_note: str | None = None

    # Spatial (bonus requirement)
    latitude: float | None = Field(None, ge=-90, le=90)
    longitude: float | None = Field(None, ge=-180, le=180)
    region_name: str | None = None

    # ------------------------------------------------------------------
    def known_variables(self) -> list[str]:
        return [k for k, v in self.model_dump().items() if v is not None]

    def merge(self, other: "EcosystemProfile") -> "EcosystemProfile":
        """Later turns overwrite earlier ones, but never with nulls."""
        base = self.model_dump()
        for key, value in other.model_dump().items():
            if value is not None:
                base[key] = value
        return EcosystemProfile(**base)


# --------------------------------------------------------------------------
# Reasoning output
# --------------------------------------------------------------------------
class MetricEffect(BaseModel):
    metric: str
    direction: Literal["increase", "decrease"]
    magnitude: str = Field(..., description="e.g. '+15-25%'")
    horizon: Horizon
    confidence: Confidence


class Citation(BaseModel):
    organisation: str
    title: str
    year: int | None = None
    locator: str | None = Field(None, description="page / section / chunk id")
    url: str | None = None
    verified: bool = Field(
        False,
        description="Set true only after a human has checked the figure against "
        "the primary source. Unverified claims are flagged in the UI.",
    )


class Recommendation(BaseModel):
    intervention_id: str
    title: str
    what_to_do: str
    why_it_works: str
    causal_chain: list[str] = Field(
        default_factory=list,
        description="Ordered multi-metric chain, e.g. "
        "['soil_organic_carbon +', 'water_holding_capacity +', 'microbial_biomass +']",
    )
    metric_effects: list[MetricEffect]
    horizon: Horizon
    confidence: Confidence
    trade_offs: str | None = None
    preconditions_met: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    retrieved_evidence: list[str] = Field(
        default_factory=list, description="Chunk ids used from the vector store"
    )
    fit_score: float = 0.0


class AssistantTurn(BaseModel):
    """What the API returns for every user message."""

    mode: Literal["clarify", "recommend", "discuss"]
    message: str
    clarifying_questions: list[str] = Field(default_factory=list)
    profile: EcosystemProfile = Field(default_factory=EcosystemProfile)
    missing_critical: list[str] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    evidence_used: list[dict[str, Any]] = Field(default_factory=list)


# --------------------------------------------------------------------------
# HTTP contracts
# --------------------------------------------------------------------------
class ChatRequest(BaseModel):
    session_id: str
    message: str = ""
    structured_input: dict[str, Any] | None = Field(
        None, description="Optional JSON profile fields, merged over parsed text"
    )


class ChatResponse(AssistantTurn):
    session_id: str
    turn_index: int
