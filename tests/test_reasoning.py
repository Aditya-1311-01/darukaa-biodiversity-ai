"""Tests for the deterministic layer. No network, no API key needed."""
import json
from pathlib import Path

import pytest

from app.config import INTERVENTIONS_FILE
from app.reasoning import detect_synergies, need_signals, rank, score_intervention
from app.conversation import missing_critical, ready, regex_extract, coerce
from app.schemas import EcosystemProfile

INTERVENTIONS = json.loads(INTERVENTIONS_FILE.read_text())["interventions"]

SEMI_ARID_WHEAT = EcosystemProfile(
    soil_organic_carbon_pct=0.3,
    rainfall_regime="low",
    climate_zone="semi_arid",
    land_use="monoculture_crop",
    current_crop="wheat",
    area_hectares=4,
)


def test_knowledge_base_shape():
    assert len(INTERVENTIONS) >= 15
    ids = [i["id"] for i in INTERVENTIONS]
    assert len(ids) == len(set(ids)), "intervention ids must be unique"
    for item in INTERVENTIONS:
        assert item["effects"], f"{item['id']} has no metric effects"
        assert item["citations"], f"{item['id']} has no citation"
        assert len(item["causal_chain"]) >= 3, f"{item['id']} chain is too shallow"


def test_gate_blocks_thin_profiles():
    assert not ready(EcosystemProfile(soil_ph=7.2))
    assert ready(SEMI_ARID_WHEAT)


def test_missing_critical_is_reported():
    assert "land_use" in missing_critical(EcosystemProfile(soil_ph=7.2))
    assert missing_critical(SEMI_ARID_WHEAT) == []


def test_need_signals_detect_degradation():
    signals = need_signals(SEMI_ARID_WHEAT)
    assert signals["soil_carbon_depletion"] > 0.5
    assert signals["water_scarcity"] == 1.0
    assert signals["structural_simplicity"] == 1.0


def test_ranking_is_multi_variable_and_relevant():
    picked = rank(INTERVENTIONS, SEMI_ARID_WHEAT, limit=3)
    assert len(picked) == 3
    ids = [item["id"] for item, _, _ in picked]
    assert any(i in ids for i in ("legume_cover_crop", "alley_cropping_agroforestry",
                                  "crop_diversification_intercrop", "residue_retention_no_till"))
    # collectively the set must engage several distinct site variables
    engaged = {r for _, _, reasons in picked for r in reasons}
    assert len(engaged) >= 3, "recommendations are not reasoning across variables"


def test_binding_constraint_is_always_addressed():
    """A strongly sodic soil must get the sodicity fix, not only generic advice."""
    sodic = EcosystemProfile(
        soil_ph=9.1, land_use="degraded_land", climate_zone="semi_arid",
        rainfall_regime="low", tree_cover_pct=2,
    )
    ids = [item["id"] for item, _, _ in rank(INTERVENTIONS, sodic, limit=3)]
    assert "gypsum_organic_amendment_sodic" in ids


def test_crop_specific_intervention_surfaces():
    paddy = EcosystemProfile(
        land_use="monoculture_crop", current_crop="rice", climate_zone="humid",
        rainfall_regime="high", irrigation_source="tubewell",
        fertiliser_use="high broadcast urea", pesticide_use="calendar spraying",
        soil_organic_carbon_pct=0.9,
    )
    ids = [item["id"] for item, _, _ in rank(INTERVENTIONS, paddy, limit=3)]
    assert "alternate_wetting_drying_rice" in ids


def test_category_diversity_guard():
    picked = rank(INTERVENTIONS, SEMI_ARID_WHEAT, limit=3)
    cats = [item.get("category") for item, _, _ in picked]
    assert max(cats.count(c) for c in set(cats)) <= 2


def test_contraindication_blocks():
    very_dry = EcosystemProfile(
        soil_organic_carbon_pct=0.3, annual_rainfall_mm=250,
        rainfall_regime="low", land_use="monoculture_crop", climate_zone="arid",
    )
    cover = next(i for i in INTERVENTIONS if i["id"] == "legume_cover_crop")
    assert score_intervention(cover, very_dry, need_signals(very_dry)) is None


def test_ph_gates_sodic_treatment():
    gypsum = next(i for i in INTERVENTIONS if i["id"] == "gypsum_organic_amendment_sodic")
    sodic = EcosystemProfile(soil_ph=9.1, land_use="degraded_land", climate_zone="semi_arid")
    neutral = EcosystemProfile(soil_ph=6.8, land_use="degraded_land", climate_zone="semi_arid")
    assert score_intervention(gypsum, sodic, need_signals(sodic)) is not None
    assert score_intervention(gypsum, neutral, need_signals(neutral)) is None


def test_synergy_detection():
    notes = detect_synergies(["legume_cover_crop", "residue_retention_no_till"])
    assert notes and "oxidised" in notes[0]
    assert detect_synergies(["legume_cover_crop"]) == []


def test_regex_extraction_handles_hinglish_and_units():
    got = regex_extract("mera 2 ha plot hai, organic carbon 0.4 %, ph 8.6, kam baarish, sirf gehu")
    assert got["soil_organic_carbon_pct"] == 0.4
    assert got["soil_ph"] == 8.6
    assert got["rainfall_regime"] == "low"
    assert got["land_use"] == "monoculture_crop"
    assert got["area_hectares"] == 2


def test_rainfall_mm_maps_to_regime():
    assert regex_extract("annual rainfall is 450 mm")["rainfall_regime"] == "low"
    assert regex_extract("we get 1400 mm")["rainfall_regime"] == "high"


def test_coerce_drops_invalid_slots():
    profile = coerce({"soil_ph": 99, "land_use": "monoculture_crop", "nonsense": 1})
    assert profile.land_use.value == "monoculture_crop"
    assert profile.soil_ph is None


def test_profile_merge_never_overwrites_with_null():
    merged = SEMI_ARID_WHEAT.merge(EcosystemProfile(soil_ph=8.1))
    assert merged.soil_ph == 8.1
    assert merged.soil_organic_carbon_pct == 0.3
