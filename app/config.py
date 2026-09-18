from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = ROOT / "knowledge"
CORPUS_DIR = KNOWLEDGE_DIR / "corpus"
INTERVENTIONS_FILE = KNOWLEDGE_DIR / "interventions.json"
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", ROOT / ".chroma"))

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_EXTRACTION_MODEL = os.getenv("GROQ_EXTRACTION_MODEL", "llama-3.1-8b-instant")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
SCIENCE_COLLECTION = "science_corpus"
INTERVENTION_COLLECTION = "intervention_cards"

TOP_K = int(os.getenv("TOP_K", "5"))
MAX_RECOMMENDATIONS = int(os.getenv("MAX_RECOMMENDATIONS", "3"))

# A profile needs at least this many known variables before the system will
# produce recommendations. The brief requires >= 3 environmental variables.
MIN_VARIABLES_FOR_RECOMMENDATION = 3

CRITICAL_SLOTS = ["land_use", "climate_zone", "soil_organic_carbon_pct"]
