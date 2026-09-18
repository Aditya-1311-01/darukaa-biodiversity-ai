# Biodiversity Intelligence — Darukaa.Earth Challenge

A knowledge-grounded conversational system that reasons about soil, water, land use and biodiversity together, and returns recommendations with a stated mechanism, quantified metric effects, a time horizon, a confidence level, a trade-off and a source.

The design position of this project is simple: **the language model does not choose the recommendations.** A deterministic reasoning engine does, using a structured knowledge base with explicit preconditions, contraindications and causal chains. The model is used for two narrow jobs — parsing a messy human message into typed fields, and narrating a decision that has already been made and can be justified field by field. An LLM asked to pick interventions produces fluent, plausible, unfalsifiable advice, which is precisely what the brief rules out.

---

## Architecture

```
                    user message (text and/or JSON)
                                │
        ┌───────────────────────▼───────────────────────┐
        │ 1. SLOT EXTRACTION                            │
        │    Groq llama-3.1-8b (JSON mode)              │
        │    + regex fallback (handles Hinglish)        │
        │    → merged into the session EcosystemProfile │
        └───────────────────────┬───────────────────────┘
                                │
        ┌───────────────────────▼───────────────────────┐
        │ 2. READINESS GATE                             │
        │    ≥ 3 environmental variables known?         │
        │    critical slots present?                    │
        └──────────┬───────────────────────┬────────────┘
                 no│                       │yes
        ┌──────────▼──────────┐            │
        │ targeted clarifying │            │
        │ questions, no guess │            │
        └─────────────────────┘            │
                          ┌────────────────▼────────────────┐
                          │ 3. RETRIEVAL (ChromaDB)         │
                          │   science_corpus   → evidence   │
                          │   intervention_cards → shortlist│
                          └────────────────┬────────────────┘
                                           │
                          ┌────────────────▼────────────────┐
                          │ 4. DETERMINISTIC REASONING      │
                          │   preconditions / contraindic.  │
                          │   degradation signals           │
                          │   scoring + diversity guard     │
                          │   binding-constraint rule       │
                          │   synergy detection             │
                          └────────────────┬────────────────┘
                                           │
                          ┌────────────────▼────────────────┐
                          │ 5. NARRATION                    │
                          │   Groq llama-3.3-70b, grounded  │
                          │   ONLY in the selected cards    │
                          │   and retrieved chunks          │
                          └────────────────┬────────────────┘
                                           │
                       structured JSON  +  prose site assessment
```

Every stage degrades instead of failing. No Groq key → regex extraction and a deterministic narrative. No ChromaDB → a transparent keyword retriever. The reasoning core has no external dependency at all, which is what lets CI test it in isolation.

### Why the reasoning layer is separate

Three behaviours fall out of this split that a prompt-only system cannot reliably give you:

1. **Contraindications actually block.** A legume cover crop is refused below ~300 mm annual rainfall because it competes for residual moisture. Gypsum is refused on a neutral soil. These are hard gates, not suggestions in a prompt.
2. **The binding constraint is always addressed.** Broadly-useful interventions outscore narrow ones by construction, because they engage more variables. A strongly sodic soil would therefore receive cover-crop advice while nothing touched the sodium causing the problem. An explicit rule guarantees that every severely degraded dimension is covered by at least one recommendation.
3. **Synergies are stated, not implied.** Pairs of interventions whose effects compound (cover cropping + zero till; hedgerow + threshold spraying; inoculation + reduced phosphorus) are detected from the selected set and surfaced as reasoning about the combination.

---

## Knowledge system

Two collections, indexed separately because they answer different questions.

| Collection | Contents | Role |
|---|---|---|
| `science_corpus` | Paragraph-chunked notes synthesised from FAO, IPCC, IPBES, UNCCD, UNEP and Ramsar material, with frontmatter metadata | Evidence retrieval, shown to the user |
| `intervention_cards` | One embedded summary per structured intervention | Semantic shortlist before the rule engine filters |

Embeddings: `BAAI/bge-small-en-v1.5` via sentence-transformers. Store: ChromaDB, persistent.

### Intervention schema

`knowledge/interventions.json` — 20 interventions, 84 quantified metric effects. This file is the substance of the system; the code around it is comparatively small.

```jsonc
{
  "id": "legume_cover_crop",
  "title": "Legume-based cover cropping in the fallow window",
  "category": "soil_biology",
  "what_to_do": "…concrete, executable instruction…",
  "preconditions":     { "soil_organic_carbon_pct": {"max": 0.75},
                         "land_use": ["monoculture_crop", "fallow"] },
  "contraindications": { "annual_rainfall_mm": {"max": 300} },
  "need_keys": ["soil_carbon_depletion"],          // optional override
  "causal_chain": [ "…", "…", "…" ],               // ≥ 3 linked steps
  "why_it_works": "…mechanism, not a slogan…",
  "effects": [
    { "metric": "soil_organic_carbon", "direction": "increase",
      "magnitude": "+0.1-0.3 percentage points",
      "horizon": "medium", "confidence": "high" }
  ],
  "horizon": "medium",
  "confidence": "high",
  "trade_offs": "…never presented as costless…",
  "citations": [ { "organisation": "FAO", "title": "…", "year": 2017,
                   "locator": "ch. 4", "url": "…", "verified": false } ]
}
```

> **On the `verified` flag.** Every citation carries it, and the API and UI surface unverified figures as unverified. Effect magnitudes are drawn from published ranges, and each one should be checked against the primary source before it is presented as settled. The flag exists so the system is honest about which claims have been through that check rather than quietly asserting all of them with equal confidence.

### Scoring

```
fit = precondition_fit   * 0.40    # how many checkable preconditions match
    + variable_coverage  * 0.20    # how many distinct site variables it engages
    + urgency            * 0.25    # how degraded the dimension it targets is
    + evidence_strength  * 0.05    # confidence grading of the intervention
    + 0.10 if urgency >= 0.6       # narrow fixes for severe problems get a step up
```

Then: at most two picks from any one category, and the binding-constraint rule described above.

---

## Evaluation

`python -m eval.evaluate` runs eight expert-labelled site cases through the full selection path and reports precision@3, hit rate on the non-negotiable intervention, and whether every severely degraded dimension was covered.

Current results:

| Metric | Value |
|---|---|
| Cases | 8 |
| Mean precision@3 | 0.917 |
| Must-include hit rate | 1.000 |
| Severe-constraint coverage | 1.000 |
| Distinct categories per answer | 2–3 |

With `ragas` and `datasets` installed and a Groq key present, the same command additionally scores the generated narrative for faithfulness to the retrieved evidence and for context precision. This is the check that catches the model drifting off its sources.

---

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/chat` | Send a message and/or a structured profile; returns the turn |
| `GET` | `/api/session/{id}/profile` | Inspect accumulated memory for a session |
| `POST` | `/api/session/{id}/reset` | Clear a session |
| `GET` | `/api/knowledge/interventions` | List the knowledge base |
| `POST` | `/api/knowledge/reindex` | Rebuild the vector index |
| `GET` | `/api/health` | Component status |

Request:

```json
{
  "session_id": "s-demo",
  "message": "Biodiversity is declining on my land",
  "structured_input": {
    "soil_organic_carbon_pct": 0.3,
    "rainfall_regime": "low",
    "climate_zone": "semi_arid",
    "land_use": "monoculture_crop",
    "current_crop": "wheat",
    "area_hectares": 4
  }
}
```

Response (abridged):

```json
{
  "mode": "recommend",
  "message": "…site assessment prose…",
  "profile": { "…accumulated across turns…" },
  "recommendations": [
    {
      "intervention_id": "legume_cover_crop",
      "title": "Legume-based cover cropping in the fallow window",
      "what_to_do": "…",
      "why_it_works": "…",
      "causal_chain": ["…", "…", "…"],
      "metric_effects": [
        { "metric": "soil_organic_carbon", "direction": "increase",
          "magnitude": "+0.1-0.3 percentage points",
          "horizon": "medium", "confidence": "high" }
      ],
      "horizon": "medium",
      "confidence": "high",
      "trade_offs": "Consumes residual soil moisture…",
      "preconditions_met": ["soil_organic_carbon_pct = 0.3", "land_use = monoculture_crop"],
      "citations": [{ "organisation": "FAO", "verified": false }],
      "retrieved_evidence": ["soil_carbon_00", "water_biodiversity_00"],
      "fit_score": 1.0
    }
  ],
  "evidence_used": [{ "chunk_id": "soil_carbon_00", "source_org": "FAO", "score": 0.81, "excerpt": "…" }]
}
```

Structured input is merged **over** anything parsed from the text, so explicit JSON always wins.

---

## Local setup

Requires Python 3.10 or newer. Everything runs inside a project-local virtual environment, so nothing is installed system-wide.

### One command

```bash
git clone <your-repo-url> && cd darukaa-biodiversity-ai

bash setup.sh          # macOS / Linux
setup.bat              # Windows
make setup             # if you have make
```

The script creates `.venv`, installs dependencies, copies `.env.example` to `.env`, builds the ChromaDB index and runs the tests. The first run downloads the sentence-transformers embedding model, which takes a few minutes.

Then add your Groq key to `.env` (free, from https://console.groq.com) and start the server:

```bash
source .venv/bin/activate      # Windows: .venv\Scripts\activate
uvicorn app.main:app --reload
```

Open `http://localhost:8000`. The frontend is served by the same FastAPI process, so there is nothing else to start.

### Manual, if you prefer

```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

cp .env.example .env           # then add your Groq key
python -m knowledge.ingest     # builds the vector index
uvicorn app.main:app --reload
```

`deactivate` leaves the environment. `.venv/` is gitignored and must never be committed — collaborators rebuild it from `requirements.txt`.

### Make targets

| Command | Does |
|---|---|
| `make setup` | venv, install, `.env`, index, tests |
| `make run` | start the dev server |
| `make test` | pytest |
| `make eval` | selection eval + RAGAS if installed |
| `make lint` | ruff |
| `make clean` | remove `.venv`, `.chroma`, caches |

### Tests

```bash
source .venv/bin/activate
pytest -q
python -m eval.evaluate
```

The test suite needs neither a Groq key nor ChromaDB — it targets the deterministic layer, which is the part where correctness is checkable.

### Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `externally-managed-environment` on pip install | You are outside the venv. Activate it first, or run `bash setup.sh`. |
| `GROQ_API_KEY is not set` | `.env` missing or empty. The app still runs — regex extraction and a deterministic narrative — but without LLM narration. |
| Health endpoint shows `keyword_fallback` | ChromaDB or sentence-transformers failed to load. Re-run `python -m knowledge.ingest` inside the venv and read the error it prints. |
| `chromadb` build fails on install | Usually a missing compiler. `sudo apt install build-essential` on Debian/Ubuntu, or use the Docker path instead. |

---

## Deployment

**Render** (`render.yaml` included): connect the repo, set `GROQ_API_KEY` in the dashboard, deploy. The build command installs dependencies and builds the index; `CHROMA_DIR` points at `/tmp/chroma` because the free tier filesystem is ephemeral.

**Docker**:

```bash
docker build -t darukaa-biodiversity .
docker run -p 8000:8000 -e GROQ_API_KEY=your_key darukaa-biodiversity
```

The image pre-builds the vector index at build time so cold starts do not pay for it.

---

## CI/CD

`.github/workflows/ci.yml` runs on every push and pull request:

1. `ruff check` across `app`, `knowledge`, `eval`, `tests`
2. `pytest -q` — 15 tests on the reasoning core, knowledge-base integrity, slot extraction and session memory
3. Import check on the FastAPI app
4. Docker image build, gated on the test job passing

Because the reasoning layer imports Groq lazily and falls back when ChromaDB is absent, CI installs only the light dependencies and still exercises the logic that matters.

---

## Repository layout

```
setup.sh / setup.bat / Makefile   one-command venv setup
app/
  main.py          FastAPI routes, static mount
  schemas.py       Pydantic contracts — profile, recommendation, API
  conversation.py  slot filling, memory, gate, orchestration
  reasoning.py     scoring, contraindications, constraint rule, synergies
  retriever.py     ChromaDB collections + keyword fallback
  llm.py           Groq wrapper (JSON extraction, narration)
  static/          frontend — index.html, styles.css, app.js
knowledge/
  interventions.json   20 structured interventions
  corpus/*.md          science notes indexed for evidence
  ingest.py            index builder
eval/
  evaluate.py          selection eval + optional RAGAS
  golden_cases.json    8 expert-labelled site cases
tests/
  test_reasoning.py    15 tests, no network required
```

---

## Requirement mapping

| Brief requirement | Where it lives |
|---|---|
| Retrievable knowledge layer, not prompts | `knowledge/`, `app/retriever.py` — two Chroma collections plus a structured intervention DB |
| Soil, land use, biodiversity, climate, human impact | `EcosystemProfile` covers all five groups; corpus has a note per group |
| Clarifying questions on incomplete input | `conversation.py` readiness gate — recommendations are withheld below three variables |
| Multi-turn memory | Session profile merges across turns, never overwritten with nulls; visible live in the UI panel |
| What / why / which metric / reference | Every `Recommendation` carries all four as typed fields |
| Multi-metric reasoning | `causal_chain` per intervention, `need_signals` across dimensions, synergy detection across the selected set |
| Text + structured input | `message` and `structured_input` on the same endpoint; geo-coordinates supported in the profile |
| Horizon and confidence | Per intervention and per individual metric effect |
