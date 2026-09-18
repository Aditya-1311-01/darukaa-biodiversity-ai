from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import llm
from .config import INTERVENTIONS_FILE, MAX_RECOMMENDATIONS
from .conversation import INTERVENTIONS, SESSIONS, handle_turn
from .retriever import load_corpus, store
from .schemas import ChatRequest, ChatResponse

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Darukaa.Earth Biodiversity Intelligence",
    description="Knowledge-grounded environmental reasoning over soil, water, land use and biodiversity.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/api/health")
def health() -> dict:
    vs = store()
    return {
        "status": "ok",
        "groq_configured": llm.available(),
        "vector_store": "chromadb" if vs.ok else "keyword_fallback",
        "interventions_loaded": len(INTERVENTIONS),
        "corpus_chunks": len(load_corpus()),
        "max_recommendations": MAX_RECOMMENDATIONS,
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    if not req.message.strip() and not req.structured_input:
        raise HTTPException(400, "Send a message, a structured_input object, or both.")
    turn = handle_turn(req.session_id, req.message, req.structured_input)
    session = SESSIONS.get(req.session_id)
    return ChatResponse(**turn.model_dump(), session_id=req.session_id, turn_index=session.turn_index)


@app.post("/api/session/{session_id}/reset")
def reset(session_id: str) -> dict:
    SESSIONS.reset(session_id)
    return {"status": "reset", "session_id": session_id}


@app.get("/api/session/{session_id}/profile")
def get_profile(session_id: str) -> dict:
    session = SESSIONS.get(session_id)
    return {
        "session_id": session_id,
        "turn_index": session.turn_index,
        "profile": session.profile.model_dump(),
        "known_variables": session.profile.known_variables(),
    }


@app.get("/api/knowledge/interventions")
def list_interventions() -> dict:
    return {
        "count": len(INTERVENTIONS),
        "source_file": str(INTERVENTIONS_FILE.name),
        "items": [
            {
                "id": i["id"],
                "title": i["title"],
                "category": i.get("category"),
                "horizon": i.get("horizon"),
                "confidence": i.get("confidence"),
                "metrics": [e["metric"] for e in i.get("effects", [])],
            }
            for i in INTERVENTIONS
        ],
    }


@app.post("/api/knowledge/reindex")
def reindex() -> dict:
    vs = store()
    if not vs.ok:
        raise HTTPException(503, "ChromaDB unavailable in this environment.")
    return vs.build(INTERVENTIONS)


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")
