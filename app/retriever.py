"""Vector retrieval layer.

Two collections, deliberately:

  science_corpus     - prose chunks from reports. Used as *evidence*.
  intervention_cards - one embedded summary per structured intervention.
                       Used as a semantic *shortlist* before the rule engine
                       does the hard filtering.

If ChromaDB is unavailable (e.g. offline CI), the retriever degrades to a
transparent keyword scorer so the rest of the system still runs and tests
still pass.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .config import (
    CHROMA_DIR,
    CORPUS_DIR,
    INTERVENTION_COLLECTION,
    SCIENCE_COLLECTION,
    TOP_K,
)

log = logging.getLogger(__name__)


@dataclass
class Chunk:
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    score: float = 0.0


# --------------------------------------------------------------------------
# Corpus loading + chunking
# --------------------------------------------------------------------------
FRONTMATTER = re.compile(r"^---\n(.*?)\n---\n", re.S)


def _parse_frontmatter(raw: str) -> tuple[dict[str, str], str]:
    match = FRONTMATTER.match(raw)
    if not match:
        return {}, raw
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()
    return meta, raw[match.end():]


def chunk_document(path: Path, *, target_words: int = 180) -> list[Chunk]:
    """Paragraph-aware chunking. Keeps paragraphs whole; merges short ones."""
    meta, body = _parse_frontmatter(path.read_text(encoding="utf-8"))
    paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]

    chunks: list[Chunk] = []
    buffer: list[str] = []
    words = 0
    for para in paragraphs:
        buffer.append(para)
        words += len(para.split())
        if words >= target_words:
            chunks.append(_make_chunk(path, len(chunks), buffer, meta))
            buffer, words = [], 0
    if buffer:
        chunks.append(_make_chunk(path, len(chunks), buffer, meta))
    return chunks


def _make_chunk(path: Path, index: int, buffer: list[str], meta: dict) -> Chunk:
    return Chunk(
        chunk_id=f"{path.stem}_{index:02d}",
        text="\n\n".join(buffer),
        metadata={
            "document": path.stem,
            "title": meta.get("title", path.stem),
            "source_org": meta.get("source_org", "unknown"),
            "source_title": meta.get("source_title", ""),
            "topics": meta.get("topics", ""),
            "verified": meta.get("verified", "false"),
        },
    )


@lru_cache(maxsize=1)
def load_corpus() -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(CORPUS_DIR.glob("*.md")):
        chunks.extend(chunk_document(path))
    log.info("Loaded %d chunks from %s", len(chunks), CORPUS_DIR)
    return chunks


# --------------------------------------------------------------------------
# Chroma-backed store
# --------------------------------------------------------------------------
class VectorStore:
    def __init__(self) -> None:
        self.ok = False
        self.client = None
        try:
            import chromadb
            from chromadb.utils import embedding_functions

            CHROMA_DIR.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            # ONNX-based, no PyTorch. sentence-transformers pulls in full torch
            # (800MB+ install, 300-500MB resident just for the model), which
            # blows the 512MB memory limit on Render's free tier once FastAPI,
            # ChromaDB and uvicorn are also resident. This default embedding
            # function does the same job at a fraction of the footprint.
            self.embed_fn = embedding_functions.DefaultEmbeddingFunction()
            self.ok = True
        except Exception as exc:  # noqa: BLE001
            log.warning("Vector store unavailable (%s). Falling back to keyword search.", exc)

    # -- indexing ---------------------------------------------------------
    def build(self, interventions: list[dict[str, Any]]) -> dict[str, int]:
        if not self.ok:
            raise RuntimeError("ChromaDB is not available; cannot build the index.")

        science = self.client.get_or_create_collection(
            SCIENCE_COLLECTION, embedding_function=self.embed_fn
        )
        chunks = load_corpus()
        if chunks:
            science.upsert(
                ids=[c.chunk_id for c in chunks],
                documents=[c.text for c in chunks],
                metadatas=[c.metadata for c in chunks],
            )

        cards = self.client.get_or_create_collection(
            INTERVENTION_COLLECTION, embedding_function=self.embed_fn
        )
        if interventions:
            cards.upsert(
                ids=[i["id"] for i in interventions],
                documents=[intervention_embedding_text(i) for i in interventions],
                metadatas=[
                    {"title": i["title"], "category": i.get("category", "")}
                    for i in interventions
                ],
            )
        return {"science_chunks": len(chunks), "intervention_cards": len(interventions)}

    # -- querying ---------------------------------------------------------
    def query_science(self, text: str, k: int = TOP_K) -> list[Chunk]:
        if not self.ok:
            return keyword_search(text, load_corpus(), k)
        try:
            col = self.client.get_or_create_collection(
                SCIENCE_COLLECTION, embedding_function=self.embed_fn
            )
            res = col.query(query_texts=[text], n_results=k)
            out: list[Chunk] = []
            for cid, doc, meta, dist in zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
            ):
                out.append(Chunk(cid, doc, meta, score=round(1 - dist, 4)))
            return out
        except Exception as exc:  # noqa: BLE001
            log.warning("Vector query failed (%s); using keyword fallback.", exc)
            return keyword_search(text, load_corpus(), k)

    def query_interventions(self, text: str, k: int = 8) -> list[str]:
        if not self.ok:
            return []
        try:
            col = self.client.get_or_create_collection(
                INTERVENTION_COLLECTION, embedding_function=self.embed_fn
            )
            res = col.query(query_texts=[text], n_results=k)
            return list(res["ids"][0])
        except Exception:  # noqa: BLE001
            return []


def intervention_embedding_text(item: dict[str, Any]) -> str:
    parts = [
        item["title"],
        item.get("what_to_do", ""),
        item.get("why_it_works", ""),
        " ".join(item.get("causal_chain", [])),
        item.get("search_query", ""),
    ]
    return "\n".join(p for p in parts if p)


# --------------------------------------------------------------------------
# Offline fallback
# --------------------------------------------------------------------------
STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "how",
    "what", "why", "can", "you", "your", "our", "into", "over", "than", "then",
    "has", "have", "not", "but", "its", "it", "in", "on", "of", "to", "a", "an",
    "is", "be", "at", "by", "or", "as", "my", "i",
}


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z]{3,}", text.lower()) if t not in STOPWORDS]


def keyword_search(query: str, chunks: list[Chunk], k: int) -> list[Chunk]:
    q = set(_tokens(query))
    scored: list[Chunk] = []
    for chunk in chunks:
        tokens = _tokens(chunk.text)
        if not tokens:
            continue
        overlap = sum(1 for t in tokens if t in q)
        score = overlap / (len(tokens) ** 0.5)
        if score > 0:
            scored.append(Chunk(chunk.chunk_id, chunk.text, chunk.metadata, round(score, 4)))
    scored.sort(key=lambda c: c.score, reverse=True)
    return scored[:k]


_store: VectorStore | None = None


def store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store


def load_interventions(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["interventions"]