"""Build the vector index. Run once after install, and after any knowledge edit.

    python -m knowledge.ingest
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import INTERVENTIONS_FILE  # noqa: E402
from app.retriever import load_corpus, load_interventions, store  # noqa: E402


def main() -> int:
    chunks = load_corpus()
    interventions = load_interventions(INTERVENTIONS_FILE)
    print(f"corpus chunks : {len(chunks)}")
    print(f"interventions : {len(interventions)}")

    vs = store()
    if not vs.ok:
        print("ChromaDB unavailable. The API will run on the keyword fallback retriever.")
        return 1

    stats = vs.build(interventions)
    print(f"indexed       : {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
