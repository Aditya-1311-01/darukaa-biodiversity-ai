"""Evaluation harness.

Two layers, because they measure different failures:

1. Golden-case retrieval/selection eval (always runs, no API key needed).
   For a set of site profiles with expert-expected interventions, measure
   precision@3, recall of the must-have intervention, and whether the binding
   constraint was addressed. This catches reasoning regressions.

2. RAGAS faithfulness / context-precision on the generated narrative (runs only
   when GROQ_API_KEY and ragas are present). This catches the LLM drifting off
   the retrieved evidence.

    python -m eval.evaluate
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import INTERVENTIONS_FILE  # noqa: E402
from app.reasoning import need_signals, rank, targets_of  # noqa: E402
from app.retriever import load_interventions  # noqa: E402
from app.schemas import EcosystemProfile  # noqa: E402

GOLDEN = Path(__file__).parent / "golden_cases.json"


def run_selection_eval() -> dict:
    cases = json.loads(GOLDEN.read_text())["cases"]
    interventions = load_interventions(INTERVENTIONS_FILE)

    rows = []
    for case in cases:
        profile = EcosystemProfile(**case["profile"])
        picked = rank(interventions, profile, limit=3)
        ids = [item["id"] for item, _, _ in picked]

        expected = set(case["acceptable"])
        must = case.get("must_include")
        signals = need_signals(profile)
        severe = [k for k, v in signals.items() if v >= 0.7]
        covered = [
            s for s in severe
            if any(s in targets_of(item) for item, _, _ in picked)
        ]

        rows.append(
            {
                "case": case["name"],
                "selected": ids,
                "precision@3": round(len(expected & set(ids)) / 3, 2),
                "must_include_hit": (must in ids) if must else None,
                "severe_constraints": len(severe),
                "constraints_covered": len(covered),
                "distinct_categories": len({item.get("category") for item, _, _ in picked}),
            }
        )

    summary = {
        "cases": len(rows),
        "mean_precision@3": round(sum(r["precision@3"] for r in rows) / len(rows), 3),
        "must_include_hit_rate": round(
            sum(1 for r in rows if r["must_include_hit"]) / sum(1 for r in rows if r["must_include_hit"] is not None),
            3,
        ),
        "constraint_coverage": round(
            sum(r["constraints_covered"] for r in rows) / max(sum(r["severe_constraints"] for r in rows), 1),
            3,
        ),
    }
    return {"rows": rows, "summary": summary}


def run_ragas_eval() -> dict | None:
    """Faithfulness of the narrative to the retrieved evidence."""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import context_precision, faithfulness
    except ImportError:
        return None

    from app.conversation import handle_turn

    cases = json.loads(GOLDEN.read_text())["cases"]
    questions, answers, contexts = [], [], []
    for i, case in enumerate(cases):
        turn = handle_turn(f"eval-{i}", case["message"], case["profile"])
        if turn.mode != "recommend":
            continue
        questions.append(case["message"])
        answers.append(turn.message)
        contexts.append([e["excerpt"] for e in turn.evidence_used] or ["no evidence retrieved"])

    if not questions:
        return None

    ds = Dataset.from_dict({"question": questions, "answer": answers, "contexts": contexts})
    result = evaluate(ds, metrics=[faithfulness, context_precision])
    return dict(result)


def main() -> int:
    selection = run_selection_eval()
    print("== Selection eval ==")
    for row in selection["rows"]:
        print(json.dumps(row))
    print("\nsummary:", json.dumps(selection["summary"], indent=2))

    ragas_result = run_ragas_eval()
    if ragas_result:
        print("\n== RAGAS ==")
        print(json.dumps(ragas_result, indent=2))
    else:
        print("\nRAGAS skipped (install ragas + datasets and set GROQ_API_KEY to run it).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
