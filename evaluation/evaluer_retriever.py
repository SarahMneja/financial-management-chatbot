"""Evaluate financial-management retrieval without calling the LLM."""

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import config
from retriever import Retriever

QUESTION_FILE = Path(__file__).resolve().parent / "questions.json"
REPORT_FILE = Path(__file__).resolve().parent / "resultats" / "retriever.md"

RANKS = (1, 3, 5)


def first_expected_rank(passages, expected_ids):
    """Return the rank of the first expected chunk, or None."""
    for rank, passage in enumerate(passages, start=1):
        if passage.get("id") in expected_ids:
            return rank
    return None


def evaluate_question(retriever, item, depth):
    """Retrieve passages and measure whether the expected chunk appears."""
    passages = retriever.chercher(
        item["question"],
        n_passages=depth,
        n_candidats=depth,
    )

    best_score = max(
        (float(p.get("score_faiss", 0.0)) for p in passages),
        default=0.0,
    )

    expected_ids = set(item.get("chunks_attendus", []))

    return {
        "question": item["question"],
        "expected": sorted(expected_ids),
        "rank": (
            first_expected_rank(passages, expected_ids)
            if expected_ids
            else None
        ),
        "best_score": best_score,
        "refusal": best_score < config.SEUIL_FAISS,
        "top_ids": [p.get("id", "?") for p in passages[:5]],
    }


def summarize(items):
    """Calculate hit@k, MRR, and out-of-scope refusal rate."""
    in_scope = [item for item in items if item["expected"]]
    out_scope = [item for item in items if not item["expected"]]

    result = {}

    if in_scope:
        for k in RANKS:
            hits = sum(
                item["rank"] is not None and item["rank"] <= k
                for item in in_scope
            )
            result[f"hit@{k}"] = hits / len(in_scope)

        reciprocal_ranks = [
            1 / item["rank"] if item["rank"] is not None else 0.0
            for item in in_scope
        ]
        result["MRR"] = sum(reciprocal_ranks) / len(in_scope)

    if out_scope:
        result["correct refusals"] = (
            sum(item["refusal"] for item in out_scope) / len(out_scope)
        )

    return result


def format_summary(summary):
    if not summary:
        return "No scored questions"

    return "  ".join(
        f"{name}={score:.2f}"
        for name, score in summary.items()
    )


def write_report(with_reranker, without_reranker, depth):
    """Write the results to a Markdown report."""
    summary_on, details_on = with_reranker
    summary_off, details_off = without_reranker

    lines = [
        f"# Retriever evaluation — {datetime.now():%Y-%m-%d %H:%M}",
        "",
        f"Embedding model: `{config.MODELE_EMBEDDING}`",
        f"Reranker: `{config.MODELE_RERANKER}`",
        f"Search depth: {depth}",
        "",
        "This report evaluates retrieval only; no answer-generation LLM is used.",
        "",
        "## With and without reranking",
        "",
        "| Metric | Reranking on | Reranking off | Difference |",
        "|---|---:|---:|---:|",
    ]

    for key, on_value in summary_on.items():
        off_value = summary_off.get(key, 0.0)
        lines.append(
            f"| {key} | **{on_value:.2f}** | {off_value:.2f} | "
            f"{on_value - off_value:+.2f} |"
        )

    lines += [
        "",
        "## In-scope question details",
        "",
        "| Question | Expected chunks | Rank on | Rank off | Best cosine score |",
        "|---|---|---:|---:|---:|",
    ]

    for on, off in zip(details_on, details_off):
        rank_on = str(on["rank"]) if on["rank"] is not None else "missing"
        rank_off = str(off["rank"]) if off["rank"] is not None else "missing"

        lines.append(
            f"| {on['question']} | {', '.join(on['expected'])} | "
            f"{rank_on} | {rank_off} | {on['best_score']:.3f} |"
        )

    out_scope = [item for item in details_on if not item["expected"]]

    if out_scope:
        lines += [
            "",
            "## Out-of-scope questions",
            "",
            f"Refusal threshold: `config.SEUIL_FAISS` "
            f"({config.SEUIL_FAISS})",
            "",
            "| Question | Best cosine score | Refusal expected? |",
            "|---|---:|---|",
        ]

        for item in out_scope:
            refusal = "yes" if item["refusal"] else "**NO**"
            lines.append(
                f"| {item['question']} | {item['best_score']:.3f} | "
                f"{refusal} |"
            )

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    if not QUESTION_FILE.exists():
        raise FileNotFoundError(
            f"Missing test set: {QUESTION_FILE}"
        )

    questions = json.loads(
        QUESTION_FILE.read_text(encoding="utf-8")
    )["questions"]

    depth = int(getattr(config, "N_CANDIDATS", 20))

    print("=" * 72)
    print(" FINANCIAL-MANAGEMENT RETRIEVER EVALUATION")
    print("=" * 72)
    print(f"{len(questions)} questions; search depth={depth}; no LLM calls.\n")

    retriever = Retriever()

    print("Pass 1: reranking in its current configured state")
    details_on = [
        evaluate_question(retriever, item, depth)
        for item in questions
    ]
    summary_on = summarize(details_on)

    print("Pass 2: reranking disabled")
    retriever.reranker = None
    details_off = [
        evaluate_question(retriever, item, depth)
        for item in questions
    ]
    summary_off = summarize(details_off)

    print("\nWith reranking:    " + format_summary(summary_on))
    print("Without reranking: " + format_summary(summary_off))

    missing = [
        item for item in details_on
        if item["expected"] and item["rank"] is None
    ]

    if missing:
        print(
            f"\nExpected chunk not retrieved for "
            f"{len(missing)} question(s):"
        )
        for item in missing:
            print(
                f" - {item['question']} "
                f"(expected: {', '.join(item['expected'])})"
            )
    else:
        print("\nAll expected chunks appeared in the returned passages.")

    write_report(
        (summary_on, details_on),
        (summary_off, details_off),
        depth,
    )

    print(f"\nReport written to: {REPORT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()