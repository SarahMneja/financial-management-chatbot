"""Evaluate answer quality with RAGAS and a Groq judge model."""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import config

QUESTION_FILE = Path(__file__).resolve().parent / "questions.json"
REPORT_FILE = Path(__file__).resolve().parent / "resultats" / "ragas.md"

# Prefer a judge model different from the answer-generation model.
DEFAULT_JUDGE = "qwen/qwen3-32b"
DEFAULT_QUESTIONS = 3
PAUSE_SECONDS = 3
JUDGE_MAX_TOKENS = 4096

METRIC_DESCRIPTIONS = {
    "faithfulness": "Does the answer stay faithful to retrieved passages?",
    "answer_relevancy": "Does the answer address the question?",
    "context_precision": "Are relevant passages ranked near the top?",
    "context_recall": "Do the passages cover the reference answer?",
}


def make_samples(questions):
    """Run the chatbot and prepare samples for RAGAS."""
    from ragas import SingleTurnSample
    from chatbot import Chatbot

    chatbot = Chatbot()
    samples = []
    metadata = []

    for i, item in enumerate(questions, start=1):
        print(f"\n[{i}/{len(questions)}] {item['question']}")

        started = time.time()
        result = chatbot.repondre(item["question"])

        passages = result.get("passages", [])
        answer = result.get("reponse", "")

        samples.append(
            SingleTurnSample(
                user_input=item["question"],
                response=answer,
                retrieved_contexts=[
                    passage.get("texte", "")
                    for passage in passages
                ],
                reference=item["reference"],
            )
        )

        metadata.append({
            "passages": len(passages),
            "llm_called": result.get("appel_llm", False),
        })

        print(
            f"  passages={len(passages)}; "
            f"LLM called={result.get('appel_llm', False)}; "
            f"{time.time() - started:.1f}s"
        )

        if i < len(questions):
            time.sleep(PAUSE_SECONDS)

    return samples, metadata, chatbot.retriever.encodeur


class LocalEmbeddings:
    """Reuse the embedding model already loaded by the Retriever."""

    def __init__(self, encoder):
        self.encoder = encoder

    def embed_documents(self, texts):
        return self.encoder.encode(
            texts,
            normalize_embeddings=True,
        ).tolist()

    def embed_query(self, text):
        return self.encoder.encode(
            [text],
            normalize_embeddings=True,
        )[0].tolist()


def build_embeddings(encoder):
    """Wrap the local embedding model in a RAGAS-compatible adapter."""
    from langchain_core.embeddings import Embeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    class CompatibleEmbeddings(LocalEmbeddings, Embeddings):
        pass

    return LangchainEmbeddingsWrapper(
        CompatibleEmbeddings(encoder)
    )


def run_ragas(samples, encoder, judge_model):
    """Run the four RAGAS metrics."""
    from langchain_groq import ChatGroq
    from ragas import EvaluationDataset, RunConfig, evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    if not config.CLE_GROQ:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Check the .env file at the project root."
        )

    print(f"\nRAGAS judge: {judge_model}")
    print(f"Answer generator: {config.MODELE_LLM}")

    judge = LangchainLLMWrapper(
        ChatGroq(
            model=judge_model,
            temperature=0,
            max_tokens=JUDGE_MAX_TOKENS,
            groq_api_key=config.CLE_GROQ,
        )
    )

    run_config = RunConfig(
        max_workers=1,
        timeout=180,
        max_retries=5,
        max_wait=45,
    )

    return evaluate(
        EvaluationDataset(samples=samples),
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=judge,
        embeddings=build_embeddings(encoder),
        run_config=run_config,
    )


def write_report(result, samples, metadata, judge_model):
    """Save average scores and individual answers to Markdown."""
    table = result.to_pandas()
    metrics = list(METRIC_DESCRIPTIONS)

    lines = [
        f"# RAGAS evaluation — {datetime.now():%Y-%m-%d %H:%M}",
        "",
        f"Generator: `{config.MODELE_LLM}`",
        f"Judge: `{judge_model}`",
        f"Embeddings: `{config.MODELE_EMBEDDING}`",
        f"Questions: {len(samples)}",
        "",
        "The judge should differ from the generator where possible.",
        "",
        "## Average scores",
        "",
        "| Metric | Score | Meaning |",
        "|---|---:|---|",
    ]

    for metric, meaning in METRIC_DESCRIPTIONS.items():
        mean = table[metric].mean()
        valid = int(table[metric].notna().sum())
        score = f"{mean:.3f}" if mean == mean else "—"

        suffix = (
            f" ({valid}/{len(table)} valid)"
            if valid < len(table)
            else ""
        )

        lines.append(
            f"| {metric} | **{score}**{suffix} | {meaning} |"
        )

    lines += [
        "",
        "## Per-question details",
        "",
        "| # | Question | Passages | "
        + " | ".join(metrics)
        + " |",
        "|---:|---|---:|" + "---:|" * len(metrics),
    ]

    for i, (sample, info) in enumerate(
        zip(samples, metadata)
    ):
        row = table.iloc[i]

        scores = " | ".join(
            f"{row[metric]:.2f}"
            if row[metric] == row[metric]
            else "—"
            for metric in metrics
        )

        question = sample.user_input.replace("|", "\\|")

        lines.append(
            f"| {i + 1} | {question} | {info['passages']} | "
            f"{scores} |"
        )

    lines += ["", "## Generated answers", ""]

    for i, sample in enumerate(samples, start=1):
        lines += [
            f"### {i}. {sample.user_input}",
            "",
            sample.response or "(No answer returned.)",
            "",
            f"**Reference answer:** {sample.reference}",
            "",
            "---",
            "",
        ]

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the financial-management RAG chatbot."
    )

    parser.add_argument(
        "--questions",
        type=int,
        default=DEFAULT_QUESTIONS,
        help=f"Number of in-scope questions (default {DEFAULT_QUESTIONS})",
    )

    parser.add_argument(
        "--juge",
        "--judge",
        dest="judge",
        default=DEFAULT_JUDGE,
        help=f"Groq judge model (default {DEFAULT_JUDGE})",
    )

    args = parser.parse_args()

    all_questions = json.loads(
        QUESTION_FILE.read_text(encoding="utf-8")
    )["questions"]

    in_scope = [
        item for item in all_questions
        if item.get("chunks_attendus") and item.get("reference")
    ]

    selected = in_scope[:max(1, args.questions)]

    if not selected:
        raise ValueError(
            "No in-scope questions with reference answers found."
        )

    print("=" * 72)
    print(" FINANCIAL-MANAGEMENT RAGAS EVALUATION")
    print("=" * 72)

    print(
        f"{len(selected)} questions selected "
        f"from {len(in_scope)} in-scope questions."
    )

    if args.judge == config.MODELE_LLM:
        print(
            "WARNING: judge and generator are the same model; "
            "scores may be biased."
        )

    samples, metadata, encoder = make_samples(selected)

    result = run_ragas(
        samples,
        encoder,
        args.judge,
    )

    write_report(
        result,
        samples,
        metadata,
        args.judge,
    )

    table = result.to_pandas()

    print("\nAverage scores:")

    for metric in METRIC_DESCRIPTIONS:
        mean = table[metric].mean()

        if mean == mean:
            print(f"  {metric:<20} {mean:.3f}")
        else:
            print(f"  {metric:<20} unavailable")

    print(
        f"\nReport written to: "
        f"{REPORT_FILE.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()