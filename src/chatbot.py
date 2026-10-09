
"""
=============================================================================
 COMPLETE RAG CHATBOT
=============================================================================

Connects the three RAG steps:

    question
       |
    RETRIEVAL      retriever.py    Finds relevant passages
       |
    AUGMENTATION   augmentation.py Builds the prompt
       |
    GENERATION     generateur.py   Generates the answer
       |
    Answer + sources

USAGE
-----
    python src/chatbot.py
    python src/chatbot.py "What is finance?"
    python src/chatbot.py --demo
    python src/chatbot.py --prompt "What is finance?"
=============================================================================
"""

import argparse
import sys

import config
from augmentation import afficher_prompt, formater_source
from generateur import Generateur
from retriever import Retriever


# Test questions adapted to the financial management book.
QUESTIONS_DEMO = [
    "What is finance?",
    "What do financial managers do?",
    "What is the objective of financial management?",
    "What are the responsibilities of a CFO?",
    "What is the recipe for couscous?",  # Intentionally off-topic
]


class Chatbot:
    """Connects the retriever and the generator."""

    def __init__(self, verbeux=True):
        self.retriever = Retriever(verbeux=verbeux)
        self.generateur = Generateur()

    def _est_pertinent(self, passages):
        """
        Checks whether at least one retrieved passage reaches
        the FAISS similarity threshold.
        """
        if not passages:
            return False

        meilleur_score = max(
            passage["score_faiss"] for passage in passages
        )

        return meilleur_score >= config.SEUIL_FAISS

    def repondre(self, question):
        """Processes a question and returns its answer and sources."""

        passages = self.retriever.chercher(question)

        # Do not call the LLM if the retrieved passages are not relevant.
        if not self._est_pertinent(passages):
            return {
                "reponse": "The document does not cover this point.",
                "passages": passages,
                "appel_llm": False,
            }

        reponse = self.generateur.generer(question, passages)

        return {
            "reponse": reponse,
            "passages": passages,
            "appel_llm": True,
        }


# =============================================================================
# DISPLAY
# =============================================================================

def afficher(resultat, question, generateur=None):
    """Displays the answer and the sources used."""

    print("=" * 76)
    print(f"QUESTION: {question}")
    print("=" * 76)
    print()
    print(resultat["reponse"])

    # Display the retrieved sources, including when the LLM is not called.
    if resultat["passages"]:
        print()
        print("-" * 76)
        print("SOURCES")
        print("-" * 76)

        for numero, passage in enumerate(
            resultat["passages"], start=1
        ):
            scores = f"cosine {passage['score_faiss']:.3f}"

            if "score_rerank" in passage:
                scores += (
                    f"   rerank {passage['score_rerank']:+.2f}"
                )

            print(
                f"  [Source {numero}] "
                f"{formater_source(passage['metadata'])}"
            )
            print(
                f"              {scores}   {passage['id']}"
            )

    if not resultat["appel_llm"]:
        print(
            "\n  (No passage reached the relevance threshold; "
            "the model was not called.)"
        )
    elif generateur and generateur.derniere_consommation:
        consommation = generateur.derniere_consommation
        print(
            f"\n  Tokens sent: {consommation['prompt']}; "
            f"tokens received: {consommation['reponse']}; "
            f"total: {consommation['total']}."
        )

    print()


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    parseur = argparse.ArgumentParser(
        description="RAG chatbot for Fundamentals of Financial Management."
    )

    parseur.add_argument(
        "question",
        nargs="*",
        help="The question to ask",
    )
    parseur.add_argument(
        "--demo",
        action="store_true",
        help="Run the test questions",
    )
    parseur.add_argument(
        "--prompt",
        action="store_true",
        help="Display the prompt without calling the LLM",
    )

    args = parseur.parse_args()

    # Prompt mode: retrieve passages and display the prompt only.
    if args.prompt:
        question = " ".join(args.question) or QUESTIONS_DEMO[0]
        retriever = Retriever()
        afficher_prompt(question, retriever.chercher(question))
        return

    chatbot = Chatbot()

    # Demonstration mode.
    if args.demo:
        for question in QUESTIONS_DEMO:
            afficher(
                chatbot.repondre(question),
                question,
                chatbot.generateur,
            )
        return

    # One question passed on the command line.
    if args.question:
        question = " ".join(args.question)
        afficher(
            chatbot.repondre(question),
            question,
            chatbot.generateur,
        )
        return

    # Interactive mode.
    print("Ask questions about Fundamentals of Financial Management.")
    print("Enter a blank line or press Ctrl+C to quit.\n")

    while True:
        try:
            question = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if not question:
            print("Goodbye.")
            break

        print()
        afficher(
            chatbot.repondre(question),
            question,
            chatbot.generateur,
        )


if __name__ == "__main__":
    sys.exit(main())