"""Streamlit interface for the Fundamentals of Financial Management chatbot."""

import re
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import config
from augmentation import formater_source
from chatbot import Chatbot


st.set_page_config(
    page_title="Financial Management Chatbot",
    page_icon="📘",
    layout="centered",
)

SUGGESTIONS = [
    "What is finance?",
    "What is financial management?",
    "What do financial managers do?",
    "What are the responsibilities of a CFO?",
]

CITATION_PATTERN = re.compile(
    r"\[\s*Source\s*(\d+)\s*\]",
    re.IGNORECASE,
)


@st.cache_resource(show_spinner=False)
def load_chatbot():
    """Load the chatbot once and reuse its models across reruns."""
    return Chatbot()


def reset_chat():
    """Clear the current conversation."""
    st.session_state.messages = []


def render_answer(answer, passages):
    """Display an answer and its retrieved source passages."""
    st.markdown(answer)

    cited_numbers = {
        int(match)
        for match in CITATION_PATTERN.findall(answer or "")
    }

    if not passages:
        return

    with st.expander(
        f"Retrieved passages ({len(passages)})",
        expanded=False,
    ):
        for number, passage in enumerate(passages, start=1):
            metadata = passage.get("metadata", {})
            source_label = formater_source(metadata)
            score = passage.get("score_faiss")

            heading = f"[Source {number}] {source_label}"

            if score is not None:
                heading += f" · cosine {float(score):.3f}"

            if number in cited_numbers:
                heading += " · cited in answer"

            st.markdown(f"**{heading}**")
            st.write(passage.get("texte", ""))

            if number < len(passages):
                st.divider()


st.title("📘 Financial Management Chatbot")

st.caption(
    "Ask questions about *Fundamentals of Financial Management* "
    "(Brigham & Houston). Answers should be grounded in the "
    "textbook and include source citations."
)


with st.sidebar:
    st.header("About this chatbot")

    st.write(
        "This project retrieves passages from the textbook, "
        "builds a context, and asks a language model to answer "
        "from those passages."
    )

    st.caption(
        "If the textbook does not support an answer, "
        "the chatbot should say so."
    )

    if st.button(
        "New conversation",
        use_container_width=True,
    ):
        reset_chat()
        st.rerun()


if "messages" not in st.session_state:
    st.session_state.messages = []


# Display previous messages.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            render_answer(
                message["content"],
                message.get("passages", []),
            )
        else:
            st.markdown(message["content"])


# Show suggested questions at the beginning of the conversation.
if not st.session_state.messages:
    st.markdown("### Try a question")

    cols = st.columns(2)

    for i, suggestion in enumerate(SUGGESTIONS):
        if cols[i % 2].button(
            suggestion,
            key=f"suggestion_{i}",
            use_container_width=True,
        ):
            st.session_state.pending_question = suggestion
            st.rerun()


typed_question = st.chat_input(
    "Ask about finance or financial management…"
)

pending_question = st.session_state.pop(
    "pending_question",
    None,
)

question = typed_question or pending_question


if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner(
                "Searching the textbook and preparing an answer…"
            ):
                bot = load_chatbot()
                result = bot.repondre(question)

            answer = result.get(
                "reponse",
                "I could not generate an answer.",
            )

            passages = result.get("passages", [])

            render_answer(answer, passages)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "passages": passages,
            })

            generator = getattr(bot, "generateur", None)
            usage = getattr(
                generator,
                "derniere_consommation",
                None,
            )

            if usage:
                st.caption(
                    f"Tokens: {usage.get('total', '?')} total "
                    f"({usage.get('prompt', '?')} prompt, "
                    f"{usage.get('reponse', '?')} response)"
                )

        except Exception as exc:
            st.error(
                "The chatbot encountered an error. "
                "Check the VS Code terminal for details."
            )
            st.exception(exc)

    st.rerun()