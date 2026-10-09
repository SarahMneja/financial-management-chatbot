# Financial Management Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions using *Fundamentals of Financial Management, 12th Edition*, by Eugene F. Brigham and Joel F. Houston.

The project transforms a financial-management textbook into searchable text, retrieves relevant passages using semantic similarity, and uses a large language model (LLM) to generate answers grounded in those passages. It also includes retrieval evaluation, answer-quality evaluation, and a web interface.

## Table of Contents

1. [Project Objectives](#1-project-objectives)
2. [How It Works](#2-how-it-works)
3. [Project Structure](#3-project-structure)
4. [Requirements](#4-requirements)
5. [Installation and Setup](#5-installation-and-setup)
6. [Session 1 — PDF Extraction and Chunking](#6-session-1--pdf-extraction-and-chunking)
7. [Session 2 — Vectorisation and FAISS](#7-session-2--vectorisation-and-faiss)
8. [Session 3 — Retrieval and Answer Generation](#8-session-3--retrieval-and-answer-generation)
9. [Session 4 — Evaluation and Web Interface](#9-session-4--evaluation-and-web-interface)
10. [Running the Chatbot](#10-running-the-chatbot)
11. [Example Questions](#11-example-questions)
12. [Evaluation Metrics](#12-evaluation-metrics)
13. [Configuration](#13-configuration)
14. [Troubleshooting](#14-troubleshooting)
15. [GitHub and Security](#15-github-and-security)
16. [Future Improvements](#16-future-improvements)

## 1. Project Objectives

The main objectives are to:

* Extract text from a financial-management textbook in PDF format.
* Divide the extracted text into manageable chunks while preserving useful metadata.
* Convert text chunks into numerical embeddings.
* Store and search embeddings using FAISS.
* Retrieve relevant passages for a user's question.
* Optionally rerank retrieved passages to improve their relevance.
* Generate answers using an LLM through the Groq API.
* Ground answers in the textbook and display source citations.
* Evaluate retrieval performance and answer quality.
* Provide an interactive web interface using Streamlit.

## 2. How It Works

The chatbot follows a Retrieval-Augmented Generation architecture.

```text
                 FINANCIAL MANAGEMENT PDF
                           |
                           v
                 SESSION 1: EXTRACTION
                Extract text and metadata
                           |
                           v
                      TEXT CHUNKS
                           |
                           v
                 SESSION 2: VECTORIZATION
                  Embedding model
                           |
                           v
                    FAISS INDEX
                           |
                           v
USER QUESTION ---> SESSION 3: RETRIEVAL
                           |
                           v
                Relevant textbook passages
                           |
                           v
                  OPTIONAL RERANKING
                           |
                           v
                    AUGMENTATION
                 Build the context prompt
                           |
                           v
                 LLM THROUGH GROQ API
                           |
                           v
               ANSWER WITH SOURCE CITATIONS
                           |
                           v
                SESSION 4: EVALUATION/UI
```

The system combines two types of processing:

* **Retrieval:** finds textbook passages that are semantically similar to the question.
* **Generation:** uses the retrieved passages as context to formulate an answer.

The chatbot is designed to avoid inventing information when the textbook does not support an answer. Its actual reliability must be checked through testing and evaluation.

## 3. Project Structure

The repository is organized as follows:

```text
financial-management-chatbot/
│
├── src/
│   ├── config.py
│   ├── session1_pdf_to_chunks.py
│   ├── vectoriser.py
│   ├── retriever.py
│   ├── augmentation.py
│   ├── generateur.py
│   └── chatbot.py
│
├── data/
│   ├── chunks.jsonl
│   ├── chunks.json
│   └── index.faiss
│
├── evaluation/
│   ├── questions.json
│   ├── evaluer_retriever.py
│   ├── evaluer_ragas.py
│   └── resultats/
│
├── .streamlit/
│   └── config.toml
│
├── app.py
├── requirements.txt
├── requirements-session4.txt
├── .env
├── .env.example
├── .gitignore
└── README.md
```

Some files are generated during execution. The exact contents of `data/` may vary depending on which stages have been run.

### Main files

| File                              | Purpose                                                                                      |
| --------------------------------- | -------------------------------------------------------------------------------------------- |
| `src/config.py`                   | Central configuration for file paths, embedding model, retrieval settings, LLM, and API key. |
| `src/session1_pdf_to_chunks.py`   | Extracts and processes text from the source PDF.                                             |
| `src/vectoriser.py`               | Generates embeddings and creates the FAISS index.                                            |
| `src/retriever.py`                | Searches for relevant passages and optionally reranks them.                                  |
| `src/augmentation.py`             | Builds the context and instructions supplied to the LLM.                                     |
| `src/generateur.py`               | Calls the language model through Groq and handles generated answers.                         |
| `src/chatbot.py`                  | Connects retrieval, relevance checks, and answer generation.                                 |
| `evaluation/questions.json`       | Contains evaluation questions, expected chunk IDs, and reference answers.                    |
| `evaluation/evaluer_retriever.py` | Measures retrieval performance with and without reranking.                                   |
| `evaluation/evaluer_ragas.py`     | Evaluates generated answers and retrieved contexts using RAGAS.                              |
| `app.py`                          | Provides the Streamlit web interface.                                                        |
| `.streamlit/config.toml`          | Contains Streamlit appearance and server settings.                                           |

## 4. Requirements

The project uses Python and the following main libraries:

* Python 3.13 in the current development environment.
* Sentence Transformers for text embeddings.
* FAISS for similarity search.
* NumPy for numerical operations.
* Groq for access to a language model.
* Python-dotenv for loading environment variables.
* Streamlit for the web interface.
* RAGAS for evaluating RAG quality.
* LangChain integrations for the RAGAS evaluation model.

The embedding model currently configured is:

`paraphrase-multilingual-MiniLM-L12-v2`

The default generation model is:

`openai/gpt-oss-120b`

The configured reranker is:

`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`

Actual model availability and compatibility depend on the installed package versions and the available API models.

## 5. Installation and Setup

### 5.1. Open the project

Open the `financial-management-chatbot` folder in VS Code.

Open a PowerShell terminal from **Terminal → New Terminal**.

Check that the terminal is in the project root, where `src/`, `data/`, and `requirements.txt` are located.

### 5.2. Activate the virtual environment

Run:

```powershell
.\venv\Scripts\Activate.ps1
```

The terminal prompt should begin with `(venv)`.

If the virtual environment does not exist, create it using the installed Python version:

```powershell
py -3.13 -m venv venv
```

Then activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

### 5.3. Install the main dependencies

Run:

```powershell
python -m pip install -r requirements.txt
```

Install the Session 4 dependencies:

```powershell
python -m pip install -r requirements-session4.txt
```

Verify the main packages:

```powershell
python -c "import sentence_transformers, faiss, groq; print('Main dependencies OK')"
```

Verify the Session 4 packages:

```powershell
python -c "import streamlit, ragas, langchain_core, langchain_groq; print('Session 4 dependencies OK')"
```

Expected output:

```text
Main dependencies OK
Session 4 dependencies OK
```

If an import fails, confirm that the virtual environment is activated and install the missing package in that same environment.

### 5.4. Configure the Groq API key

Create a `.env` file in the project root, next to `requirements.txt`.

Add your own Groq API key:

```dotenv
GROQ_API_KEY="your_actual_api_key"
```

Replace the placeholder with your real key. Do not include the key in this README or commit it to GitHub.

The `.env.example` file should contain only a placeholder:

```dotenv
GROQ_API_KEY="gsk_votre_cle_ici"
```

The configuration module loads the `.env` file from the project root.

### 5.5. Check the configuration

Open `src/config.py` and confirm that its paths and model settings match the project.

Important settings include:

* `CHUNKS`: path to the JSONL chunk file.
* `INDEX`: path to the FAISS index.
* `METADONNEES`: path to the chunk metadata JSON file.
* `MODELE_EMBEDDING`: embedding model.
* `MODELE_RERANKER`: reranking model.
* `MODELE_LLM`: generation model.
* `N_CANDIDATS`: number of retrieval candidates.
* `N_PASSAGES`: number of passages supplied to the generator.
* `SEUIL_FAISS`: minimum similarity threshold used by the chatbot.
* `UTILISER_RERANKING`: enables or disables reranking.

Do not change the embedding model without considering whether the existing FAISS index must be rebuilt.

## 6. Session 1 — PDF Extraction and Chunking

### Objective

Extract text from the financial-management textbook and divide it into chunks suitable for semantic search.

### Input

The source PDF:

`Fundamentals of Financial Management 12th edition - Brigham Houston.pdf`

Place the PDF where the Session 1 script expects it, or update the configured file path.

### Process

1. Open and read the PDF.
2. Extract text from its pages.
3. Process the extracted text.
4. Divide the text into chunks.
5. Preserve relevant information such as chunk IDs and page metadata.
6. Save the processed chunks for Session 2.

### Output

The primary chunk output is:

`data/chunks.jsonl`

The project also uses `data/chunks.json` for chunk metadata associated with retrieval and evaluation.

### Run Session 1

If the script supports command-line execution and its input path is configured correctly, run:

```powershell
python src\session1_pdf_to_chunks.py
```

The exact output depends on the script's current configuration. Session 1 is already completed in the current workflow, so **do not rerun it just to start Session 4**.

### Previously obtained results

The initial extraction reported:

* 755 PDF pages.
* Approximately 2.34 million extracted characters.
* 1,841 chunks used in the vectorisation stage.

These figures describe the previous successful run; they are not guaranteed for a new extraction with different settings.

## 7. Session 2 — Vectorisation and FAISS

### Objective

Convert text chunks into embeddings and make them searchable using FAISS.

### Process

1. Load the processed chunks.
2. Encode the text with the configured Sentence Transformers model.
3. Normalize embeddings for cosine-similarity search.
4. Build a FAISS index.
5. Save the index and associated chunk metadata.

### Main files

* Script: `src/vectoriser.py`
* Chunk data: `data/chunks.jsonl`
* Metadata: `data/chunks.json`
* FAISS index: `data/index.faiss`

### Run vectorisation

Only rerun this step if the chunk data or embedding configuration has changed, or if the index is missing.

```powershell
python src\vectoriser.py
```

### Previously obtained results

The previous run produced:

* 1,841 chunk embeddings.
* 384 dimensions per embedding.
* A FAISS index containing 1,841 vectors.

The embedding model has a configured maximum sequence length of 512 tokens. Some existing chunks were longer than that limit, so chunk-length improvements are a possible future enhancement.

**Important:** if you change the embedding model, rebuild the FAISS index using the same model that will be used for subsequent queries.

## 8. Session 3 — Retrieval and Answer Generation

### Objective

Retrieve relevant passages for a question, construct a context prompt, and generate a grounded answer.

### 8.1. Retrieval

`src/retriever.py` searches the FAISS index and returns relevant passages with their scores and metadata.

The pipeline can optionally rerank candidate passages using the configured cross-encoder.

To test retrieval directly:

```powershell
python src\retriever.py "What is finance?"
```

Expected behavior:

* A list of retrieved chunks.
* Chunk IDs and page information.
* FAISS similarity scores.
* Reranking information when enabled.

Exact scores and ranking can vary depending on the configuration and model versions.

### 8.2. Augmentation

`src/augmentation.py` constructs the context supplied to the language model.

The prompt instructs the chatbot to:

* Use the retrieved passages as its evidence.
* Cite sources using the `[Source N]` format.
* Avoid inventing information that is not supported by the context.
* Respond in English, concisely and clearly.

### 8.3. Answer generation

`src/generateur.py` sends the prompt to the configured Groq model and processes the returned answer.

List the available models supported by the current generator:

```powershell
python src\generateur.py --modeles
```

Test generation with a question:

```powershell
python src\generateur.py "What is financial management?"
```

This tests the generation component. It is not the same as testing the full retrieval-and-generation pipeline.

### 8.4. Complete chatbot

`src/chatbot.py` combines retrieval, relevance checks, and generation.

Run the demonstration questions:

```powershell
python src\chatbot.py --demo
```

Ask one question:

```powershell
python src\chatbot.py "What is finance?"
```

Start an interactive session:

```powershell
python src\chatbot.py
```

The chatbot returns an answer, retrieved passages, source information, and token-usage information when available.

### Important limitation

A similarity score alone does not guarantee that a passage answers a question. The chatbot's relevance threshold, chunk quality, and retrieval ranking must be tested. The evaluation scripts in Session 4 help measure these issues.

## 9. Session 4 — Evaluation and Web Interface

### Objective

Evaluate retrieval quality, assess answer quality, and provide an interactive interface.

### 9.1. Evaluation dataset

File:

`evaluation/questions.json`

The dataset contains:

* Questions about financial management.
* Expected chunk IDs for in-scope questions.
* Reference answers for answer-quality evaluation.
* Out-of-scope questions that the chatbot should refuse to answer.

Before evaluating, verify that the expected chunk IDs exist in `data/chunks.json` and genuinely support the corresponding answers. Reference answers should also be checked against the textbook.

### 9.2. Evaluate retrieval

Run:

```powershell
python evaluation\evaluer_retriever.py
```

The script evaluates whether expected chunks appear in the retrieved results and compares retrieval with reranking enabled and disabled.

For a meaningful comparison, set `UTILISER_RERANKING = True` in `src/config.py` before starting the evaluation. The script then disables reranking for its second pass.

The evaluation report is written to:

`evaluation/resultats/retriever.md`

The report includes retrieval metrics and details about questions whose expected chunks were not found.

### 9.3. Evaluate answer quality with RAGAS

Start with two questions:

```powershell
python evaluation\evaluer_ragas.py --questions 2
```

The evaluation script uses the chatbot to obtain answers and retrieved contexts, then asks a separate judge model to score the results.

The process makes API calls and can take several minutes. It may fail if a configured judge model is unavailable, the API key is missing, or the installed library versions are incompatible.

The report is written to:

`evaluation/resultats/ragas.md`

After the first run succeeds, increase the number of questions to evaluate a larger sample.

### 9.4. Launch the web interface

Run:

```powershell
streamlit run app.py
```

Open the local URL printed in the terminal. With the project’s configured port, the address is usually:

`http://localhost:8502`

The interface includes:

* Suggested questions.
* A chat input.
* Generated answers.
* Retrieved source passages.
* Similarity scores where available.
* A button to start a new conversation.

Keep the terminal open while using the application. Stop the local server with `Ctrl+C` when finished.

### 9.5. Manual testing checklist

Test the application with the following questions:

1. `What is finance?`
2. `What is financial management?`
3. `What do financial managers do?`
4. `What are the responsibilities of a CFO?`
5. `What is the recipe for couscous?`

Check that:

* Relevant questions receive useful answers.
* Source citations correspond to the retrieved passages.
* Retrieved passages genuinely support the answer.
* The out-of-scope question is refused rather than answered with invented information.
* The new-conversation control clears the visible chat.
* API or model errors are reported clearly.

A successful launch confirms that the application runs; it does not by itself prove that the chatbot's answers are accurate.

## 10. Running the Chatbot

For normal use, activate the virtual environment and launch the web application:

```powershell
.\venv\Scripts\Activate.ps1
streamlit run app.py
```

Open the local URL printed by Streamlit and enter a question.

For command-line testing, use:

```powershell
python src\chatbot.py "What is finance?"
```

You do not need to rerun PDF extraction or vectorisation every time you ask a question. Those stages produce the files used by the retrieval pipeline.

## 11. Example Questions

The chatbot is intended for questions about concepts covered in the textbook, including:

* What is finance?
* What is financial management?
* What do financial managers do?
* What are the responsibilities of a CFO?
* What is the objective of financial management?

The quality of an answer depends on whether the relevant material was extracted, chunked, indexed, and retrieved correctly.

For questions outside the textbook's scope, the chatbot should indicate that the document does not provide the answer.

## 12. Evaluation Metrics

### Retrieval metrics

| Metric               | Meaning                                                                                          |
| -------------------- | ------------------------------------------------------------------------------------------------ |
| Hit@1                | Proportion of in-scope questions for which an expected chunk appears first.                      |
| Hit@3                | Proportion of in-scope questions for which an expected chunk appears in the first three results. |
| Hit@5                | Proportion of in-scope questions for which an expected chunk appears in the first five results.  |
| MRR                  | Mean Reciprocal Rank; rewards expected chunks appearing closer to the top.                       |
| Correct refusal rate | Proportion of out-of-scope test questions for which the configured threshold triggers refusal.   |

These metrics depend on the correctness of the expected chunk IDs and the coverage of the test set. They should not be interpreted as a complete measure of real-world chatbot quality.

### RAGAS metrics

| Metric            | Purpose                                                                         |
| ----------------- | ------------------------------------------------------------------------------- |
| Faithfulness      | Measures whether the generated answer is supported by the retrieved context.    |
| Answer relevancy  | Measures whether the answer addresses the question.                             |
| Context precision | Assesses how relevant the retrieved context is.                                 |
| Context recall    | Assesses whether the context covers information needed by the reference answer. |

RAGAS scores are evaluation signals, not guarantees of factual correctness. Results depend on the test dataset, references, judge model, and evaluation configuration.

## 13. Configuration

The main configuration file is:

`src/config.py`

Use it to manage:

* File paths.
* Embedding model.
* Reranking model.
* Generation model.
* Number of retrieval candidates.
* Number of passages returned.
* Similarity threshold.
* Temperature and response token limit.
* Groq API key loaded from `.env`.

Change one setting at a time and test the results before making other changes.

If the embedding model changes, the existing embeddings and FAISS index may no longer be compatible. If the retrieval threshold changes, test both relevant and out-of-scope questions to understand the effect.

## 14. Troubleshooting

### `ModuleNotFoundError: No module named 'ragas'`

Activate the project environment and install Session 4 dependencies:

```powershell
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements-session4.txt
```

Verify:

```powershell
python -c "import ragas; print(ragas.__version__)"
```

### `GROQ_API_KEY` is missing

Check that:

* `.env` exists in the project root.
* It contains `GROQ_API_KEY` with a valid key.
* `src/config.py` loads the root `.env` file.
* The key has not expired or been revoked.

Never paste your actual API key into GitHub or this README.

### The FAISS index or chunk file is missing

Confirm that `data/chunks.jsonl`, `data/chunks.json`, and `data/index.faiss` exist where the configuration expects them.

If an artifact is missing, rerun the relevant earlier stage in order: extraction and chunking, then vectorisation.

### The chatbot refuses a relevant question

Inspect the retrieved passages and their scores. Possible causes include:

* The answer is not present in the retrieved chunks.
* The expected chunk was not retrieved.
* The similarity threshold is too restrictive.
* The chunks are too long or contain unrelated text.
* Reranking changes the order of relevant passages.

Use the retrieval evaluation report before changing the threshold or rebuilding the index.

### A model is unavailable

List the available models using:

```powershell
python src\generateur.py --modeles
```

Select a supported model in `src/config.py` or the evaluation command where appropriate.

### Streamlit uses a different port

If the configured port is already occupied, use the local URL printed in the terminal. Do not assume the application always runs on port 8502.

### GitHub still shows ignored files

Adding a file to `.gitignore` prevents new untracked files from being added, but it does not automatically remove files already tracked by Git. Review tracked files in GitHub Desktop and untrack generated artifacts when appropriate.

## 15. GitHub and Security

The repository contains source code and project documentation. Generated indexes, intermediate data, and API credentials should be handled carefully.

The `.gitignore` file should include:

```gitignore
.env
venv/
.venv/
__pycache__/
*.pyc
data/index.faiss
data/chunks.json
evaluation/resultats/
data/historique.json
```

Review this list against the current project before committing. Keep files needed to reproduce the project, but avoid committing large generated artifacts unless there is a clear reason to version them.

### Before every commit

* Check the changed files in GitHub Desktop.
* Confirm `.env` is not included.
* Check that no API key or other secret appears in source code, logs, or documentation.
* Review changes to configuration and requirements.
* Commit with a clear message.
* Push to the intended GitHub repository.

If an API key has ever been committed publicly, removing the file alone is not enough; revoke or rotate the key.

## 16. Future Improvements

Possible next steps include:

* Improve chunk boundaries and reduce chunks that exceed the embedding model's token limit.
* Improve retrieval using the evaluation results rather than changing settings blindly.
* Compare retrieval with and without reranking.
* Expand the evaluation dataset with verified expected chunks and reference answers.
* Test more in-scope and out-of-scope questions.
* Improve source citation validation.
* Add conversation history if follow-up questions require it.
* Track response latency and API usage.
* Improve error handling and deployment documentation.

## Conclusion

This project implements the main stages of a PDF-based RAG chatbot:

1. **Session 1:** extract and chunk the textbook.
2. **Session 2:** embed the chunks and build the FAISS index.
3. **Session 3:** retrieve passages, construct the context, and generate answers.
4. **Session 4:** evaluate retrieval and answer quality, then interact with the chatbot through Streamlit.

The final goal is a usable chatbot whose answers are grounded in the financial-management textbook, with traceable sources and measurable retrieval and answer quality.
