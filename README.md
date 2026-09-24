# Financial Management Chatbot

## 📚 Project Overview

This project aims to develop a chatbot capable of answering questions based on the book:

**Fundamentals of Financial Management — 12th Edition**
Eugene F. Brigham and Joel F. Houston

The project follows a Retrieval-Augmented Generation (RAG) approach. The document is first transformed into structured chunks, then converted into numerical vectors and indexed for semantic search. The retrieved information will later be used to generate relevant answers to user questions.

---

## 🎯 Project Objectives

The main objectives of the project are to:

* Extract and process information from the Financial Management PDF.
* Clean and structure the extracted text.
* Reconstruct the document hierarchy using titles and sections.
* Extract tables from the PDF.
* Divide the document into meaningful semantic chunks.
* Convert the chunks into vector embeddings.
* Build a FAISS index for semantic search.
* Retrieve relevant passages for a user's question.
* Use the retrieved passages in the future chatbot generation stage.

---

## 🏗️ Project Pipeline

The project follows this general pipeline:

```text
Financial Management PDF
          │
          ▼
   PDF Text Extraction
          │
          ▼
    Text Cleaning
          │
          ▼
 Document Hierarchy
   Reconstruction
          │
          ▼
    Table Extraction
          │
          ▼
   Semantic Chunking
          │
          ▼
     chunks.jsonl
          │
          ▼
     Vectorisation
          │
          ▼
   SentenceTransformer
          │
          ▼
  Vector Embeddings
          │
          ▼
      FAISS Index
          │
          ▼
   Semantic Search
          │
          ▼
 Relevant Retrieved Chunks
          │
          ▼
    Chatbot Generation
```

---

# 📌 Sessions

## Session 1 — PDF to Chunks

### Objective

The objective of Session 1 was to transform the Financial Management PDF into structured semantic chunks that can be used in the following stages of the RAG pipeline.

### Main steps

1. Upload and inspect the PDF.
2. Extract text using **PyMuPDF**.
3. Analyze font sizes to identify titles and headings.
4. Clean the extracted text.
5. Reconstruct the document hierarchy.
6. Extract tables using **pdfplumber**.
7. Create semantic chunks.
8. Add metadata to each chunk.
9. Save the resulting chunks in JSONL format.

### Main parameters

```text
Chunk target:       512 tokens
Overlap:             50 tokens
```

### Results

```text
PDF pages:           755
Tables extracted:    172
Final chunks:        1,841
```

### Main output

```text
chunks.jsonl
```

This file contains the semantic chunks that are used as the input for Session 2.

---

## Session 2 — Vectorisation

### Objective

The objective of Session 2 was to transform the chunks produced in Session 1 into numerical vector representations and create a FAISS index for semantic search.

### Main steps

1. Load `chunks.jsonl`.
2. Load the SentenceTransformer model.
3. Check chunk lengths.
4. Generate vector embeddings.
5. Normalize the embeddings.
6. Create a FAISS index.
7. Add the vectors to the index.
8. Save the FAISS index.
9. Save the chunks and their metadata.
10. Test semantic search with Financial Management questions.

### Model

```text
paraphrase-multilingual-MiniLM-L12-v2
```

### Vector dimensions

```text
384
```

### Maximum sequence length

```text
512
```

### Batch size

```text
32
```

### FAISS index

```text
IndexFlatIP
```

Normalized embeddings are used so that the inner product corresponds to cosine similarity.

### Results

```text
Chunks:        1,841
Vectors:       1,841
Dimensions:      384
FAISS index:     Created successfully
```

Semantic search was tested using questions such as:

```text
What is finance?
What is the time value of money?
What is the relationship between risk and return?
What are financial statements?
```

The search successfully retrieved relevant sections from the Financial Management book.

---

# 📂 Project Structure

```text
Financial-Management-Chatbot/
│
├── README.md
│
├── Session1_PDF_to_Chunks/
│   ├── README.md
│   ├── session1_pdf_to_chunks.py
│   └── chunks.jsonl
│
└── Session2_Vectorisation/
    ├── README.md
    ├── session2_vectorisation.py
    ├── chunks.json
    └── index.faiss
```

---

# 🛠️ Technologies Used

### Python

Main programming language used throughout the project.

### Google Colab

Used as the development and execution environment.

### PyMuPDF

Used for PDF text extraction and analysis of PDF formatting.

### pdfplumber

Used for extracting tables from the PDF.

### Sentence Transformers

Used to transform text chunks into vector embeddings.

### FAISS

Used to store and search the vector embeddings.

### GitHub

Used to organize and store the project code and generated files.

---

# 🔄 Current Project Status

| Session   | Description              | Status      |
| --------- | ------------------------ | ----------- |
| Session 1 | PDF → Semantic Chunks    | ✅ Completed |
| Session 2 | Chunks → Vectors → FAISS | ✅ Completed |
| Session 3 | Retrieval + Generation   | ⏳ Next      |

---

# 📖 Current Knowledge Base

The chatbot's knowledge base is based on:

**Fundamentals of Financial Management, 12th Edition**
**Eugene F. Brigham and Joel F. Houston**

The processed document contains **755 pages** and has been transformed into **1,841 semantic chunks**.

---

# 🚀 Future Development

The next stage of the project will use the semantic search results to retrieve relevant passages and provide them to a generation model.

The planned workflow is:

```text
User Question
      │
      ▼
Semantic Search
      │
      ▼
Relevant Chunks
      │
      ▼
Generation Model
      │
      ▼
Chatbot Answer
```

The final chatbot should use the retrieved information from the Financial Management book to formulate answers to users' questions.

---

## 👩‍💻 Project Status

This repository contains the progressive development of the Financial Management chatbot, with each session documenting a stage of the RAG pipeline.
