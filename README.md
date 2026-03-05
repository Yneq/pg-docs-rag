# pg-docs-rag

A local Retrieval-Augmented Generation (RAG) system built using PostgreSQL official documentation as the knowledge base.

This project demonstrates an end-to-end RAG pipeline running fully locally (no external APIs), focusing on system design clarity rather than UI complexity.

---

## 🎯 Project Goal

Build a minimal but production-style RAG system that:

- Uses real technical documentation as knowledge base
- Runs entirely locally (zero API cost)
- Demonstrates vector search + LLM grounding
- Is easy to explain in an interview

---

## 🧠 System Architecture

Document → Clean → Chunk → Embed → Store → Retrieve → Ground → Generate

### 1. Data Ingestion

- Source: PostgreSQL official documentation
- HTML cleaned using BeautifulSoup
- Text chunked into fixed-size segments
- Embeddings generated using `nomic-embed-text` via Ollama
- Stored in ChromaDB (persistent vector store)

### 2. Retrieval

- User query embedded using the same embedding model
- Top-K similar chunks retrieved via vector similarity search
- Similarity scores inspected for debugging

### 3. Grounded Generation

- Retrieved chunks injected into prompt
- LLM (`llama3.2` via Ollama) generates answer
- Prompt constrains model to use only retrieved context

---

## 🏗 Tech Stack

| Component | Technology |
|------------|------------|
| Language | Python |
| Embedding | Ollama (`nomic-embed-text`) |
| LLM | Ollama (`llama3.2`) |
| Vector DB | ChromaDB (persistent mode) |
| Data Source | PostgreSQL Documentation |

---

## 📂 Project Structure
</> Plain text
pg-docs-rag/
├── data/
│   └── raw/                 # Raw PostgreSQL documentation
│
├── scripts/
│   ├── ingest_docs.py       # Build vector database from docs
│   └── demo_rag.py          # Ask questions using RAG
│
├── chroma/                  # Chroma vector store
│
├── requirements.txt
└── README.md


---

## 🚀 How It Works

### Ingestion

```bash
python scripts/ingest_docs.py

Steps performed:

Parse HTML

Extract readable text

Chunk text

Generate embeddings

Persist into ChromaDB

Run Demo
python scripts/demo_rag.py

Example query:

What does SELECT do in PostgreSQL?

System flow:

Embed query

Retrieve top matching chunks

Inject context into prompt

Generate grounded answer

🧩 Design Decisions
Why Local Models?

Avoid API cost

Remove external dependency

Easier to test and debug

Demonstrates understanding of model deployment

Why ChromaDB?

Lightweight

Simple persistent storage

Good for local RAG prototype

Why PostgreSQL Docs?

Technical, structured content

Suitable for semantic search

Easier to discuss in backend/system interviews

🛠 Engineering Considerations

Persistent vector storage

HTML cleaning before embedding

Chunk overlap strategy

Distance inspection for debugging

Context size control to reduce latency

Simple guardrail to reduce hallucination

📊 Current Status

~1500+ chunks indexed

Retrieval working with similarity scores

Fully local RAG pipeline

CLI demo functional

🔮 Potential Improvements

Similarity threshold guardrail

Metadata filtering (section-based search)

Hybrid search (keyword + vector)

FastAPI endpoint

Multi-document ingestion

Evaluation dataset

Token usage tracking

💡 What This Project Demonstrates

Understanding of RAG architecture

Embedding workflow

Vector search fundamentals

Prompt grounding strategy

Trade-offs between latency and context size

Local LLM deployment workflow

📝 Notes

Fully local

CPU inference (slower on lightweight machines)

Designed for clarity and interview discussion rather than production scale
