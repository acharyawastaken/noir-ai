# Hybrid Search RAG Demo

A production-ready RAG proof-of-concept that solves the "exact keyword" problem by utilizing LangChain's `EnsembleRetriever` to combine vector semantic search with BM25 keyword matching.

## Why this exists
Pure vector search often fails on names, dates, and product codes. By combining BM25 (keyword) and Vectors (semantic), this pipeline guarantees highly accurate retrieval for client demonstrations.

## Setup

1. Clone the repository and navigate to the directory.
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - Mac/Linux: `source venv/bin/activate`
   - Windows: `venv\Scripts\activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Create a `.env` file and add your LLM API keys (e.g., `OPENAI_API_KEY=your_key`).

## Usage (CLI PoC)

**1. Ingest a Document**
Drop an exported Notion `.md` file into the directory, then run:
```bash
python ingest.py your_document.md
```

*Note: This will create a `./chroma_db` folder to persist your vectors.*

**2. Query the Document**

```bash
python query.py "What is the exact product code mentioned for the new feature?"
```
