# Technical Architecture

## Core Stack
* **Language:** Python 3.10+
* **Framework:** LangChain (for orchestration and retrievers)
* **Vector Store:** ChromaDB (Running in local persistent mode)
* **Keyword Search:** BM25 (via `rank_bm25`)
* **LLM & Embeddings:** OpenAI (or Anthropic/Local via standard LangChain wrappers)

## System Components

### 1. Data Ingestion (`ingest.py`)
* **Input:** Exported Notion Markdown (`.md`) files.
* **Processing:** Read file -> Split into semantic chunks -> Generate Vector Embeddings.
* **Storage:** * Vectors -> `chromadb.PersistentClient(path="./chroma_db")`
    * Keyword Index -> BM25 Index (serialized and saved locally to pair with Chroma).

### 2. Retrieval & Generation (`query.py`)
* **Query Parsing:** Accept natural language query.
* **Hybrid Search:**
    * *Path A:* Vector Search (semantic meaning).
    * *Path B:* BM25 Search (exact keywords, names, IDs).
* **Ensemble:** LangChain `EnsembleRetriever` merges Path A and Path B (typically 0.5/0.5 weight).
* **Generation:** Top-K merged documents injected into LLM context window to generate the final answer.
