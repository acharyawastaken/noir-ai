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

## Next Phase Architecture Additions

### 1. Multi-Agent RAG
* **Router Agent:** Analyzes the query intent and decides if it is a general chat, single-document RAG lookup, or multi-document retrieval.
* **Q&A Agent:** Specialized generator that crafts responses from document context.
* **Synthesizer Agent:** Cleans, refines, and formats the output answer.

### 2. Authentication with JWT
* **Authentication Middleware:** Intercepts REST calls to validate client JWT signatures.
* **User Identity Mapping:** Isolates vector store collections or directories by tenant/user id.

### 3. Citations, Reranking & Query Expansion
* **Query Expansion:** Uses an LLM to generate synonym-based sub-queries.
* **Reranking (Cross-Encoder):** Cohere or HuggingFace Cross-Encoder filters the top retrieved document candidates to select the most matching top-K context.
* **Citations:** Appends metadata containing document filenames, slide/page numbers, and matching text snippets.

### 4. Multi-Doc Support & PPTX Ingestion
* **Parser Factory:** Extends data ingestion to support PowerPoint files (`.pptx`) using libraries like `python-pptx` to extract structured slides.
* **Multi-Doc Collection:** Organizes vectors with a metadata `doc_id` field for target filtering.

### 5. Chat History & Memory
* **ConversationBufferMemory / Redis Store:** Stores and formats past messages, passing recent chat turns as context to the LLM to sustain coherent discussions.
