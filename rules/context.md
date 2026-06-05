# Project Context & Progress Tracking

## 🧠 What is Understood (Project Context)

**Project Goal:** Build a production-grade, demo-ready Retrieval-Augmented Generation (RAG) Proof of Concept (PoC) that explicitly solves the "exact keyword" problem common in standard RAG tutorials.

**Core Differentiator:**
- **Hybrid Search is Mandatory:** The system uses LangChain's `EnsembleRetriever` to combine `OllamaEmbeddings` (vector semantic search) with `BM25Retriever` (exact keyword matching).

**Key Constraints & Rules:**
- **Local-First:** Fully local stack using Ollama. No OpenAI API or paid models.
  - Embeddings: `nomic-embed-text` via `OllamaEmbeddings`
  - LLM: `llama3` via `ChatOllama`
- **Local Persistence:** Uses `chromadb.PersistentClient` via `./chroma_db` directory. No in-memory databases.
- **MVP Scope:** Strictly limited to local `.md` files. No Notion OAuth, no cloud databases.

**Final Tech Stack:**
- Python 3.11
- `langchain 1.3.4` + `langchain_classic` + `langchain_community` + `langchain_ollama`
- ChromaDB (Local Persistent Vector Store via `langchain_community.vectorstores.Chroma`)
- BM25 (`rank_bm25` via `langchain_community.retrievers.BM25Retriever`)
- Ollama (Local LLM & Embedding server)
- FastAPI + Uvicorn (REST API backend)
- Streamlit (Frontend UI)

---

## ✅ What is Completed

### Phase 0: Project Scaffolding
- [x] `rules/PLAN.md`, `ARCHITECTURE.md`, `PRD.md`, `MVP.md`, `AI_RULES.md`, `CLAUDE.md`, `README.md`, `requirements.txt`
- [x] `rules/context.md` initialized

### Phase 1: CLI Proof of Concept
- [x] `ingest.py`: Loads `.md` → splits into chunks → generates embeddings via Ollama → saves to persistent ChromaDB + pickled BM25 index.
- [x] `query.py`: Loads ChromaDB + BM25 index → `EnsembleRetriever` (0.5/0.5) → `ChatOllama` → prints answer + source chunks.
- [x] `.env` created (API key no longer needed for local Ollama stack)
- [x] `venv` created and all dependencies installed.

### Phase 1 Polish: Compatibility Fixes
- [x] Migrated from OpenAI to Ollama (`OllamaEmbeddings`, `ChatOllama`).
- [x] Fixed all broken imports for LangChain `1.3.4`:
  - `EnsembleRetriever` → `langchain_classic.retrievers`
  - `create_retrieval_chain` → `langchain_classic.chains`
  - `create_stuff_documents_chain` → `langchain_classic.chains.combine_documents`
- [x] Fixed `api.py` subprocess to use `sys.executable` instead of `"python"` to ensure the venv is used.

### Phase 2: API & UI Wrapping ✅ FULLY WORKING
- [x] `api.py` (FastAPI): `/upload` and `/query` endpoints, running at `http://localhost:8000`.
- [x] `app.py` (Streamlit): Upload sidebar + chat interface, running at `http://localhost:8501`.
- [x] `requirements.txt` updated with `fastapi`, `uvicorn`, `python-multipart`, `streamlit`.

---

## 🚀 Next Steps (Phase 3 - Future Scope)
- [ ] Notion OAuth Integration (Deferred per `PLAN.md`)
- [ ] Consider adding streaming responses to the Streamlit chat UI
- [ ] Consider adding multi-document support (ingest multiple files into the same ChromaDB collection)
- [ ] Consider adding a "clear index" button to the Streamlit UI to reset ChromaDB and BM25 index
