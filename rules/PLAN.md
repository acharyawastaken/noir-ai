# Execution Plan

## Phase 1: The Proof of Concept (CLI First)
* **Step 1.1: Environment Setup.** Initialize virtual environment, install dependencies from `requirements.txt`, and set up `.env` for LLM API keys.
* **Step 1.2: Ingestion Pipeline (`ingest.py`).** Write a script that takes a local `.md` file, chunks it, generates vector embeddings, generates a BM25 index, and saves both to a persistent ChromaDB instance.
* **Step 1.3: Retrieval Pipeline (`query.py`).** Write a script that accepts a user query via CLI, uses LangChain's `EnsembleRetriever` to combine vector similarity and BM25 exact keyword matching, and passes the context to an LLM for generation.

## Phase 2: API & UI Wrapping (Post-PoC)
* **Step 2.1: API Layer.** Wrap the ingestion and query logic into a simple REST API (e.g., FastAPI) to decouple the backend from the frontend.
* **Step 2.2: User Interface.** Create a lightweight frontend (e.g., Streamlit or Gradio) allowing users to upload a `.md` file visually and chat with their document.

## Phase 3: Future Scope (Do Not Build Now)
* Notion OAuth Integration (Deferred to avoid 2 weeks of zero-value complexity).
