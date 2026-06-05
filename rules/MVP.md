# Minimum Viable Product (MVP) Scope

## IN SCOPE for MVP
* **CLI Scripts:** `ingest.py` (for loading data) and `query.py` (for asking questions).
* **File Support:** Single `.md` file ingestion.
* **Database:** `ChromaDB` initialized EXPLICITLY with a local directory path (`./chroma_db`).
* **Retrieval:** LangChain `EnsembleRetriever` combining standard vector search with BM25.
* **Output:** Console-printed text answers with cited source chunks.

## OUT OF SCOPE for MVP (Strictly Avoid)
* Notion OAuth or any API-based data connectors.
* Cloud vector databases (Pinecone, Weaviate, etc.).
* Complex UI or Authentication systems.
* Advanced chunking strategies (keep to standard recursive character splitting for now).

## NEXT PHASE SCOPE
* **Multi-Agent RAG Orchestration**
* **Authentication with JWT tokens**
* **Citations, Reranking & Query Expansion**
* **Multi-Doc index query routing & PPTX input parsing**
* **Chat history and memory stores**
