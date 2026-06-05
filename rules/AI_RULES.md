# AI Coding Assistant Rules

**CRITICAL DIRECTIVES FOR CODE GENERATION:**

1.  **HYBRID SEARCH IS MANDATORY:** Do not write standard, vector-only RAG. You must use LangChain's `EnsembleRetriever` combining BM25 and Vector search. This is the core value proposition.
2.  **CHROMA PERSISTENCE:** Never use in-memory ChromaDB. Always instantiate with `chromadb.PersistentClient(path="./chroma_db")`. Vectors must survive restarts.
3.  **MVP SCOPE:** Do not implement Notion APIs, OAuth, or external web scrapers. Only write logic to parse local `.md` files.
4.  **CLI FIRST:** Build pure Python CLI scripts (`ingest.py` and `query.py`) first. Do not scaffold web frameworks (FastAPI/Streamlit) until the CLI PoC is verified working.
5.  **SIMPLICITY:** Keep chunking and prompt logic simple. Focus on getting the BM25 + Vector integration working flawlessly in under 100 lines of code per script.
