# Project Context for Claude/AI Agent

You are building a production-grade, demo-able RAG pipeline. The user wants to avoid the pitfalls of "tutorial RAG."

**Your specific constraints for this workspace:**
* Read `MVP.md` before generating any feature code.
* Adhere strictly to `AI_RULES.md`.
* We are prioritizing robustness in retrieval (Hybrid Search) over UI/UX for Phase 1. 
* If asked to "build the ingestion script", output a Python CLI script that handles `.md` -> text splitting -> embeddings -> Persistent ChromaDB & BM25 saving.
* If asked to "build the query script", load the persistent DBs and use `EnsembleRetriever`.
* Do not over-engineer. Write clean, well-commented, modular Python.
