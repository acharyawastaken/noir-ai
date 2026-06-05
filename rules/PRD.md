# Product Requirements Document (PRD)

## Problem Statement
Standard tutorial-level RAG applications fail in real-world scenarios because pure vector search misses exact keywords (names, dates, product codes). Furthermore, building complex integrations (like Notion OAuth) early on delays the ability to demo core value to clients.

## Product Vision
A demo-ready RAG application that provides highly accurate, context-aware answers over user documents by utilizing a hybrid retrieval system (semantic + keyword). 

## Target Audience
Clients and internal stakeholders evaluating the efficacy of custom LLM solutions on proprietary data.

## Core Requirements
1.  **File Upload:** The system must accept Markdown (`.md`) files representing exported Notion pages.
2.  **Hybrid Retrieval:** The system must retrieve information using both vector similarity and BM25 exact keyword matching to ensure no factual data is missed.
3.  **Local Persistence:** Database state must survive restarts to prevent time-consuming re-indexing and "ghost" debugging during development.
4.  **CLI Accessibility:** The foundational system must be fully operable via the command line to validate logic before UI development.
