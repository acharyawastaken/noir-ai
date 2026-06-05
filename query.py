import os
import sys
import pickle
import time
import shutil
from dotenv import load_dotenv

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

class RAGEngine:
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.vector_retriever = None
        self.bm25_retriever = None
        self.ensemble_retriever = None
        self.llm = None
        self.prompt = None
        self.is_loaded = False

    def reset(self):
        # 1. Clean up resources/files on disk
        if os.path.exists("./chroma_db"):
            try:
                shutil.rmtree("./chroma_db")
            except Exception as e:
                print(f"Error removing ./chroma_db: {e}")
        if os.path.exists("bm25_index.pkl"):
            try:
                os.remove("bm25_index.pkl")
            except Exception as e:
                print(f"Error removing bm25_index.pkl: {e}")

        # 2. Reset in-memory states
        self.embeddings = None
        self.vectorstore = None
        self.vector_retriever = None
        self.bm25_retriever = None
        self.ensemble_retriever = None
        self.is_loaded = False

    def load(self, force=False):
        # Always initialize LLM and prompt first so conversational queries always work
        if self.llm is None:
            self.llm = ChatOllama(
                model="qwen2:1.5b",
                temperature=0,
                num_predict=256,      # Cap output at 256 tokens for speed
                num_ctx=2048,         # Smaller context window = faster inference
            )

        if self.prompt is None:
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a sarcastic, witty AI chatbot.
Guidelines:
1. Be funny, sarcastic, and concise (under 60 words).
2. Answer the user's question. If context is provided, use it. Otherwise, chat with the user generally.
3. Keep it brief.

Context:
{context}"""),
                ("human", "{input}")
            ])

        if self.is_loaded and not force:
            return

        if not os.path.exists("./chroma_db") or not os.path.exists("bm25_index.pkl"):
            self.is_loaded = False
            return

        # Initialize retrievers in-process for speed
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=self.embeddings)
        self.vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 2})

        with open("bm25_index.pkl", "rb") as f:
            self.bm25_retriever = pickle.load(f)
        self.bm25_retriever.k = 2

        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.vector_retriever],
            weights=[0.5, 0.5]
        )
        self.is_loaded = True

    def query(self, query_text):
        self.load()

        if self.is_loaded:
            # ── Normal RAG Retrieval & Prompting ──
            total_start = time.perf_counter()

            # 1. Benchmark Chroma Only
            t_chroma_start = time.perf_counter()
            chroma_docs = self.vector_retriever.invoke(query_text)
            chroma_time = time.perf_counter() - t_chroma_start

            # 2. Benchmark BM25 Only
            t_bm25_start = time.perf_counter()
            bm25_docs = self.bm25_retriever.invoke(query_text)
            bm25_time = time.perf_counter() - t_bm25_start

            # 3. Ensemble Retrieval (Used for RAG)
            t_ensemble_start = time.perf_counter()
            docs = self.ensemble_retriever.invoke(query_text)
            ensemble_time = time.perf_counter() - t_ensemble_start

            # 4. Context Formatting
            t_format_start = time.perf_counter()
            context_str = "\n\n".join([doc.page_content for doc in docs])
            context_format_time = time.perf_counter() - t_format_start

            # 5. Prompt Construction
            t_prompt_start = time.perf_counter()
            formatted_prompt = self.prompt.format_messages(context=context_str, input=query_text)
            prompt_construction_time = time.perf_counter() - t_prompt_start

            # 6. LLM Invocation
            t_llm_start = time.perf_counter()
            llm_response = self.llm.invoke(formatted_prompt)
            llm_time = time.perf_counter() - t_llm_start

            total_time = time.perf_counter() - total_start

            # Gather metrics
            total_chars = len(context_str)
            total_prompt_chars = sum(len(m.content) for m in formatted_prompt)
            est_tokens_sent = total_prompt_chars // 4

            # Format Performance Report
            report_lines = [
                "--------------------------------------------------",
                "PERFORMANCE REPORT & BENCHMARKS",
                "--------------------------------------------------",
                f"ChromaDB Retrieval (only): {chroma_time:.4f}s ({len(chroma_docs)} docs)",
                f"BM25 Retrieval (only):     {bm25_time:.4f}s ({len(bm25_docs)} docs)",
                f"Ensemble Retrieval (RAG):  {ensemble_time:.4f}s ({len(docs)} docs)",
                f"Context Formatting:        {context_format_time:.4f}s",
                f"Prompt Construction:       {prompt_construction_time:.4f}s",
                f"LLM Invocation:            {llm_time:.4f}s",
                f"Total Chain Execution:     {total_time:.4f}s",
                "",
                "RAG Metadata:",
                f"- Number of retrieved docs: {len(docs)}",
                f"- Total characters retrieved: {total_chars}",
                f"- Estimated tokens sent to LLM: {est_tokens_sent}",
                "--------------------------------------------------"
            ]
            report = "\n".join(report_lines)
            print(f"\n{report}\n")

            return llm_response.content
        else:
            # ── Fallback Conversational LLM Wrapper (No Index Loaded) ──
            total_start = time.perf_counter()

            # 1. Prompt Construction
            t_prompt_start = time.perf_counter()
            formatted_prompt = self.prompt.format_messages(context="[No document ingested. Answer generally.]", input=query_text)
            prompt_construction_time = time.perf_counter() - t_prompt_start

            # 2. LLM Invocation
            t_llm_start = time.perf_counter()
            llm_response = self.llm.invoke(formatted_prompt)
            llm_time = time.perf_counter() - t_llm_start

            total_time = time.perf_counter() - total_start

            # Format Performance Report
            report_lines = [
                "--------------------------------------------------",
                "PERFORMANCE REPORT & BENCHMARKS (DIRECT CONVERSATION)",
                "--------------------------------------------------",
                f"Prompt Construction:       {prompt_construction_time:.4f}s",
                f"LLM Invocation:            {llm_time:.4f}s",
                f"Total Chain Execution:     {total_time:.4f}s",
                "--------------------------------------------------"
            ]
            report = "\n".join(report_lines)
            print(f"\n{report}\n")

            return llm_response.content

# Instantiate global engine
rag_engine = RAGEngine()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python query.py "<your_query_here>"')
        sys.exit(1)
        
    query_text = sys.argv[1]
    engine = RAGEngine()
    engine.load()
    print(engine.query(query_text))
