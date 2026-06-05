import os
import sys
import pickle
import time
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever

def get_loader(file_path):
    """Return the appropriate LangChain loader based on file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader
        return PyPDFLoader(file_path)
    elif ext in (".docx", ".doc"):
        from langchain_community.document_loaders import Docx2txtLoader
        return Docx2txtLoader(file_path)
    elif ext == ".md" or ext == ".txt":
        from langchain_community.document_loaders import TextLoader
        return TextLoader(file_path, encoding='utf-8')
    else:
        raise ValueError(f"Unsupported file type: {ext}. Supported: .pdf, .docx, .doc, .md, .txt")

def ingest_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    print(f"Loading document: {file_path} (type: {ext})...")
    
    loader = get_loader(file_path)
    docs = loader.load()
    print(f"Loaded {len(docs)} page(s)/section(s).")

    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    splits = text_splitter.split_documents(docs)
    print(f"Created {len(splits)} chunks.")

    print("Generating Vector Embeddings and saving to Chroma in batches...")
    # Make sure you have pulled this model via `ollama pull nomic-embed-text`
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    # Process in batches to avoid rate limits
    batch_size = 50
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    
    for i in range(0, len(splits), batch_size):
        batch = splits[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1} (chunks {i} to {min(i+batch_size, len(splits))})...")
        vectorstore.add_documents(documents=batch)
        
    print("Vectors saved to ./chroma_db")

    print("Generating BM25 Index...")
    bm25_retriever = BM25Retriever.from_documents(splits)
    
    with open("bm25_index.pkl", "wb") as f:
        pickle.dump(bm25_retriever, f)
    print("BM25 index saved to ./bm25_index.pkl")

    print("Ingestion complete!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <path_to_file>")
        print("Supported formats: .pdf, .docx, .doc, .md, .txt")
        sys.exit(1)
        
    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist.")
        sys.exit(1)
        
    ingest_document(file_path)
