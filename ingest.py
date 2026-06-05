import os
import sys
import pickle
import time
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever

def ingest_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    print(f"Loading document: {file_path} (type: {ext})...")
    
    from langchain_core.documents import Document
    docs = []
    
    if ext == ".pdf":
        import pypdf
        reader = pypdf.PdfReader(file_path)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                docs.append(Document(page_content=text, metadata={"source": file_path, "page": i + 1}))
                
    elif ext == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader
        loader = Docx2txtLoader(file_path)
        docs = loader.load()
        
    elif ext == ".doc":
        try:
            from langchain_community.document_loaders import Docx2txtLoader
            loader = Docx2txtLoader(file_path)
            docs = loader.load()
        except Exception:
            raise ValueError("Word .doc format is not supported directly. Please save your file as .docx and try again.")
            
    elif ext in (".md", ".txt"):
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(file_path, encoding='utf-8')
        docs = loader.load()
        
    elif ext == ".csv":
        import csv
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row_idx, row in enumerate(reader):
                row_text = ", ".join([f"{k}: {v}" for k, v in row.items() if v])
                if row_text.strip():
                    docs.append(Document(page_content=row_text, metadata={"source": file_path, "row": row_idx}))
                    
    elif ext == ".xlsx":
        import pandas as pd
        xl = pd.ExcelFile(file_path)
        for sheet_name in xl.sheet_names:
            df = xl.parse(sheet_name)
            df = df.fillna("")
            for row_idx, row in df.iterrows():
                row_text = ", ".join([f"{col}: {val}" for col, val in row.items() if str(val).strip()])
                if row_text.strip():
                    docs.append(Document(
                        page_content=row_text, 
                        metadata={"source": file_path, "sheet": sheet_name, "row": row_idx}
                    ))
    else:
        raise ValueError(f"Unsupported file type: {ext}")
        
    if not docs:
        raise ValueError(f"No text content could be extracted from {file_path}")
        
    print(f"Loaded {len(docs)} document section(s).")
    
    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    splits = text_splitter.split_documents(docs)
    
    # Filter out empty splits
    splits = [s for s in splits if s.page_content.strip()]
    print(f"Created {len(splits)} chunks.")

    print("Generating Vector Embeddings and saving to Chroma in batches...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
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
        print("Supported formats: .pdf, .docx, .doc, .csv, .xlsx, .md, .txt")
        sys.exit(1)
        
    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist.")
        sys.exit(1)
        
    ingest_document(file_path)
