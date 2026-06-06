import os
import sys
import pickle
import time
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever

def sanitise_text(text: str) -> str:
    """Sanitise text content to protect against prompt injection, safety override tricks, and formatting errors."""
    if not text:
        return ""
    import re
    patterns_to_neutralize = [
        r"ignore\s+(?:all\s+)?previous\s+instructions",
        r"ignore\s+(?:any\s+)?instructions\s+above",
        r"ignore\s+the\s+(?:context|rules|guidelines)",
        r"system\s+override",
        r"you\s+are\s+now\s+a",
        r"instead\s+of\s+answering",
        r"forget\s+your\s+previous",
        r"bypass\s+safety",
        r"delete\s+all",
        r"reset\s+database",
        r"execute\s+command",
    ]
    sanitised = text
    for pattern in patterns_to_neutralize:
        sanitised = re.sub(pattern, "[CLEANED SECURE DATA]", sanitised, flags=re.IGNORECASE)
    sanitised = sanitised.replace("{", "[").replace("}", "]")
    sanitised = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", sanitised)
    return sanitised

def ingest_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    print(f"Loading document: {file_path} (type: {ext})...")
    
    from langchain_core.documents import Document
    docs = []
    
    easyocr_reader = None
    
    if ext == ".pdf":
        import pypdf
        reader = pypdf.PdfReader(file_path)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip() and len(text.strip()) > 30:
                print(f"Extracted direct text from page {i + 1}")
                docs.append(Document(
                    page_content=sanitise_text(text), 
                    metadata={"source": file_path, "page": i + 1}
                ))
            else:
                print(f"Page {i + 1} appears to be scanned. Running OCR...")
                try:
                    import fitz  # PyMuPDF
                    import tempfile
                    
                    doc = fitz.open(file_path)
                    fitz_page = doc.load_page(i)
                    pix = fitz_page.get_pixmap()
                    
                    with tempfile.TemporaryDirectory() as temp_dir:
                        image_path = os.path.join(temp_dir, f"temp_page_{i}.png")
                        pix.save(image_path)
                        
                        if easyocr_reader is None:
                            import easyocr
                            print("Initializing EasyOCR reader...")
                            easyocr_reader = easyocr.Reader(['en'], gpu=False)
                            
                        result = easyocr_reader.readtext(image_path, detail=0)
                        ocr_text = "\n".join(result)
                        if ocr_text.strip():
                            docs.append(Document(
                                page_content=sanitise_text(ocr_text), 
                                metadata={"source": file_path, "page": i + 1, "ocr": True}
                            ))
                except Exception as e:
                    print(f"OCR failed for page {i + 1}: {e}")
                    
    elif ext in (".png", ".jpg", ".jpeg"):
        try:
            import easyocr
            print("Initializing EasyOCR reader...")
            easyocr_reader = easyocr.Reader(['en'], gpu=False)
            print(f"Running OCR on image {file_path}...")
            result = easyocr_reader.readtext(file_path, detail=0)
            ocr_text = "\n".join(result)
            if ocr_text.strip():
                docs.append(Document(
                    page_content=sanitise_text(ocr_text), 
                    metadata={"source": file_path, "ocr": True}
                ))
        except Exception as e:
            raise ValueError(f"Failed to run OCR on image: {e}")
            
    elif ext == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader
        loader = Docx2txtLoader(file_path)
        raw_docs = loader.load()
        for d in raw_docs:
            docs.append(Document(page_content=sanitise_text(d.page_content), metadata=d.metadata))
            
    elif ext == ".doc":
        try:
            from langchain_community.document_loaders import Docx2txtLoader
            loader = Docx2txtLoader(file_path)
            raw_docs = loader.load()
            for d in raw_docs:
                docs.append(Document(page_content=sanitise_text(d.page_content), metadata=d.metadata))
        except Exception:
            raise ValueError("Word .doc format is not supported directly. Please save your file as .docx and try again.")
            
    elif ext in (".md", ".txt"):
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(file_path, encoding='utf-8')
        raw_docs = loader.load()
        for d in raw_docs:
            docs.append(Document(page_content=sanitise_text(d.page_content), metadata=d.metadata))
            
    elif ext == ".csv":
        import csv
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row_idx, row in enumerate(reader):
                row_text = ", ".join([f"{k}: {v}" for k, v in row.items() if v])
                if row_text.strip():
                    docs.append(Document(
                        page_content=sanitise_text(row_text), 
                        metadata={"source": file_path, "row": row_idx}
                    ))
                    
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
                        page_content=sanitise_text(row_text), 
                        metadata={"source": file_path, "sheet": sheet_name, "row": row_idx}
                    ))

    elif ext == ".pptx":
        try:
            from pptx import Presentation
            prs = Presentation(file_path)
            for i, slide in enumerate(prs.slides):
                slide_text_parts = []
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text.strip():
                        slide_text_parts.append(shape.text.strip())
                    if shape.has_table:
                        for row in shape.table.rows:
                            for cell in row.cells:
                                if cell.text.strip():
                                    slide_text_parts.append(cell.text.strip())
                slide_text = "\n".join(slide_text_parts)
                if slide_text.strip():
                    docs.append(Document(
                        page_content=sanitise_text(slide_text),
                        metadata={"source": file_path, "slide": i + 1}
                    ))
        except Exception as e:
            raise ValueError(f"Failed to parse PowerPoint presentation: {e}")
    else:
        raise ValueError(f"Unsupported file type: {ext}")
        
    if not docs:
        raise ValueError(f"No text content could be extracted from {file_path}")
        
    # Assign doc_id to every loaded document
    doc_id = os.path.basename(file_path)
    for d in docs:
        d.metadata["doc_id"] = doc_id
        
    print(f"Loaded {len(docs)} document section(s) for doc_id: {doc_id}.")
    
    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n\n", "\n", " ", ""]
    )
    splits = text_splitter.split_documents(docs)
    
    # Filter out empty splits
    splits = [s for s in splits if s.page_content.strip()]
    for s in splits:
        s.metadata["doc_id"] = doc_id
    print(f"Created {len(splits)} chunks.")

    print("Generating Vector Embeddings and saving to Chroma in batches...")
    chroma_db_path = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    bm25_index_path = os.getenv("BM25_INDEX_PATH", "bm25_index.pkl")
    ollama_embed_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

    embeddings = OllamaEmbeddings(model=ollama_embed_model)

    # Use existing Chroma database, clear old version of this document if it exists, and append new chunks.
    vectorstore = Chroma(persist_directory=chroma_db_path, embedding_function=embeddings)
    try:
        print(f"Checking for existing index entries for {doc_id} to prevent duplication...")
        vectorstore.delete(where={"doc_id": doc_id})
        print(f"Cleared existing chunks for {doc_id}.")
    except Exception as e:
        print(f"No previous index entry to clean or collection is empty: {e}")
    
    batch_size = 50
    for i in range(0, len(splits), batch_size):
        batch = splits[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1} (chunks {i} to {min(i+batch_size, len(splits))})...")
        vectorstore.add_documents(documents=batch)
        
    print(f"Vectors saved to {chroma_db_path}")

    print("Generating Multi-Doc BM25 Indices...")
    # Retrieve all documents currently in the collection to build global and per-doc BM25 indexes
    all_data = vectorstore.get(include=["documents", "metadatas"])
    all_docs = []
    docs_by_id = {}
    
    for doc_text, meta in zip(all_data["documents"], all_data["metadatas"]):
        doc = Document(page_content=doc_text, metadata=meta)
        all_docs.append(doc)
        d_id = meta.get("doc_id", "unknown")
        if d_id not in docs_by_id:
            docs_by_id[d_id] = []
        docs_by_id[d_id].append(doc)
        
    if all_docs:
        bm25_data = {
            "global": BM25Retriever.from_documents(all_docs),
            "documents": {}
        }
        for d_id, doc_splits in docs_by_id.items():
            bm25_data["documents"][d_id] = BM25Retriever.from_documents(doc_splits)
            
        with open(bm25_index_path, "wb") as f:
            pickle.dump(bm25_data, f)
        print(f"BM25 indices saved to {bm25_index_path} ({len(docs_by_id)} documents indexed)")
    else:
        print("No documents found in store, skipping BM25 indexing.")

    print("Ingestion complete!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <path_to_file>")
        print("Supported formats: .pdf, .docx, .doc, .csv, .xlsx, .md, .txt, .png, .jpg, .jpeg")
        sys.exit(1)
        
    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist.")
        sys.exit(1)
        
    ingest_document(file_path)
