from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import shutil
import subprocess
import sys

# Import in-process RAG engine for instant query retrieval times (<5s)
from query import rag_engine

app = FastAPI(title="Multi-Source RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".csv", ".xlsx", ".md", ".txt"}

class QueryRequest(BaseModel):
    query: str

@app.on_event("startup")
async def startup_event():
    # Warm up / preload the RAGEngine components on API startup
    print("Warming up RAG Engine and caching model embeddings/vectorstore...")
    try:
        rag_engine.load()
        print("RAG Engine loaded successfully!")
    except Exception as e:
        print(f"RAG Engine preload failed: {e}. It will load dynamically on first prompt.")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )
    
    # Save the file temporarily
    file_path = f"./{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        result = subprocess.run(
            [sys.executable, "ingest.py", file_path], 
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Ingestion failed: {result.stderr}")
        
        # Ingestion succeeded, force-reload the cache to update the indexes in memory
        print("Ingestion complete, updating RAG Engine components...")
        rag_engine.load(force=True)
            
    finally:
        # Clean up the file
        if os.path.exists(file_path):
            os.remove(file_path)
            
    return {"message": "File successfully ingested and indexed.", "details": result.stdout}

@app.post("/query")
async def query_document(request: QueryRequest):
    try:
        # Run directly in-process - completely bypassing subprocess overhead
        response = rag_engine.query(request.query)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.post("/reset")
async def reset_rag():
    try:
        rag_engine.reset()
        return {"message": "RAG engine index reset successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
