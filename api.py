from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import os
import shutil
import subprocess
import sys
import jwt
import datetime
from dotenv import load_dotenv

load_dotenv()

# Import in-process RAG engine for instant query retrieval times (<5s)
from query import rag_engine

app = FastAPI(title="Multi-Source RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".csv", ".xlsx", ".md", ".txt", ".png", ".jpg", ".jpeg", ".pptx"}
SECRET_KEY = os.getenv("SECRET_KEY", "noir-super-secret-key-12345")
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin")
DEFAULT_PASSWORD = os.getenv("DEFAULT_PASSWORD", "password")

security = HTTPBearer()

class QueryRequest(BaseModel):
    query: str
    session_id: str = "default"
    model_profile: str = "flash"

class LoginRequest(BaseModel):
    username: str
    password: str

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        username: str = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401, detail="Invalid token: missing username claim")
        return username
    except jwt.PyJWTError as e:
        raise HTTPException(status_code=401, detail=f"Invalid or expired credentials: {str(e)}")

@app.on_event("startup")
async def startup_event():
    # Warm up / preload the RAGEngine components on API startup
    print("Warming up RAG Engine and caching model embeddings/vectorstore...")
    try:
        rag_engine.load()
        print("RAG Engine loaded successfully!")
    except Exception as e:
        print(f"RAG Engine preload failed: {e}. It will load dynamically on first prompt.")

@app.post("/login")
async def login(credentials: LoginRequest):
    if (credentials.username == ADMIN_USERNAME and credentials.password == ADMIN_PASSWORD) or credentials.password == DEFAULT_PASSWORD:
        payload = {
            "sub": credentials.username,
            "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=24)
        }
        token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
        return {"token": token, "username": credentials.username}
    else:
        raise HTTPException(status_code=401, detail="Invalid username or password")

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...), 
    session_id: str = Form("default"),
    current_user: str = Depends(get_current_user)
):
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
        # Release the in-memory ChromaDB connection FIRST so ingest.py can safely
        # write to chroma_db without hitting Windows file lock errors.
        rag_engine.unload()

        backend_session_id = f"{current_user}:{session_id}"

        result = subprocess.run(
            [sys.executable, "ingest.py", file_path, "--session-id", backend_session_id], 
            capture_output=True, text=True,
            env=dict(os.environ, PYTHONIOENCODING="utf-8")
        )
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Ingestion failed: {result.stderr}")
        
        # Ingestion succeeded, force-reload the cache to update the indexes in memory
        print("Ingestion complete, updating RAG Engine components...")
        rag_engine.load(force=True)
            
    finally:
        # Clean up the uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)
            
    return {"message": "File successfully ingested and indexed.", "details": result.stdout}

class DeleteDocumentRequest(BaseModel):
    doc_id: str
    session_id: str = "default"

@app.post("/query")
async def query_document(request: QueryRequest, current_user: str = Depends(get_current_user)):
    try:
        # Combine current_user and session_id to isolate memory space securely
        backend_session_id = f"{current_user}:{request.session_id}"
        
        # Check if the chat history is empty for this session (meaning first turn)
        is_first_turn = len(rag_engine.get_history(backend_session_id)) == 0
        
        result = rag_engine.query(request.query, session_id=backend_session_id, model_profile=request.model_profile)
        response_text = result["response"]
        route = result["route"]
        target_doc = result["target_doc"]
        
        title = None
        if is_first_turn:
            title = rag_engine.generate_chat_title(request.query, response_text)
            
        return {
            "response": response_text,
            "title": title,
            "route": route,
            "target_doc": target_doc,
            "sources": result.get("sources", []),
            "benchmarks": result.get("benchmarks", {})
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.post("/reset")
async def reset_rag(
    session_id: str = "default",
    current_user: str = Depends(get_current_user)
):
    try:
        backend_session_id = f"{current_user}:{session_id}"
        # Delete only documents belonging to this session
        if rag_engine.vectorstore is not None:
            try:
                rag_engine.vectorstore.delete(where={"session_id": backend_session_id})
                print(f"[RAG ENGINE] Deleted documents for session {backend_session_id} from vectorstore.")
            except Exception as e:
                print(f"[RAG ENGINE] Error deleting documents for session {backend_session_id}: {e}")
        # Clear chat history for this session only
        rag_engine.clear_session_history(backend_session_id)
        return {"message": "RAG engine index and chat history for this session reset successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")

@app.get("/documents")
async def list_documents(
    session_id: str = "default",
    current_user: str = Depends(get_current_user)
):
    try:
        backend_session_id = f"{current_user}:{session_id}"
        if not rag_engine.is_loaded:
            rag_engine.load()
        if not rag_engine.is_loaded:
            return {"documents": []}
            
        docs = []
        if rag_engine.vectorstore is not None:
            session_chunks = rag_engine.vectorstore.get(where={"session_id": backend_session_id}, include=["metadatas"])
            if session_chunks and "metadatas" in session_chunks:
                docs = list(set([m.get("doc_id") for m in session_chunks["metadatas"] if m.get("doc_id")]))
        return {"documents": docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents/delete")
async def delete_document(request: DeleteDocumentRequest, current_user: str = Depends(get_current_user)):
    try:
        backend_session_id = f"{current_user}:{request.session_id}"
        rag_engine.delete_document(request.doc_id, backend_session_id)
        return {"message": f"Document '{request.doc_id}' deleted successfully from chat session."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    uvicorn.run(app, host=host, port=port)
