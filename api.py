from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import os
import shutil
import subprocess
import sys
import jwt
import datetime

# Import in-process RAG engine for instant query retrieval times (<5s)
from query import rag_engine

app = FastAPI(title="Multi-Source RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".csv", ".xlsx", ".md", ".txt", ".png", ".jpg", ".jpeg"}
SECRET_KEY = "noir-super-secret-key-12345"
security = HTTPBearer()

class QueryRequest(BaseModel):
    query: str

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
    # Simplistic check: Allow admin/admin or any user with password 'password'
    if (credentials.username == "admin" and credentials.password == "admin") or credentials.password == "password":
        payload = {
            "sub": credentials.username,
            "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=24)
        }
        token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
        return {"token": token, "username": credentials.username}
    else:
        raise HTTPException(status_code=401, detail="Invalid username or password")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), current_user: str = Depends(get_current_user)):
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
        # delete and recreate chroma_db without hitting Windows file lock errors.
        rag_engine.reset()

        result = subprocess.run(
            [sys.executable, "ingest.py", file_path], 
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

@app.post("/query")
async def query_document(request: QueryRequest, current_user: str = Depends(get_current_user)):
    try:
        # Pass current_user as the session_id so that each user gets their own memory!
        response = rag_engine.query(request.query, session_id=current_user)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.post("/reset")
async def reset_rag(current_user: str = Depends(get_current_user)):
    try:
        rag_engine.reset()
        # Also clear the current user's specific history
        rag_engine.history.pop(current_user, None)
        return {"message": "RAG engine index and chat history reset successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
