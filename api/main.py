import sys
import os

# Add project root to path so we can import src modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from src.chain import get_qa_chain, format_docs
from src.ingest import ingest_pdf

app = FastAPI(
    title="Customer Support AI Chatbot",
    description="RAG-powered customer support chatbot using Ollama",
    version="0.2.0",
)

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the QA chain at startup
qa_chain = None
retriever = None

UPLOAD_DIR = "Data/uploads"


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[str] = []


class UploadResponse(BaseModel):
    message: str
    files: list[str] = []
    total_chunks: int = 0


@app.on_event("startup")
async def startup_event():
    global qa_chain, retriever
    # Create upload directory if it doesn't exist
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    try:
        qa_chain, retriever = get_qa_chain()
        print("✅ QA chain initialized successfully")
    except Exception as e:
        print(f"⚠️ QA chain initialization failed: {e}")
        print("Upload documents via the UI to create the vector database.")


def reload_chain():
    """Reload the QA chain after new documents are ingested."""
    global qa_chain, retriever
    try:
        qa_chain, retriever = get_qa_chain()
        print("✅ QA chain reloaded successfully")
    except Exception as e:
        print(f"⚠️ QA chain reload failed: {e}")


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "chain_loaded": qa_chain is not None,
    }


@app.post("/upload", response_model=UploadResponse)
async def upload_documents(files: list[UploadFile] = File(...)):
    """Upload up to 3 PDF documents, ingest them into Weaviate."""
    if len(files) > 3:
        raise HTTPException(
            status_code=400,
            detail="Maximum 3 files can be uploaded at a time.",
        )

    # Validate all files are PDFs
    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail=f"Only PDF files are supported. '{f.filename}' is not a PDF.",
            )

    uploaded_names = []
    total_chunks = 0

    for f in files:
        # Save uploaded file to disk
        file_path = os.path.join(UPLOAD_DIR, f.filename)
        try:
            contents = await f.read()
            with open(file_path, "wb") as fp:
                fp.write(contents)
            uploaded_names.append(f.filename)

            # Ingest the PDF into Weaviate
            chunks = ingest_pdf(file_path)
            total_chunks += chunks
            print(f"✅ Uploaded and ingested '{f.filename}' ({chunks} chunks)")
        except Exception as e:
            print(f"❌ Failed to process '{f.filename}': {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process '{f.filename}': {str(e)}",
            )

    # Reload the QA chain to pick up new documents
    reload_chain()

    return UploadResponse(
        message=f"Successfully uploaded and ingested {len(uploaded_names)} document(s).",
        files=uploaded_names,
        total_chunks=total_chunks,
    )


# [OPT-1] Streaming endpoint — sends tokens as they're generated
@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    """Stream the LLM response token-by-token using Server-Sent Events."""
    if qa_chain is None:
        raise HTTPException(
            status_code=503,
            detail="QA chain not initialized. Please upload documents first.",
        )

    def token_generator():
        """Generator that yields tokens as they are produced by the LLM."""
        try:
            for chunk in qa_chain.stream(request.question):
                yield chunk
        except Exception as e:
            yield f"\n\n⚠️ Error: {str(e)}"

    return StreamingResponse(
        token_generator(),
        media_type="text/plain",
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Non-streaming query endpoint (fallback)."""
    if qa_chain is None:
        raise HTTPException(
            status_code=503,
            detail="QA chain not initialized. Please upload documents first.",
        )

    try:
        # LCEL chain takes a string and returns a string
        answer = qa_chain.invoke(request.question)

        # Fetch source documents separately via retriever
        sources = []
        if retriever:
            docs = retriever.invoke(request.question)
            sources = list(set(
                doc.metadata.get("source", "unknown") for doc in docs
            ))

        return QueryResponse(answer=answer, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
