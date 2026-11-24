import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ingest import ingest_pdf, ingest_directory
from rag_pipeline import add_documents, generate_answer

app = FastAPI(title="RAG Document Assistant", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",")],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    query: str


@app.on_event("startup")
def load_existing_documents():
    add_documents(ingest_directory())


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    _, uploaded_chunks = ingest_pdf(file)
    from ingest import PyPDFLoader, RecursiveCharacterTextSplitter
    from pathlib import Path

    pdf_path = Path(os.getenv("DATA_PATH", "./data")) / Path(file.filename).name
    chunks = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100).split_documents(
        PyPDFLoader(str(pdf_path)).load()
    )
    for chunk in chunks:
        chunk.metadata["source"] = pdf_path.name
        chunk.metadata["document_name"] = pdf_path.name

    indexed = add_documents(chunks)
    return {"filename": pdf_path.name, "chunks_uploaded": uploaded_chunks, "chunks_indexed": indexed}


@app.post("/ask")
def ask(request: AskRequest):
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    return generate_answer(query)
