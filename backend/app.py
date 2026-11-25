import os
from contextlib import asynccontextmanager
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from openai import APIError
from pydantic import BaseModel

from .ingest import DATA_DIR, InvalidPDFError, ingest_directory, ingest_pdf
from .rag_pipeline import AnswerUnavailableError, add_documents, generate_answer, index_lock


@asynccontextmanager
async def lifespan(app: FastAPI):
    add_documents(ingest_directory())
    yield


app = FastAPI(title="RAG Document Assistant", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",")],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    query: str


@app.get("/", include_in_schema=False)
def home():
    return RedirectResponse("/docs")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    with TemporaryDirectory(dir=DATA_DIR) as staging:
        try:
            staged_pdf, chunks = ingest_pdf(file, Path(staging))
        except InvalidPDFError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        destination = DATA_DIR / staged_pdf.name
        previous_pdf = Path(staging) / "previous"
        with index_lock:
            if destination.is_file():
                copyfile(destination, previous_pdf)
            staged_pdf.replace(destination)
            try:
                indexed = add_documents(chunks)
            except Exception:
                if previous_pdf.exists():
                    previous_pdf.replace(destination)
                else:
                    destination.unlink()
                raise

    return {
        "filename": destination.name,
        "chunks_uploaded": len(chunks),
        "chunks_indexed": indexed,
    }


@app.post("/ask")
def ask(request: AskRequest):
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        return generate_answer(query)
    except AnswerUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except APIError as exc:
        raise HTTPException(status_code=502, detail="The answer provider could not complete the request.") from exc
