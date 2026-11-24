from pathlib import Path
import os

from dotenv import load_dotenv
from fastapi import UploadFile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DATA_DIR = Path(os.getenv("DATA_PATH", "./data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)


def ingest_pdf(file: UploadFile) -> tuple[str, int]:
    """Save and ingest a PDF, returning its filename and chunk count."""
    filename = Path(file.filename or "document.pdf").name
    destination = DATA_DIR / filename

    if hasattr(file, "file"):
        with destination.open("wb") as output:
            output.write(file.file.read())

    loader = PyPDFLoader(str(destination))
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    for chunk in chunks:
        chunk.metadata["source"] = filename
        chunk.metadata["document_name"] = filename

    if not chunks:
        raise ValueError("The PDF contained no readable text.")

    return str(destination), len(chunks)


def ingest_directory(data_dir: str | Path | None = None) -> list:
    """Ingest every PDF in a directory for CLI or startup use."""
    directory = Path(data_dir or DATA_DIR)
    directory.mkdir(parents=True, exist_ok=True)

    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)

    for pdf_path in directory.glob("*.pdf"):
        documents = PyPDFLoader(str(pdf_path)).load()
        chunks = splitter.split_documents(documents)
        for chunk in chunks:
            chunk.metadata["source"] = pdf_path.name
            chunk.metadata["document_name"] = pdf_path.name
        all_chunks.extend(chunks)

    return all_chunks
