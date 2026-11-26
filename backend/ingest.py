import logging
import os
from pathlib import Path
from shutil import copyfileobj

from dotenv import load_dotenv
from fastapi import UploadFile
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf.errors import PyPdfError

load_dotenv()

DATA_DIR = Path(os.getenv("DATA_PATH", "./data"))
logger = logging.getLogger(__name__)


class InvalidPDFError(ValueError):
    pass


def load_pdf_chunks(path: Path) -> list[Document]:
    try:
        pages = PyPDFLoader(str(path)).load()
    except (PyPdfError, ValueError) as exc:
        raise InvalidPDFError("The PDF could not be read.") from exc

    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    chunks = splitter.split_documents(pages)
    if not chunks:
        raise InvalidPDFError("The PDF contained no readable text.")

    for chunk in chunks:
        chunk.metadata["source"] = path.name
        chunk.metadata["document_name"] = path.name

    return chunks


def ingest_pdf(file: UploadFile, staging_dir: Path) -> tuple[Path, list[Document]]:
    """Copy an upload to staging and extract its chunks before committing it."""
    path = staging_dir / Path(file.filename).name
    with path.open("wb") as output:
        copyfileobj(file.file, output)

    return path, load_pdf_chunks(path)


def ingest_directory(data_dir: str | Path | None = None) -> list[Document]:
    directory = Path(data_dir) if data_dir is not None else DATA_DIR
    directory.mkdir(parents=True, exist_ok=True)

    all_chunks = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.name.lower().endswith(".pdf"):
            try:
                all_chunks.extend(load_pdf_chunks(path))
            except InvalidPDFError as exc:
                logger.warning("Skipping %s: %s", path.name, exc)

    return all_chunks
