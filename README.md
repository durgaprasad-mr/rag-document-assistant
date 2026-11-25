# RAG Document Assistant

A grounded RAG assistant for college syllabi, policies, manuals, medical guides, and other PDF documents.

## Features

- FastAPI upload and question-answer endpoints
- PDF text extraction with preserved document/page metadata
- 700-character chunking with 100-character overlap
- Hugging Face embeddings stored locally in ChromaDB
- BM25 keyword retrieval combined with semantic retrieval
- Cross-Encoder re-ranking to select the strongest contexts
- Strict grounded generation with document and page citations
- Streamlit chat UI with source snippets
- Docker and Docker Compose support
- RAGAS evaluation scaffold

## Architecture

```text
PDF → PyPDFLoader → RecursiveChunking → Hugging Face Embeddings
                                      ↓
                                  ChromaDB
Query → Hybrid Retrieval (BM25 + Vector) → Cross-Encoder → Top 3
                                      ↓
                         Grounded LLM Answer + Source citations
```

## Setup

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp env.example .env
```

Set `OPENAI_API_KEY` in `.env` to enable answers. Upload text-based PDFs in the UI or place them in `data/` before starting the backend. The first indexing and question requests download the local embedding and reranking models.

### Run with Python

```bash
python -m uvicorn backend.app:app --reload
streamlit run frontend/ui.py
```

Open `http://localhost:8501` for the UI and `http://localhost:8000/docs` for API documentation.

### Run with Docker

On the first run, copy `env.example` to `.env` and set `OPENAI_API_KEY` in `.env`.

```bash
docker compose up -d --build --wait
```

Open `http://localhost:8501` for the chat UI and `http://localhost:8000/docs` for the API. PDFs and the search index persist in `data/` and `chroma_db/`. Docker keeps downloaded models in a named volume.

On a Linux server, point your reverse proxy at `127.0.0.1:8501`. Use `RAG_UI_PORT` and `RAG_API_PORT` to change the host ports, or set `RAG_UI_BIND` to choose the UI’s listening address.

## API

### Upload

```bash
curl -F "file=@data/example.pdf" http://localhost:8000/upload
```

### Ask

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"What is the main topic of this document?"}'
```

Response:

```json
{
  "answer": "...",
  "sources": [{"document":"example.pdf","page":"3","snippet":"..."}]
}

```

## Evaluation

Add 15–20 real questions and reference answers from your PDFs to `TEST_CASES` in `backend/evaluate.py`, then run:

```bash
python -m backend.evaluate
```

The evaluation reports:

- Faithfulness
- Answer relevance
- Context precision

Do not claim a score until it is generated from a real test dataset.

## Security

- Never commit `.env` or API keys.
- Keep the vector database out of source control.
- Use an allowlisted LLM API key server-side.
- For public deployment, add authentication and per-user rate limiting.
