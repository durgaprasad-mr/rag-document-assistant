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
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp env.example .env
```

Add `OPENAI_API_KEY` to `.env`, then copy sample PDFs into `data/`.

### Run with Python

```bash
python -m uvicorn backend.app:app --reload
streamlit run frontend/ui.py
```

Open `http://localhost:8501` for the UI and `http://localhost:8000/docs` for API documentation.

### Run with Docker

```bash
docker compose up --build
```

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

Replace the placeholder entries in `backend/evaluate.py` with 15–20 real questions and reference answers from your PDFs, then run:

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
