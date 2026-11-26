from io import BytesIO
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from httpx import Request, Response
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import AIMessage
from openai import RateLimitError
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from backend import app, ingest, rag_pipeline


def pdf_bytes(*page_texts):
    writer = PdfWriter()
    for text in page_texts:
        page = writer.add_blank_page(width=612, height=792)
        font = DictionaryObject({
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        })
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): font}),
        })
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode())
        page[NameObject("/Contents")] = stream
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


@pytest.fixture
def isolated_index(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(app, "DATA_DIR", data_dir)
    monkeypatch.setattr(ingest, "DATA_DIR", data_dir)
    monkeypatch.setattr(rag_pipeline, "CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.setattr(rag_pipeline, "_vector_store", None)
    monkeypatch.setattr(
        rag_pipeline, "HuggingFaceEmbeddings", lambda **kwargs: DeterministicFakeEmbedding(size=16)
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANSWER_MODE", raising=False)
    return data_dir


def upload(client, filename, content):
    return client.post("/upload", files={"file": (filename, content, "application/pdf")})


def test_upload_and_answer_with_page_citation(isolated_index, monkeypatch):
    model = Mock()
    model.invoke.return_value = AIMessage(content="Friday (policy.pdf, page 2).")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(rag_pipeline, "ChatOpenAI", lambda **kwargs: model)
    monkeypatch.setattr(
        rag_pipeline,
        "_reranker",
        Mock(predict=lambda pairs, **kwargs: [int("deadline" in text) for _, text in pairs]),
    )

    with TestClient(app.app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.post("/ask", json={"query": "Deadline?"}).json() == {
            "answer": rag_pipeline.NO_ANSWER,
            "sources": [],
        }
        response = upload(client, "policy.pdf", pdf_bytes("Overview", "The deadline is Friday."))
        assert response.status_code == 200
        assert response.json() == {"filename": "policy.pdf", "chunks_uploaded": 2, "chunks_indexed": 2}

        # Only keyword retrieval can supply the matching page in this test.
        store = rag_pipeline.get_vector_store()
        monkeypatch.setattr(store, "similarity_search", lambda *args, **kwargs: [])
        response = client.post("/ask", json={"query": "What is the deadline?"})
        assert response.status_code == 200
        assert response.json() == {
            "answer": "Friday (policy.pdf, page 2).",
            "sources": [{"document": "policy.pdf", "page": "2", "snippet": "The deadline is Friday."}],
        }
        assert "[Source 1: policy.pdf, page 2]" in model.invoke.call_args.args[0][1][1]
        assert client.post("/ask", json={"query": "   "}).status_code == 400


@pytest.mark.parametrize("mode", ["no_key", "extractive", "exhausted"])
def test_passage_answers_when_openai_is_unavailable(isolated_index, monkeypatch, mode):
    provider = Mock()
    monkeypatch.setattr(rag_pipeline, "ChatOpenAI", provider)
    if mode != "no_key":
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    if mode == "extractive":
        monkeypatch.setenv("ANSWER_MODE", "extractive")
    if mode == "exhausted":
        provider.return_value.invoke.side_effect = RateLimitError(
            "Out of credits",
            response=Response(429, request=Request("POST", "https://api.openai.com")),
            body={"code": "credit_balance_exhausted"},
        )
    ranker = Mock()
    ranker.predict.side_effect = lambda pairs, **kwargs: [
        0.95 if "deadline" in query and "Friday" in text else 0.01 for query, text in pairs
    ]
    monkeypatch.setattr(rag_pipeline, "_reranker", ranker)
    with TestClient(app.app) as client:
        assert upload(client, "policy.pdf", pdf_bytes("Overview", "The deadline is Friday.")).is_success
        answer = client.post("/ask", json={"query": "What is the deadline?"})
        assert answer.is_success
        assert answer.json() == {
            "answer": "The deadline is Friday.\n\n(policy.pdf, page 2)",
            "sources": [{"document": "policy.pdf", "page": "2", "snippet": "The deadline is Friday."}],
        }
        assert client.post("/ask", json={"query": "What is the cafeteria password?"}).json() == {
            "answer": rag_pipeline.NO_ANSWER, "sources": []
        }
    if mode != "exhausted":
        provider.assert_not_called()


def test_reupload_and_restart_replace_old_chunks(isolated_index):
    path = isolated_index / "policy.PDF"
    path.write_bytes(pdf_bytes("Intro", "Old policy"))
    (isolated_index / "corrupt.pdf").write_bytes(b"not a PDF")
    legacy_chunks = ingest.load_pdf_chunks(path)
    store = rag_pipeline.get_vector_store()
    store.add_documents(legacy_chunks)

    with TestClient(app.app) as client:
        assert len(store.get()["ids"]) == 2
        assert upload(client, path.name, b"broken PDF").status_code == 400
        assert upload(client, path.name, pdf_bytes("")).status_code == 400
        assert upload(client, "notes.txt", b"notes").status_code == 400
        assert path.read_bytes().startswith(b"%PDF")
        assert len(store.get()["ids"]) == 2
        replacement = pdf_bytes("Current policy")
        assert upload(client, path.name, replacement).status_code == 200
        assert path.read_bytes() == replacement
        assert store.get()["documents"] == ["Current policy"]

    with TestClient(app.app):
        assert store.get()["documents"] == ["Current policy"]


def test_failed_indexing_preserves_previous_pdf_and_chunks(isolated_index, monkeypatch):
    original = pdf_bytes("Original policy")
    path = isolated_index / "policy.pdf"
    path.write_bytes(original)

    with TestClient(app.app, raise_server_exceptions=False) as client:
        store = rag_pipeline.get_vector_store()
        write = store.add_documents
        monkeypatch.setattr(rag_pipeline, "INDEX_BATCH_SIZE", 1)
        # Persist one new batch before forcing the next batch to fail.
        outcomes = iter([None, RuntimeError("indexing failed")])

        def fail_second_write(chunks, ids):
            failure = next(outcomes)
            if failure is not None:
                raise failure
            return write(chunks, ids=ids)

        monkeypatch.setattr(store, "add_documents", fail_second_write)
        assert upload(client, path.name, pdf_bytes("New introduction", "New policy")).status_code == 500
        assert path.read_bytes() == original
        assert store.get()["documents"] == ["Original policy"]
