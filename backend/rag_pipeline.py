import os
import re

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")

_embeddings = None
_vector_store = None
_reranker = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings


def get_vector_store():
    global _vector_store
    if _vector_store is None:
        _vector_store = Chroma(
            collection_name="documents",
            embedding_function=get_embeddings(),
            persist_directory=CHROMA_PATH,
        )
    return _vector_store


def add_documents(chunks: list[Document]) -> int:
    if not chunks:
        return 0
    get_vector_store().add_documents(chunks)
    return len(chunks)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\w]+", text.lower())


def hybrid_retrieve(query: str, k: int = 12) -> list[Document]:
    """Combine Chroma semantic search with BM25 keyword search."""
    store = get_vector_store()
    try:
        collection = store._collection
        corpus = collection.get(include=["documents", "metadatas"])
    except Exception:
        return store.similarity_search(query, k=k)

    documents = corpus.get("documents") or []
    metadatas = corpus.get("metadatas") or []
    if not documents:
        return []

    candidates = [Document(page_content=text, metadata=meta or {}) for text, meta in zip(documents, metadatas)]
    tokenized_corpus = [tokenize(doc.page_content) for doc in candidates]
    bm25 = BM25Okapi(tokenized_corpus)
    keyword_scores = bm25.get_scores(tokenize(query))

    keyword_ranked = sorted(range(len(candidates)), key=lambda i: keyword_scores[i], reverse=True)[:k]
    semantic = store.similarity_search(query, k=k)

    combined = {}
    for rank, doc in enumerate(keyword_ranked):
        combined[(doc.page_content, str(doc.metadata))] = candidates[doc]
    for rank, doc in enumerate(semantic):
        combined.setdefault((doc.page_content, str(doc.metadata)), doc)

    return list(combined.values())[:k * 2]


def rerank(query: str, documents: list[Document], top_n: int = 3) -> list[Document]:
    global _reranker
    if not documents:
        return []
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL)
    pairs = [(query, doc.page_content) for doc in documents]
    scores = _reranker.predict(pairs)
    ranked = sorted(zip(documents, scores), key=lambda item: item[1], reverse=True)
    return [doc for doc, _ in ranked[:top_n]]


def generate_answer(query: str) -> dict:
    retrieved = hybrid_retrieve(query)
    relevant = rerank(query, retrieved)

    if not relevant:
        return {
            "answer": "I do not have enough information in the provided documents.",
            "sources": [],
        }

    context_parts = []
    sources = []
    for index, doc in enumerate(relevant, start=1):
        name = doc.metadata.get("document_name") or doc.metadata.get("source") or "Unknown document"
        page = doc.metadata.get("page", doc.metadata.get("page_number"))
        page_label = str(int(page) + 1) if isinstance(page, (int, float)) else str(page or "unknown")
        snippet = doc.page_content.strip()
        context_parts.append(f"[Source {index}: {name}, page {page_label}]\n{snippet}")
        sources.append({"document": name, "page": page_label, "snippet": snippet})

    system_prompt = """Answer the user's question using ONLY the provided context.
If the answer is not in the context, state exactly: 'I do not have enough information in the provided documents.'
Always cite the document name and page number for factual claims.
Do not invent or use information outside the retrieved context."""

    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    response = llm.invoke([
        ("system", system_prompt),
        ("human", f"Context:\n{'\n\n'.join(context_parts)}\n\nQuestion: {query}"),
    ])

    return {"answer": response.content, "sources": sources}
