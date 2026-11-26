import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")


def backend_post(endpoint: str, **kwargs):
    try:
        response = requests.post(f"{BACKEND_URL}/{endpoint}", timeout=120, **kwargs)
    except requests.RequestException:
        st.error("Could not reach the backend. Check that the API is running and try again.")
        return None

    try:
        result = response.json()
    except ValueError:
        st.error(f"The backend returned an unexpected response (HTTP {response.status_code}).")
        return None

    if not response.ok:
        st.error(result.get("detail", "The backend could not complete the request."))
        return None

    return result


def show_sources(sources):
    if sources:
        with st.expander("View sources"):
            for source in sources:
                st.markdown(f"**{source['document']} — page {source['page']}**")
                st.caption(source["snippet"])


st.set_page_config(page_title="RAG Document Assistant", page_icon="📚", layout="wide")
st.title("📚 RAG Document Assistant")

with st.sidebar:
    st.header("Upload documents")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])
    if st.button("Index PDF", disabled=uploaded_file is None):
        with st.spinner("Indexing PDF..."):
            result = backend_post(
                "upload",
                files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
            )
        if result is not None:
            st.success(f"Indexed {result['filename']} ({result['chunks_indexed']} chunks).")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        show_sources(message.get("sources"))

query = st.chat_input("Ask a question about your documents")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            result = backend_post("ask", json={"query": query})
        if result is not None:
            st.markdown(result["answer"])
            show_sources(result["sources"])
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
            })
