import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="RAG Document Assistant", page_icon="📚", layout="wide")
st.title("📚 RAG Document Assistant")

with st.sidebar:
    st.header("Upload documents")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])
    if uploaded_file and st.button("Index PDF"):
        response = requests.post(
            f"{BACKEND_URL}/upload",
            files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
            timeout=120,
        )
        if response.ok:
            st.success(response.json())
        else:
            st.error(response.text)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("View sources"):
                for source in message["sources"]:
                    st.markdown(f"**{source['document']} — page {source['page']}**")
                    st.caption(source["snippet"])

query = st.chat_input("Ask a question about your documents")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/ask",
                    json={"query": query},
                    timeout=60,
                )
                response.raise_for_status()
                result = response.json()
                st.markdown(result["answer"])
                sources = result.get("sources", [])
                if sources:
                    with st.expander("View sources"):
                        for source in sources:
                            st.markdown(f"**{source['document']} — page {source['page']}**")
                            st.caption(source["snippet"])
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": sources,
                })
            except requests.RequestException as exc:
                st.error(f"Could not reach the backend: {exc}")
