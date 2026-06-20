import sys
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import streamlit as st
import google.generativeai as genai
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings

st.set_page_config(page_title="RAG BASED AI CHATBOT", layout="wide")

api_key = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-2.5-flash')

EMBED_MODEL = "models/gemini-embedding-001"


class GeminiEmbeddingFunction(EmbeddingFunction):
    """Wraps Gemini's embedding API so chromadb can call it directly."""

    def __init__(self, task_type="retrieval_document"):
        self.task_type = task_type

    def __call__(self, input: Documents) -> Embeddings:
        result = genai.embed_content(
            model=EMBED_MODEL,
            content=input,
            task_type=self.task_type,
        )
        return result["embedding"]


def chunk_text(text, chunk_size=500, overlap=50):
    """Naive sliding-window chunker. Swap for a smarter splitter
    (e.g. split on section headers) once campus_data.txt grows."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return [c.strip() for c in chunks if c.strip()]


@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_or_create_collection(
        name="campus_data",
        embedding_function=GeminiEmbeddingFunction(task_type="retrieval_document"),
    )

    if collection.count() == 0:
        try:
            with open("campus_data.txt", "r") as f:
                text = f.read()
        except FileNotFoundError:
            text = ""

        chunks = chunk_text(text)
        if chunks:
            collection.add(
                documents=chunks,
                ids=[f"chunk_{i}" for i in range(len(chunks))],
            )

    return collection

collection = get_collection()
query_embedder = GeminiEmbeddingFunction(task_type="retrieval_query")

st.sidebar.title("Settings")
if st.sidebar.button("Clear History"):
    st.session_state.messages = []
    st.session_state.chat = None
    st.rerun()

st.title("RAG BASED AI CHATBOT")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat" not in st.session_state or st.session_state.chat is None:
    st.session_state.chat = model.start_chat(history=[])

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Ask me anything you want.")

if user_input:
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    query_embedding = query_embedder([user_input])[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=3)
    retrieved_chunks = results["documents"][0] if results["documents"] else []
    context = "\n\n".join(retrieved_chunks)

    prompt = (
        "You are a helpful assistant for VIT Vellore students. "
        "Use the following campus data to answer the question. "
        "If the answer isn't in the data, say you don't know.\n\n"
        f"Campus data:\n{context}\n\n"
        f"Question: {user_input}"
    )

    response = st.session_state.chat.send_message(prompt)

    with st.chat_message("assistant"):
        st.write(response.text)
    st.session_state.messages.append({"role": "assistant", "content": response.text})
