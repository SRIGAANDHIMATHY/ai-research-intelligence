import streamlit as st
import tempfile
import numpy as np
import faiss
import re
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="Lightweight AI Learning Assistant",
    layout="wide"
)

st.title("Smart Learning Assistant")
st.markdown("Upload study material and get structured explanations instantly.")

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()
if "vector_ready" not in st.session_state:
    st.session_state.vector_ready = False

if "current_file" not in st.session_state:
    st.session_state.current_file = None

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
def extract_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content + "\n"
    return clean_text(text)

def chunk_text(text, chunk_size=300):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk.split()) > 50:
            chunks.append(chunk)
    return chunks

@st.cache_resource(show_spinner=False)
def build_vector_store(chunks):
    embeddings = model.encode(chunks)
    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index
def retrieve_context(query, top_k=2):
    query_embedding = model.encode([query]).astype("float32")
    distances, indices = st.session_state.index.search(query_embedding, top_k)

    selected_chunks = [
        st.session_state.chunks[idx]
        for idx in indices[0]
    ]

    return " ".join(selected_chunks)

def format_answer(query, context):

    sentences = re.split(r'(?<=[.!?]) +', context)

    definition = sentences[0] if sentences else context

    key_points = []
    for sentence in sentences[1:6]:
        if len(sentence.split()) > 6:
            key_points.append(sentence.strip())

    explanation = " ".join(sentences[:5])

    important_words = query.split()
    for word in important_words:
        explanation = re.sub(
            rf'\b{word}\b',
            f"**{word}**",
            explanation,
            flags=re.IGNORECASE
        )

    formatted = f"""
## Definition
{definition}

## Key Points
"""
    for point in key_points:
        formatted += f"- {point}\n"

    formatted += f"""
## Explanation
{explanation}

## Exam Insight
Understanding this topic is important for conceptual clarity and exam-based questions.
"""
    return formatted
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:

    if uploaded_file.name != st.session_state.current_file:
        st.session_state.vector_ready = False
        st.session_state.current_file = uploaded_file.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    if not st.session_state.vector_ready:

        with st.spinner("🔎 Processing document..."):

            full_text = extract_text(pdf_path)

            if not full_text:
                st.error("No readable text found.")
                st.stop()

            chunks = chunk_text(full_text)

            if not chunks:
                st.error("Document too small.")
                st.stop()

            index = build_vector_store(chunks)

            st.session_state.index = index
            st.session_state.chunks = chunks
            st.session_state.vector_ready = True

        st.success("--> Document indexed successfully!")

    st.divider()

    query = st.text_input("Ask any question about the document")

    if st.button("Generate Answer") and query:

        with st.spinner("Generating structured explanation..."):
            context = retrieve_context(query)
            answer = format_answer(query, context)

        st.markdown(answer)

else:
    st.info("Upload a PDF to begin.")
