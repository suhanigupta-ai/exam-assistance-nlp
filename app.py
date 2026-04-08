import PyPDF2
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import logging
import streamlit as st

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

pdf_path = "notes.pdf"

# -------------------------------
# Step 1: Extract text from PDF
# -------------------------------
def extract_text(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    return text


# -------------------------------
# Step 2: Clean text
# -------------------------------
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9.,!? ]', '', text)
    return text


# -------------------------------
# Step 3: Chunk text
# -------------------------------
def chunk_text(text, chunk_size=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks


# -------------------------------
# Step 4: Generate answer
# -------------------------------
def generate_answer(results, marks):
    context = " ".join(results)
    
    # Word limits instead of character limits
    limits = {"1": 50, "2": 100, "3": 150, "5": 250, "10": 400}
    word_limit = limits.get(marks, len(context.split()))
    
    words = context.split()
    trimmed = " ".join(words[:word_limit])
    
    # End at last complete sentence
    for punct in ['. ', '? ', '! ']:
        last = trimmed.rfind(punct)
        if last != -1:
            trimmed = trimmed[:last + 1]
            break
    
    return trimmed


# -------------------------------
# Load everything once using cache
# -------------------------------
@st.cache_resource
def load_system():
    raw_text = extract_text(pdf_path)
    cleaned_text = clean_text(raw_text)
    chunks = chunk_text(cleaned_text)

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = np.array(model.encode(chunks)).astype('float32')

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return model, index, chunks


# -------------------------------
# Search function
# -------------------------------
def search(query, model, index, chunks, k=3):
    query_vector = np.array(model.encode([query])).astype('float32')
    distances, indices = index.search(query_vector, k)
    results = [chunks[i] for i in indices[0]]
    return results


# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Exam Prep Chatbot", page_icon="📚")
st.title("📚 Exam Preparation Chatbot")
st.caption("Ask questions from your notes and get answers based on marks!")

# Load system
with st.spinner("Loading your notes..."):
    model, index, chunks = load_system()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Marks selector
marks = st.selectbox("Select marks for answer length:", ["1", "2", "3", "5", "10"])

# Chat input
if prompt := st.chat_input("Ask your question here..."):

    # Show user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate answer
    results = search(prompt, model, index, chunks)
    answer = generate_answer(results, marks)

    # Show bot message
    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})