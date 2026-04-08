import PyPDF2
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import logging

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
# MAIN PIPELINE
# -------------------------------
raw_text = extract_text(pdf_path)
cleaned_text = clean_text(raw_text)
chunks = chunk_text(cleaned_text)

print("Total chunks:", len(chunks))


# -------------------------------
# Step 4: Create embeddings
# -------------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings = model.encode(chunks)

# IMPORTANT FIX (for FAISS)
embeddings = np.array(embeddings).astype('float32')

print("Embedding shape:", embeddings.shape)


# -------------------------------
# Step 5: FAISS Index
# -------------------------------
# Step 5: FAISS Index
try:
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    print("FAISS index created!")
except Exception as e:
    print(f"FAISS ERROR: {e}")
    input("Press Enter to exit...")


# -------------------------------
# Step 6: Search function
# -------------------------------
def search(query, k=3):
    query_vector = model.encode([query])
    query_vector = np.array(query_vector).astype('float32')
    
    distances, indices = index.search(query_vector, k)
    
    results = [chunks[i] for i in indices[0]]
    return results


# -------------------------------
# Step 7: Exam Answer Generator
# -------------------------------
def generate_answer(results, marks):
    context = " ".join(results)
    
    if marks == "1":
        return context[:100]
    
    elif marks == "2":
        return context[:200]
    
    elif marks == "3":
        return context[:300]
    
    elif marks == "5":
        return context[:500]
    
    elif marks == "10":
        return context[:800]
    
    else:
        return context


# -------------------------------
# USER INPUT
# -------------------------------
query = input("\nEnter your question: ")
marks = input("Enter marks (1/2/3/5/10): ")

results = search(query)

print("\n--- Relevant Content ---\n")
for r in results:
    print(r)
    print("-----")


# -------------------------------
# FINAL ANSWER
# -------------------------------
final_answer = generate_answer(results, marks)

print("\n=== EXAM ANSWER ===\n")
print(final_answer)