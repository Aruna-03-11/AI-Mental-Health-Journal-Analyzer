import streamlit as st
import os
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np

# 👉 Import Endee (make sure you installed it properly)
from endee import Client

# ---------------------------
# Setup
# ---------------------------
st.set_page_config(page_title="AI Document Q&A", layout="wide")
st.title("📄 AI Document Q&A using Endee")

# Initialize embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Endee client
client = Client()
collection = client.get_or_create_collection("documents")

# ---------------------------
# Functions
# ---------------------------

def extract_text_from_pdf(pdf_file):
    text = ""
    reader = PdfReader(pdf_file)
    for page in reader.pages:
        text += page.extract_text()
    return text


def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks


def store_in_endee(chunks):
    embeddings = model.encode(chunks)
    
    for i, chunk in enumerate(chunks):
        collection.add(
            ids=[str(i)],
            embeddings=[embeddings[i].tolist()],
            metadatas=[{"text": chunk}]
        )


def query_endee(question, top_k=3):
    query_embedding = model.encode([question])[0]
    
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )
    
    texts = [item["text"] for item in results["metadatas"][0]]
    return texts


def generate_answer(context, question):
    # Simple response (you can replace with OpenAI API later)
    combined_context = " ".join(context)
    return f"Answer based on document:\n\n{combined_context[:500]}..."


# ---------------------------
# UI
# ---------------------------

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    with st.spinner("Processing document..."):
        text = extract_text_from_pdf(uploaded_file)
        chunks = chunk_text(text)
        store_in_endee(chunks)
    st.success("Document processed and stored in Endee!")

# Question input
question = st.text_input("Ask a question from the document")

if question:
    with st.spinner("Searching answer..."):
        context = query_endee(question)
        answer = generate_answer(context, question)
    
    st.subheader("📌 Answer")
    st.write(answer)

    st.subheader("📚 Retrieved Context")
    for i, ctx in enumerate(context):
        st.write(f"**Chunk {i+1}:** {ctx[:200]}...")
