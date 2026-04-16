# 📄 AI Document Q&A using Endee Vector Database

## 🚀 Overview
This project is an AI-powered Document Question Answering system built using a Retrieval Augmented Generation (RAG) pipeline.

Users can upload documents (PDF), and the system allows them to ask questions. The application retrieves relevant content using semantic search and generates answers based on the document context.

---

## 💡 Features
- 📂 Upload PDF documents
- 🔍 Semantic search using vector embeddings
- 🤖 Context-based answer generation
- ⚡ Fast retrieval using Endee Vector Database
- 🧠 RAG (Retrieval Augmented Generation) pipeline

---

## 🏗️ System Architecture

1. **Document Upload**
   - User uploads PDF file

2. **Text Extraction**
   - Extract text using PyPDF2

3. **Chunking**
   - Split text into smaller chunks

4. **Embedding Generation**
   - Convert text into vector embeddings

5. **Storage (Endee)**
   - Store embeddings in Endee Vector Database

6. **Query Processing**
   - Convert user query into embedding

7. **Retrieval**
   - Fetch top relevant chunks from Endee

8. **Answer Generation**
   - Generate answer using retrieved context

---

## 🛠️ Tech Stack

- Python
- Streamlit (UI)
- Endee (Vector Database)
- Sentence Transformers (Embeddings)
- PyPDF2 (PDF Processing)

---

## 📦 Project Structure
