# RAG Chatbot – Prototype V1

This chatbot project is designed for **educational institutions** to answer student queries. Students can ask questions related to their university/institute and receive accurate, real-time responses.

## 💡 What It Does

This chatbot uses **Retrieval-Augmented Generation (RAG)** to provide contextual answers from a given set of documents. For this version, all documents are specifically tailored to VIT (Vellore Institute of Technology), making it a functional internal assistant for university-related inquiries.

### Current Supported Topics
- 📅 Academic Calendar (Fall and Winter 2024–2025)
- 🏫 General Information about VIT
- 👨‍🏫 Faculty Information (SCOPE School only)
- 📘 Syllabi for M.Tech and MCA Programs

## ⚙️ How It Works

- Documents are converted to text and embedded into a vector database.
- When a student asks a question, the system retrieves the most relevant pieces of text using a vector search.
- A **history-aware retriever** is used to improve multi-turn conversations by incorporating previous chat history.
- The language model uses the retrieved context to generate a final response.

## 📁 Structure Overview
data/ - text files
vector_db/ - vectorized embeddings
main_v1.py - core chatbot logic

## 🚀 Getting Started
See the main project README for setup and instructions.