# 🤖 Multidocument RAG-based Conversational Chatbot

This chatbot project is specifically designed for educational institutions to answer students queries. The students can ask any question related to their university / institute and can get accurate real time responses. 

## What does it do?
This chatbot is based on Retrieval Augmented Generation(RAG), which means we can provide certain documents to the chatbot and it can answer any question related to those documents. RAG is an execllent way to design a custom chatbot. 
This chatbot has been developed to serve requests of students of my university, and that is why the knowledge base (documents) that is provided to the chatbot is completely related to my university. The first prototype of the chatbot is able to answer queries related to:
- Academic Calender (Fall and winter 2024-2025)
- General information related to VIT (Vellore Institute of Technology)
- Faculty information (of one school - SCORE)
- Syllabi for Mtech and MCA courses

## How does it works?
The chatbot retrieves relevant documents, processes user inputs while maintaining session history, and formulates accurate responses using the context from the documents. It utilizes a history-aware retriever to maintain the chat flow and provide detailed responses based on prior questions or user references.

## 🧑‍💻 Technologies Used
LangChain, for implementing retrieval-based pipelines.
Chromadb, for document vector storage and similarity search.
Streamlit, for creating the chatbot interface.
Google Generative AI API, for embeddings and question-answering.
Python, for all backend operations.
dotenv, for API key management.
Streamlit community cloud, for deployment

## How to Use the Chatbot
Refer to the text documents in "data" directory to get an idea of what type of questions (from the documents) can be asked.
Enter your query in the input box.
The chatbot will retrieve relevant documents and provide detailed answers based on the context.
The chat history is maintained throughout the session to ensure continuity in conversation.

🎯**This is an ongoing project. I am actively learning and building side by side making it give more accurate answers. So feel free to contribute, report issues, or suggest enhancements to help improve its performance.

