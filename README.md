# 🎓 RAG Chatbot for Universities

Welcome to the RAG-based Chatbot project — a two-stage prototype built to answer student queries using documents from educational institutions.

This chatbot is designed for **internal use by universities**, allowing students to ask questions about academic schedules, course details, faculty info, and more — all based on custom document knowledge.

---

## 🧩 What is RAG?

**Retrieval-Augmented Generation (RAG)** is a framework that retrieves relevant context from a document store before generating an answer using a language model.

Instead of relying solely on pre-trained knowledge, RAG allows you to **inject your own data** into the chatbot pipeline.

---

## 🔄 Project Evolution

| Prototype | Summary | Techniques Used |
|----------|---------|------------------|
| **V1** | Basic chatbot using RAG on university docs | History-aware retriever, plain vector search |
| **V2** | Enhanced accuracy with advanced RAG components | Query rewriting, Reciprocal Rank Fusion, Few-shot prompting |

Each version is maintained in its own folder and can be tested independently.

---

## 🧪 Paper & Research

To understand the advanced design choices in Prototype V2, refer to the full-length research paper included in the [`/full-length paper`](./paper/advanced_rag_academic_paper.pdf) folder, or refer to shorter version of document explaining thr gist of both versions of the chatbot here: [`/gist`](./paper/RAG_document.pdf)
> 📘 **Chapter 4: Design and Details** in full-length paper, explains the logic and reasoning behind each enhancement.

---

## 📁 Repository Structure
common/ - Shared utilities (vectorization, PDF-to-txt, etc.)
prototype_v1/ - Basic RAG chatbot (first version)
prototype_v2/ - Advanced RAG chatbot (second version)
paper/ - Unpublished research paper for prototype 2, and a short document explaining the gist of the project.
requirements.txt - Python dependencies

---

## 📸 Demo & Usage
Here is the recorded explanation of the project: ([Video](https://www.linkedin.com/feed/update/urn:li:activity:7329031871680389120/))
---

## 👥 Contributions
Interested in collaborating or adapting this for your own university? Feel free to raise an issue or fork the project. Licensing details are in the [LICENSE](./LICENSE.md) file.

---

## 📬 Contact

For suggestions, academic collaboration, or freelance integration:
**Email**: harshita026.sharma@gmail.com  
**GitHub**: https://github.com/HarshitaSharma026