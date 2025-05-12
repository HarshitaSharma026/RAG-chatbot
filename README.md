# 🤖 Multidocument RAG-based Conversational Chatbot

This chatbot project is specifically designed for educational institutions to answer students queries. The students can ask any question related to their university / institute and can get accurate real time responses. 

## What does it do?
This chatbot is based on Retrieval Augmented Generation(RAG), which means we can provide certain documents to the chatbot and it can answer any question related to those documents. RAG is an execllent way to design a custom chatbot. 
This chatbot has been developed to serve requests of students of my university, and that is why the knowledge base (documents) that is provided to the chatbot is completely related to my university. The first prototype of the chatbot is able to answer queries related to:
- Academic Calender (Fall and winter 2024-2025)
- General information related to VIT (Vellore Institute of Technology)
- Faculty information (of one school - SCORE)
- Syllabi for Mtech and MCA courses

## How does it work?
The chatbot retrieves relevant documents, processes user inputs while maintaining session history, and formulates accurate responses using the context from the documents. It utilizes a history-aware retriever to maintain the chat flow and provide detailed responses based on prior questions or user references.

## 🧑‍💻 Technologies Used
LangChain, for implementing retrieval-based pipelines.
Chromadb, for document vector storage and similarity search.
Streamlit, for creating the chatbot interface.
Google Generative AI API, for embeddings and question-answering.
Python, for all backend operations.
dotenv, for API key management.

## How to Run the Chatbot
1. Clone the repository
```
git clone https://github.com/HarshitaSharma026/RAG-chatbot.git
```

2. Install Dependencies
```
pip install -r requirements.txt
```

3. Before moving forward, you need to get the Gemini API key from the Google from here: [For creating Gemini API key](https://aistudio.google.com/app/apikey), and get your credentials.json file. Copy and paste this file in your working directory.

4. Vectorize documents
Before running the chatbot, you need to vectorize the documents. To do this, execute the vectorize.py script:
```
python3 vectorize.py
```
This will process and convert the text documents into embeddings, which will be used by the chatbot for retrieving relevant information.

5. Run the main script
```
streamlit run main.py
```
This will launch the chatbot, and you can begin interacting with it.

## What type of question can be asked?
Refer to the text documents in "data" directory to get an idea of what type of questions (from the documents) can be asked.
Enter your query in the input box.
The chatbot will retrieve relevant documents and provide detailed answers based on the context.
The chat history is maintained throughout the session to ensure continuity in conversation.

### Some important point for you 
1. **Langsmith** is used to track each and every interaction with the chatbot.
To enable it, you need to create an account in langsmith, get the API key, and paste it in .env file in the following format: 
```
LANGCHAIN_API_KEY="<your-langchain-api-key>"
LANGCHAIN_PROJECT="<name-of-the-project>"
```
If you wish to go without tracing, go ahead and comment out line 21 and 22 in main.py before running the chatbot.

2. The code from line 23 - line 31 is written to convert the **credentials.json** file to **secrets.toml** file which is a standard used by Streamlit to get your environment variable for deployment. To avoid this conversion creating problems for you, comment out these lines and uncomment line 32. Add the absolute path to your **credentials.json**, and you're good to go!! (Only if you want to deploy the chatbot on streamlit, otherwise ignore this point.)

🎯**This is an ongoing project. I am actively learning and building side by side making it give more accurate answers. So feel free to contribute, report issues, or suggest enhancements to help improve its performance.


💬 This project is licensed for non-commercial use only. If you're interested in using it commercially or collaborating on an open version, feel free to reach out!