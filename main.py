import os
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv

load_dotenv()
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/harshitawork/Documents/new-project/credentials.json'
genai.configure()

working_dir = os.path.dirname(os.path.abspath(__file__))

def setup_vector_store():
    persist_directory = f"{working_dir}/vector_db"
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma(persist_directory=persist_directory,
    embedding_function = embeddings)

    return vectorstore

# create chain
def chat_chain(vectorstore):
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    retriever = vectorstore.as_retriever()

    
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        model, retriever, contextualize_q_prompt
    )

    system_prompt = (
        "You are an assistant for question-answering tasks."
        "Use the following pieces of retrieved context to answer the question."
        "When answering, provide as much relevant detail as possible from the available context, but avoid unnecessary information."
        "If the user's question is unclear, ask for clarification before proceeding to ensure you give the correct the correct answer."
        "For syllabus-related questions, return only the full syllabus as it appears in the provided context. Do not elaborate on each topic unless explicitly requested by the user."
        "If the information requested is not available in the context, respond with: 'Answer is not available in the provided context.'"
        "Avoid making assumptions or giving incorrect answers.\n\n"
        "Context: {context}\n"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(model, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain

st.set_page_config(
    page_title = "Multichat chatbot",
    page_icon="📑",
    layout="centered"
)

st.title("📑 Multi-document RAG chatbot")

# session state in streamlit
# when the user is using the app, that time all the history will be stored there, but as soon as the user presses refresh the history of the last session will be lost and new session will be created.
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = setup_vector_store()

if "conversational_chain" not in st.session_state:
    st.session_state.conversational_chain = chat_chain(st.session_state.vectorstore)


# here we are displaying the history of all the messages in one session
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("How can I help you today?")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response = st.session_state.conversational_chain.invoke({"input": user_input, "chat_history": st.session_state.chat_history})
        assistant_response = response["answer"]
        st.markdown(assistant_response)

        print(response["context"])
        print("\n\n")
       

        # Add the assistant's response to the chat history
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})