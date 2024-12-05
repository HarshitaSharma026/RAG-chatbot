import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader
import google.generativeai as genai


load_dotenv()
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/harshitawork/Documents/new-project/credentials.json'
genai.configure()
# genai.configure(api_key=os.environ["GEMINI_API_KEY"])
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

loader = DirectoryLoader(path="/Users/harshitawork/Documents/new-project/data", glob = "./*.txt", loader_cls =TextLoader)

documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=500)
text_chunks = text_splitter.split_documents(documents)

vectordb = Chroma.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    persist_directory="./vector_db"
)

print("Documents vectorized !!")

query = "when is Riviera 2025?"
matched_docs = vectordb.similarity_search(query)
for ind, doc in enumerate(matched_docs):
    print(f"------------- Document {ind}: \n Context: \n {doc.page_content}")
