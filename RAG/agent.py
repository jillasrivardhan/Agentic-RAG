
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langchain_groq import ChatGroq

# ---------------------------------------------------------
# Document Loading
# ---------------------------------------------------------

loader = PyPDFLoader("M:\\Agentic-RAG\\Data\\LLM.pdf")
documents = loader.load()
# len(documents)

# ---------------------------------------------------------
# Text Splitting
# ---------------------------------------------------------

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(documents)
# len(chunks)

# ---------------------------------------------------------
# Embedding Model
# ---------------------------------------------------------

embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

#---------------------------------------------------------
# Vector Store
#---------------------------------------------------------

vector_store = Chroma.from_documents(
   documents=chunks, 
   embedding=embeddings,
   )

similar = vector_store.similarity_search("What is the difference between a language model and a large language model?", k=3)

print(similar[0].page_content)