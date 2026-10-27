
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

#---------------------------------------------------------
# Custom Tool
#---------------------------------------------------------

@tool
def retrieve_answer(query):
    """Retrieves the answer to a query from the vector store."""
    similar_docs = vector_store.similarity_search(query, k=3)
    for doc in similar_docs:
        return doc.page_content

#---------------------------------------------------------
# Memory Saver
#---------------------------------------------------------

memory = InMemorySaver()

#---------------------------------------------------------
# LLm Model
#---------------------------------------------------------

llm = ChatGroq(
   model = "openai/gpt-oss-120b",
   temperature = 0.7,
)

#---------------------------------------------------------
# Agent Creation
#---------------------------------------------------------

agent = create_agent(
   model = llm,
   tools = [retrieve_answer],
   checkpointer = memory,
   system_prompt = "You are a helpful assistant that retrieves information from a PDF document to answer questions by calling the retrieve_answer tool.",
)

while True:
      user_input = input("User: ")
      if user_input.lower() in ["exit", "quit"]:
         break
      response = agent.invoke({"messages": [{"role": "user", "content": user_input}]},
             {"configurable": {"thread_id": "user-123"}}
              )
      print(f"Agent: {response}")