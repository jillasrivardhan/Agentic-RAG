
import os
import hashlib
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langchain_groq import ChatGroq


# Load environment variables
load_dotenv()

st.set_page_config(
    page_title="Agentic RAG Chatbot",
    page_icon="📚",
    layout="centered"
)

st.title("📚 Agentic RAG Chatbot")
st.write("Upload a PDF and ask questions about its content.")


# Check API key
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("GROQ_API_KEY is missing. Please add it to your .env file.")
    st.stop()


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    st.session_state.agent = None

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "user-123"

if "pdf_id" not in st.session_state:
    st.session_state.pdf_id = None


# PDF upload
uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


# Process the uploaded PDF
if uploaded_file is not None:

    pdf_bytes = uploaded_file.getvalue()
    pdf_id = hashlib.sha256(pdf_bytes).hexdigest()

    # Process only when a new PDF is uploaded
    if st.session_state.pdf_id != pdf_id:

        try:
            with st.spinner("Processing your PDF..."):

                # Save uploaded PDF temporarily
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pdf"
                ) as temp_file:
                    temp_file.write(pdf_bytes)
                    pdf_path = temp_file.name

                try:
                    # Document loading
                    loader = PyPDFLoader(pdf_path)
                    documents = loader.load()

                    if not documents:
                        st.error("The uploaded PDF contains no readable pages.")
                        st.stop()

                    # Text splitting
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=100
                    )
                    chunks = splitter.split_documents(documents)

                    # Embedding model
                    embeddings = OllamaEmbeddings(
                        model="nomic-embed-text:latest"
                    )

                    # Vector store
                    vector_store = Chroma.from_documents(
                        documents=chunks,
                        embedding=embeddings
                    )

                finally:
                    os.remove(pdf_path)

                # Custom retrieval tool
                @tool
                def retrieve_answer(query: str) -> str:
                    """Retrieve relevant information from the uploaded PDF."""
                    similar_docs = vector_store.similarity_search(
                        query, k=3
                    )

                    if not similar_docs:
                        return "No relevant information was found in the PDF."

                    return "\n\n".join(
                        doc.page_content for doc in similar_docs
                    )

                # Memory saver
                memory = InMemorySaver()

                # LLM model
                llm = ChatGroq(
                    model="openai/gpt-oss-120b",
                    temperature=0.7,
                    api_key=api_key
                )

                # Agent creation
                agent = create_agent(
                    model=llm,
                    tools=[retrieve_answer],
                    checkpointer=memory,
                    system_prompt=(
                        "You are a helpful assistant that answers questions "
                        "using the uploaded PDF. Call retrieve_answer to "
                        "retrieve relevant information before answering "
                        "questions about the document. If the PDF does not "
                        "contain the answer, say so clearly."
                    )
                )

                # Save the initialized agent
                st.session_state.agent = agent
                st.session_state.pdf_id = pdf_id

                # Start a fresh conversation for a new PDF
                st.session_state.messages = []
                st.session_state.thread_id = pdf_id

            st.success(f"PDF processed: {uploaded_file.name}")

        except Exception as e:
            st.session_state.agent = None
            st.session_state.pdf_id = None
            st.error(f"Could not process the PDF: {e}")


# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Chat input
if st.session_state.agent is None:
    st.info("Upload a PDF first to start chatting.")
else:
    user_input = st.chat_input("Ask a question about your PDF...")

    if user_input:

        # Save and display the user message
        st.session_state.messages.append(
            {"role": "user", "content": user_input}
        )

        with st.chat_message("user"):
            st.markdown(user_input)

        try:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):

                    response = st.session_state.agent.invoke(
                        {
                            "messages": [
                                {
                                    "role": "user",
                                    "content": user_input
                                }
                            ]
                        },
                        {
                            "configurable": {
                                "thread_id": st.session_state.thread_id
                            }
                        }
                    )

                    # Get the latest AI response
                    answer = response["messages"][-1].content

                    st.markdown(answer)

            # Save AI response to history
            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )

        except Exception as e:
            st.error(
                "The AI request failed. Check your Groq API key, "
                "internet connection, and model availability."
            )
            st.exception(e)


# Clear chat history
if st.session_state.messages:
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()
