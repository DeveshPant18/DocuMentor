import streamlit as st
import os
import shutil
import time
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain.chains import create_retrieval_chain
from langchain_core.messages import AIMessage, HumanMessage

# Import the new function from rag_core
from src.rag_core import create_rag_chain

# --- App Configuration ---
st.set_page_config(page_title="DocuMentor", page_icon="🎓", layout="wide")

st.title("🎓 DocuMentor")

# --- Helper Functions ---
def clear_chat_history():
    """Clears the chat history and the RAG chain."""
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Upload some documents and I'll be ready to help you study."}
    ]
    st.session_state.chat_history = []
    if 'rag_chain' in st.session_state:
        del st.session_state['rag_chain']
    if os.path.exists("temp_docs"):
        shutil.rmtree("temp_docs")
    if os.path.exists("vector_db"):
        shutil.rmtree("vector_db")
    st.rerun()

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Upload some documents and I'll be ready to help you study."}
    ]
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# --- Sidebar for File Upload and Controls ---
with st.sidebar:
    st.header("📚 Your Documents")
    uploaded_files = st.file_uploader(
        "Upload your PDF files here and click 'Process'",
        accept_multiple_files=True,
        type="pdf"
    )

    if st.button("Process Documents"):
        if not uploaded_files:
            st.warning("Please upload at least one PDF file.")
        else:
            # Create a temporary directory to store uploaded files
            temp_dir = "temp_docs"
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            
            for uploaded_file in uploaded_files:
                with open(os.path.join(temp_dir, uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())

            # --- Ingestion Process with Progress Indicators ---
            with st.spinner("⏳ Processing documents... This may take a moment."):
                # 1. Load documents
                progress_bar = st.progress(0, text="Loading documents...")
                loader = PyPDFDirectoryLoader(temp_dir)
                docs = loader.load()
                time.sleep(0.5) # For UX
                
                # 2. Setup splitters and stores
                progress_bar.progress(25, text="Structuring content...")
                parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
                child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                vectorstore = Chroma(collection_name="doc_mentor_ai", embedding_function=embeddings)
                store = InMemoryStore()
                time.sleep(0.5)

                # 3. Create and configure the retriever
                progress_bar.progress(50, text="Building knowledge base...")
                retriever = ParentDocumentRetriever(
                    vectorstore=vectorstore,
                    docstore=store,
                    child_splitter=child_splitter,
                    parent_splitter=parent_splitter,
                )
                retriever.add_documents(docs)
                time.sleep(0.5)

                # 4. Create the RAG chain and store it in session state
                progress_bar.progress(75, text="Creating AI assistant...")
                st.session_state.rag_chain = create_rag_chain(retriever)
                time.sleep(0.5)
                
                progress_bar.progress(100, text="✅ Assistant is ready!")
                progress_bar.empty()
            
            st.success("Documents processed successfully! You can now ask questions.")

    st.divider()
    st.button('Clear Conversation', on_click=clear_chat_history, use_container_width=True)


# --- Main Chat Interface ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    if 'rag_chain' not in st.session_state:
        st.warning("Please process some documents first before asking questions.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("🧠 Thinking..."):
                response = st.session_state.rag_chain.invoke(
                    {"input": prompt, "chat_history": st.session_state.chat_history}
                )
                answer = response.get('answer', 'Sorry, I encountered an error.')
                st.write(answer)
                
                # Update chat history
                st.session_state.chat_history.extend([
                    HumanMessage(content=prompt),
                    AIMessage(content=answer)
                ])
                st.session_state.messages.append({"role": "assistant", "content": answer})