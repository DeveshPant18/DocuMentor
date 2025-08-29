import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever

# Define paths
DOCUMENTS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'documents')
VECTOR_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'vector_db')
DOCSTORE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'docstore') # We don't use this, but retriever needs it

def ingest_data():
    """
    Loads docs, creates a ParentDocumentRetriever-compatible vector store.
    It splits docs into large "parent" chunks and small "child" chunks.
    """
    print("🚀 Starting data ingestion with ParentDocumentRetriever setup...")

    # 1. Load documents
    loader = PyPDFDirectoryLoader(DOCUMENTS_PATH)
    docs = loader.load()
    if not docs:
        print("⚠️ No documents found. Please add PDF files to the 'documents' folder.")
        return
    print(f"✅ Loaded {len(docs)} document(s).")

    # 2. Define Splitters
    # This splitter creates the "parent" documents
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
    # This splitter creates the smaller "child" documents used for searching
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

    # 3. Create the vector store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        collection_name="split_parents",
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embeddings,
    )

    # 4. Create the document store for the parents
    store = InMemoryStore()

    # 5. Initialize the ParentDocumentRetriever
    # This is the main tool that handles splitting and storing
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    # 6. Add documents to the retriever
    # This automatically handles the splitting, embedding, and storing
    print("Adding documents to the retriever...")
    retriever.add_documents(docs, ids=None)
    print("✅ Documents added successfully.")
    
    # Note: The vectorstore is persisted automatically by Chroma.
    # The in-memory docstore is not persisted, so we rebuild it on app startup.
    # For production, you'd use a persistent store like Redis or SQL.
    
    print(f"✅ Vector database is ready at: {VECTOR_DB_PATH}")
    print("🎉 Ingestion complete!")

if __name__ == '__main__':
    ingest_data()