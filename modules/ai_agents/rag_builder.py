# =============================================
#              MODULE IMPORTS
# =============================================

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# =============================================
#              ARCHITECTURE
# =============================================

def build_vector_database():
    print("Initializing RAG Ingestion Pipeline...")
    
    # 1. Path Routing
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent.parent
    
    pdf_path = project_root / "data" / "knowledge_base" / "IEGC_Code.pdf"
    vector_db_path = project_root / "data" / "knowledge_base" / "faiss_index"
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"Could not find the IEGC document at {pdf_path}")

    # 2. Load and Chunk the Document
    print("Loading and chunking the grid code document...")
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()
    
    # We use a 200-character overlap to ensure protocol sentences aren't cut in half
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    print(f"Document split into {len(chunks)} operational chunks.")

    # 3. Generate Embeddings 
    # Using a fast, lightweight open-source embedding model that runs easily on CPU
    print("Generating mathematical embeddings (this may take a minute)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 4. Build and Save the FAISS Vector Database
    print("Building FAISS Vector Index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(vector_db_path))
    
    print(f"Success! Vector database saved locally to: {vector_db_path}")


# ==================================================
#              TESTING & EXECUTION
# ==================================================

if __name__ == "__main__":
    build_vector_database()