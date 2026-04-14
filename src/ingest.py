# RAG Pipeline — Data Ingestion
# Loads PDFs, chunks them, embeds, and stores in Weaviate
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_weaviate.vectorstores import WeaviateVectorStore
import weaviate


# Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "localhost")
WEAVIATE_PORT = int(os.getenv("WEAVIATE_PORT", "8080"))
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
COLLECTION_NAME = "Documents"


def get_weaviate_client():
    """Create and return a Weaviate client connection."""
    client = weaviate.connect_to_custom(
        http_host=WEAVIATE_URL,
        http_port=WEAVIATE_PORT,
        http_secure=False,
        grpc_host=WEAVIATE_URL,
        grpc_port=WEAVIATE_GRPC_PORT,
        grpc_secure=False,
    )
    return client


def ingest_pdf(file_path: str) -> int:
    """
    Ingest a single PDF file into the Weaviate vector store.

    Args:
        file_path: Path to the PDF file to ingest.

    Returns:
        Number of chunks ingested.
    """
    # Step 1: Load the PDF
    loader = PyPDFLoader(file_path)
    document = loader.load()

    # Step 2: Chunk the data (OPT-5: smaller chunks = faster embedding + retrieval)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    texts = text_splitter.split_documents(document)

    if not texts:
        print(f"⚠️ No text extracted from {file_path}")
        return 0

    # Step 3: Create embeddings using Ollama (local, no API key needed)
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )

    # Step 4: Connect to Weaviate and store embeddings
    client = get_weaviate_client()
    try:
        WeaviateVectorStore.from_documents(
            texts,
            embeddings,
            client=client,
            index_name=COLLECTION_NAME,
        )
        print(f"✅ Ingested {len(texts)} chunks from '{file_path}' into Weaviate")
    finally:
        client.close()

    return len(texts)


if __name__ == "__main__":
    # CLI usage: ingest all PDFs in Data/uploads/ (if any exist)
    uploads_dir = "Data/uploads"
    if os.path.exists(uploads_dir):
        pdf_files = [f for f in os.listdir(uploads_dir) if f.lower().endswith(".pdf")]
        if pdf_files:
            for pdf_file in pdf_files:
                file_path = os.path.join(uploads_dir, pdf_file)
                ingest_pdf(file_path)
        else:
            print("ℹ️ No PDF files found in Data/uploads/. Upload documents via the UI.")
    else:
        print("ℹ️ Data/uploads/ directory not found. Upload documents via the UI.")