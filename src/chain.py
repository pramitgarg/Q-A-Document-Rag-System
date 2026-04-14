import os
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import weaviate


# Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "localhost")
WEAVIATE_PORT = int(os.getenv("WEAVIATE_PORT", "8080"))
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:3b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
COLLECTION_NAME = "Documents"


def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


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


def get_qa_chain():
    """Build and return the RAG QA chain using Ollama + LLM (LCEL).

    Performance optimizations applied:
    - num_ctx=2048: Reduced context window for faster processing
    - num_predict=512: Cap output length to prevent verbose responses
    - num_thread: Use available CPU cores for parallel inference
    - k=4 retriever: Balanced relevance vs. speed
    """

    # Initializing the embeddings (same model used during ingestion)
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )

    # Connect to Weaviate and load the existing vector store
    client = get_weaviate_client()
    db = WeaviateVectorStore(
        client=client,
        index_name=COLLECTION_NAME,
        embedding=embeddings,
        text_key="text",
    )

    # [OPT-5] Retriever with optimized chunk count (4 instead of 5)
    retriever = db.as_retriever(
        search_type='similarity',
        search_kwargs={'k': 4},
    )

    # [OPT-3, OPT-4, OPT-7] LLM with performance tuning
    llm = OllamaLLM(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
        num_ctx=2048,        # OPT-3: Smaller context window = faster
        num_predict=512,     # OPT-4: Cap output tokens
        num_thread=4,        # OPT-7: Use multiple CPU threads
    )

    # Creating prompt template for LLM
    template = """You are a helpful AI assistant that answers questions based on the user's uploaded documents.

Below are relevant passages extracted from the user's uploaded PDF documents. Use ONLY the information in these passages to answer the question. Do not say you cannot access files — the text below IS the content from those files.

--- START OF DOCUMENT PASSAGES ---
{context}
--- END OF DOCUMENT PASSAGES ---

Question: {question}

Instructions:
- Answer the question thoroughly using the document passages above.
- If the passages contain relevant information, summarize and explain it clearly.
- Use bullet points or numbered lists when appropriate for readability.
- If the passages truly do not contain information related to the question, say "The uploaded documents don't contain information about this topic."
- NEVER say you cannot access or view files. The passages above are already extracted from the user's files.
- Keep your answer concise and focused.

Answer:"""

    prompt = PromptTemplate(
        input_variables=['context', 'question'],
        template=template,
    )

    # Build RAG chain using LCEL (LangChain Expression Language)
    qa_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return qa_chain, retriever
