# 🤖 Q/A Document Rag System

A fully local, privacy-first **RAG (Retrieval-Augmented Generation)** chatbot that answers questions from your PDF documents. Built with LangChain, Ollama, Weaviate, and Gradio — runs entirely on your machine with **zero API keys or cloud dependencies**.

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)
![LangChain](https://img.shields.io/badge/LangChain-LCEL-green)
![Weaviate](https://img.shields.io/badge/Weaviate-VectorDB-FF6F61)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## ✨ Features

- **100% Local** — No OpenAI, no cloud APIs. Everything runs on your machine via Ollama
- **RAG Pipeline** — Upload PDFs, auto-chunk, embed, and retrieve relevant context for every question
- **Document Upload** — Upload up to 3 PDF files directly from the UI (drag & drop or click)
- **Streaming Responses** — Real-time token-by-token answer delivery (no waiting for full response)
- **Weaviate Vector DB** — Production-grade vector database running as a Docker container
- **Dockerized** — One command to spin up the entire stack (Ollama + Weaviate + FastAPI + Gradio)
- **Gradio Chat UI** — Premium dark glassmorphism UI at `localhost:7860`
- **REST API** — FastAPI backend with Swagger docs at `localhost:8000/docs`
- **Performance Optimized** — Model keep-alive, context window tuning, output capping, and thread optimization

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                       Docker Compose                         │
│                                                              │
│  ┌──────────────┐  ┌─────────────┐  ┌──────────────────────┐│
│  │   Ollama      │  │  Weaviate   │  │    App Container     ││
│  │   :11434      │  │   :8080     │  │                      ││
│  │               │  │             │  │  ┌────────┐ ┌──────┐ ││
│  │  ┌──────────┐ │  │  Vector DB  │  │  │FastAPI │ │Gradio│ ││
│  │  │Llama 3.2 │ │  │  (persist)  │  │  │ :8000  │ │:7860 │ ││
│  │  │  (LLM)   │ │  │             │  │  └───┬────┘ └──┬───┘ ││
│  │  └──────────┘ │  └──────┬──────┘  │      │         │     ││
│  │  ┌──────────┐ │         │         │  ┌───▼─────────▼───┐ ││
│  │  │ nomic-   │ │         │         │  │  LangChain LCEL │ ││
│  │  │embed-text│◄├─────────┼─────────│  │  RAG Chain      │ ││
│  │  │ (Embed)  │ │         │         │  └────────┬────────┘ ││
│  │  └──────────┘ │         │         │           │          ││
│  └──────────────┘  ┌───────▼─────┐   │  ┌────────▼────────┐││
│                    │  Weaviate   │◄──│  │   Retriever     │ ││
│                    │  gRPC:50051 │   │  │  (Similarity)   │ ││
│                    └─────────────┘   │  └─────────────────┘ ││
│                                      └──────────────────────┘│
└──────────────────────────────────────────────────────────────┘
```

| Service | Container | Port | Purpose |
|---------|-----------|------|---------|
| **Ollama** | `ollama` | 11434 | Serves Llama 3.2:3b (LLM) + nomic-embed-text (Embeddings) |
| **Weaviate** | `weaviate` | 8080, 50051 | Vector database (HTTP + gRPC) |
| **FastAPI** | `chatbot-app` | 8000 | REST API backend + RAG chain |
| **Gradio** | `chatbot-app` | 7860 | Chat UI frontend |

---

## 📁 Project Structure

```
AI_assistant_RAG/
├── api/
│   ├── __init__.py
│   └── main.py              # FastAPI server (/query, /query/stream, /upload, /health)
├── src/
│   ├── __init__.py
│   ├── chain.py              # LangChain LCEL RAG chain (Llama 3.2 + Weaviate)
│   └── ingest.py             # PDF ingestion pipeline (chunk → embed → store)
├── frontend/
│   └── app.py                # Gradio chat UI (streaming + file upload)
├── Data/
│   └── uploads/              # User-uploaded PDFs (runtime, gitignored)
├── Dockerfile                # Python app container
├── docker-compose.yml        # Orchestrates Ollama + Weaviate + App
├── .dockerignore             # Files excluded from Docker build
├── start.sh                  # Entrypoint script (runs FastAPI + Gradio)
├── requirements.txt          # Python dependencies
├── pyproject.toml            # Project metadata
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- ~4 GB free disk space (for Llama 3.2 model + embedding model)
- ~8 GB RAM recommended (increase Docker memory allocation in settings if needed)

### Step 1: Clone the repository

```bash
git clone https://github.com/pramitgarg/Customer-Support-AI-Chatbot.git
cd Customer-Support-AI-Chatbot
```

### Step 2: Start all services

```bash
docker compose up -d --build
```

This will:
1. Pull the `ollama/ollama` and `weaviate` Docker images
2. Start the Ollama server and Weaviate vector database
3. Auto-pull the **Llama 3.2:3b** (LLM) and **nomic-embed-text** (embeddings) models
4. Build and start the FastAPI + Gradio app

> ⏳ **First run takes 3-5 minutes** to download models (~2.3 GB total). Subsequent starts are instant.

### Step 3: Upload documents & chat

1. Open the **Chat UI** at [http://localhost:7860](http://localhost:7860)
2. Upload up to **3 PDF files** using the upload section
3. Click **"Upload & Index Documents"** — documents are chunked, embedded, and indexed automatically
4. Start chatting! Responses stream in real-time

No manual ingestion commands needed — everything happens through the UI.

---

## 🔌 API Reference

### `GET /health`

Returns the health status of the API and whether the QA chain is loaded.

```json
{
  "status": "healthy",
  "chain_loaded": true
}
```

### `POST /upload`

Upload up to 3 PDF files for ingestion into Weaviate.

**Request:** `multipart/form-data` with `files` field

**Response:**
```json
{
  "message": "Successfully uploaded and ingested 1 document(s).",
  "files": ["document.pdf"],
  "total_chunks": 25
}
```

### `POST /query`

Ask a question against the loaded documents (non-streaming).

**Request:**
```json
{
  "question": "Summarize this document"
}
```

**Response:**
```json
{
  "answer": "Based on the document passages, ...",
  "sources": ["Data/uploads/document.pdf"]
}
```

### `POST /query/stream`

Ask a question with **streaming response** (tokens delivered in real-time).

**Request:** Same as `/query`
**Response:** `text/plain` stream of tokens

---

## 🐳 Docker Commands

```bash
# Start everything
docker compose up -d --build

# View logs (all services)
docker compose logs -f

# View only app logs
docker compose logs -f app

# Stop everything
docker compose down

# Full reset (removes volumes, models, and data)
docker compose down -v
```

---

## 🧠 How It Works

### RAG Pipeline

1. **Upload & Ingestion** (`frontend/app.py` → `api/main.py` → `src/ingest.py`)
   - User uploads PDF files via the Gradio UI
   - Files are saved to `Data/uploads/` and processed
   - Splits into 400-character chunks with 50-character overlap
   - Generates embeddings using `nomic-embed-text` via Ollama
   - Stores vectors in Weaviate (persisted via Docker volume)

2. **Retrieval + Generation** (`src/chain.py`)
   - User question → embedded using `nomic-embed-text`
   - Top-4 most similar chunks retrieved from Weaviate
   - Retrieved context + question → sent to `Llama 3.2:3b` LLM
   - LLM generates a grounded answer using [LangChain LCEL](https://python.langchain.com/docs/concepts/lcel/)

3. **Streaming Delivery** (`api/main.py` → `frontend/app.py`)
   - Tokens stream from LLM → FastAPI → Gradio UI in real-time
   - User sees the response being typed out (first token in ~2-3s)

### Models Used

| Model | Size | Purpose |
|-------|------|---------|
| **Llama 3.2:3b** | 2.0 GB | Text generation (answering questions) |
| **nomic-embed-text** | 274 MB | Text embeddings (vector search) |

### Performance Optimizations

| Optimization | Setting | Impact |
|---|---|---|
| Streaming responses | `/query/stream` endpoint | First token in ~2-3s |
| Model keep-alive | `OLLAMA_KEEP_ALIVE=-1` | No cold-start delay |
| Context window | `num_ctx=2048` | ~10-20% faster |
| Output cap | `num_predict=512` | Prevents verbose answers |
| CPU threading | `num_thread=4` | ~5-15% faster |
| Chunk optimization | `chunk_size=400`, `k=4` | Less input tokens |

---

## 🛠️ Tech Stack

| Technology | Purpose |
|-----------|---------|
| [LangChain](https://www.langchain.com/) | RAG orchestration (LCEL chains) |
| [Ollama](https://ollama.com/) | Local LLM serving |
| [Llama 3.2](https://ai.meta.com/llama/) | Language model for generation |
| [Weaviate](https://weaviate.io/) | Vector database |
| [FastAPI](https://fastapi.tiangolo.com/) | REST API backend |
| [Gradio](https://www.gradio.app/) | Chat UI frontend |
| [Docker](https://www.docker.com/) | Containerization & deployment |

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Pramit Garg**

---

## 🙏 Acknowledgments

- [Ollama](https://ollama.com/) for making local LLM inference simple
- [LangChain](https://www.langchain.com/) for the RAG framework
- [Weaviate](https://weaviate.io/) for the production-grade vector database
- [Meta Llama](https://ai.meta.com/llama/) for the open-source language model
