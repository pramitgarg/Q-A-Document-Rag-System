# 🤖 Customer Support AI Chatbot

A fully local, privacy-first **RAG (Retrieval-Augmented Generation)** chatbot that answers questions from your PDF documents. Built with LangChain, Ollama, ChromaDB, and Gradio — runs entirely on your machine with **zero API keys or cloud dependencies**.

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)
![LangChain](https://img.shields.io/badge/LangChain-LCEL-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## ✨ Features

- **100% Local** — No OpenAI, no cloud APIs. Everything runs on your machine via Ollama
- **RAG Pipeline** — Ingests PDFs, chunks them, creates embeddings, and retrieves relevant context for every question
- **Dockerized** — One command to spin up the entire stack (Ollama + FastAPI + Gradio)
- **Gradio Chat UI** — Clean, interactive chat interface at `localhost:7860`
- **REST API** — FastAPI backend with Swagger docs at `localhost:8000/docs`
- **Extensible** — Swap models, add more documents, or plug in a different frontend

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Docker Compose                       │
│                                                         │
│  ┌──────────────┐    ┌──────────────────────────────┐   │
│  │   Ollama      │    │        App Container          │   │
│  │   :11434      │◄───│                              │   │
│  │              │    │  ┌────────┐   ┌───────────┐  │   │
│  │  ┌─────────┐ │    │  │FastAPI │   │  Gradio   │  │   │
│  │  │Qwen 3.5 │ │    │  │ :8000  │   │  :7860    │  │   │
│  │  │  (LLM)  │ │    │  └───┬────┘   └─────┬─────┘  │   │
│  │  └─────────┘ │    │      │              │        │   │
│  │  ┌─────────┐ │    │  ┌───▼──────────────▼─────┐  │   │
│  │  │ nomic-  │ │    │  │    LangChain LCEL      │  │   │
│  │  │embed-txt│ │    │  │    RAG Chain           │  │   │
│  │  │(Embed)  │ │    │  └───────────┬────────────┘  │   │
│  │  └─────────┘ │    │              │               │   │
│  └──────────────┘    │  ┌───────────▼────────────┐  │   │
│                      │  │     ChromaDB            │  │   │
│                      │  │   (Vector Store)        │  │   │
│                      │  └────────────────────────┘  │   │
│                      └──────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

| Service | Container | Port | Purpose |
|---------|-----------|------|---------|
| **Ollama** | `ollama` | 11434 | Serves Qwen 3.5 (LLM) + nomic-embed-text (Embeddings) |
| **FastAPI** | `chatbot-app` | 8000 | REST API backend + RAG chain |
| **Gradio** | `chatbot-app` | 7860 | Chat UI frontend |

---

## 📁 Project Structure

```
AI_assistant_RAG/
├── api/
│   ├── __init__.py
│   └── main.py              # FastAPI server with /query and /health endpoints
├── src/
│   ├── __init__.py
│   ├── chain.py              # LangChain LCEL RAG chain (Qwen + ChromaDB)
│   └── ingest.py             # PDF ingestion pipeline (chunk → embed → store)
├── frontend/
│   └── app.py                # Gradio chat interface
├── Data/
│   └── documents/
│       └── Think_Python.pdf  # Source PDF document
├── vectordb/
│   └── chroma_db/            # ChromaDB persistent storage (auto-generated)
├── Dockerfile                # Python app container
├── docker-compose.yml        # Orchestrates Ollama + App
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
- ~8 GB free disk space (for Qwen 3.5 model + embedding model)
- ~16 GB RAM recommended

### Step 1: Clone the repository

```bash
git clone https://github.com/yourusername/Customer-Support-AI-Chatbot.git
cd Customer-Support-AI-Chatbot
```

### Step 2: Start all services

```bash
docker compose up -d --build
```

This will:
1. Pull the `ollama/ollama` Docker image
2. Start the Ollama server
3. Auto-pull the **Qwen 3.5:9b** (LLM) and **nomic-embed-text** (embeddings) models
4. Build and start the FastAPI + Gradio app

> ⏳ **First run takes 5-10 minutes** to download models (~7 GB total). Subsequent starts are instant.

### Step 3: Ingest your PDF documents

```bash
docker exec chatbot-app python src/ingest.py
```

This loads `Data/documents/Think_Python.pdf`, chunks it into 500-character pieces, creates embeddings, and stores them in ChromaDB.

### Step 4: Restart the app to load the vector database

```bash
docker compose restart app
```

### Step 5: Open the chatbot

- **Chat UI:** [http://localhost:7860](http://localhost:7860)
- **API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check:** [http://localhost:8000/health](http://localhost:8000/health)

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

### `POST /query`

Ask a question against the loaded documents.

**Request:**
```json
{
  "question": "What is a variable in Python?"
}
```

**Response:**
```json
{
  "answer": "A variable is a name that refers to a value...",
  "sources": ["Data/documents/Think_Python.pdf"]
}
```

---

## 🐳 Docker Commands

```bash
# Start everything
docker compose up -d --build

# View logs (all services)
docker compose logs -f

# View only app logs
docker compose logs -f app

# Run PDF ingestion
docker exec chatbot-app python src/ingest.py

# Restart app (after ingestion or code changes)
docker compose restart app

# Stop everything
docker compose down

# Full reset (removes volumes, models, and data)
docker compose down -v
```

---

## 🧠 How It Works

### RAG Pipeline

1. **Ingestion** (`src/ingest.py`)
   - Loads PDF documents using `PyPDFLoader`
   - Splits into 500-character chunks with 50-character overlap
   - Generates embeddings using `nomic-embed-text` via Ollama
   - Stores vectors in ChromaDB (persisted to disk)

2. **Retrieval + Generation** (`src/chain.py`)
   - User question → embedded using `nomic-embed-text`
   - Top-3 most similar chunks retrieved from ChromaDB
   - Retrieved context + question → sent to `Qwen 3.5:9b` LLM
   - LLM generates a grounded answer using [LangChain LCEL](https://python.langchain.com/docs/concepts/lcel/)

3. **Serving** (`api/main.py` + `frontend/app.py`)
   - FastAPI exposes a `/query` REST endpoint
   - Gradio provides a chat UI that calls the FastAPI backend

### Models Used

| Model | Size | Purpose |
|-------|------|---------|
| **Qwen 3.5:9b** | 6.6 GB | Text generation (answering questions) |
| **nomic-embed-text** | 274 MB | Text embeddings (vector search) |

---

## 📄 Adding Your Own Documents

1. Place your PDF files in the `Data/documents/` folder
2. Update the file path in `src/ingest.py` (line 17):
   ```python
   loader = PyPDFLoader('Data/documents/your_file.pdf')
   ```
3. Re-run ingestion:
   ```bash
   docker exec chatbot-app python src/ingest.py
   ```
4. Restart the app:
   ```bash
   docker compose restart app
   ```

---

## 🛠️ Tech Stack

| Technology | Purpose |
|-----------|---------|
| [LangChain](https://www.langchain.com/) | RAG orchestration (LCEL chains) |
| [Ollama](https://ollama.com/) | Local LLM serving |
| [Qwen 3.5](https://huggingface.co/Qwen) | Language model for generation |
| [ChromaDB](https://www.trychroma.com/) | Vector database |
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

- [Think Python](https://greenteapress.com/wp/think-python-2e/) by Allen B. Downey — used as the sample document
- [Ollama](https://ollama.com/) for making local LLM inference simple
- [LangChain](https://www.langchain.com/) for the RAG framework
