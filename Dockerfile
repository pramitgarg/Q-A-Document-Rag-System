# ==========================================
# Customer Support AI Chatbot - Dockerfile
# Uses Qwen 3.5 via Ollama for local LLM
# ==========================================
FROM python:3.13-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OLLAMA_BASE_URL=http://ollama:11434 \
    WEAVIATE_URL=weaviate \
    WEAVIATE_PORT=8080 \
    WEAVIATE_GRPC_PORT=50051 \
    LLM_MODEL=llama3.2:3b \
    EMBEDDING_MODEL=nomic-embed-text

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first (for Docker cache optimization)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Create uploads directory
RUN mkdir -p Data/uploads

# Expose ports: FastAPI (8000) + Gradio (7860)
EXPOSE 8000 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Copy startup script
COPY start.sh .
RUN chmod +x start.sh

# Run both FastAPI + Gradio
CMD ["./start.sh"]
