#!/bin/bash
# Start both FastAPI backend and Gradio frontend

echo "🚀 Starting FastAPI backend on port 8000..."
uvicorn api.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

sleep 3

echo "🎨 Starting Gradio frontend on port 7860..."
python frontend/app.py &
FRONTEND_PID=$!

# Keep the container alive — if the backend dies, exit
wait $BACKEND_PID
exit $?
