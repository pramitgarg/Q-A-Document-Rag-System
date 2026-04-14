"""
Customer Support AI Chatbot — Gradio UI
Connects to the FastAPI backend to serve RAG-powered answers.
Features: Document upload (up to 3 PDFs) + Chat interface with streaming.
Premium dark theme with glassmorphism and animations.
"""
import os
import requests
import gradio as gr

# Backend API URL (points to FastAPI inside Docker, or localhost for dev)
API_URL = os.getenv("API_URL", "http://localhost:8000")


def query_chatbot_stream(message: str, history: list):
    """
    [OPT-1] Stream the response from the FastAPI backend token-by-token.
    Uses the /query/stream endpoint for real-time token delivery.
    Yields partial responses so the user sees words appear in real-time.
    """
    if not message.strip():
        yield history
        return

    # Add user message to history
    history = history + [{"role": "user", "content": message}]
    # Add empty assistant message that we'll fill progressively
    history = history + [{"role": "assistant", "content": ""}]
    yield history

    try:
        # Use streaming endpoint
        response = requests.post(
            f"{API_URL}/query/stream",
            json={"question": message},
            timeout=120,
            stream=True,  # Enable streaming
        )
        response.raise_for_status()

        full_response = ""
        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
            if chunk:
                full_response += chunk
                # Update the last assistant message with accumulated text
                history[-1]["content"] = full_response
                yield history

        # After streaming completes, fetch sources separately
        try:
            source_resp = requests.post(
                f"{API_URL}/query",
                json={"question": message},
                timeout=10,
            )
            if source_resp.ok:
                sources = source_resp.json().get("sources", [])
                if sources:
                    unique_sources = list(set(sources))
                    source_text = "\n\n---\n📄 **Sources:** " + ", ".join(unique_sources)
                    history[-1]["content"] = full_response + source_text
                    yield history
        except Exception:
            pass  # Sources are optional, don't fail the response

    except requests.exceptions.ConnectionError:
        history[-1]["content"] = "❌ Could not connect to the backend. Make sure the API server is running."
        yield history
    except requests.exceptions.Timeout:
        history[-1]["content"] = "⏳ Request timed out. The model may still be loading — please try again."
        yield history
    except requests.exceptions.HTTPError as e:
        error_detail = ""
        try:
            error_detail = e.response.json().get("detail", "")
        except Exception:
            pass
        if error_detail:
            history[-1]["content"] = f"⚠️ {error_detail}"
        else:
            history[-1]["content"] = f"⚠️ Server error: {e.response.status_code}"
        yield history
    except Exception as e:
        history[-1]["content"] = f"⚠️ Unexpected error: {str(e)}"
        yield history


def upload_files(files):
    """Upload PDF files to the FastAPI backend for ingestion."""
    if not files:
        return "⚠️ No files selected. Please choose up to 3 PDF files."

    if len(files) > 3:
        return "⚠️ Maximum 3 files allowed. Please select fewer files."

    try:
        # Prepare multipart file upload
        file_tuples = []
        for file_path in files:
            filename = os.path.basename(file_path)
            if not filename.lower().endswith(".pdf"):
                return f"⚠️ Only PDF files are supported. '{filename}' is not a PDF."
            file_tuples.append(
                ("files", (filename, open(file_path, "rb"), "application/pdf"))
            )

        response = requests.post(
            f"{API_URL}/upload",
            files=file_tuples,
            timeout=300,
        )

        # Close all file handles
        for _, (_, fh, _) in file_tuples:
            fh.close()

        response.raise_for_status()
        data = response.json()

        file_list = ", ".join(data.get("files", []))
        chunks = data.get("total_chunks", 0)
        return (
            f"✅ **Upload successful!**\n\n"
            f"📁 **Files:** {file_list}\n"
            f"🧩 **Total chunks indexed:** {chunks}\n\n"
            f"You can now ask questions about these documents below."
        )

    except requests.exceptions.ConnectionError:
        return "❌ Could not connect to the backend. Make sure the API server is running."
    except requests.exceptions.Timeout:
        return "⏳ Upload timed out. The file may be too large or the server is busy."
    except requests.exceptions.HTTPError as e:
        error_detail = ""
        try:
            error_detail = e.response.json().get("detail", "")
        except Exception:
            pass
        if error_detail:
            return f"⚠️ {error_detail}"
        return f"⚠️ Upload failed with error: {e.response.status_code}"
    except Exception as e:
        return f"⚠️ Unexpected error: {str(e)}"


# ── Premium CSS ──

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ─── Global Reset ─── */
* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

body, .gradio-container {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%) !important;
    min-height: 100vh;
}

.gradio-container {
    max-width: 900px !important;
    margin: 0 auto !important;
    padding: 1rem !important;
}

/* ─── Glass Card Base ─── */
.glass-card {
    background: rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 20px !important;
    padding: 1.8rem !important;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
}

.glass-card:hover {
    border-color: rgba(139, 92, 246, 0.3) !important;
    box-shadow: 0 8px 40px rgba(139, 92, 246, 0.15), 0 8px 32px rgba(0, 0, 0, 0.3) !important;
}

/* ─── Header ─── */
.hero-section {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem 1rem;
    animation: fadeInDown 0.8s ease-out;
}

.hero-section h1 {
    font-size: 2.4rem !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #a78bfa 0%, #818cf8 30%, #6366f1 60%, #c084fc 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    margin-bottom: 0.5rem !important;
    letter-spacing: -0.5px !important;
}

.hero-section p, .hero-section span {
    color: rgba(203, 213, 225, 0.8) !important;
    font-size: 0.92rem !important;
    font-weight: 400 !important;
    letter-spacing: 0.3px !important;
}

/* ─── Badges in header ─── */
.badge-row {
    display: flex;
    justify-content: center;
    gap: 0.6rem;
    margin-top: 1rem;
    flex-wrap: wrap;
}

.badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.35rem 0.85rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.3px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    background: rgba(255, 255, 255, 0.05);
    color: rgba(203, 213, 225, 0.9);
    transition: all 0.3s ease;
}

.badge:hover {
    background: rgba(255, 255, 255, 0.1);
    border-color: rgba(139, 92, 246, 0.4);
    transform: translateY(-1px);
}

/* ─── Section labels ─── */
.section-label {
    color: rgba(203, 213, 225, 0.95) !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    margin-bottom: 0.8rem !important;
    letter-spacing: -0.2px !important;
}

.section-sublabel {
    color: rgba(148, 163, 184, 0.7) !important;
    font-size: 0.85rem !important;
    font-weight: 400 !important;
    margin-bottom: 1rem !important;
}

/* ─── Upload Area ─── */
.upload-zone {
    animation: fadeInUp 0.6s ease-out 0.2s both;
}

.upload-zone .glass-card {
    position: relative;
    overflow: hidden;
}

.upload-zone .glass-card::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: conic-gradient(from 180deg, transparent 0%, rgba(139, 92, 246, 0.06) 25%, transparent 50%);
    animation: rotate 8s linear infinite;
    pointer-events: none;
}

/* File upload component styling */
.upload-zone input[type="file"],
.upload-zone .wrap {
    border: 2px dashed rgba(139, 92, 246, 0.25) !important;
    border-radius: 14px !important;
    background: rgba(139, 92, 246, 0.04) !important;
    transition: all 0.3s ease !important;
}

.upload-zone input[type="file"]:hover,
.upload-zone .wrap:hover {
    border-color: rgba(139, 92, 246, 0.5) !important;
    background: rgba(139, 92, 246, 0.08) !important;
}

/* Upload button */
.upload-btn button {
    background: linear-gradient(135deg, #7c3aed 0%, #6366f1 50%, #818cf8 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.85rem 1.5rem !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.3px !important;
    cursor: pointer !important;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 20px rgba(124, 58, 237, 0.3) !important;
    position: relative;
    overflow: hidden;
}

.upload-btn button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(124, 58, 237, 0.45) !important;
}

.upload-btn button:active {
    transform: translateY(0) !important;
}

/* Status area */
.status-area {
    background: rgba(139, 92, 246, 0.06) !important;
    border: 1px solid rgba(139, 92, 246, 0.15) !important;
    border-radius: 12px !important;
    padding: 1rem 1.2rem !important;
    margin-top: 0.8rem !important;
}

.status-area p, .status-area span {
    color: rgba(203, 213, 225, 0.8) !important;
    font-size: 0.88rem !important;
}

/* ─── Chat Area ─── */
.chat-zone {
    animation: fadeInUp 0.6s ease-out 0.4s both;
}

/* Chatbot messages container */
.chat-zone .chatbot {
    background: rgba(15, 12, 41, 0.4) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 16px !important;
}

/* User messages */
.chat-zone .message.user {
    background: linear-gradient(135deg, rgba(124, 58, 237, 0.35), rgba(99, 102, 241, 0.25)) !important;
    border: 1px solid rgba(139, 92, 246, 0.2) !important;
    border-radius: 16px 16px 4px 16px !important;
    color: #e2e8f0 !important;
}

/* Bot messages */
.chat-zone .message.bot {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 16px 16px 16px 4px !important;
    color: #cbd5e1 !important;
}

/* Message text */
.chat-zone .message p, .chat-zone .message span {
    color: #e2e8f0 !important;
    font-size: 0.92rem !important;
    line-height: 1.65 !important;
}

/* Chat input */
.chat-input textarea {
    background: rgba(255, 255, 255, 0.06) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 14px !important;
    color: #e2e8f0 !important;
    font-size: 0.92rem !important;
    padding: 0.9rem 1.1rem !important;
    transition: all 0.3s ease !important;
}

.chat-input textarea:focus {
    border-color: rgba(139, 92, 246, 0.5) !important;
    box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.1) !important;
    outline: none !important;
}

.chat-input textarea::placeholder {
    color: rgba(148, 163, 184, 0.5) !important;
}

/* Clear button */
.clear-btn button {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px !important;
    color: rgba(203, 213, 225, 0.7) !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    padding: 0.6rem 1.2rem !important;
    transition: all 0.3s ease !important;
    cursor: pointer !important;
}

.clear-btn button:hover {
    background: rgba(239, 68, 68, 0.1) !important;
    border-color: rgba(239, 68, 68, 0.3) !important;
    color: rgba(252, 165, 165, 0.9) !important;
}

/* ─── Divider ─── */
.divider {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent, rgba(139, 92, 246, 0.2), transparent) !important;
    margin: 1.5rem 0 !important;
}

/* ─── Accordion ─── */
.gradio-accordion {
    border: none !important;
    background: transparent !important;
}

.gradio-accordion > .label-wrap {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 14px !important;
    padding: 0.8rem 1rem !important;
    color: rgba(203, 213, 225, 0.9) !important;
    transition: all 0.3s ease !important;
}

.gradio-accordion > .label-wrap:hover {
    background: rgba(255, 255, 255, 0.06) !important;
    border-color: rgba(139, 92, 246, 0.2) !important;
}

/* ─── File component labels ─── */
label, .label span {
    color: rgba(203, 213, 225, 0.8) !important;
    font-weight: 500 !important;
}

/* ─── Scrollbar ─── */
::-webkit-scrollbar {
    width: 6px;
}
::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.02);
}
::-webkit-scrollbar-thumb {
    background: rgba(139, 92, 246, 0.3);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(139, 92, 246, 0.5);
}

/* ─── Keyframes ─── */
@keyframes fadeInDown {
    from { opacity: 0; transform: translateY(-20px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes rotate {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}

@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 20px rgba(139, 92, 246, 0.1); }
    50% { box-shadow: 0 0 40px rgba(139, 92, 246, 0.2); }
}

/* ─── Responsive ─── */
@media (max-width: 768px) {
    .hero-section h1 { font-size: 1.8rem !important; }
    .glass-card { padding: 1.2rem !important; }
    .badge-row { gap: 0.4rem; }
}

/* ─── Footer override ─── */
footer { display: none !important; }

/* ─── Various Gradio overrides for dark theme ─── */
.gradio-container .prose {
    color: rgba(203, 213, 225, 0.8) !important;
}

.gradio-container .prose strong {
    color: rgba(233, 213, 255, 0.95) !important;
}

.gradio-container .prose em {
    color: rgba(148, 163, 184, 0.7) !important;
}

.block {
    background: transparent !important;
    border: none !important;
}

.wrap {
    background: rgba(255, 255, 255, 0.03) !important;
}

.chatbot .placeholder {
    color: rgba(148, 163, 184, 0.5) !important;
}
"""

# ── Gradio UI Layout ──

with gr.Blocks(title="🤖 Customer Support AI Chatbot") as demo:

    # ── Hero Header ──
    with gr.Column(elem_classes="hero-section"):
        gr.HTML("""
            <h1>🤖 Customer Support AI Chatbot</h1>
            <p>Intelligent document Q&A powered by local AI</p>
            <div class="badge-row">
                <span class="badge">⚡ Llama 3.2</span>
                <span class="badge">🧠 RAG Pipeline</span>
                <span class="badge">🔒 100% Local</span>
                <span class="badge">🗄️ Weaviate</span>
                <span class="badge">🔥 Streaming</span>
            </div>
        """)

    # ── Upload Section ──
    with gr.Column(elem_classes="upload-zone"):
        with gr.Column(elem_classes="glass-card"):
            gr.HTML('<p class="section-label">📁 Upload Documents</p>')
            gr.HTML('<p class="section-sublabel">Upload up to <strong>3 PDF files</strong> to build your knowledge base. Documents are chunked, embedded, and indexed automatically.</p>')

            file_input = gr.File(
                label="Drop PDFs here or click to browse",
                file_count="multiple",
                file_types=[".pdf"],
                type="filepath",
            )

            with gr.Column(elem_classes="upload-btn"):
                upload_btn = gr.Button(
                    "📤  Upload & Index Documents",
                    variant="primary",
                    size="lg",
                )

            with gr.Column(elem_classes="status-area"):
                upload_status = gr.Markdown(
                    value="*Waiting for documents...*",
                )

            upload_btn.click(
                fn=upload_files,
                inputs=[file_input],
                outputs=[upload_status],
            )

    # ── Divider ──
    gr.HTML('<hr class="divider">')

    # ── Chat Section ──
    with gr.Column(elem_classes="chat-zone"):
        with gr.Column(elem_classes="glass-card"):
            gr.HTML('<p class="section-label">💬 Chat with your Documents</p>')
            gr.HTML('<p class="section-sublabel">Ask anything about your uploaded documents. Responses stream in real-time as the AI generates them.</p>')

            chatbot_display = gr.Chatbot(
                height=420,
                placeholder="Upload documents above, then start chatting...",
                show_label=False,
            )

            with gr.Column(elem_classes="chat-input"):
                msg_input = gr.Textbox(
                    placeholder="Ask a question about your documents...",
                    container=False,
                    show_label=False,
                    lines=1,
                    max_lines=3,
                )

            # [OPT-1] Streaming: use the generator function for real-time output
            msg_input.submit(
                fn=query_chatbot_stream,
                inputs=[msg_input, chatbot_display],
                outputs=[chatbot_display],
            ).then(
                fn=lambda: "",
                outputs=[msg_input],
            )

            with gr.Row(elem_classes="clear-btn"):
                clear_btn = gr.Button("🗑️  Clear Conversation", size="sm")
                clear_btn.click(lambda: ([], ""), outputs=[chatbot_display, msg_input])


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        css=custom_css,
    )