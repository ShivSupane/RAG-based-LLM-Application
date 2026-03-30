"""
Gradio Web UI for RAG LLM
Run locally:  python gradio_app.py
"""

import os
from pathlib import Path
import gradio as gr
from dotenv import load_dotenv
from src.rag import RAGPipeline

load_dotenv()

# Global pipeline
rag = None


# 🔧 Build Index Function
def build_index(api_key, files, model):
    global rag

    if not api_key.strip():
        return "❌ Please enter your Groq API key."
    if not files:
        return "❌ Please upload at least one .txt file."

    os.environ["GROQ_API_KEY"] = api_key.strip()

    # Save files
    docs_path = Path("docs")
    docs_path.mkdir(exist_ok=True)

    for f in files:
        file_path = docs_path / f.name
        with open(file_path, "wb") as out:
            out.write(f.read())

    try:
        rag = RAGPipeline(docs_dir="docs", model=model)
        rag.load_and_index()
        return f"✅ Index built from {len(files)} file(s). Start chatting!"
    except Exception as e:
        return f"❌ Error: {str(e)}"


# 💬 Chat Function
def chat(message, history):
    if rag is None:
        return "⚠️ Please build the index first."

    try:
        return rag.ask(message)
    except Exception as e:
        return f"❌ Error: {str(e)}"


# 🖥️ UI
with gr.Blocks(title="RAG LLM") as demo:

    gr.Markdown("# 🤖 RAG LLM — Chat With Your Documents")
    gr.Markdown("Upload `.txt` files → Build Index → Ask questions")

    with gr.Row():

        # LEFT PANEL
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Setup")

            api_key_box = gr.Textbox(
                label="Groq API Key",
                placeholder="gsk_...",
                type="password"
            )

            model_dropdown = gr.Dropdown(
                label="Model",
                choices=[
                    "llama-3.1-8b-instant",
                    "llama-3.3-70b-versatile",
                    "mixtral-8x7b-32768",
                ],
                value="llama-3.1-8b-instant",
            )

            file_upload = gr.File(
                label="Upload .txt Documents",
                file_types=[".txt"],
                file_count="multiple"
            )

            build_btn = gr.Button("🔢 Build Index", variant="primary")

            status_box = gr.Textbox(
                label="Status",
                value="Waiting for setup...",
                interactive=False
            )

        # RIGHT PANEL
        with gr.Column(scale=2):
            gr.Markdown("### 💬 Chat")

            chat_ui = gr.ChatInterface(
                fn=chat,
                chatbot=gr.Chatbot(height=500),
                examples=[
                    "What is this document about?",
                    "Summarize the content",
                    "Explain AI in simple terms",
                    "How to become an AI engineer?"
                ]
            )

    # Button Action
    build_btn.click(
        fn=build_index,
        inputs=[api_key_box, file_upload, model_dropdown],
        outputs=[status_box],
    )


# 🚀 Run App
if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())