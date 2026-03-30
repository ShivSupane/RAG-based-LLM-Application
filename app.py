"""
Streamlit RAG Chat App
Run: streamlit run app.py
"""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
#MainMenu, header, footer { visibility: hidden; }

.stApp { background: #0f1117; color: #e8e8e8; }

[data-testid="stSidebar"] {
    background: #161b27 !important;
    border-right: 1px solid #1e2535;
}
[data-testid="stSidebar"] * { color: #c9d1e0 !important; }

.top-bar {
    display: flex; align-items: center; gap: 12px;
    padding: 18px 0 10px 0;
    border-bottom: 1px solid #1e2535;
    margin-bottom: 20px;
}
.top-bar .logo {
    width: 38px; height: 38px;
    background: linear-gradient(135deg, #4f8ef7, #7c5ff7);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
}
.top-bar h1 {
    font-size: 20px !important; font-weight: 600 !important;
    color: #f0f2f8 !important; margin: 0 !important; padding: 0 !important;
}
.top-bar .badge {
    background: #1a2540; border: 1px solid #2a3a5c;
    border-radius: 20px; padding: 3px 10px;
    font-size: 11px; color: #4f8ef7 !important;
    font-family: 'DM Mono', monospace;
}

.status-pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: #0d2218; border: 1px solid #1a4a2e;
    border-radius: 20px; padding: 4px 12px;
    font-size: 12px; color: #4caf82 !important; margin-bottom: 16px;
}
.status-dot {
    width: 7px; height: 7px; background: #4caf82;
    border-radius: 50%; animation: pulse 2s infinite;
}
.status-pill-warn {
    display: inline-flex; align-items: center; gap: 6px;
    background: #1a1a0d; border: 1px solid #3a3a1a;
    border-radius: 20px; padding: 4px 12px;
    font-size: 12px; color: #b0a040 !important; margin-bottom: 16px;
}
.status-dot-warn {
    width: 7px; height: 7px; background: #b0a040; border-radius: 50%;
}
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }

[data-testid="stChatMessage"] {
    background: #161b27 !important;
    border: 1px solid #1e2535 !important;
    border-radius: 14px !important;
    padding: 14px 18px !important;
    margin-bottom: 10px !important;
}

.stButton > button {
    background: linear-gradient(135deg, #4f8ef7, #7c5ff7) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    padding: 10px 20px !important; width: 100% !important;
}

[data-testid="stFileUploader"] {
    background: #1a2035 !important;
    border: 1.5px dashed #2a3a5c !important;
    border-radius: 12px !important;
}

.sidebar-section {
    font-size: 10px; font-weight: 600;
    letter-spacing: 1.2px; text-transform: uppercase;
    color: #4f8ef7 !important; margin: 16px 0 8px 0;
}
.doc-chip {
    background: #1a2535; border: 1px solid #2a3a5c;
    border-radius: 8px; padding: 6px 10px;
    font-size: 12px; color: #a0b0cc !important;
    margin-bottom: 5px;
}
.empty-state {
    text-align: center; padding: 60px 20px; color: #4a5568;
}
.empty-state .icon { font-size: 44px; margin-bottom: 14px; }
.empty-state h3 { color: #6b7a99 !important; font-weight: 500; font-size: 17px; }
.empty-state p { font-size: 13px; color: #4a5568; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "rag" not in st.session_state:
    st.session_state.rag = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "ready" not in st.session_state:
    st.session_state.ready = False
if "doc_names" not in st.session_state:
    st.session_state.doc_names = []

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-section">Model</div>', unsafe_allow_html=True)
    st.markdown(
        '<span style="font-family:monospace;font-size:11px;background:#1a2035;'
        'color:#7c5ff7;padding:3px 8px;border-radius:6px;border:1px solid #2a1f50;">'
        'llama-3.1-8b-instant · Groq</span>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="sidebar-section">Knowledge Base</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Drop .txt files here",
        type=["txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if st.button("⚡ Build Index"):
        if not uploaded_files:
            st.warning("Upload at least one .txt file first.")
        else:
            docs_path = Path("docs")
            docs_path.mkdir(exist_ok=True)
            names = []
            for f in uploaded_files:
                (docs_path / f.name).write_bytes(f.read())
                names.append(f.name)

            from src.rag import RAGPipeline
            with st.spinner("Building knowledge base..."):
                rag = RAGPipeline(docs_dir="docs", model="llama-3.1-8b-instant")
                rag.load_and_index()
                st.session_state.rag = rag
                st.session_state.ready = True
                st.session_state.doc_names = names
                st.session_state.messages = []
            st.success("✅ Ready! Start chatting.")

    if st.session_state.doc_names:
        st.markdown('<div class="sidebar-section">Indexed Documents</div>', unsafe_allow_html=True)
        for name in st.session_state.doc_names:
            st.markdown(f'<div class="doc-chip">📄 {name}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.caption("• Upload docs → answers from your files\n• No upload → normal AI chat\n• Add GROQ_API_KEY in .env")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="top-bar">
    <div class="logo">🤖</div>
    <h1>RAG Chat Assistant</h1>
    <span class="badge">LLaMA · FAISS · Groq</span>
</div>
""", unsafe_allow_html=True)

if st.session_state.ready:
    st.markdown(
        f'<div class="status-pill"><div class="status-dot"></div>'
        f'{len(st.session_state.doc_names)} doc(s) indexed — RAG mode active</div>',
        unsafe_allow_html=True
    )
else:
    st.markdown(
        '<div class="status-pill-warn"><div class="status-dot-warn"></div>'
        'No documents — using base AI knowledge</div>',
        unsafe_allow_html=True
    )

if not st.session_state.messages:
    st.markdown("""
    <div class="empty-state">
        <div class="icon">💬</div>
        <h3>Start a conversation</h3>
        <p>Ask anything below. Upload .txt documents in the sidebar<br>to chat with your own files.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

if prompt := st.chat_input("Ask anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if st.session_state.rag:
                response = st.session_state.rag.ask(prompt)
            else:
                from langchain_groq import ChatGroq
                llm = ChatGroq(
                    model="llama-3.1-8b-instant",
                    api_key=os.getenv("GROQ_API_KEY"),
                    temperature=0,
                )
                response = llm.invoke(prompt).content
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})