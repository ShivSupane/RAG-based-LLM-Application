import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# 🔥 IMPROVED PROMPT (SMART HYBRID)
PROMPT = ChatPromptTemplate.from_template("""
You are a helpful AI assistant.

- If the context is relevant to the question, use it.
- If the context is irrelevant or empty, IGNORE it and answer using your own knowledge.

Context:
{context}

Question:
{question}

Answer:
""")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


class RAGPipeline:
    def __init__(self, docs_dir: str = "docs"):
        self.docs_dir = Path(docs_dir)

        # ✅ Embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        # ✅ Groq LLM
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",  # or mixtral if needed
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0
        )

        self.vectorstore = None

    def load_and_index(self):
        print(f"📂 Loading documents from '{self.docs_dir}/'...")

        loader = DirectoryLoader(
            str(self.docs_dir),
            glob="**/*.txt",
            loader_cls=TextLoader
        )

        documents = loader.load()

        if not documents:
            raise ValueError("❌ No .txt files found in docs folder")

        print(f"✅ Loaded {len(documents)} document(s)")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        chunks = splitter.split_documents(documents)
        print(f"✂️ Split into {len(chunks)} chunks")

        print("🔢 Creating FAISS index...")
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)

        print("✅ RAG Ready!\n")

    def ask(self, question: str):
        if not self.vectorstore:
            raise RuntimeError("❌ Run load_and_index() first")

        # 🔥 SMART RETRIEVAL WITH SCORE
        docs_and_scores = self.vectorstore.similarity_search_with_score(
            question, k=4
        )

        # 🔥 FILTER RELEVANT DOCS ONLY
        relevant_docs = []
        for doc, score in docs_and_scores:
            # Lower score = better match
            if score < 0.7:   # 🔥 KEY FIX
                relevant_docs.append(doc)

        # DEBUG (optional)
        # print("DEBUG:", relevant_docs)

        # 🔥 IF NO GOOD DOCS → USE GENERAL LLM
        if not relevant_docs:
            print("⚠️ No relevant docs found → using general knowledge\n")
            return self.llm.invoke(question).content

        # 🔥 ELSE USE RAG
        context = format_docs(relevant_docs)

        chain = PROMPT | self.llm | StrOutputParser()

        return chain.invoke({
            "context": context,
            "question": question
        })