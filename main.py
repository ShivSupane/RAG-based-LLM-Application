from src.rag import RAGPipeline


def main():
    print("=" * 50)
    print("  🤖  RAG LLM — Chat With Your Documents")
    print("=" * 50)

    rag = RAGPipeline(docs_dir="docs")
    rag.load_and_index()

    print("Type your question (or 'quit' to exit).\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Bye!")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("👋 Bye!")
            break

        print("🤔 Thinking...")
        answer = rag.ask(question)
        print(f"\n🤖 {answer}\n")
        print("-" * 50)


if __name__ == "__main__":
    main()