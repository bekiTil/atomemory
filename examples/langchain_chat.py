"""LangChain: use atomir as memory (recall before, remember after).

Runs offline with atomir's fake backends and a stub model, so it needs no keys.
Swap `stub_llm` for a real ChatGroq/ChatOpenAI to make it a real chatbot.

    pip install "atomir[langchain]"
    python examples/langchain_chat.py
"""

from atomir.assembly import build_memory_service
from atomir.config import Settings
from atomir.integrations.langchain import AtomirMemory


def stub_llm(system: str, user: str) -> str:
    # Replace with a real LangChain chat model. Here we just echo the context.
    return f"(using memory)\n{system}\n\nUser asked: {user}"


def main() -> None:
    # Offline config so the example runs with no external keys.
    svc = build_memory_service(
        Settings(llm_backend="fake", embed_backend="fake",
                 store_backend="json", store_path="./example_store.json")
    )
    mem = AtomirMemory(svc, user_id="user:123")
    mem.memory.reset("user:123")

    # Seed some memory (in a real app this is your users' past messages).
    mem.remember("I'm vegetarian. My manager is Dana Lopez. I work on Project Atlas.")

    # A turn: recall -> answer -> remember.
    question = "who is my manager?"
    context = mem.recall(question)                       # atomir retrieval
    answer = stub_llm(f"Known facts:\n{context}", question)
    mem.remember(question)

    print("QUESTION:", question)
    print("RECALLED:\n" + context)
    print("ANSWER:", answer)


if __name__ == "__main__":
    main()
