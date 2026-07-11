"""LangGraph multi-agent: two agents coordinating through SHARED atomir memory.

A researcher writes findings to shared memory; a writer reads them back — the
coordination happens through persistent facts, not just graph state. Runs
offline (fake backends, stub agents), no keys needed.

    pip install "atomir[langgraph]"
    python examples/langgraph_multiagent.py
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from atomir.assembly import build_memory_service
from atomir.config import Settings
from atomir.integrations.langgraph import recall_node, scope

mem = build_memory_service(
    Settings(llm_backend="fake", embed_backend="fake",
             store_backend="json", store_path="./example_store.json")
)

USER = scope("user:123")  # shared namespace for the whole crew


class State(TypedDict):
    user_id: str
    input: str
    memories: str
    research: str
    draft: str


def researcher(state: State) -> dict:
    # A confirmed finding — written ONCE to shared memory (write policy: store
    # durable findings, not chatter). The writer will read it back below.
    finding = "The user's manager is Dana Lopez and they work on Project Atlas."
    mem.add(state["user_id"], finding)
    return {"research": finding}


def writer(state: State) -> dict:
    # Coordinates with the researcher purely through SHARED memory, not state.
    hits = mem.search(state["user_id"], "who leads the project?")
    facts = "; ".join(r["text"] for r in hits["results"])
    return {"draft": f"Draft email addressed using: {facts}"}


def build_app():
    g = StateGraph(State)
    g.add_node("recall", recall_node(mem))          # shared recall feeds the crew
    g.add_node("researcher", researcher)
    g.add_node("writer", writer)
    g.add_edge(START, "recall")
    g.add_edge("recall", "researcher")
    g.add_edge("researcher", "writer")
    g.add_edge("writer", END)
    return g.compile()


if __name__ == "__main__":
    mem.reset(USER)
    out = build_app().invoke({"user_id": USER, "input": "help me email my project lead"})
    print("research:", out["research"])
    print("draft   :", out["draft"])
    print("memory  :", [f["text"] for f in mem.get_all(USER)])
