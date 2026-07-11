"""LangGraph integration for atomir — ready-made recall/remember nodes.

These are plain callables matching LangGraph's node signature (state -> partial
state update), so this module imports NO framework: drop the nodes into any
StateGraph. `scope()` builds consistent user_id namespaces for multi-agent /
multi-tenant memory. Works with a MemoryService or a MemoryClient.

`pip install atomir[langgraph]` pulls langgraph itself; the integration code here
depends only on atomir.

Example:
    from langgraph.graph import StateGraph, START, END
    from atomir.assembly import build_memory_service
    from atomir.integrations.langgraph import recall_node, remember_node, scope

    mem = build_memory_service()
    g = StateGraph(dict)
    g.add_node("recall", recall_node(mem))       # -> state["memories"]
    g.add_node("agent", my_agent)                # reads state["memories"]
    g.add_node("remember", remember_node(mem))   # stores state["input"]
    g.add_edge(START, "recall"); g.add_edge("recall", "agent")
    g.add_edge("agent", "remember"); g.add_edge("remember", END)
    app = g.compile()
"""

from __future__ import annotations

from typing import Any, Callable

from atomir.integrations import format_memories

Node = Callable[[dict], dict]


def scope(user: str, agent: str | None = None, tenant: str | None = None) -> str:
    """Build a consistent atomir user_id namespace for memory topology.

    scope("u1")                     -> "u1"                    (shared user memory)
    scope("u1", agent="researcher") -> "u1#agent:researcher"   (agent-private)
    scope("u1", tenant="acme")      -> "acme|u1"               (multi-tenant)
    """
    ns = f"{tenant}|{user}" if tenant else user
    return f"{ns}#agent:{agent}" if agent else ns


def recall_node(
    memory: Any,
    *,
    user_id_key: str = "user_id",
    query_key: str = "input",
    output_key: str = "memories",
    k: int = 6,
    decompose: bool = True,
    as_text: bool = True,
) -> Node:
    """Node that searches memory for state[query_key] into state[output_key].

    as_text=True (default) writes a prompt-ready bullet string; False writes the
    raw list of result dicts.
    """

    def _recall(state: dict) -> dict:
        res = memory.search(state[user_id_key], state[query_key], k=k, decompose=decompose)
        results = res.get("results", [])
        return {output_key: format_memories(results) if as_text else results}

    return _recall


def remember_node(
    memory: Any,
    *,
    user_id_key: str = "user_id",
    text_key: str = "input",
    output_key: str | None = None,
) -> Node:
    """Node that extracts + reconciles state[text_key] into memory.

    Returns {} by default (no state change); set output_key to also record the
    add() result (operations + facts) into state.
    """

    def _remember(state: dict) -> dict:
        result = memory.add(state[user_id_key], state[text_key])
        return {output_key: result} if output_key else {}

    return _remember
