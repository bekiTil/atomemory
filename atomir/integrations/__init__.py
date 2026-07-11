"""Optional framework integrations for using atomir with agents.

atomir is the MEMORY, not the model: your app/agent reasons, atomir supplies the
relevant facts. The universal pattern is **recall before, remember after**:

    1. recall relevant facts for the incoming message -> inject into the prompt
    2. the LLM/agent produces the answer
    3. remember the turn -> atomir extracts atomic facts and reconciles them

Memory topology is chosen by the `user_id` namespace (see `langgraph.scope`):

    "user:123"                     shared  — the whole crew sees one user memory
    "user:123#agent:researcher"    private — an agent's own scratchpad
    "acme|user:123"                tenant  — multi-tenant hierarchy

Deployment:
    - in-process : build_memory_service()  (direct calls, fastest)
    - as service : uvicorn atomir.api:app + MemoryClient(url) in each agent
                   (distributed agents share one memory; per-user lock keeps
                   concurrent writes safe)

Multi-agent note: agents coordinate through SHARED memory (one writes findings,
another reads them), persisting across runs — not just within graph state. Write
policy: remember confirmed/durable findings only, not every message.

Submodules (`langchain`, `langgraph`) import their framework at load time, so a
framework is pulled in only when you import that module and install the matching
extra (`atomir[langchain]` / `atomir[langgraph]`). The core package and this
package never depend on any framework. Runnable examples live in `examples/`.
"""


def format_memories(results: list[dict]) -> str:
    """Render atomir search results into a bullet list for a prompt."""
    if not results:
        return "(no relevant memories)"
    return "\n".join(f"- {r['text']}" for r in results)
