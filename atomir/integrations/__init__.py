"""Optional framework integrations.

Each submodule imports its framework at module load, so it's only pulled in when
you import it (and install the matching extra, e.g. `atomir[langchain]`). The
core package never depends on any framework.
"""


def format_memories(results: list[dict]) -> str:
    """Render atomir search results into a bullet list for a prompt."""
    if not results:
        return "(no relevant memories)"
    return "\n".join(f"- {r['text']}" for r in results)
