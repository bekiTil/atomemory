# atomir

Atomic memory infrastructure for agents. **Memory is atomic on both ends:**
atomic facts on write (extract → reconcile), atomic sub-question decomposition
on read (decompose → retrieve per sub-question → compose).

## Status
Built inside-out, one step at a time. Currently: **Step 0 — project skeleton.**

## Design principle: vendor-neutral
The LLM, the embedder, and the vector store are each an **interface** chosen at
runtime by config (`{provider, config}` blocks). No vendor is privileged. The
engine imports only the interfaces — never a provider SDK. Defaults use `fake`
backends so everything runs with **no external keys**.

## Quickstart (skeleton)
```bash
pip install -e .
cp .env.example .env            # optional; defaults run fully offline
python -c "from atomir.config import settings; print(settings)"
```
