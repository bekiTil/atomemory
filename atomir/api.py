"""HTTP surface over the engine (FastAPI). A thin adapter, no business logic.

Endpoints unpack a request, call ONE `MemoryService` method, and return its
result. Handlers are plain `def` (not `async def`) so Starlette runs the
blocking engine work in a threadpool instead of stalling the event loop; the
per-user lock inside `MemoryService` keeps same-user writes correct across those
threads.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from atomir.assembly import build_memory_service
from atomir.config import settings

service = build_memory_service()
app = FastAPI(title="atomir", version="0.4.1")


class AddBody(BaseModel):
    user_id: str
    text: str


class SearchBody(BaseModel):
    user_id: str
    query: str
    k: int = 6
    decompose: bool = True


@app.post("/memories")
def add_memories(body: AddBody) -> dict:
    return service.add(body.user_id, body.text)


@app.post("/search")
def search(body: SearchBody) -> dict:
    return service.search(body.user_id, body.query, k=body.k, decompose=body.decompose)


@app.post("/answer")
def answer(body: SearchBody) -> dict:
    return service.answer(body.user_id, body.query, k=body.k, decompose=body.decompose)


@app.get("/memories")
def get_all(user_id: str = Query(...)) -> list[dict]:
    return service.get_all(user_id)


@app.delete("/memories/{fact_id}")
def delete_one(fact_id: str, user_id: str = Query(...)) -> dict:
    if not service.delete(user_id, fact_id):
        raise HTTPException(status_code=404, detail="fact not found for this user")
    return {"deleted": True, "id": fact_id}


@app.delete("/memories")
def reset(user_id: str = Query(...)) -> dict:
    return {"reset": service.reset(user_id)}


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "store": settings.store_backend,
        "llm": settings.llm_backend,
        "embedder": settings.embed_backend,
    }
