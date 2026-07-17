"""Path A: atomir vs mem0 on the SAME eval_set, SAME LLM + embedder + judge.

Fairness: both systems use one Groq LLM (extraction/reasoning) and one shared
embedder; both are graded by the same LLM judge. The only variable is the memory
engine. mem0 doesn't support Jina/Voyage, so the shared embedder defaults to
Ollama (free, local) — run `ollama pull nomic-embed-text` first.

Setup (one time):
    pip install mem0ai ollama                    # mem0 + its ollama client lib
    # install Ollama from https://ollama.com/download, then:
    ollama pull nomic-embed-text                 # the shared embedder model
    # Groq is the shared LLM (free tier works); no embedder key needed.

Run:
    LLM_API_KEY=gsk_... PYTHONPATH=. python eval/compare_mem0.py [--answers] [--limit N]

Start with --limit 2 to smoke-test cheaply before the full set.

Store is Qdrant for both (atomir in-memory; mem0 local path). expect_count is
skipped — fact granularity differs across engines, so it's not a fair check.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
import uuid

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# COMPARE_PROVIDER=openai uses gpt-4o-mini + text-embedding-3-small for BOTH
# engines (paid, no free-tier caps). Default 'groq' uses Groq LLM + Ollama embed.
PROVIDER = os.environ.get("COMPARE_PROVIDER", "groq")
OLLAMA = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

if PROVIDER == "openai":
    LLM_KEY = os.environ.get("OPENAI_API_KEY", "")
    LLM_MODEL = os.environ.get("MODEL", "gpt-4o-mini")
    EMB_MODEL = os.environ.get("COMPARE_EMBED_MODEL", "text-embedding-3-small")
    EMB_DIM = int(os.environ.get("EMBED_DIM", "1536"))
else:
    LLM_KEY = os.environ.get("LLM_API_KEY") or os.environ.get("GROQ_API_KEY", "")
    LLM_MODEL = os.environ.get("MODEL", "llama-3.3-70b-versatile")
    EMB_MODEL = os.environ.get("COMPARE_EMBED_MODEL", "nomic-embed-text")
    EMB_DIM = int(os.environ.get("EMBED_DIM", "768"))


def pct(xs, p):
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def build_llm():
    if PROVIDER == "openai":
        from atomir.llm.openai import OpenAILLM
        return OpenAILLM(api_key=LLM_KEY, model=LLM_MODEL)
    from atomir.llm.groq import GroqLLM
    return GroqLLM(api_key=LLM_KEY, model=LLM_MODEL)


def build_embedder():
    if PROVIDER == "openai":
        from atomir.embeddings.openai import OpenAIEmbedder
        return OpenAIEmbedder(api_key=LLM_KEY, embed_dim=EMB_DIM, model=EMB_MODEL)
    from atomir.embeddings.ollama import OllamaEmbedder
    return OllamaEmbedder(embed_dim=EMB_DIM, model=EMB_MODEL, base_url=OLLAMA)


def make_atomir(llm, emb):
    """Fresh atomir MemoryService (in-memory Qdrant)."""
    from qdrant_client import QdrantClient
    from atomir.memory import MemoryService
    from atomir.stores.qdrant_store import QdrantMemoryStore
    store = QdrantMemoryStore(QdrantClient(":memory:"), "atomir_cmp", EMB_DIM)
    return MemoryService(store, llm, emb)


class Mem0Memory:
    """Adapter giving mem0 the same add/search/get_all/reset shape as atomir."""

    def __init__(self) -> None:
        from mem0 import Memory
        if PROVIDER == "openai":
            os.environ.setdefault("OPENAI_API_KEY", LLM_KEY)
            llm_cfg = {"provider": "openai", "config": {"model": LLM_MODEL, "api_key": LLM_KEY}}
            emb_cfg = {"provider": "openai", "config": {"model": EMB_MODEL, "api_key": LLM_KEY}}
        else:
            llm_cfg = {"provider": "groq", "config": {"model": LLM_MODEL, "api_key": LLM_KEY}}
            emb_cfg = {"provider": "ollama", "config": {
                "model": EMB_MODEL, "ollama_base_url": OLLAMA, "embedding_dims": EMB_DIM}}
        config = {
            "llm": llm_cfg,
            "embedder": emb_cfg,
            "vector_store": {"provider": "qdrant", "config": {
                "collection_name": "mem0_cmp", "path": tempfile.mkdtemp(),
                "embedding_model_dims": EMB_DIM, "on_disk": False}},
        }
        self.m = Memory.from_config(config)

    def add(self, user_id, text):
        return self.m.add(text, user_id=user_id, infer=True)

    def search(self, user_id, query, k=6, decompose=True):
        res = self.m.search(query, filters={"user_id": user_id}, top_k=k, threshold=0.0)
        rows = res.get("results", []) if isinstance(res, dict) else res
        return {"results": [{"text": r.get("memory", ""), "id": r.get("id"),
                             "score": r.get("score")} for r in rows]}

    def get_all(self, user_id):
        res = self.m.get_all(filters={"user_id": user_id})
        rows = res.get("results", []) if isinstance(res, dict) else res
        return [{"text": r.get("memory", "")} for r in rows]


def grade_case(case, memory, judge, answers):
    """Same grading for both engines. Returns (passed, add_latencies, search_latencies)."""
    uid = "cmp-" + uuid.uuid4().hex[:8]
    add_lat, search_lat, checks = [], [], []

    for msg in case.get("history", []):
        t = time.perf_counter()
        memory.add(uid, msg)
        add_lat.append((time.perf_counter() - t) * 1000)

    active = " || ".join(f["text"].lower() for f in memory.get_all(uid))
    for sub in case.get("expect_facts", []):
        checks.append(sub.lower() in active)
    for sub in case.get("expect_absent", []):
        checks.append(sub.lower() not in active)
    if case.get("judge_active"):
        ok, _ = judge.judge(case["judge_active"], active or "(none)")
        checks.append(ok)

    if case.get("query"):
        t = time.perf_counter()
        res = memory.search(uid, case["query"])
        search_lat.append((time.perf_counter() - t) * 1000)
        retr = " || ".join(r["text"].lower() for r in res["results"])
        for sub in case.get("expect_retrieved", []):
            checks.append(sub.lower() in retr)
        for sub in case.get("expect_not_retrieved", []):
            checks.append(sub.lower() not in retr)
        if answers and case.get("judge"):
            ctx = "; ".join(r["text"] for r in res["results"]) or "(none)"
            ans = judge.chat_text(
                "Answer using ONLY the known facts; if they don't contain it, say you don't know.",
                f"FACTS: {ctx}\nQUESTION: {case['query']}")
            ok, _ = judge.judge(case["judge"], ans)
            checks.append(ok)

    return all(checks), add_lat, search_lat


def run(name, cases, memory, judge, answers):
    passed, done, errors, add_lat, search_lat = 0, 0, 0, [], []
    for case in cases:
        try:
            ok, a, s = grade_case(case, memory, judge, answers)
        except Exception as e:  # e.g. rate limit — record and keep going
            errors += 1
            print(f"  [{name}] {case['id']:<26} ERROR: {str(e)[:70]}")
            continue
        done += 1
        passed += ok
        add_lat += a
        search_lat += s
        print(f"  [{name}] {case['id']:<26} {'PASS' if ok else 'FAIL'}")
    return {"name": name, "passed": passed, "done": done, "errors": errors,
            "total": len(cases), "add": add_lat, "search": search_lat}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="run only the first N cases")
    ap.add_argument("--set", default=os.path.join(os.path.dirname(__file__), "eval_set.json"))
    args = ap.parse_args()

    cases = json.load(open(args.set))
    # expect_count is engine-specific; drop it for a fair cross-engine comparison.
    for c in cases:
        c.pop("expect_count", None)
    if args.limit:
        cases = cases[: args.limit]

    judge = build_llm()
    emb = build_embedder()
    print(f"shared config [{PROVIDER}]: llm={LLM_MODEL}  embedder={EMB_MODEL}({EMB_DIM})\n")

    print("== atomir ==")
    r_atomir = run("atomir", cases, make_atomir(judge, emb), judge, args.answers)
    print("\n== mem0 ==")
    try:
        r_mem0 = run("mem0", cases, Mem0Memory(), judge, args.answers)
    except Exception as e:
        print("mem0 setup failed:", str(e)[:120])
        r_mem0 = {"name": "mem0", "passed": 0, "done": 0, "errors": len(cases),
                  "total": len(cases), "add": [], "search": []}

    print("\n" + "=" * 66)
    print(f"{'system':<10}{'accuracy (of completed)':<26}{'add p50/p95':<18}search p50/p95")
    print("-" * 66)
    for r in (r_atomir, r_mem0):
        d = r["done"] or 1
        acc = f"{r['passed']}/{r['done']} ({100*r['passed']//d}%)  +{r['errors']} err"
        add = f"{pct(r['add'],50):.0f}/{pct(r['add'],95):.0f}ms"
        srch = f"{pct(r['search'],50):.0f}/{pct(r['search'],95):.0f}ms"
        print(f"{r['name']:<10}{acc:<26}{add:<18}{srch}")


if __name__ == "__main__":
    main()
