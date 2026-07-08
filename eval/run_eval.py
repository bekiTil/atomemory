"""Tiny eval harness — the daily driver.

For each case: ingest its history into a FRESH user_id (and a fresh in-memory
store for isolation), then grade the resulting ACTIVE facts. Grading is by
substring where that's unambiguous, and by the LLM `judge` (`judge_active`) where
phrasing varies — e.g. UPDATE/DELETE, where a correct memory may keep a negated
or historical mention. Retrieval is graded by substring. With --answers, also
compose an answer from the retrieved facts and grade it. Reports accuracy AND
latency (p50/p95 per op).

Run:   LLM_BACKEND=groq python eval/run_eval.py [--answers] [--min-sim 0.5]
"""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid

from qdrant_client import QdrantClient

from atomir.config import settings
from atomir.memory import MemoryService
from atomir.providers import EmbedderFactory, LLMFactory
from atomir.stores.qdrant_store import QdrantMemoryStore


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def _fresh_service(llm, emb, min_sim: float) -> MemoryService:
    store = QdrantMemoryStore(QdrantClient(":memory:"), "eval", settings.embed_dim)
    return MemoryService(store, llm, emb, reconcile_min_sim=min_sim)


def evaluate(cases, llm, emb, min_sim: float, answers: bool = False) -> dict:
    """Run every case at a given min_sim. Returns rows + latency samples."""
    add_lat: list[float] = []
    search_lat: list[float] = []
    rows = []
    for case in cases:
        svc = _fresh_service(llm, emb, min_sim)
        uid = "eval-" + uuid.uuid4().hex[:8]
        for msg in case.get("history", []):
            t = time.perf_counter()
            svc.add(uid, msg)
            add_lat.append((time.perf_counter() - t) * 1000)

        active = svc.get_all(uid)
        active_text = " || ".join(f["text"].lower() for f in active)
        checks: list[tuple[str, bool]] = []
        for sub in case.get("expect_facts", []):
            checks.append((f"has:{sub}", sub.lower() in active_text))
        for sub in case.get("expect_absent", []):
            checks.append((f"absent:{sub}", sub.lower() not in active_text))
        if "expect_count" in case:
            checks.append((f"count={case['expect_count']}", len(active) == case["expect_count"]))
        if case.get("judge_active"):
            ok, _ = llm.judge(case["judge_active"], active_text or "(no facts)")
            checks.append(("judge_active", ok))

        if case.get("query"):
            t = time.perf_counter()
            res = svc.search(uid, case["query"])
            search_lat.append((time.perf_counter() - t) * 1000)
            retrieved = " || ".join(r["text"].lower() for r in res["results"])
            for sub in case.get("expect_retrieved", []):
                checks.append((f"retr:{sub}", sub.lower() in retrieved))
            for sub in case.get("expect_not_retrieved", []):
                checks.append((f"no-retr:{sub}", sub.lower() not in retrieved))
            if answers and case.get("judge"):
                ctx = "; ".join(r["text"] for r in res["results"]) or "(no relevant facts)"
                ans = llm.chat_text(
                    "Answer using ONLY the known facts; if they don't contain the "
                    "answer, say you don't know.",
                    f"KNOWN FACTS: {ctx}\nQUESTION: {case['query']}",
                )
                ok, _ = llm.judge(case["judge"], ans)
                checks.append(("judge", ok))

        passed = all(v for _, v in checks)
        rows.append({
            "id": case["id"], "type": case["type"], "passed": passed,
            "failed": [n for n, v in checks if not v],
            "active": active_text, "n_active": len(active),
        })
    return {"rows": rows, "add_lat": add_lat, "search_lat": search_lat}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", action="store_true", help="also compose + judge answers")
    ap.add_argument("--min-sim", type=float, default=settings.reconcile_min_sim)
    ap.add_argument("--set", default=os.path.join(os.path.dirname(__file__), "eval_set.json"))
    args = ap.parse_args()

    cases = json.load(open(args.set))
    llm = LLMFactory.create(settings.llm)
    emb = EmbedderFactory.create(settings.embedder)
    print(f"config: llm={settings.llm_backend} embedder={settings.embed_backend} "
          f"dim={settings.embed_dim} min_sim={args.min_sim}\n")

    out = evaluate(cases, llm, emb, args.min_sim, answers=args.answers)
    rows, add_lat, search_lat = out["rows"], out["add_lat"], out["search_lat"]

    print(f"{'case':<28}{'type':<12}{'result':<8}failed checks")
    print("-" * 78)
    n_pass = 0
    for r in rows:
        n_pass += r["passed"]
        print(f"{r['id']:<28}{r['type']:<12}{'PASS' if r['passed'] else 'FAIL':<8}{', '.join(r['failed'])}")
        if not r["passed"]:
            print(f"    active facts: {r['active'][:160]}")
    print("-" * 78)
    print(f"accuracy: {n_pass}/{len(rows)} = {100*n_pass/len(rows):.0f}%")
    print(f"add    latency  p50={pct(add_lat,50):.0f}ms  p95={pct(add_lat,95):.0f}ms  (n={len(add_lat)})")
    print(f"search latency  p50={pct(search_lat,50):.0f}ms  p95={pct(search_lat,95):.0f}ms  (n={len(search_lat)})")


if __name__ == "__main__":
    main()
