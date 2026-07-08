"""Tiny eval harness — the daily driver.

For each case: ingest its history into a FRESH user_id (and a fresh in-memory
store for isolation), then check the resulting ACTIVE facts by substring, and
retrieval by substring. With --answers, also generate an answer from the
retrieved facts and grade it with the LLM `judge`. Reports accuracy AND latency
(p50/p95 per op) so every change shows quality and milliseconds together.

Run (real providers):  LLM_BACKEND=groq python eval/run_eval.py [--answers]
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
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def fresh_service(llm, emb) -> MemoryService:
    store = QdrantMemoryStore(QdrantClient(":memory:"), "eval", settings.embed_dim)
    return MemoryService(store, llm, emb, reconcile_min_sim=settings.reconcile_min_sim)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", action="store_true", help="also generate + judge answers")
    ap.add_argument("--set", default=os.path.join(os.path.dirname(__file__), "eval_set.json"))
    args = ap.parse_args()

    cases = json.load(open(args.set))
    llm = LLMFactory.create(settings.llm)
    emb = EmbedderFactory.create(settings.embedder)
    print(f"config: llm={settings.llm_backend} embedder={settings.embed_backend} "
          f"dim={settings.embed_dim} min_sim={settings.reconcile_min_sim}\n")

    add_lat: list[float] = []
    search_lat: list[float] = []
    rows = []

    for case in cases:
        svc = fresh_service(llm, emb)
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

        if case.get("query"):
            t = time.perf_counter()
            res = svc.search(uid, case["query"])
            search_lat.append((time.perf_counter() - t) * 1000)
            retrieved = " || ".join(r["text"].lower() for r in res["results"])
            for sub in case.get("expect_retrieved", []):
                checks.append((f"retr:{sub}", sub.lower() in retrieved))
            for sub in case.get("expect_not_retrieved", []):
                checks.append((f"no-retr:{sub}", sub.lower() not in retrieved))
            if args.answers and case.get("judge"):
                ctx = "; ".join(r["text"] for r in res["results"])
                answer = llm.chat_text(
                    "Answer the question using ONLY the known facts. If the facts do "
                    "not contain the answer, say you don't know.",
                    f"KNOWN FACTS: {ctx}\nQUESTION: {case['query']}",
                )
                ok, _ = llm.judge(case["judge"], answer)
                checks.append(("judge", ok))
        elif args.answers and case.get("judge"):
            # absence/constraint cases may not have a query -> judge active facts
            ok, _ = llm.judge(case["judge"], active_text)
            checks.append(("judge", ok))

        passed = all(v for _, v in checks)
        failed = [n for n, v in checks if not v]
        rows.append((case["id"], case["type"], passed, failed, active_text))

    # ---- report ----
    print(f"{'case':<28}{'type':<12}{'result':<8}failed checks")
    print("-" * 78)
    n_pass = 0
    for cid, ctype, passed, failed, active in rows:
        n_pass += passed
        mark = "PASS" if passed else "FAIL"
        print(f"{cid:<28}{ctype:<12}{mark:<8}{', '.join(failed)}")
        if not passed:
            print(f"    active facts: {active[:160]}")
    print("-" * 78)
    print(f"accuracy: {n_pass}/{len(rows)} = {100*n_pass/len(rows):.0f}%")
    print(f"add    latency  p50={pct(add_lat,50):.0f}ms  p95={pct(add_lat,95):.0f}ms  (n={len(add_lat)})")
    print(f"search latency  p50={pct(search_lat,50):.0f}ms  p95={pct(search_lat,95):.0f}ms  (n={len(search_lat)})")


if __name__ == "__main__":
    main()
