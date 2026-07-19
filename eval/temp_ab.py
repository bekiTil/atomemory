"""A/B: does temperature=0 help? Measures accuracy AND run-to-run determinism.

Runs the eval set twice per arm and reports, per arm, the accuracy of each run
plus the number of cases whose verdict FLIPPED between two identical runs. Flips
are the determinism signal: a deterministic planner should produce zero.

    LLM_BACKEND=openai MODEL=gpt-4o-mini EMBED_BACKEND=openai EMBED_DIM=1536 \
    PYTHONPATH=. python eval/temp_ab.py
"""

from __future__ import annotations

import json
import os

from atomir.config import settings
from atomir.providers import EmbedderFactory, LLMFactory
from run_eval import evaluate  # same directory


def build_llm(temp):
    cfg = dict(settings.llm["config"])
    cfg["temperature"] = temp
    return LLMFactory.create({"provider": settings.llm_backend, "config": cfg})


def main() -> None:
    path = os.path.join(os.path.dirname(__file__), "eval_set.json")
    cases = json.load(open(path))
    emb = EmbedderFactory.create(settings.embedder)

    arms = {"temp=0": 0.0, "provider default": None}
    results = {}

    for name, temp in arms.items():
        llm = build_llm(temp)
        runs = []
        for i in range(2):
            out = evaluate(cases, llm, emb, settings.reconcile_min_sim)
            verdicts = {r["id"]: r["passed"] for r in out["rows"]}
            runs.append(verdicts)
            n = sum(verdicts.values())
            print(f"  [{name}] run {i+1}: {n}/{len(verdicts)}")
        flips = [cid for cid in runs[0] if runs[0][cid] != runs[1][cid]]
        results[name] = (runs, flips)

    print("\n" + "=" * 62)
    print(f"{'arm':<20}{'run1':>7}{'run2':>7}{'flips':>8}   flipped cases")
    print("-" * 62)
    for name, (runs, flips) in results.items():
        a, b = sum(runs[0].values()), sum(runs[1].values())
        print(f"{name:<20}{a:>7}{b:>7}{len(flips):>8}   {', '.join(flips) or '-'}")
    print("\nflips = cases that changed verdict between two IDENTICAL runs "
          "(lower is more deterministic)")


if __name__ == "__main__":
    main()
