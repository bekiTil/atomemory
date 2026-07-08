"""Tune RECONCILE_MIN_SIM against the eval set.

The write-side gate threshold is embedder-dependent: too high and same-attribute
updates never reach the LLM (stale facts linger); too low and distinct facts get
over-merged. This sweeps a few values and reports accuracy + how compact the
resulting memory is (fewer active facts on UPDATE/dedup = cleaner collapse),
so the default can be chosen from data rather than guessed.

Run:  LLM_BACKEND=groq python eval/tune.py 0.4 0.5 0.6 0.7
"""

from __future__ import annotations

import json
import os
import sys

from atomir.config import settings
from atomir.providers import EmbedderFactory, LLMFactory

sys.path.insert(0, os.path.dirname(__file__))
from run_eval import evaluate, pct  # noqa: E402


def main() -> None:
    values = [float(x) for x in sys.argv[1:]] or [0.4, 0.5, 0.6, 0.7]
    cases = json.load(open(os.path.join(os.path.dirname(__file__), "eval_set.json")))
    llm = LLMFactory.create(settings.llm)
    emb = EmbedderFactory.create(settings.embedder)
    print(f"sweeping RECONCILE_MIN_SIM over {values} "
          f"(llm={settings.llm_backend}, embedder={settings.embed_backend})\n")

    print(f"{'min_sim':<10}{'accuracy':<12}{'active_facts':<14}add_p50")
    print("-" * 44)
    best = None
    for v in values:
        out = evaluate(cases, llm, emb, v, answers=False)
        rows = out["rows"]
        acc = sum(r["passed"] for r in rows) / len(rows)
        total_active = sum(r["n_active"] for r in rows)  # lower = more collapse/dedup
        add_p50 = pct(out["add_lat"], 50)
        print(f"{v:<10}{f'{100*acc:.0f}% ({sum(r['passed'] for r in rows)}/{len(rows)})':<12}"
              f"{total_active:<14}{add_p50:.0f}ms")
        # Best = highest accuracy, tie-broken by most compact memory.
        key = (acc, -total_active)
        if best is None or key > best[0]:
            best = (key, v)
    print("-" * 44)
    print(f"recommended RECONCILE_MIN_SIM = {best[1]}")


if __name__ == "__main__":
    main()
