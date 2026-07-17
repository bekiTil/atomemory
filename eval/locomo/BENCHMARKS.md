# atomir on LOCOMO — results

A partial, honest evaluation of atomir on the [LOCOMO](https://github.com/snap-research/locomo)
long-conversation memory benchmark, run through
[mem0's memory-benchmarks harness](https://github.com/mem0ai/memory-benchmarks)
via the atomir adapter in this folder.

## Setup

- **Scope:** 2 of 10 conversations (conv 0 + conv 1), **39 questions**.
- **Model (both answerer and judge):** `gpt-4o-mini`.
- **Embedder:** `text-embedding-3-small` (1536-d).
- **Retrieval:** atomir hybrid (dense + BM25, fused with RRF), decomposition on.
- Grading: LLM-as-judge at each retrieval cutoff (the harness default).

## Results (top-50 cutoff)

| Category | Score |
|---|---|
| multi-hop | **11/12 (92%)** |
| open-domain | 4/4 (100%) |
| single-hop | 2/2 (100%) |
| **non-temporal total** | **17/18 (94%)** |
| temporal | 4/21 (19%) |
| **overall** | **21/39 (53%)** |

By cutoff, overall: top-10 48%, top-50 53%, top-200 53%.

## Reading this honestly

- **atomir is strong on multi-hop and factual recall** (94% non-temporal) — the
  atomic decomposition + hybrid retrieval do what they're meant to.
- **atomir has no temporal reasoning.** Temporal questions (19%) are the entire
  gap and, being the majority of these questions, they set the 53% overall
  ceiling. This is an architectural limitation, not a tuning issue.

## What this is NOT

- **Not comparable to mem0's published 92.5%.** That number is on GPT-4o over all
  1,540 questions; this is gpt-4o-mini over 39 questions from 2 conversations.
  Comparing the two directly would be misleading.
- **A subset**, run this small because of API request-per-day limits — not a
  full-benchmark result.

## Reproduce

See [README.md](README.md) for the adapter + run recipe, then aggregate with:

```bash
python eval/locomo/aggregate.py <predicted_dir>
```
