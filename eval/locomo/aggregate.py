"""Aggregate LOCOMO per-question result files into per-category accuracy.

The benchmark writes one JSON per answered question as it goes, so this works on
a PARTIAL run too — you get the multi-hop / temporal / open-domain numbers across
whatever conversations completed, even if the run died or ran out of credit.

    python eval/locomo/aggregate.py <predicted_dir>
"""

from __future__ import annotations

import collections
import glob
import json
import sys

d = sys.argv[1] if len(sys.argv) > 1 else "results/locomo/predicted_atomir-full"
files = sorted(glob.glob(f"{d}/conv*_q*.json"))

overall: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
by_cat: dict[str, dict[str, list[int]]] = collections.defaultdict(
    lambda: collections.defaultdict(lambda: [0, 0])
)
convs: set = set()

for p in files:
    try:
        x = json.load(open(p))
    except Exception:
        continue
    convs.add(x.get("conversation_idx"))
    cat = x.get("category_name", "?")
    for cut, r in (x.get("cutoff_results") or {}).items():
        ok = 1 if str(r.get("judgment", "")).upper() == "CORRECT" else 0
        overall[cut][0] += ok
        overall[cut][1] += 1
        by_cat[cut][cat][0] += ok
        by_cat[cut][cat][1] += 1

print(f"dir: {d}")
print(f"conversations with data: {sorted(c for c in convs if c is not None)}  "
      f"({len(files)} questions)")
for cut in ("top_10", "top_20", "top_50", "top_200"):
    if cut not in overall:
        continue
    o = overall[cut]
    print(f"\n--- {cut} ---  overall {o[0]}/{o[1]} ({100*o[0]/o[1]:.0f}%)")
    for cat, (c, t) in sorted(by_cat[cut].items()):
        print(f"    {cat:<13} {c}/{t} ({100*c/t:.0f}%)")
