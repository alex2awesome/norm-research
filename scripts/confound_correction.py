"""Quantify + correct the L3-inflation (jargon) confound across the full
2-axis classification.

The 5-agent audit established: gpt-5-mini systematically over-rates clusters as
articulability=3 ("needs domain expert") when a strong LLM-judge could actually
apply them (=2), reflexively triggered by technical vocabulary. Endpoints (1, 4)
are reliable; the 2-vs-3 boundary is the biased one.

Audit-validated signature of a SUSPECT (inflated) level-3:
    articulability == 3  AND  indeterminacy <= 2  AND  reasoning_depth <= 3
i.e. a "3" with no structural free parameter (indeterminacy>=3) and no deep
holistic reasoning (reasoning_depth==4) — the only thing making it a 3 is the
"domain expertise" claim, which the audit found is usually spurious jargon.

A SOUND level-3 has reasoning_depth==4 OR indeterminacy>=3.

This script reports raw vs. corrected per-task articulability (suspect-3 -> 2).
The correction is an ESTIMATE; the definitive fix is a v7 re-classification.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path("/Users/spangher/Projects/stanford-research/norm-research")
OUT = ROOT / "outputs" / "analyses"
FULL = OUT / "cluster_2axis_full.jsonl"


def load_dedup():
    seen, rows = set(), []
    for line in FULL.open():
        try:
            r = json.loads(line)
        except Exception:
            continue
        res = r.get("result", {})
        if "_error" in res or not res:
            continue
        key = (r.get("medoid_name", ""), r.get("medoid_description", ""))
        if key in seen:
            continue
        seen.add(key)
        rows.append(r)
    return rows


def is_suspect_l3(res: dict) -> bool:
    return (res.get("articulability") == 3
            and (res.get("indeterminacy") or 0) <= 2
            and (res.get("reasoning_depth") or 0) <= 3)


def main():
    rows = load_dedup()
    by_task = {}
    for r in rows:
        by_task.setdefault(r["task"], []).append(r["result"])

    n_l3 = sum(1 for r in rows if r["result"].get("articulability") == 3)
    n_suspect = sum(1 for r in rows if is_suspect_l3(r["result"]))
    print(f"{len(rows)} clusters | {n_l3} are L3 | {n_suspect} suspect-L3 "
          f"({n_suspect/max(1,n_l3)*100:.0f}% of all L3)\n")

    print("=" * 86)
    print("RAW vs CORRECTED articulability per task (corrected = suspect-L3 reassigned to L2)")
    print("=" * 86)
    print(f"{'task':<24} {'L1':>10} {'L2 raw>corr':>16} {'L3 raw>corr':>16} {'L4':>9}")
    out = []
    for t in sorted(by_task):
        res = by_task[t]
        n = len(res)
        raw = Counter(x.get("articulability") for x in res)
        suspect = sum(1 for x in res if is_suspect_l3(x))
        # corrected: suspect L3 -> L2
        c2_raw, c3_raw = raw.get(2, 0), raw.get(3, 0)
        c2_cor, c3_cor = c2_raw + suspect, c3_raw - suspect
        f = lambda x: f"{x/n*100:.0f}%"
        print(f"{t:<24} {f(raw.get(1,0)):>10} "
              f"{f(c2_raw)+'>'+f(c2_cor):>16} {f(c3_raw)+'>'+f(c3_cor):>16} {f(raw.get(4,0)):>9}")
        out.append({
            "task": t, "n": n,
            "L1_pct": raw.get(1,0)/n*100,
            "L2_raw_pct": c2_raw/n*100, "L2_corrected_pct": c2_cor/n*100,
            "L3_raw_pct": c3_raw/n*100, "L3_corrected_pct": c3_cor/n*100,
            "L4_pct": raw.get(4,0)/n*100,
            "suspect_l3_pct": suspect/n*100,
        })

    df = pd.DataFrame(out)
    df.to_parquet(OUT / "articulability_corrected_per_task.parquet")
    print(f"\nwrote {OUT/'articulability_corrected_per_task.parquet'}")
    print()
    print("KEY: L4 (the tacit-gap signal) is UNAFFECTED by this correction — the audit")
    print("found endpoints reliable. The creative-writing/humor L4 finding stands; only")
    print("the L2/L3 split shifts.")


if __name__ == "__main__":
    main()
