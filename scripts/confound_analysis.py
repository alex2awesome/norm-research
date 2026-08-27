"""Structural-confound hunt on the 2-axis classification.

Checks whether the per-task articulability picture is skewed by structural
properties rather than the task itself:
  A. source orientation (formal_guideline / blog_post / research_article / ...)
  B. medoid-description length (verbosity confound)
  C. cluster size (multi-member clusters)
  D. whether the L4 (tacit) finding is itself a source-type artifact

Joins each classified cluster back to its source page(s) via member rubric keys
-> page_id -> pages_df.orientation.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/spangher/Projects/stanford-research/norm-research")
HIER = ROOT / "outputs" / "hierarchy"
OUT = ROOT / "outputs" / "analyses"
FULL = OUT / "cluster_2axis_full.jsonl"
PAGES = ROOT / "notebooks" / "_explore_cache" / "pages.parquet"

TASKS = ["code-review", "creative-writing", "grant-funding", "humor",
         "legal-outcome-prediction", "math-stackexchange", "news-homepages",
         "notice-and-comment", "patents", "peer-review", "press-releases"]


def load_classified():
    seen, by_key = set(), {}
    for line in FULL.open():
        try:
            r = json.loads(line)
        except Exception:
            continue
        res = r.get("result", {})
        if "_error" in res or not res:
            continue
        k = (r.get("medoid_name", ""), r.get("medoid_description", ""))
        if k in seen:
            continue
        seen.add(k)
        by_key[k] = r
    return by_key


def page_id_of(key: str) -> str:
    parts = key.split("::")
    return "::".join(parts[:3]) if len(parts) >= 3 else ""


def main():
    classified = load_classified()
    pages = pd.read_parquet(PAGES)
    orient = dict(zip(pages.page_id, pages.orientation))

    # join: enumerate R1-refined clusters, attach orientation + classification
    recs = []
    for task in TASKS:
        p = HIER / f"{task}_general_r1_refined.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        for par in d.get("parented_trees", []):
            for ch in par.get("children", []):
                k = (ch.get("medoid_name", ""), ch.get("medoid_description", ""))
                if k not in classified:
                    continue
                res = classified[k]["result"]
                os_ = [orient.get(page_id_of(r.get("key", ""))) for r in ch.get("rubrics", [])]
                os_ = [o for o in os_ if o]
                modal = Counter(os_).most_common(1)[0][0] if os_ else "unknown"
                recs.append({
                    "task": task,
                    "articulability": res.get("articulability"),
                    "reasoning_depth": res.get("reasoning_depth"),
                    "indeterminacy": res.get("indeterminacy"),
                    "orientation": modal,
                    "desc_len": len(ch.get("medoid_description", "") or ""),
                    "cluster_size": len(ch.get("rubrics", [])),
                })
    df = pd.DataFrame(recs).drop_duplicates(
        subset=["task", "articulability", "desc_len", "cluster_size", "orientation"])
    print(f"joined {len(df)} clusters to source orientation\n")

    def artic_dist(sub):
        n = len(sub)
        c = sub.articulability.value_counts()
        return " ".join(f"L{k}:{c.get(k,0)/n*100:>3.0f}%" for k in (1, 2, 3, 4)) + f"  (n={n})"

    # A. articulability by orientation
    print("=" * 80)
    print("A. ARTICULABILITY by SOURCE ORIENTATION (pooled across tasks)")
    print("=" * 80)
    for o, sub in sorted(df.groupby("orientation"), key=lambda kv: -len(kv[1])):
        if len(sub) < 100:
            continue
        print(f"  {o:<24} {artic_dist(sub)}")

    # B. orientation mix per task
    print("\n" + "=" * 80)
    print("B. ORIENTATION MIX per task (% of clusters)")
    print("=" * 80)
    top_orients = [o for o, _ in Counter(df.orientation).most_common(5)]
    print(f"  {'task':<24} " + " ".join(f"{o[:14]:>15}" for o in top_orients))
    for task in TASKS:
        sub = df[df.task == task]
        if sub.empty:
            continue
        n = len(sub)
        cells = " ".join(f"{(sub.orientation==o).sum()/n*100:>14.0f}%" for o in top_orients)
        print(f"  {task:<24} {cells}")

    # C. articulability by desc-length quartile
    print("\n" + "=" * 80)
    print("C. ARTICULABILITY by MEDOID-DESCRIPTION-LENGTH quartile (verbosity confound)")
    print("=" * 80)
    df["len_q"] = pd.qcut(df.desc_len, 4, labels=["Q1 short", "Q2", "Q3", "Q4 long"])
    for q, sub in df.groupby("len_q", observed=True):
        print(f"  {str(q):<24} {artic_dist(sub)}  mean_artic={sub.articulability.mean():.2f}")

    # D. articulability by cluster size
    print("\n" + "=" * 80)
    print("D. ARTICULABILITY by CLUSTER SIZE")
    print("=" * 80)
    for lo, hi, lab in [(1, 1, "singletons"), (2, 2, "size 2"), (3, 5, "size 3-5"), (6, 999, "size 6+")]:
        sub = df[(df.cluster_size >= lo) & (df.cluster_size <= hi)]
        if len(sub) < 30:
            continue
        print(f"  {lab:<24} {artic_dist(sub)}  mean_artic={sub.articulability.mean():.2f}")

    # E. is the L4 (tacit) finding a source-type artifact?
    print("\n" + "=" * 80)
    print("E. L4 (tacit) RATE by orientation — is the creative/humor tacit finding confounded?")
    print("=" * 80)
    for o, sub in sorted(df.groupby("orientation"), key=lambda kv: -len(kv[1])):
        if len(sub) < 100:
            continue
        l4 = (sub.articulability == 4).mean() * 100
        print(f"  {o:<24} L4={l4:>4.0f}%  (n={len(sub)})")
    print()
    print("  L4 rate within creative-writing+humor, by orientation:")
    ch = df[df.task.isin(["creative-writing", "humor"])]
    for o, sub in sorted(ch.groupby("orientation"), key=lambda kv: -len(kv[1])):
        if len(sub) < 50:
            continue
        l4 = (sub.articulability == 4).mean() * 100
        print(f"    {o:<22} L4={l4:>4.0f}%  (n={len(sub)})")

    df.to_parquet(OUT / "confound_analysis_clusters.parquet")
    print(f"\nwrote {OUT/'confound_analysis_clusters.parquet'}")


if __name__ == "__main__":
    main()
