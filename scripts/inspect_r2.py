"""Quick sanity look at R2 expanded outputs."""
import json, sys
from pathlib import Path

ROOT = Path("/lfs/skampere3/0/alexspan/norm-research/outputs/hierarchy")
cells = sys.argv[1:] or ["grant-funding_hyper_specific", "code-review_hyper_specific", "humor_hyper_specific"]

for cell in cells:
    r1 = ROOT / f"{cell}_r1_refined.json"
    r2 = ROOT / f"{cell}_r2_expanded.json"

    if r1.exists():
        d1 = json.loads(r1.read_text())
        n_p1 = len(d1.get("parented_trees", []))
        n_c1 = sum(len(p.get("children", [])) for p in d1.get("parented_trees", []))
    else:
        n_p1 = n_c1 = "?"

    if not r2.exists():
        print(f"\n=== {cell}: R2 not yet ===")
        continue

    d2 = json.loads(r2.read_text())
    merged = d2.get("merged_groups", [])
    grand = d2.get("grandparents", [])

    print(f"\n=== {cell} ===")
    print(f"  R1-refined: {n_p1} parents, {n_c1} children")
    print(f"  R2 merged_groups:  {len(merged)}")
    print(f"  R2 grandparents:   {len(grand)}")

    if merged:
        print(f"  TOP 5 merged_groups by total_leaf_rubrics:")
        for m in sorted(merged, key=lambda x: -(x.get("total_leaf_rubrics") or 0))[:5]:
            n_leaves = m.get("total_leaf_rubrics") or len(m.get("all_leaves", []))
            n_srcs = len(m.get("source_clusters", []))
            name = m.get("merged_name", "")[:80]
            print(f"    [{n_srcs} R1-parents → {n_leaves} leaves] {name}")

    if grand:
        print(f"  TOP 5 grandparents by n_children:")
        for g in sorted(grand, key=lambda x: -x.get("n_children", 0))[:5]:
            nc = g.get("n_children", 0)
            nl = g.get("total_leaf_rubrics", 0)
            name = g.get("grandparent_name", "")[:80]
            print(f"    [{nc} children, {nl} leaves] {name}")
