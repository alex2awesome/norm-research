"""Quick sanity look at R3 outputs vs R2."""
import json, sys
from pathlib import Path

ROOT = Path("/lfs/skampere3/0/alexspan/norm-research/outputs/hierarchy")
cells = sys.argv[1:] or ["creative-writing_general", "humor_general", "code-review_general"]

for cell in cells:
    r2 = ROOT / f"{cell}_r2_expanded.json"
    r3 = ROOT / f"{cell}_r3_expanded.json"

    if r2.exists():
        d2 = json.loads(r2.read_text())
        m2 = len(d2.get("merged_groups", []))
        g2 = len(d2.get("grandparents", []))
    else:
        m2 = g2 = "?"

    if not r3.exists():
        print(f"\n=== {cell}: R3 skipped (insufficient parents — pipeline correctly short-circuited) ===")
        print(f"   R2: {m2} merged_groups, {g2} grandparents")
        continue

    d3 = json.loads(r3.read_text())
    m3 = d3.get("merged_groups", [])
    g3 = d3.get("grandparents", [])

    print(f"\n=== {cell} ===")
    print(f"  R2: {m2} merged_groups, {g2} grandparents")
    print(f"  R3: {len(m3)} merged_groups, {len(g3)} grandparents")

    if m3:
        print(f"  TOP 5 R3 merged_groups by total_leaf_rubrics:")
        for m in sorted(m3, key=lambda x: -(x.get("total_leaf_rubrics") or 0))[:5]:
            nl = m.get("total_leaf_rubrics", 0)
            print(f"    [{nl} leaves] {m.get('merged_name', '')[:90]}")
    if g3:
        print(f"  TOP 5 R3 grandparents by n_children:")
        for g in sorted(g3, key=lambda x: -x.get("n_children", 0))[:5]:
            nc = g.get("n_children", 0)
            nl = g.get("total_leaf_rubrics", 0)
            print(f"    [{nc} ch, {nl} l] {g.get('grandparent_name', '')[:90]}")
