"""Sanity inspection of R1 expanded.json files.

For each task × bucket: print top-N parents (by leaf count) with their
top child names. Catches obvious issues like overly-generic parents,
conflated concepts, near-duplicate parents that should have been merged.
"""
import json, sys, argparse
from pathlib import Path


def inspect(path: Path, top_n: int = 12, children_per_parent: int = 5):
    d = json.loads(path.read_text())
    parents = d.get("parented_trees", [])
    n_parents = len(parents)
    n_children = sum(len(p.get("children", [])) for p in parents)
    n_leaves = sum(p.get("total_leaf_rubrics", sum(c.get("size", len(c.get("rubrics", []))) for c in p.get("children", []))) for p in parents)

    print(f"\n{'='*90}")
    print(f"FILE: {path.name}")
    print(f"  parents={n_parents}  children={n_children}  leaves={n_leaves}")
    print(f"  avg children/parent={n_children/max(1,n_parents):.1f}")
    print(f"{'='*90}")

    def parent_leaves(p):
        return p.get("total_leaf_rubrics", sum(c.get("size", len(c.get("rubrics", []))) for c in p.get("children", [])))

    sorted_parents = sorted(parents, key=lambda p: -parent_leaves(p))
    print(f"\nTOP {top_n} PARENTS by leaf count:")
    for i, p in enumerate(sorted_parents[:top_n], 1):
        nl = parent_leaves(p)
        nc = len(p.get("children", []))
        name = (p.get("parent_name") or "")[:80]
        desc = (p.get("parent_description") or "")[:120]
        print(f"\n  [{i}] {nl:>4} leaves, {nc:>3} children: {name}")
        print(f"        desc: {desc}")
        for j, c in enumerate(p.get("children", [])[:children_per_parent], 1):
            cn = (c.get("medoid_name") or "")[:70]
            cs = c.get("size", len(c.get("rubrics", [])))
            print(f"        - [{cs:>3}] {cn}")
        if len(p.get("children", [])) > children_per_parent:
            print(f"        ... +{len(p['children']) - children_per_parent} more children")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/lfs/skampere3/0/alexspan/norm-research/outputs/hierarchy")
    ap.add_argument("--task", default=None, help="filter to one task")
    ap.add_argument("--bucket", default=None, help="filter to one bucket")
    ap.add_argument("--top-n", type=int, default=12)
    ap.add_argument("--children-per-parent", type=int, default=5)
    args = ap.parse_args()

    files = sorted(Path(args.dir).glob("*_r1_expanded.json"))
    cw_extra = Path(args.dir) / "creative-writing_general_gpt5_k200_expanded.json"
    if cw_extra.exists():
        files.append(cw_extra)
    for f in files:
        name = f.name
        if args.task and args.task not in name: continue
        if args.bucket and args.bucket not in name: continue
        inspect(f, top_n=args.top_n, children_per_parent=args.children_per_parent)


if __name__ == "__main__":
    main()
