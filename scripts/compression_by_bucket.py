"""Compute compression ratios per (task, bucket) and look for the
general-vs-specific pattern."""
import json
from pathlib import Path

ROOT = Path("/lfs/skampere3/0/alexspan/norm-research/outputs/hierarchy")

TASKS = ["code-review", "creative-writing", "grant-funding", "humor",
         "legal-outcome-prediction", "math-stackexchange", "news-homepages",
         "notice-and-comment", "patents", "peer-review", "press-releases"]
BUCKETS = ["general", "specific", "hyper_specific"]

rows = []
for task in TASKS:
    for bucket in BUCKETS:
        r1 = ROOT / f"{task}_{bucket}_r1_refined.json"
        if not r1.exists() and task == "creative-writing" and bucket == "general":
            r1 = ROOT / "creative-writing_general_r1_refined.json"
        r2 = ROOT / f"{task}_{bucket}_r2_expanded.json"
        if not r1.exists() or not r2.exists():
            continue
        d1 = json.loads(r1.read_text())
        d2 = json.loads(r2.read_text())
        p1 = len(d1.get("parented_trees", []))
        c1 = sum(len(p.get("children", [])) for p in d1.get("parented_trees", []))
        m2 = len(d2.get("merged_groups", []))
        g2 = len(d2.get("grandparents", []))
        rows.append({
            "task": task, "bucket": bucket,
            "p1": p1, "c1": c1, "m2": m2, "g2": g2,
            "merge_rate": m2 / max(1, p1),
            "gp_rate": g2 / max(1, p1),
        })

# Print pivot table: task × bucket → merge_rate
print(f"\nR2 merged_groups / R1-refined parents (lower → less collapsing):\n")
print(f"{'task':<25} {'general':>10} {'specific':>10} {'hyper_specific':>16}")
for task in TASKS:
    cells = {r["bucket"]: r for r in rows if r["task"] == task}
    g = cells.get("general", {})
    s = cells.get("specific", {})
    h = cells.get("hyper_specific", {})
    gr = f"{g.get('m2','-')}/{g.get('p1','-')} ({g.get('merge_rate',0)*100:.0f}%)" if g else "-"
    sr = f"{s.get('m2','-')}/{s.get('p1','-')} ({s.get('merge_rate',0)*100:.0f}%)" if s else "-"
    hr = f"{h.get('m2','-')}/{h.get('p1','-')} ({h.get('merge_rate',0)*100:.0f}%)" if h else "-"
    print(f"{task:<25} {gr:>10} {sr:>10} {hr:>16}")

print(f"\n--- Averages by bucket ---")
for bucket in BUCKETS:
    cells = [r for r in rows if r["bucket"] == bucket]
    if not cells: continue
    avg_merge = sum(r["merge_rate"] for r in cells) / len(cells)
    avg_gp = sum(r["gp_rate"] for r in cells) / len(cells)
    print(f"  {bucket:<16} avg merge_rate = {avg_merge*100:.1f}%   avg gp_rate = {avg_gp*100:.1f}%   (n={len(cells)})")
