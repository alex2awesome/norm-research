"""Scan all R2 outputs for timeouts and other batch-level failures."""
import json
from pathlib import Path

root = Path("/lfs/skampere3/0/alexspan/norm-research/outputs/hierarchy")
print(f"{'cell':<50} {'batches':>7} {'TO':>4} {'ERR':>4} {'merges':>7} {'gp':>4}  flag")
print("-" * 90)

for f in sorted(root.glob("*_r2.json")):
    if "prerefinement" in f.name:
        continue
    d = json.loads(f.read_text())
    batches = d.get("batches", [])
    n_to = sum(1 for b in batches if isinstance(b.get("result"), dict) and "_error" in b["result"] and "timeout" in str(b["result"].get("_error", "")))
    n_oerr = sum(1 for b in batches if isinstance(b.get("result"), dict) and "_error" in b["result"] and "timeout" not in str(b["result"].get("_error", "")))
    expanded = root / f.name.replace(".json", "_expanded.json")
    if expanded.exists():
        e = json.loads(expanded.read_text())
        m = len(e.get("merged_groups", []))
        g = len(e.get("grandparents", []))
    else:
        m = g = "?"
    flag = ""
    if n_to > 0:
        flag = "*** TIMEOUTS ***"
    if n_oerr > 0:
        flag += f" ERR"
    cell = f.name.replace("_r2.json", "")
    print(f"{cell:<50} {len(batches):>7} {n_to:>4} {n_oerr:>4} {str(m):>7} {str(g):>4}  {flag}")

print()
print("=" * 90)
print("R3 scan:")
print(f"{'cell':<50} {'batches':>7} {'TO':>4} {'merges':>7} {'gp':>4}")
print("-" * 90)
for f in sorted(root.glob("*_r3.json")):
    if "prerefinement" in f.name:
        continue
    d = json.loads(f.read_text())
    batches = d.get("batches", [])
    n_to = sum(1 for b in batches if isinstance(b.get("result"), dict) and "_error" in b["result"] and "timeout" in str(b["result"].get("_error", "")))
    expanded = root / f.name.replace(".json", "_expanded.json")
    if expanded.exists():
        e = json.loads(expanded.read_text())
        m = len(e.get("merged_groups", []))
        g = len(e.get("grandparents", []))
    else:
        m = g = "?"
    flag = "*** TIMEOUTS ***" if n_to > 0 else ""
    cell = f.name.replace("_r3.json", "")
    print(f"{cell:<50} {len(batches):>7} {n_to:>4} {str(m):>7} {str(g):>4}  {flag}")
