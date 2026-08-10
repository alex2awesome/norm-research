"""Per-fixture score table for diagnostics."""
import json
from pathlib import Path
from .metrics import load_all

F = Path(__file__).parent / "fixtures" / "sample_prs.json"
metrics = load_all()
fixtures = json.loads(F.read_text())

# How much diff is in each truncated text?
print(f"{'ix':>2} {'lang':<11} {'y':>1} {'len':>5} {'#diff':>5} {'title':<30}")
print("-" * 65)
for i, f in enumerate(fixtures):
    t = f["text"]
    n_diff = t.count("diff --git")
    print(f"{i:>2} {f['language']:<11} {f['label']:>1} {len(t):>5} "
          f"{n_diff:>5} {(f.get('title') or '')[:30]:<30}")

print(f"\n{'ix':<3} {'lang':<11} {'y':>1}  " + " ".join(
    f"{m.ASPECT_ID:>6}" for m in metrics))
print("-" * 80)
for i, f in enumerate(fixtures):
    row = []
    for m in metrics:
        a = m.applies(f["text"])
        if not a:
            row.append("  .   ")
            continue
        s = m.score(f["text"])
        row.append(f"{s:>6.3f}" if s is not None else "  -   ")
    print(f"{i:<3} {f['language']:<11} {f['label']:>1}  " + " ".join(row))
