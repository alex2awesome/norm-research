"""Test a408 on synthetic LC diffs built from balanced_v2 rows."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pandas as pd

REPO = Path("/lfs/skampere3/0/alexspan/norm-research")
sys.path.insert(0, str(REPO))

from methods.existing_metrics_runner.coded.metrics import a408_edit_distance_editorial as M

EXT_BY_LANG = {
    "python": "py", "java": "java", "cpp": "cpp", "c": "c",
    "javascript": "js", "typescript": "ts", "go": "go", "rust": "rs",
    "ruby": "rb", "csharp": "cs", "swift": "swift", "kotlin": "kt",
    "sql": "sql",
}


def make_diff(slug: str, lang: str, code: str) -> str:
    ext = EXT_BY_LANG.get(lang, "txt")
    body = "\n".join("+" + ln for ln in code.split("\n"))
    n = len(code.split("\n"))
    header = (
        f"## PR Title\nLeetCode {slug}\n\n"
        f"## Code Diff (1/1 files)\n"
        f"diff --git a/leetcode/{slug}.{ext} b/leetcode/{slug}.{ext}\n"
        f"new file mode 100644\n"
        f"index 0000000..1111111\n"
        f"--- /dev/null\n"
        f"+++ b/leetcode/{slug}.{ext}\n"
        f"@@ -0,0 +1,{n} @@\n"
    )
    return header + body + "\n"


def main():
    b = pd.read_parquet(REPO / "datasets/leetcode_balanced/balanced_v2.parquet")
    random.seed(42)
    rows = b.sample(n=200, random_state=42).reset_index(drop=True)

    results = []
    for r in rows.itertuples():
        diff = make_diff(r.question_slug, r.language, r.code)
        ap = M.applies(diff)
        sc = M.score(diff) if ap else None
        results.append((r.question_slug, r.language, ap, sc, r.taste_label))

    n_appl = sum(1 for *_a, ap, _sc, _y in results if ap)
    scored = [(s, y) for *_a, ap, s, y in results if ap and s is not None]
    print(f"sample={len(results)}, applied={n_appl}, scored={len(scored)}")

    if scored:
        scs = [s for s, _y in scored]
        print(f"score min={min(scs):.3f}, "
              f"max={max(scs):.3f}, "
              f"mean={sum(scs)/len(scs):.3f}")
        # By taste label
        pos = [s for s, y in scored if y == 1]
        neg = [s for s, y in scored if y == 0]
        if pos:
            print(f"  label=1 n={len(pos)} mean={sum(pos)/len(pos):.3f}")
        if neg:
            print(f"  label=0 n={len(neg)} mean={sum(neg)/len(neg):.3f}")
        # Quartiles
        s_sorted = sorted(scs)
        q = [s_sorted[int(len(s_sorted) * p)]
             for p in (0.25, 0.5, 0.75)]
        print(f"  quartiles: 25%={q[0]:.3f} 50%={q[1]:.3f} 75%={q[2]:.3f}")

    print("\nFirst 25 sampled rows:")
    for slug, lang, ap, sc, y in results[:25]:
        sc_str = f"{sc:.3f}" if sc is not None else "  -  "
        print(f"  {slug:<40} {lang:<11} applies={ap} score={sc_str} label={y}")


if __name__ == "__main__":
    main()
