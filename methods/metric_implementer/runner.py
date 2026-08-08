"""Apply all loaded metrics to a set of PR diffs; report per-metric stats."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List

from .metrics import load_all

FIXTURES = Path(__file__).parent / "fixtures" / "sample_prs.json"


def main():
    metrics = load_all()
    print(f"loaded {len(metrics)} metrics: " +
          ", ".join(m.ASPECT_ID for m in metrics))
    fixtures: List[dict] = json.loads(FIXTURES.read_text())
    print(f"fixtures: {len(fixtures)} PRs "
          f"(labels: {sum(f['label'] for f in fixtures)}/{len(fixtures)})\n")

    # per-metric results
    print(f"{'metric':<30} {'tier':>4} {'class':>16} {'appl':>5} "
          f"{'score':>7} {'std':>7} {'auc':>7} {'tools'}")
    print("-" * 100)

    for m in metrics:
        appl, scs, ys = [], [], []
        for f in fixtures:
            a = m.applies(f["text"])
            s = m.score(f["text"]) if a else None
            appl.append(int(a))
            if s is not None:
                scs.append(s)
                ys.append(f["label"])
        n_appl = sum(appl)
        mean_s = sum(scs) / len(scs) if scs else float("nan")
        if len(scs) >= 2:
            mu = sum(scs) / len(scs)
            var = sum((s - mu) ** 2 for s in scs) / max(len(scs) - 1, 1)
            std = math.sqrt(var)
        else:
            std = float("nan")
        # cheap "AUC" estimate when both classes present
        try:
            pos = [s for s, y in zip(scs, ys) if y == 1]
            neg = [s for s, y in zip(scs, ys) if y == 0]
            if pos and neg:
                wins = 0
                ties = 0
                for p in pos:
                    for n in neg:
                        if p > n:
                            wins += 1
                        elif p == n:
                            ties += 1
                auc = (wins + 0.5 * ties) / (len(pos) * len(neg))
            else:
                auc = float("nan")
        except Exception:
            auc = float("nan")
        tools = ",".join(m.TOOLS) or "(stdlib)"
        print(f"{m.ASPECT_ID + ' ' + m.ASPECT_NAME[:20]:<30} "
              f"{m.TIER:>4} {m.CLASSIFICATION:>16} {n_appl:>5} "
              f"{mean_s:>7.3f} {std:>7.3f} {auc:>7.3f} {tools}")


if __name__ == "__main__":
    main()
