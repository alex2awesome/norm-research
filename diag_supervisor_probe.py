"""Does a STRONGER supervisor (GLM-5.4 / GLM-5.2 / Opus) escape the Gemma collapse on real rubrics?

Scores the two collapsed rubrics from diag_judge_real (citation=all-1, anonymization NA) over a
small sample with each candidate supervisor via the z.ai anthropic proxy, reports the score
distribution. If a stronger model discriminates (non-degenerate), GEPA-supervised-by-it is the
right fix; if everything stays collapsed, the wall is the rubric/corpus mismatch, not the model.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "methods"))

from methods.metrics_tree_infilling.config import InfillConfig
from methods.metrics_tree_infilling.io_metrics import load_rubric_metrics, make_vllm_judge_scorer
from methods.metrics_tree_infilling.run import DATASET_CONFIGS
from methods.metrics_tree_infilling.io_metrics import REPO_ROOT

# z.ai anthropic proxy (GLM). ANTHROPIC_BASE_URL / ANTHROPIC_AUTH_TOKEN expected in env.
SUPERVISORS = ["glm-5.4", "glm-5.2", "claude-sonnet-4-20250514"]


def _load_texts(task, n, seed):
    dcfg = DATASET_CONFIGS[task]
    base = REPO_ROOT / dcfg["split"]
    fp = next((Path(str(base) + e) for e in (".csv.gz", ".csv") if Path(str(base) + e).exists()), None)
    df = pd.read_csv(fp, low_memory=False)
    s = df[dcfg["text"]].dropna().astype(str)
    return s.sample(n=min(n, len(s)), random_state=seed).tolist()


def _dist(col):
    if len(col) == 0:
        return "n/a"
    u, c = np.unique(np.round(col, 2), return_counts=True)
    return " ".join(f"{ui}:{ci}" for ui, ci in zip(u, c)) + f" std={np.std(col):.2f}"


def main() -> int:
    rubrics = load_rubric_metrics("peer-review", limit=6)
    # pick the all-NA ones (0,1,2,5) + the all-1 one (4); score a few
    picks = [rubrics[k] for k in (4, 0, 3) if k < len(rubrics)]
    texts = _load_texts("peer-review", 25, 7)
    print(f"texts={len(texts)} rubrics={[r.name[:30] for r in picks]}\n")

    for mdl in SUPERVISORS:
        cfg = InfillConfig(materialize_backend="anthropic", materialize_model=mdl,
                           llm_concurrency=4, max_text_tokens=700)
        try:
            judge = make_vllm_judge_scorer(cfg)
            lv, ap = judge(picks, texts)
        except Exception as e:  # noqa: BLE001
            print(f"[{mdl}] ERROR {type(e).__name__}: {str(e)[:90]}  (likely not served / quota)")
            continue
        print(f"[{mdl}]")
        for j, r in enumerate(picks):
            col = lv[ap[:, j], j]
            print(f"   {r.name[:44]:44} appl={ap[:, j].mean():.2f}  {_dist(col)}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
