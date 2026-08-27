"""Does the CTREE's own judge (make_vllm_judge_scorer) collapse on REAL peer-review rubrics?

The metric_implementer distillation judge collapsed to all-zeros on a real rubric. But that's a
different code path. The ctree probe uses make_vllm_judge_scorer (LLMClient). This loads a few
real peer-review rubrics, materializes them via Gemma-4 over a sample, and reports the score
distribution per rubric (collapse = all-0 / all-0.5 / zero variance).

Read-out decides the probe approach:
  * collapses too  -> apply GEPA with a strong supervisor (GLM-5.4/Opus) per user direction
  * does not        -> the collapse was metric_implementer-specific; probe proceeds with Gemma
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "methods"))

from methods.metrics_tree_infilling.config import InfillConfig
from methods.metrics_tree_infilling.io_metrics import load_rubric_metrics, make_vllm_judge_scorer
from methods.metrics_tree_infilling.run import DATASET_CONFIGS
from methods.metrics_tree_infilling.io_metrics import REPO_ROOT

import pandas as pd

_KEY = Path.home() / ".openrouter-api-key.txt"
os.environ.setdefault("OPENAI_API_KEY", _KEY.read_text().strip())
os.environ.setdefault("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")


def _load_texts(task, n, seed):
    dcfg = DATASET_CONFIGS[task]
    base = REPO_ROOT / dcfg["split"]
    fp = next((Path(str(base) + e) for e in (".csv.gz", ".csv") if Path(str(base) + e).exists()), None)
    df = pd.read_csv(fp, low_memory=False)
    s = df[dcfg["text"]].dropna().astype(str)
    return s.sample(n=min(n, len(s)), random_state=seed).tolist()


def main() -> int:
    cfg = InfillConfig(
        materialize_backend="openai_compatible", materialize_model="google/gemma-4-31B-it",
        openai_base_url="https://openrouter.ai/api/v1", llm_concurrency=8,
        max_text_tokens=700,
    )
    rubrics = load_rubric_metrics("peer-review", limit=6)
    texts = _load_texts("peer-review", 60, 7)
    print(f"rubrics={len(rubrics)} texts={len(texts)} model={cfg.materialize_model}")
    judge = make_vllm_judge_scorer(cfg)
    lv, ap = judge(rubrics, texts)
    print(f"\n{'#':>2} {'rubric':50} {'appl':>5} {'mean':>6} {'std':>6} {'%0':>5} {'%0.5':>6} {'%1':>5} collapsed?")
    any_collapse = False
    for j, r in enumerate(rubrics):
        col = lv[ap[:, j], j]
        if len(col) == 0:
            print(f"{j:>2} {r.name[:50]:50} (no applicable items)")
            continue
        uniq, cnts = np.unique(np.round(col, 2), return_counts=True)
        dist = {u: c for u, c in zip(uniq, cnts)}
        p0 = dist.get(0.0, 0) / len(col)
        ph = dist.get(0.5, 0) / len(col)
        p1 = dist.get(1.0, 0) / len(col)
        std = float(np.std(col))
        collapsed = std < 0.05 or (p0 > 0.9 or p1 > 0.9 or ph > 0.9)
        any_collapse = any_collapse or collapsed
        print(f"{j:>2} {r.name[:50]:50} {ap[:, j].mean():5.2f} {np.mean(col):6.2f} {std:6.2f} "
              f"{p0:5.2f} {ph:6.2f} {p1:5.2f}   {'YES' if collapsed else 'no'}")
    print(f"\n=> ctree judge collapses on real rubrics? {'YES' if any_collapse else 'NO'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
