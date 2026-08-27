"""Rubric-viability scan: which peer-review rubrics actually carry signal on the corpus?

The collapse is model-independent (glm-5.2 == sonnet == gemma), so it's a rubric/corpus mismatch.
This scores ~30 rubrics over ~100 texts (Gemma, cheap; collapse is model-independent so Gemma is
fine for screening) and reports, per rubric: applicability, # distinct scores, collapsed flag.
Survivors (appl>0.3 AND non-degenerate) are the usable subset for the probe.

Also prints a sample corpus text so we can see whether texts are reviews, abstracts, or process docs.
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

_KEY = Path.home() / ".openrouter-api-key.txt"
os.environ.setdefault("OPENAI_API_KEY", _KEY.read_text().strip())
os.environ.setdefault("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")


def _load_texts(task, n, seed):
    dcfg = DATASET_CONFIGS[task]
    base = REPO_ROOT / dcfg["split"]
    fp = next((Path(str(base) + e) for e in (".csv.gz", ".csv") if Path(str(base) + e).exists()), None)
    df = pd.read_csv(fp, low_memory=False)
    print(f"corpus columns: {list(df.columns)[:12]}")
    print(f"sample text:\n  {(df[dcfg['text']].dropna().astype(str).iloc[0])[:400]}\n")
    s = df[dcfg["text"]].dropna().astype(str)
    return s.sample(n=min(n, len(s)), random_state=seed).tolist()


def main() -> int:
    cfg = InfillConfig(materialize_backend="openai_compatible",
                      materialize_model="google/gemma-4-31B-it",
                      openai_base_url="https://openrouter.ai/api/v1", llm_concurrency=8,
                      max_text_tokens=700)
    rubrics = load_rubric_metrics("peer-review", limit=30)
    texts = _load_texts("peer-review", 100, 7)
    print(f"rubrics={len(rubrics)} texts={len(texts)}\n")
    judge = make_vllm_judge_scorer(cfg)
    lv, ap = judge(rubrics, texts)
    viable, na_only, collapsed = [], [], []
    print(f"{'#':>2} {'rubric':52} {'appl':>5} {'uniq':>4} {'std':>5}  verdict")
    for j, r in enumerate(rubrics):
        col = lv[ap[:, j], j]
        appl = float(ap[:, j].mean())
        if len(col) == 0 or appl < 0.05:
            na_only.append(r.name); verdict = "ALL-NA"; uniq, std = 0, 0.0
        else:
            uniq = len(np.unique(np.round(col, 1)))
            std = float(np.std(col))
            if std < 0.1 or uniq <= 1:
                collapsed.append(r.name); verdict = "collapsed"
            else:
                viable.append((r.name, appl, std, uniq)); verdict = "VIABLE"
        print(f"{j:>2} {r.name[:52]:52} {appl:5.2f} {uniq:4d} {std:5.2f}  {verdict}")
    print(f"\nSUMMARY: viable={len(viable)}  all-NA={len(na_only)}  collapsed={len(collapsed)}  "
          f"(of {len(rubrics)})")
    if viable:
        print("viable rubrics:")
        for n, a, s, u in viable:
            print(f"   {n[:60]:60} appl={a:.2f} std={s:.2f} uniq={u}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
