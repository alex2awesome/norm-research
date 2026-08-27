"""Measure the GOAL metric: collective predictive POWER of the articulated metric set (train->test).

Loads a task's rubrics, materializes them over a train/test split, drops degenerate (all-NA /
collapsed) metrics, fits logistic-regression(label ~ metric levels) on TRAIN, and reports TEST
AUC + per-metric marginal contribution. This is the number the ctree is supposed to maximize.

    python diag_metric_power.py peer-review 60   # task, # rubrics
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent / "methods"))

from methods.metrics_tree_infilling.config import InfillConfig
from methods.metrics_tree_infilling.io_metrics import (
    REPO_ROOT, discover_test_split, load_rubric_metrics, make_design, make_vllm_judge_scorer,
    materialize,
)
from methods.metrics_tree_infilling.run import DATASET_CONFIGS

_KEY = Path.home() / ".openrouter-api-key.txt"
os.environ.setdefault("OPENAI_API_KEY", _KEY.read_text().strip())
os.environ.setdefault("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")


def _load_df(task):
    dcfg = DATASET_CONFIGS[task]
    base = REPO_ROOT / dcfg["split"]
    fp = next((Path(str(base) + e) for e in (".csv.gz", ".csv") if Path(str(base) + e).exists()), None)
    df = pd.read_csv(fp, low_memory=False)
    df = df.dropna(subset=[dcfg["id"], dcfg["text"], dcfg["label"]]).reset_index(drop=True)
    df[dcfg["label"]] = pd.to_numeric(df[dcfg["label"]], errors="coerce")
    df = df.dropna(subset=[dcfg["label"]]).reset_index(drop=True)
    df[dcfg["label"]] = df[dcfg["label"]].astype(int)
    return df, dcfg


def main() -> int:
    task = sys.argv[1] if len(sys.argv) > 1 else "peer-review"
    n_metrics = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    cfg = InfillConfig(materialize_backend="openai_compatible",
                      materialize_model="google/gemma-4-31B-it",
                      openai_base_url="https://openrouter.ai/api/v1", llm_concurrency=10,
                      max_text_tokens=700, id_column="id", text_column="text", label_column="judgement")
    df, dcfg = _load_df(task)
    # align cfg columns to the dataset
    cfg.id_column, cfg.text_column, cfg.label_column = dcfg["id"], dcfg["text"], dcfg["label"]
    df = df.sample(n=min(400, len(df)), random_state=7).reset_index(drop=True)
    print(f"task={task} rows={len(df)} label_rate={df[dcfg['label']].mean():.2f}")

    rubrics = load_rubric_metrics(task, limit=n_metrics)
    print(f"rubrics={len(rubrics)}")
    df_d, df_t = discover_test_split(df, cfg)
    judge = make_vllm_judge_scorer(cfg)
    sm_d = materialize(rubrics, df_d, cfg, judge)
    sm_t = materialize(rubrics, df_t, cfg, judge)
    # shared DesignSpec so discover and test share the SAME column layout (NA-indicator columns
    # can otherwise differ between splits and mis-align Xd[:, keep] -> Xt[:, keep]).
    Xd, _, _, spec = make_design(sm_d, df_d, cfg)
    Xt, _, _, _ = make_design(sm_t, df_t, cfg, spec=spec)
    yd = df_d[dcfg["label"]].to_numpy()
    yt = df_t[dcfg["label"]].to_numpy()

    # drop degenerate columns (no variance on train)
    std = Xd.std(axis=0)
    keep = np.where(std > 1e-6)[0]
    print(f"non-degenerate metric columns: {len(keep)}/{Xd.shape[1]}")

    base_auc = _auc(yt, np.full_like(yt, yt.mean(), dtype=float))
    if len(keep) >= 1 and len(np.unique(yd)) == 2:
        lr = LogisticRegression(max_iter=2000, C=1.0).fit(Xd[:, keep], yd)
        p = lr.predict_proba(Xt[:, keep])[:, 1]
        auc = _auc(yt, p)
        print(f"\nTEST AUC (all non-degenerate metrics): {auc:.3f}  (base-rate AUC={base_auc:.3f})")
        # per-metric marginal contribution (drop-one)
        coefs = np.abs(lr.coef_[0])
        order = np.argsort(-coefs)[:10]
        print("top contributing metrics:")
        for k in order:
            print(f"   {rubrics[keep[k]].name[:50]:50} |coef|={coefs[k]:.3f}")
    else:
        print("too few non-degenerate metrics to fit")
    return 0


def _auc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


if __name__ == "__main__":
    raise SystemExit(main())
