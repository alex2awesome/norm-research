"""Held-out evaluation of the per-task cross-encoders.

For each task, the CE predicts a 0-1 sameness score on the HELD-OUT eval pairs
(split=="eval" -- rubrics never seen in CE training). We compare the CE score
to the v6 judge's label:
  - Spearman correlation (CE score vs. judge score/2)
  - merge-classifier view: CE >= 0.75 called "same" vs. judge score == 2
"""
from __future__ import annotations

import os

os.environ["HOME"] = "/lfs/skampere3/0/alexspan"
os.environ["HF_HOME"] = "/lfs/skampere3/0/alexspan/hf_cache"
os.environ["HF_MODULES_CACHE"] = "/lfs/skampere3/0/alexspan/hf_cache/modules"
os.environ["XDG_CACHE_HOME"] = "/lfs/skampere3/0/alexspan/.cache"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

WORK = Path("/lfs/skampere3/0/alexspan/norm_embed")
VERDICTS = WORK / "all_verdicts.jsonl"
CE_DIR = WORK / "ce_models"

TASKS = ["code-review", "creative-writing", "grant-funding", "humor",
         "legal-outcome-prediction", "math-stackexchange", "news-homepages",
         "notice-and-comment", "patents", "peer-review", "press-releases"]


def main():
    from sentence_transformers import CrossEncoder
    from scipy.stats import spearmanr

    ev = defaultdict(list)
    for line in VERDICTS.open():
        v = json.loads(line)
        if v.get("split") == "eval" and v.get("score") in (0, 1, 2):
            ev[v["task"]].append(v)

    print(f"{'task':<26}{'n_eval':>8}{'spearman':>10}{'merge P':>9}{'merge R':>9}")
    rho_all, P_all, R_all = [], [], []
    for task in TASKS:
        pairs = ev.get(task, [])
        d = CE_DIR / task
        if len(pairs) < 20 or not d.exists():
            continue
        ce = CrossEncoder(str(d), device="cuda")
        scores = ce.predict([[p["canonical_a"], p["canonical_b"]] for p in pairs],
                            show_progress_bar=False)
        gold = np.array([p["score"] / 2.0 for p in pairs])
        scores = np.asarray(scores, dtype=float)
        rho = spearmanr(scores, gold).statistic
        pred_same = scores >= 0.75
        gold_same = gold >= 1.0
        tp = int((pred_same & gold_same).sum())
        fp = int((pred_same & ~gold_same).sum())
        fn = int((~pred_same & gold_same).sum())
        P = tp / (tp + fp) if (tp + fp) else 0.0
        R = tp / (tp + fn) if (tp + fn) else 0.0
        rho_all.append(rho); P_all.append(P); R_all.append(R)
        print(f"{task:<26}{len(pairs):>8}{rho:>10.3f}{P*100:>8.0f}%{R*100:>8.0f}%")
    print("-" * 62)
    print(f"{'MEAN':<26}{'':<8}{np.mean(rho_all):>10.3f}"
          f"{np.mean(P_all)*100:>8.0f}%{np.mean(R_all)*100:>8.0f}%")


if __name__ == "__main__":
    main()
