"""Judge-half diagnosis (option 4): is the live Gemma JUDGE accurate + reliable for a CORRECT
feature? Closes the other half of the smoke diagnosis.

Gives the live Gemma-4 judge the CORRECT glow/song rubrics (so the proposer is taken out of the
picture), materializes them over a 100-item sample, and compares to ground truth
(``world.detect``). Reports accuracy, score distribution (watching for the all-0.5 collapse),
and test-retest reliability.

Read-out:
  * judge accurate + reliable -> a proposer fix ALONE suffices; distillation not needed for quality
  * judge noisy / collapsed   -> GEPA judge-optimization (distillation) is needed for quality too

Run from repo root (reads ~/.openrouter-api-key.txt):
    python diag_judge_probe.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "methods"))

from methods.metrics_tree_infilling.config import InfillConfig
from methods.metrics_tree_infilling.io_metrics import MetricSpec, make_vllm_judge_scorer
from methods.metrics_tree_infilling.feature_gen import estimate_reliability

from methods.metrics_tree_infilling.tests.test_scenario.generate import build_corpus
from methods.metrics_tree_infilling.tests.test_scenario import world

_KEY = Path.home() / ".openrouter-api-key.txt"
os.environ.setdefault("OPENAI_API_KEY", _KEY.read_text().strip())
os.environ.setdefault("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")


def _cfg() -> InfillConfig:
    return InfillConfig(
        materialize_backend="openai_compatible",
        materialize_model="google/gemma-4-31B-it",
        openai_base_url="https://openrouter.ai/api/v1", llm_concurrency=8,
    )


GLOW = MetricSpec(
    metric_id="glow", name="Bioluminescent", kind="judge", role="feature",
    description="Whether the creature glows / is bioluminescent vs dark.",
    guidance=("Return 1 if the creature is described as bioluminescent, glowing, luminous, or "
              "casting light / a shine; return 0 if it is dark, shadowed, dim, or gives off no light."),
)
SONG = MetricSpec(
    metric_id="song", name="MelodiousCall", kind="judge", role="feature",
    description="Whether the creature's call is melodious vs harsh.",
    guidance=("Return 1 if the creature's call / song / voice is described as melodious, musical, "
              "sweet, or pleasant; return 0 if harsh, discordant, grating, or screeching."),
)


def _truth(texts, attr):
    return np.array([world.detect(t, attr) for t in texts])


def main() -> int:
    df, _ = build_corpus(n=500, seed=7)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(df), 100, replace=False)
    texts = df.iloc[idx]["text"].astype(str).tolist()

    c = _cfg()
    judge = make_vllm_judge_scorer(c)

    # one judge call per item scores BOTH features
    lv, ap = judge([GLOW, SONG], texts)
    glow_scores, glow_ap = lv[:, 0], ap[:, 0]
    song_scores, song_ap = lv[:, 1], ap[:, 1]

    rows = [
        ("Bioluminescent", "glow", glow_scores, glow_ap),
        ("MelodiousCall", "song", song_scores, song_ap),
    ]
    print(f"{'feature':14} {'appl':>5} {'acc':>6} {'n':>4} {'mean':>6} {'std':>6} "
          f"{'%0':>5} {'%0.5':>6} {'%1':>5} {'reliab':>7}")
    for name, attr, scores, appl in rows:
        tr = _truth(texts, attr)
        pos = next(iter(world.ATTRIBUTES[attr]))            # 'luminous' / 'melodious'
        y = (tr == pos).astype(int)
        mask = appl & np.isfinite(scores)
        yhat = (scores[mask] > 0.5).astype(int)
        acc = float((yhat == y[mask]).mean()) if mask.any() else float("nan")
        s = scores[mask]
        uniq, cnts = np.unique(np.round(s, 2), return_counts=True)
        dist = {u: c for u, c in zip(uniq, cnts)}
        metric = GLOW if attr == "glow" else SONG
        rel = estimate_reliability(metric, texts, judge, 60, rng)
        p0 = dist.get(0.0, 0) / max(len(s), 1)
        ph = dist.get(0.5, 0) / max(len(s), 1)
        p1 = dist.get(1.0, 0) / max(len(s), 1)
        print(f"{name:14} {appl.mean():5.2f} {acc:6.3f} {int(mask.sum()):4d} "
              f"{np.nanmean(scores):6.2f} {np.nanstd(scores):6.2f} "
              f"{p0:5.2f} {ph:6.2f} {p1:5.2f} {rel:7.3f}")
        # per-truth-condition means (does the judge separate luminous vs dim?)
        for v in world.ATTRIBUTES[attr]:
            vmask = mask & (tr == v)
            if vmask.any():
                print(f"    truth={v:10} mean_score={np.nanmean(scores[vmask]):.2f} "
                      f"(n={int(vmask.sum())})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
