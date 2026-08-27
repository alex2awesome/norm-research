"""GEPA-style operationalization of proposed metrics (2026-07-09, user directive).

Every newly proposed metric gets an ITERATED articulation pass before gate scoring: score a
calibration subsample, diagnose the rubric as an instrument (test-retest reliability, score
distribution, MI-recovery = can a blind reconstructor re-derive the rubric from score
exemplars and reproduce the scores), and if the diagnostics are poor, have the proposer
REWRITE the rubric given those diagnostics. Keep the best variant by (retest + recovery)/2.

All objectives are LABEL-FREE (reliability, recovery, applicability spread) — the gate owns
all label contact (reconstruction-only discipline). This is the missing OPERATIONALIZE stage:
the existing reconstruction_accuracy is diagnostic-only; here it drives improvement.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
from scipy.stats import spearmanr

from .io_metrics import MetricSpec
from .loop import _score_one

_REWRITE_PROMPT = """You are improving an evaluation rubric so a careful but literal judge can
score any single item 0-1 RELIABLY and FAITHFULLY. The rubric's intent must not change.

METRIC: {name}
CURRENT RUBRIC:
{rubric}

MEASURED DIAGNOSTICS of the current rubric as an instrument (label-free):
- test-retest reliability (same judge, independent rereads): {retest:.2f} {retest_note}
- score distribution: mean {mean:.2f}, std {std:.2f} {dist_note}
- blind recovery: a reader shown only (item, score) pairs re-derived the criterion as:
  "{recovered}"
  and reproduced the scores with agreement {recovery:.2f} {recovery_note}

REWRITE the rubric to fix the weakest diagnostic while preserving the metric's meaning:
make the scored property concrete and observable in one reading; give anchored guidance for
1.0 / 0.5 / 0.0; disambiguate from the neighboring property the blind reader drifted toward.
Under 250 words. Return ONLY the rubric text."""


@dataclass
class OpResult:
    rubric: str
    iterations: int
    retest: float
    recovery: float
    std: float
    trajectory: List[dict] = field(default_factory=list)


def _diagnose(spec: MetricSpec, cal_texts: List[str], judge_scorer, proposer, cfg,
              seed: int = 0) -> dict:
    lv, ap = _score_one(spec, cal_texts, judge_scorer)
    ok = ap & np.isfinite(lv)
    d = dict(applicability=float(ok.mean()),
             mean=float(np.nanmean(lv[ok])) if ok.any() else float("nan"),
             std=float(np.nanstd(lv[ok])) if ok.any() else 0.0,
             retest=float("nan"), recovery=float("nan"), recovered="")
    if ok.sum() < 30:
        return d
    # retest: salted independent reread of a slice (same engine; the salt forces a fresh read)
    sub = np.where(ok)[0][:60]
    salted = MetricSpec(metric_id=spec.metric_id + "_rt", name=spec.name,
                        description=spec.description, kind="judge",
                        guidance=spec.guidance + "\n(Re-evaluate carefully from scratch.)")
    lv2, ap2 = _score_one(salted, [cal_texts[i] for i in sub], judge_scorer)
    both = np.isfinite(lv2) & ap2
    if both.sum() > 20 and np.nanstd(lv[sub][both]) > 0 and np.nanstd(lv2[both]) > 0:
        d["retest"] = float(spearmanr(lv[sub][both], lv2[both]).statistic)
    # MI-recovery: blind re-derivation + score reproduction (reuse the loop's machinery)
    from .global_infill import reconstruction_accuracy
    agree, _auc, recovered = reconstruction_accuracy(
        spec, lv, ok, cal_texts, judge_scorer, proposer, cfg, n_show=16, n_eval=40,
        n_tries=1, seed=seed)
    d["recovery"], d["recovered"] = float(agree), recovered[:300]
    return d


def _score_of(d: dict) -> float:
    r = 0.0 if np.isnan(d["retest"]) else d["retest"]
    v = 0.0 if np.isnan(d["recovery"]) else d["recovery"]
    return (r + v) / 2


def operationalize_rubric(
    name: str, description: str, rubric: str,
    cal_texts: List[str], judge_scorer, proposer: Callable[[str], Optional[str]], cfg,
    min_retest: float = 0.8, min_recovery: float = 0.65, max_rewrites: int = 2,
) -> OpResult:
    """Iterate the rubric against label-free instrument diagnostics; return the best variant."""
    variants = [rubric]
    diags = []
    spec0 = MetricSpec(metric_id="op0", name=name, description=description, kind="judge",
                       guidance=rubric)
    d = _diagnose(spec0, cal_texts, judge_scorer, proposer, cfg, seed=0)
    diags.append(d)
    for it in range(max_rewrites):
        cur = diags[-1]
        good = ((np.isnan(cur["retest"]) or cur["retest"] >= min_retest)
                and (np.isnan(cur["recovery"]) or cur["recovery"] >= min_recovery)
                and cur["std"] >= 0.08)
        if good:
            break
        prompt = _REWRITE_PROMPT.format(
            name=name, rubric=variants[-1][:1500],
            retest=0 if np.isnan(cur["retest"]) else cur["retest"],
            retest_note="(LOW — judges disagree across rereads)" if (
                not np.isnan(cur["retest"]) and cur["retest"] < min_retest) else "",
            mean=cur["mean"] if not np.isnan(cur["mean"]) else 0.5, std=cur["std"],
            dist_note="(COLLAPSED — nearly constant, cannot discriminate)" if cur["std"] < 0.08 else "",
            recovered=cur["recovered"] or "(no recovery available)",
            recovery=0 if np.isnan(cur["recovery"]) else cur["recovery"],
            recovery_note="(LOW — scores drift from the stated semantics)" if (
                not np.isnan(cur["recovery"]) and cur["recovery"] < min_recovery) else "")
        try:
            resp = proposer(prompt)
        except Exception:
            break
        new_rubric = (resp or "").strip()
        if len(new_rubric) < 80:
            break
        variants.append(new_rubric)
        spec = MetricSpec(metric_id=f"op{it+1}", name=name, description=description,
                          kind="judge", guidance=new_rubric)
        diags.append(_diagnose(spec, cal_texts, judge_scorer, proposer, cfg, seed=it + 1))
    best = int(np.argmax([_score_of(d) for d in diags]))
    return OpResult(rubric=variants[best], iterations=len(variants) - 1,
                    retest=diags[best]["retest"], recovery=diags[best]["recovery"],
                    std=diags[best]["std"],
                    trajectory=[{k: v for k, v in d.items() if k != "recovered"}
                                for d in diags])
