"""§9 fix end-to-end: the loop recovers an absent-feature ROOT XOR via the composite path.

A world where the label is the XOR of two textual properties NEITHER of which is a known metric
and neither of which has marginal signal. The single-feature proposer provably fails here (no one
attribute separates -> it returns nothing). With ``enable_composite_proposer`` the loop proposes a
2-primitive composite, fits the boolean rule on data, and closes the gap.

No LLM: an offline oracle composite proposer + oracle judge (deterministic).
"""

from __future__ import annotations

import json
import re
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import pytest

from methods.metrics_tree_infilling.config import InfillConfig
from methods.metrics_tree_infilling.feature_gen import ProposedComposite, ProposedFeature
from methods.metrics_tree_infilling.io_metrics import (
    MetricSpec, _stable_id, discover_test_split, materialize,
)
from methods.metrics_tree_infilling.loop import run_infill
from methods.metrics_tree_infilling.interactions import RULES, apply_rule

# --- a tiny world ------------------------------------------------------------

ATTRS = {
    "size": {"tiny": ["small and slight", "no bigger than a hare"],
             "hulking": ["a towering brute", "massive and heavy-set"]},          # known (noise here)
    "stripe": {"striped": ["banded with bold stripes", "striped all over"],
               "plain": ["a uniform single hue", "plain and unpatterned"]},      # absent primitive A
    "crest": {"crested": ["crowned by a tall crest", "bearing a vivid crest"],
              "bald": ["smooth-headed and bare", "lacking any crest"]},           # absent primitive B
}
KNOWN = ["size"]
_TAG = re.compile(r"\[\[oracle:(\w+)=(\w+)\]\]")


def _detect(text: str, attr: str) -> Optional[str]:
    t = text.lower()
    for value, phrasings in ATTRS[attr].items():
        if any(p.lower() in t for p in phrasings):
            return value
    return None


def _make_corpus(n=2000, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        a = {k: str(rng.choice(list(v))) for k, v in ATTRS.items()}
        y = int((a["stripe"] == "striped") ^ (a["crest"] == "crested"))   # XOR, ~50% base rate
        frags = [str(rng.choice(ATTRS[k][a[k]])) for k in ATTRS]
        rng.shuffle(frags)
        rows.append({"id": i, "text": "The creature is " + ", ".join(frags) + ".", "judgement": y})
    return pd.DataFrame(rows)


def _code_metric(attr: str, value: str, name: str) -> MetricSpec:
    def fn(t):
        v = _detect(t, attr)
        return None if v is None else float(v == value)
    return MetricSpec(_stable_id("code", attr, value), name, name, "code", code_fn=fn, role="feature")


def companion_code() -> List[MetricSpec]:
    return [_code_metric("size", "tiny", "Diminutive size")]   # noise w.r.t. the XOR label


def _oracle_judge(metrics, texts):
    n, M = len(texts), len(metrics)
    lv, ap = np.full((n, M), np.nan), np.zeros((n, M), bool)
    for j, m in enumerate(metrics):
        tag = _TAG.search(m.guidance or "")
        if not tag:
            continue
        attr, value = tag.group(1), tag.group(2)
        for i, t in enumerate(texts):
            d = _detect(t, attr)
            if d is not None:
                lv[i, j], ap[i, j] = float(d == value), True
    return lv, ap


def _oracle_single(prompt: str) -> str:
    """The single-feature proposer: returns '' here because NO single attr separates an XOR
    (each is ~50/50 in both classes) — this is exactly the §9 wall."""
    return ""


def _oracle_composite(prompt: str) -> str:
    """Search pairs of absent attrs + boolean rules; return the composite that separates."""
    def block(header):
        i = prompt.find(header)
        if i == -1:
            return []
        rest = prompt[i + len(header):]
        end = rest.find("\n\n")
        return [s.strip() for s in (rest if end == -1 else rest[:end]).split("\n---\n") if s.strip()]
    pos = block("POSITIVES (label 1):\n")
    neg = block("NEGATIVES (label 0):\n")
    if not pos or not neg:
        return ""
    cands = [a for a in ATTRS if a not in KNOWN]

    def sig(texts, attr):
        v0 = next(iter(ATTRS[attr]))
        return np.array([1.0 if _detect(t, attr) == v0 else 0.0 for t in texts])

    best = None  # (sep, attr_a, attr_b, rule)
    for x in range(len(cands)):
        for y in range(x + 1, len(cands)):
            ax, ay = cands[x], cands[y]
            sap, sbp = sig(pos, ax), sig(pos, ay)
            san, sbn = sig(neg, ax), sig(neg, ay)
            base = max(abs(sap.mean() - san.mean()), abs(sbp.mean() - sbn.mean()))
            for r in RULES:
                pp = apply_rule(sap, sbp, r).mean()
                pn = apply_rule(san, sbn, r).mean()
                sep = abs(pp - pn)
                if sep > base + 0.1 and (best is None or sep > best[0]):
                    best = (sep, ax, ay, r)
    if best is None:
        return ""
    _, ax, ay, rule = best
    vax = next(iter(ATTRS[ax]))
    vay = next(iter(ATTRS[ay]))
    prims = [
        ProposedFeature(name=f"{ax}_{vax}", description=f"is {ax} {vax}",
                        rubric=f"Return 1 if {vax}; else 0. [[oracle:{ax}={vax}]]"),
        ProposedFeature(name=f"{ay}_{vay}", description=f"is {ay} {vay}",
                        rubric=f"Return 1 if {vay}; else 0. [[oracle:{ay}={vay}]]"),
    ]
    return json.dumps({"primitives": [{"name": p.name, "description": p.description, "rubric": p.rubric}
                                      for p in prims], "rule": rule})


def test_loop_recovers_xor_composite():
    cfg = InfillConfig(
        n_permutations=199, min_node_size=40, max_depth=4, random_seed=0,
        max_outer_rounds=3, reliability_sample_size=60,
        gap_deviance_per_item=1.20, gap_auc_threshold=0.60,   # noise size metric -> AUC ~0.5 -> gap
        include_text_length_in_z=False, enable_composite_proposer=True,
    )
    df = _make_corpus()
    df_d, df_t = discover_test_split(df, cfg)
    metrics = companion_code()
    sm_d = materialize(metrics, df_d, cfg, _oracle_judge)
    sm_t = materialize(metrics, df_t, cfg, _oracle_judge)
    res = run_infill(df_d, df_t, metrics, sm_d, sm_t, cfg, _oracle_single, _oracle_judge,
                     log=lambda *a, **k: None, composite_proposer=_oracle_composite)

    kept = [r for r in res.records if r.status == "kept"]
    # no single feature should be kept (the single oracle proposer always returns "")
    assert all("composite" in r.origin for r in kept), [r.origin for r in kept]
    assert len(kept) >= 1, f"no composite kept; records={[(r.name, r.status) for r in res.records]}"
    assert any("xor" in r.name.lower() for r in kept), [r.name for r in kept]
    assert res.final_gap_count == 0, f"gap not closed; final_gaps={res.final_gap_count}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
