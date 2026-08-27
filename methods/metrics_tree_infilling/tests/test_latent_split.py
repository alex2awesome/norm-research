"""Can a *discovered* feature become a new splitting variable? (latent region, no context)

The creature-dossier scenario gave the tree an observed context covariate (habitat) to split
on. This test removes that crutch entirely: the known metrics are model terms only (they are
*not* offered as splitting covariates), and there is no context column at all. The region
where the published rules apply is marked by a property that lives ONLY in the prose. The loop
must discover that property and — with ``discovered_feature_role="both"`` — the tree must then
*split on the discovered feature* to carve out the latent region.

Construction: a discovered feature that GATES the whole Code.
  - known model metrics: ``size``, ``feeding``, ``pelt`` (role=feature, X only — never split on)
  - hidden feature: ``crest`` (crested / plain), present only in the text
  - verdict:  for CRESTED creatures the published Code applies (small + gentle + soft are kept);
              for PLAIN creatures the Code is irrelevant (a low base rate).
Because ``crest`` gates all three criteria at once, splitting on it resolves far more
instability than splitting on any single criterion — so the tree should split on the
*discovered* ``crest``. ``crest`` also carries a marginal effect (crested kept more overall),
so the contrast can surface it in the first place. ``color`` is a decoy.
"""

from __future__ import annotations

import json
import re
from typing import List, Optional

import numpy as np
import pandas as pd
import pytest

from methods.metrics_tree_infilling.config import InfillConfig
from methods.metrics_tree_infilling.io_metrics import MetricSpec, _stable_id, materialize, discover_test_split
from methods.metrics_tree_infilling.loop import run_infill

# --- a tiny world (self-contained) ----------------------------------------------------

ATTRS = {
    "size": {"tiny": ["no bigger than a hare", "small and slight of frame"],
             "hulking": ["a towering brute", "massive and heavy-set"]},
    "feeding": {"grazer": ["grazes on moss and leaves", "browses gently on fruit"],
                "hunter": ["stalks and devours smaller beasts", "hunts live prey by night"]},
    "pelt": {"furred": ["covered in soft fur", "wrapped in a downy pelt"],
             "scaled": ["sheathed in hard scales", "clad in glossy plates"]},
    "crest": {"crested": ["crowned with a tall feathered crest", "bearing an ornate plumed crest",
                          "its head topped by a vivid crest"],
              "plain": ["with a smooth, bare head", "plain-headed and unadorned",
                        "its head lacking any crest"]},
    "color": {"azure": ["washed in deep azure", "with a cool blue sheen"],
              "ochre": ["a dull ochre tone", "muddy yellow-brown all over"]},  # decoy
}
KNOWN = ["size", "feeding", "pelt"]


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
        if a["crest"] == "crested":          # the Code applies
            logit = (0.6
                     + 1.2 * (1 if a["size"] == "tiny" else -1)
                     + 1.2 * (1 if a["feeding"] == "grazer" else -1)
                     + 1.2 * (1 if a["pelt"] == "furred" else -1))
        else:                                # plain: Code irrelevant, low base rate
            logit = -0.8
        y = int(rng.random() < 1.0 / (1.0 + np.exp(-logit)))
        frags = [str(rng.choice(ATTRS[k][a[k]])) for k in ATTRS]
        rng.shuffle(frags)
        rows.append({"id": i, "text": "The creature is " + ", ".join(frags) + ".", "judgement": y})
    return pd.DataFrame(rows)


# --- known metrics (model terms only) + deterministic offline oracle ------------------

def _code_metric(attr: str, value: str, name: str) -> MetricSpec:
    def fn(t):
        v = _detect(t, attr)
        return None if v is None else float(v == value)
    return MetricSpec(_stable_id("code", attr, value), name, name, "code", code_fn=fn, role="feature")


def companion_code() -> List[MetricSpec]:
    return [_code_metric("size", "tiny", "Diminutive size"),
            _code_metric("feeding", "grazer", "Gentle feeder"),
            _code_metric("pelt", "furred", "Soft pelt")]


_TAG = re.compile(r"\[\[oracle:(\w+)=(\w+)\]\]")


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


def _oracle_proposer(prompt: str) -> str:
    def block(header):
        i = prompt.find(header)
        if i == -1:
            return []
        rest = prompt[i + len(header):]
        end = rest.find("\n\n")
        chunk = (rest if end == -1 else rest[:end]).strip()
        return [s.strip() for s in chunk.split("\n---\n") if s.strip()]
    pos = block("POSITIVES (label 1):\n")
    neg = block("NEGATIVES (label 0):\n")
    if not pos or not neg:
        return ""
    best = None
    for attr, values in ATTRS.items():
        if attr in KNOWN:        # honor "find a property not already covered by known criteria"
            continue
        v0 = next(iter(values))
        pp = float(np.mean([_detect(t, attr) == v0 for t in pos]))
        pn = float(np.mean([_detect(t, attr) == v0 for t in neg]))
        sep = abs(pp - pn)
        if best is None or sep > best[2]:
            best = (attr, v0 if pp >= pn else [v for v in values if v != v0][0], sep)
    if best is None or best[2] < 0.15:
        return ""
    attr, value, _ = best
    return json.dumps({"name": f"{attr}_{value}", "description": f"whether the creature is {value}",
                       "rubric": f"Return 1 if the creature is {value}; else 0. [[oracle:{attr}={value}]]"})


# --- the test -------------------------------------------------------------------------

def _split_variables(tree) -> List[str]:
    return [n.split.variable for n in tree.all_nodes() if n.split is not None]


def test_discovered_feature_becomes_a_splitting_variable():
    cfg = InfillConfig(
        n_permutations=199, min_node_size=40, max_depth=4, random_seed=0, max_outer_rounds=4,
        gap_deviance_per_item=1.15, gap_auc_threshold=0.56, include_text_length_in_z=False,
        discovered_feature_role="both",   # <-- let a discovered feature ALSO become a split
    )
    df = _make_corpus()
    df_d, df_t = discover_test_split(df, cfg)
    metrics = companion_code()
    sm_d = materialize(metrics, df_d, cfg, _oracle_judge)
    sm_t = materialize(metrics, df_t, cfg, _oracle_judge)
    res = run_infill(df_d, df_t, metrics, sm_d, sm_t, cfg, _oracle_proposer, _oracle_judge,
                     log=lambda *a, **k: None)

    kept = {r.name for r in res.records if r.status == "kept"}
    assert any("crest" in n.lower() for n in kept), f"crest not discovered; kept={kept}"
    assert not any("color" in n.lower() for n in kept), f"decoy discovered: {kept}"

    # The discovered crest feature must appear as a SPLIT variable — a partitioning axis the
    # tree gained ONLY by discovering it (no context column ever encoded the latent region;
    # the known criteria are model terms and are never split on).
    crest_name = next(n for n in kept if "crest" in n.lower())
    split_vars = _split_variables(res.tree)
    assert crest_name in split_vars, \
        f"discovered feature {crest_name!r} did not become a split; splits were {split_vars}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
