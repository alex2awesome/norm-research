#!/usr/bin/env python3
"""Per-answer V features from sympy verification verdicts + AUC vs the floor.

Joins claim-level verdicts (run_verification.py --results) back to answers,
builds per-answer features, restricts to rows present in the v3.3 canonical
dataset, and reports per-feature + combined AUCs on the v3.3 eval/test rows.

Fidelity filter (2026-06-11 spot-check finding): REFUTED is contaminated by
extraction infidelity — definitions ("So if A = ...") and context-dependent
equations extracted as free-variable identities. We therefore compute refuted
features in two flavors:
  raw        — all REFUTED claims
  filtered   — drop claims whose source_quote opens with a binding/hypothesis
               cue (if/let/suppose/define/...) and claims not marked
               load_bearing by the extractor
Headline numbers use the filtered flavor.

Usage (sk3):
  python3 build_v_features.py \
      --claims claims_eval.jsonl --results verif_eval.jsonl \
      --canonical ../math_se_v3_3_propensity_balanced.csv.gz \
      --out v_features_eval.csv
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict

import numpy as np
import pandas as pd

# binding/hypothesis cues: the quote introduces notation or an assumption,
# so "refuting" it as an identity is meaningless
BINDING_RE = re.compile(
    r"^\s*(?:so\s+|now\s+|and\s+|then\s+)?"
    r"(if|let|suppose|assume|define|denote|set|put|write|consider|"
    r"given|say|where)\b", re.I)


def is_binding_quote(quote: str | None) -> bool:
    return bool(quote) and bool(BINDING_RE.match(quote.strip()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--canonical", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    claims = {}
    for line in open(args.claims):
        d = json.loads(line)
        claims[d["claim_id"]] = d

    per_answer = defaultdict(lambda: defaultdict(int))
    meta = {}
    for line in open(args.results):
        r = json.loads(line)
        c = claims.get(r["claim_id"])
        if c is None:
            continue
        rid = c["row_id"]
        meta[rid] = c["judgement"]
        f = per_answer[rid]
        v = r["verdict"]
        if v == "NO_CLAIM":
            f["n_none"] += 1
            continue
        f["n_claims"] += 1
        binding = is_binding_quote(c.get("source_quote"))
        load = bool(c.get("load_bearing"))
        if v.startswith("VERIFIED"):
            f["n_verified"] += 1
            f["n_verified_sym"] += v == "VERIFIED_SYMBOLIC"
        elif v == "REFUTED":
            f["n_refuted_raw"] += 1
            if not binding:
                f["n_refuted_filt"] += 1
                f["n_refuted_load"] += load
        elif v == "INCONCLUSIVE":
            f["n_inconclusive"] += 1
        elif v == "PARSE_FAIL":
            f["n_parse_fail"] += 1

    rows = []
    for rid, f in per_answer.items():
        nc = f["n_claims"]
        rows.append(dict(
            row_id=rid, judgement=meta[rid],
            has_checkable=int(nc > 0), n_claims=nc,
            n_verified=f["n_verified"], n_verified_sym=f["n_verified_sym"],
            n_refuted_raw=f["n_refuted_raw"],
            n_refuted_filt=f["n_refuted_filt"],
            n_refuted_load=f["n_refuted_load"],
            n_inconclusive=f["n_inconclusive"],
            n_parse_fail=f["n_parse_fail"],
            any_verified=int(f["n_verified"] > 0),
            any_refuted_filt=int(f["n_refuted_filt"] > 0),
            any_refuted_load=int(f["n_refuted_load"] > 0),
            frac_verified=f["n_verified"] / nc if nc else 0.0,
        ))
    feat = pd.DataFrame(rows)
    print(f"answers with verdicts: {len(feat):,} "
          f"(checkable: {int(feat.has_checkable.sum()):,} "
          f"= {feat.has_checkable.mean():.1%})")

    # claims' row_id is the SE answer post id == canonical answer_id
    canon = pd.read_csv(args.canonical,
                        usecols=["answer_id", "split", "judgement"])
    canon = canon.rename(columns={"answer_id": "row_id"})
    feat = feat.merge(canon[["row_id", "split"]], on="row_id", how="left")
    feat.to_csv(args.out, index=False)
    print(f"wrote {args.out}")

    sub = feat[feat.split.notna()]
    print(f"\nrows in v3.3 canonical: {len(sub):,} "
          f"({sub.split.value_counts().to_dict()}) "
          f"y-balance: {sub.judgement.mean():.3f}")
    if len(sub) < 200:
        print("too few joined rows for AUC; rerun when extraction covers more")
        return

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    feats = ["has_checkable", "n_claims", "n_verified", "n_verified_sym",
             "n_refuted_raw", "n_refuted_filt", "n_refuted_load",
             "n_inconclusive", "n_parse_fail", "any_verified",
             "any_refuted_filt", "any_refuted_load", "frac_verified"]
    y = sub.judgement.values
    print("\n| feature | AUC |\n|---|---|")
    for c in feats:
        x = sub[c].values
        auc = roc_auc_score(y, x) if np.std(x) > 0 else 0.5
        print(f"| {c} | {auc:.3f} |")
    X = sub[feats].values
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000))
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    aucs = []
    for tr, te in cv.split(X, y):
        clf.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    print(f"\ncombined LR 5-fold CV AUC: {np.mean(aucs):.3f} "
          f"± {np.std(aucs):.3f}  (question-only floor: 0.461)")


if __name__ == "__main__":
    main()
