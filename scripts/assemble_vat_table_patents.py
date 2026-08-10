#!/usr/bin/env python3
"""Assemble the final patents VAT table (task #50).  RUN ON SK3.

Pulls together every wrap-up artifact:
  V claim-level : datasets/patents/processed/truecite_testbed_v1/
                    v61_ablation_results.txt        (v6a vs v6.1a min_of_max)
  V app-level   : datasets/patents/processed/phase2_dataset_v0/app_level_v/
                    app_v_scores.json               (+ control falsification)
  V per-class   : app_v_scores joined to raw/oard/oard_rejections_by_app.csv
  A per-class   : runs/validity_full/v2/patents/per_class_a_results.json
  Dense ceiling : hardcoded from sweep history (+ 0_7 if available)

Output: notes/2026-06-12__patents-vat-final-table.md (markdown, printed too).

DENSE-CEILING CAVEAT (dual audit Codex+Fable 2026-07-13): the DENSE dict below is
PROVENANCE-ORPHANED and CROSS-COHORT. The .70-.74 values trace to old baseline-sweep
validation_metrics.csv files with no dataset/split provenance; naming indicates they were
trained on the NAIVE globally-balanced files (07_balance) where CPC/era/length are freely
predictive — NOT the cite-bearing CPC×length-balanced cohort the V/A numbers live on. The
training protocol also used row-random label-stratified splits (duplicate texts + families
straddle train/test), test-split checkpoint selection by default, and max_length=1024 tokens
(44% of rows longer). "Gap to dense ceiling = taste + headroom" is therefore NOT sound as
cross-cohort arithmetic: no dense run on the actual residual-V cohort exists. Do not quote
the gap until one does (see scripts/patents_y_power.py for the honest-baseline program).
"""
import csv
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = os.path.expanduser("~/norm-research")
TB = f"{ROOT}/datasets/patents/processed/truecite_testbed_v1"
AV = f"{ROOT}/datasets/patents/processed/phase2_dataset_v0/app_level_v"
OARD = f"{ROOT}/datasets/patents/raw/oard/oard_rejections_by_app.csv"
A_RES = f"{ROOT}/runs/validity_full/v2/patents/per_class_a_results.json"
OUT = f"{ROOT}/notes/2026-06-12__patents-vat-final-table.md"

DENSE = {"first_draft (llama-8b)": 0.7413, "first_draft_cpc": 0.7349,
         "final_outcome": 0.7005, "+applicant cites": 0.7445,
         "+examiner cites": 0.712}
CLAIM_V = {"v0 max-cos v3": 0.502, "v0 max-cos v6a": 0.569,
           "v2 min_of_max (v6a retr + qwen judge)": 0.574}
CLASSES = ["rejected_101", "rejected_102", "rejected_103",
           "rejected_112a", "rejected_112b"]

lines = ["# Patents VAT decomposition — final table (2026-06-12)", ""]


def add(s=""):
    lines.append(s)
    print(s)


# ---- V claim-level: v6.1a ablation ----
add("## V claim-level (true-cites testbed, independent claims)")
add()
for k, v in CLAIM_V.items():
    add(f"- {k}: **{v}**")
abl = f"{TB}/v61_ablation_results.txt"
if os.path.exists(abl):
    add()
    add("### v6.1a ablation (overnight)")
    add("```")
    add(open(abl).read().strip())
    add("```")
else:
    add(f"- v6.1a ablation: NOT LANDED YET ({abl})")
add()

# ---- V app-level + control + per-class ----
add("## V app-level (3K main + 1K zero-resolved control apps)")
add()
scores_f = f"{AV}/app_v_scores.json"
if os.path.exists(scores_f):
    rows = json.load(open(scores_f))
    main = [r for r in rows if not r["is_ctrl"]]
    ctrl = [r for r in rows if r["is_ctrl"]]

    def auc(rs):
        y = np.array([r["judgement"] for r in rs])
        v = np.array([1 - r["v_min_of_max"] / 100 for r in rs])
        return roc_auc_score(y, v) if len(set(y)) == 2 else None

    add(f"- main: n={len(main)}, **AUC={auc(main):.4f}**")
    if ctrl:
        add(f"- control (falsification, should be ~0.5): n={len(ctrl)}, "
            f"AUC={auc(ctrl):.4f}")
    # per-class slices via OARD
    if os.path.exists(OARD):
        flags = {}
        with open(OARD) as f:
            for row in csv.DictReader(f):
                flags[row["app_id"]] = row
        add()
        add("### V per rejection class (main apps, accepted vs class-rejected)")
        add()
        add("| class | n_rej | V AUC |")
        add("|---|---|---|")
        acc = [r for r in main if r["judgement"] == 1]
        for cls in CLASSES:
            rej = [r for r in main if r["judgement"] == 0
                   and str(flags.get(str(r["app_id"]), {}).get(cls, "0"))
                   in ("1", "True", "1.0")]
            if len(rej) < 5:
                add(f"| {cls} | {len(rej)} | too few |")
                continue
            add(f"| {cls} | {len(rej)} | {auc(acc + rej):.4f} |")
else:
    add(f"- NOT LANDED YET ({scores_f})")
add()

# ---- A per-class ----
add("## A per rejection class (qwen judge, pooled CV logistic over aspects)")
add()
if os.path.exists(A_RES):
    a = [r for r in json.load(open(A_RES))
         if r["judge"].startswith("qwen")][0]
    add("| class | n_rej | A AUC | 95% CI |")
    add("|---|---|---|---|")
    for cls, e in a["classes"].items():
        if "pooled_auc" in e:
            add(f"| {cls} | {e['n_rejected']} | {e['pooled_auc']:.3f} | "
                f"[{e['ci'][0]}, {e['ci'][1]}] |")
else:
    add(f"- missing {A_RES} (run scripts/per_class_a_analysis.py on laptop, "
        "scp here)")
add()

# ---- dense ceiling ----
add("## Dense ceiling (Llama-8B reward model)")
add()
for k, v in DENSE.items():
    add(f"- {k}: {v}")
hist = (f"{ROOT}/outputs/dense_sweeps/patents_first_draft_cpc/subset_0_7/"
        "training_history.json")
if os.path.exists(hist):
    h = json.load(open(hist))
    best = max((e.get("eval_auc", 0) for e in h
                if isinstance(e, dict)), default=None)
    add(f"- first_draft_cpc subset_0_7 (text-only, this run): {best}")
add()
add("## Reading")
add()
add("- SS101 is ARTICULABLE (A~0.78) but V-blind. SS102/103 at ~0.50-0.56 is a "
    "DOC-ONLY-judge measurement — WITHDRAWN as an A-blindness claim (2026-07-10): "
    "prior-art grounds need the search record in the judge's context (WS3, "
    "datasets/patents/2026-07-10__evidence_aware_judge_ws3.md); doc-only judges "
    "are structurally blind there, so chance AUC is an instrument artifact.")
add("- Control falsification: app-level V on zero-resolved apps should "
    "collapse to ~0.5; if it doesn't, V is leaking app metadata.")
add("- Gap to dense ceiling (~0.74) = taste + un-implemented V/A headroom.")

with open(OUT, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"\nwrote {OUT}")
