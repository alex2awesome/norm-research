#!/usr/bin/env python3
"""TRAIN-BIG / EVAL-CANONICAL split for the mathlib accept/reject VERDICT cell
(user-directed correction 2026-08-08; registry entry "MATHLIB AND HOMEPAGE TERMINAL
VERDICTS RETRACTED").

WHY THIS BUILD EXISTS
---------------------
The canonical de-confounded slice (accept_reject_clean.parquet, n=7,956, 94.3% positive
=> only ~450 negatives, ~360 of them in the area-grouped train fold) STARVED the dense
arm: three class-weighted seeds returned eval .6328/.5534/.5066 (mean .5643, spread .126)
and test .4599/.4346/.5065 (mean .467, at/below chance) -- while a class-weighted TF-IDF
LogisticRegression on the IDENTICAL rows/split scored eval .6774 / test .7856. A
bag-of-words beating an 8B LoRA on the same rows means the TRAINING RUN was starved, not
that the cell has no signal.

FIX (the CW stage-0 / "more data" pattern): train on the FULL PRE-AUDIT mathlib
population, EXCLUDING every row whose area-group appears in the canonical eval/test
folds, and EVALUATE on the canonical de-confounded rows unchanged.

  train  = pre-audit population (accept_reject_dataset.csv.gz, n=35,796, joined to
           status-200 diffs in pr_diffs.jsonl) minus rows in the held-out area groups
           minus rows with an empty diff (no artifact to read).
  eval   = VERBATIM copy of dense_standard_cw/split/eval.csv  (canonical, area=Analysis)
  test   = VERBATIM copy of dense_standard_cw/split/test.csv   (canonical, areas=
           CategoryTheory, Control)

Area-grouped holdout integrity is ASSERTED in code: zero area overlap and zero row_id
overlap between train and eval/test.

LABEL-SEMANTICS CAVEAT (recorded in the manifest, bounds interpretation)
------------------------------------------------------------------------
The canonical slice was produced by two filters, both of which the train-big population
deliberately re-admits:
  1. save_deconfounded.py: keep conv_prefix=='feat' AND year>=2025 (drops the mathlib3
     port era and the change-type confound; title-only AUC collapses .638 -> .566 there,
     i.e. that signal WAS confound).
  2. finalize_slice.py:    keep 0 < additions <= 1000 AND drop rejects with
     n_review_threads == 0 (size hygiene + "abandoned, never actually reviewer-rejected"
     negatives).
Filter 2's second clause changes what y=0 MEANS: a pre-audit negative may be an abandoned
PR rather than a reviewer rejection. Training on those rows is acceptable here ONLY
because every EVALUATED row is a canonical de-confounded row, so nothing confounded can
leak into the readout; the cost is a train/eval distribution shift, which makes the
resulting T a conservative (lower) bound on the canonical-row dense ceiling rather than a
best-effort estimate. The manifest records how many train rows come from each re-admitted
stratum.

Text/group definitions are verbatim the canonical C-leg (build_dense_standard.py /
mathlib_remeasure2.py / save_deconf.py): text = diff_noauth (author-stripped diff, no
title), group = top-level Mathlib/<Area>/ path most touched by the diff.

VARIANTS
--------
  --variant bigtrain  (default) the full pre-audit train fold described above, n=29,324.
  --variant regime    the same fold restricted to the audit's REGIME filter
                      (conv_prefix=='feat' AND year>=2025), n=6,651. Motivated by the
                      TF-IDF ablation in notes/2026-08-08__mathlib_homepage_corrections.md:
                      the whole distribution-shift cost of training big is carried by the
                      regime stratum, while the hygiene strata are nearly free -- so this
                      arm keeps 2.1x the canonical fold's negatives (780 vs 363) at a
                      matched regime, and separates "more negatives" from "shifted rows".

Usage (CPU only):
  python3 build_dense_bigtrain.py [--variant bigtrain|regime]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT_BY_VARIANT = {"bigtrain": HERE / "dense_standard_bigtrain",
                  "regime": HERE / "dense_standard_regimematched"}
CANON = HERE / "dense_standard_cw" / "split"  # canonical de-confounded area-grouped folds

RAW_CANDIDATES = [
    HERE / "accept_reject_dataset.csv.gz",
    Path("/lfs/skampere3/0/alexspan/norm-research/datasets/math/mathlib/accept_reject_dataset.csv.gz"),
]
DIFF_CANDIDATES = [
    HERE / "pr_diffs.jsonl",
    Path("/lfs/skampere3/0/alexspan/norm-research/datasets/math/mathlib/pr_diffs.jsonl"),
]

AUTHOR_RE = re.compile(
    r"Copyright|Authors:|Released under|Apache 2\.0|described in the file LICENSE|SPDX-License|maintainer",
    re.I,
)
AREA_RE = re.compile(r"(?:a|b)/Mathlib/([A-Za-z0-9_]+)/")


def strip_author(d: str) -> str:
    return "\n".join(l for l in str(d).split("\n") if not AUTHOR_RE.search(l))


def area(d: str) -> str:
    ms = AREA_RE.findall(str(d))
    return Counter(ms).most_common(1)[0][0] if ms else "NONE"


def pick(cands):
    p = next((c for c in cands if c.exists()), None)
    assert p is not None, f"none of {cands} exist"
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=sorted(OUT_BY_VARIANT), default="bigtrain")
    variant = ap.parse_args().variant
    OUT = OUT_BY_VARIANT[variant]
    print(f"variant={variant} -> {OUT}")
    raw_p, diff_p = pick(RAW_CANDIDATES), pick(DIFF_CANDIDATES)
    print(f"raw={raw_p}\ndiffs={diff_p}")

    diffs = {}
    with open(diff_p) as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("status") == 200:
                diffs[r["number"]] = r["diff"]
    print(f"status-200 diffs: {len(diffs)}")

    raw = pd.read_csv(raw_p)
    n_raw = len(raw)
    raw["diff"] = raw["number"].map(diffs.get)
    n_with_diff = int(raw["diff"].notna().sum())
    raw = raw[raw["diff"].notna()].copy()
    raw["diff_noauth"] = raw["diff"].astype(str).map(strip_author)
    raw["area"] = raw["diff"].astype(str).map(area)
    n_empty = int((raw["diff"].astype(str).str.strip() == "").sum())
    raw = raw[raw["diff"].astype(str).str.strip() != ""].copy()
    print(f"pre-audit rows: raw={n_raw} with_diff={n_with_diff} empty_diff_dropped={n_empty} "
          f"usable={len(raw)} pos_rate={raw.judgement.mean():.4f}")

    # ---- canonical eval/test folds, taken VERBATIM ------------------------------------
    ev = pd.read_csv(CANON / "eval.csv")
    te = pd.read_csv(CANON / "test.csv")
    held_areas = sorted(set(ev["group"]) | set(te["group"]))
    held_ids = set(ev["row_id"].astype(str)) | set(te["row_id"].astype(str))
    print(f"canonical eval n={len(ev)} areas={sorted(set(ev['group']))} | "
          f"test n={len(te)} areas={sorted(set(te['group']))}")

    # ---- PROVENANCE ASSERTION: the pre-audit join reproduces the canonical text --------
    by_num = {str(k): v for k, v in zip(raw["number"], raw["diff_noauth"])}
    by_area = {str(k): v for k, v in zip(raw["number"], raw["area"])}
    n_checked = n_text_match = n_area_match = 0
    for frame in (ev, te):
        for rid, txt, grp in zip(frame["row_id"].astype(str), frame["text"].astype(str),
                                 frame["group"].astype(str)):
            n_checked += 1
            n_text_match += int(by_num.get(rid) == txt)
            n_area_match += int(by_area.get(rid) == grp)
    assert n_text_match == n_checked, (
        f"text provenance FAIL: {n_text_match}/{n_checked} canonical rows reproduce from "
        f"pr_diffs.jsonl + strip_author")
    assert n_area_match == n_checked, f"area provenance FAIL: {n_area_match}/{n_checked}"
    print(f"ASSERTION PASS (provenance): all {n_checked} canonical eval+test rows reproduce "
          f"byte-identically (text AND area) from the pre-audit source")

    # ---- TRAIN-BIG: drop the held-out area groups --------------------------------------
    tr = raw[~raw["area"].isin(held_areas)].copy()
    if variant == "regime":
        keep = (tr["conv_prefix"] == "feat") & (tr["year"] >= 2025)
        print(f"variant=regime: keeping {int(keep.sum())} of {len(tr)} rows "
              f"(conv_prefix=='feat' AND year>=2025)")
        tr = tr[keep].copy()
    assert len(set(tr["area"]) & set(held_areas)) == 0, "AREA OVERLAP -- holdout integrity FAIL"
    overlap_ids = set(tr["number"].astype(str)) & held_ids
    assert not overlap_ids, f"ROW OVERLAP -- {len(overlap_ids)} train rows also in eval/test"
    print(f"ASSERTION PASS (holdout integrity): 0 area overlap "
          f"({len(set(tr['area']))} train areas vs held {held_areas}), 0 row_id overlap")

    train_rows = [
        {"text": r.diff_noauth, "judgement": int(r.judgement), "group": r.area,
         "row_id": str(r.number)}
        for r in tr.itertuples()
    ]

    # ---- re-admitted-confound census (bounds interpretation) ---------------------------
    is_feat_2025 = (tr["conv_prefix"] == "feat") & (tr["year"] >= 2025)
    size_ok = (tr["additions"] > 0) & (tr["additions"] <= 1000)
    abandoned_rej = (tr["judgement"] == 0) & (tr["n_review_threads"] == 0)
    canonical_like = is_feat_2025 & size_ok & ~abandoned_rej
    strata = {
        "train_rows_total": int(len(tr)),
        "train_negatives_total": int((tr["judgement"] == 0).sum()),
        "readmitted_pre2025_or_non_feat": int((~is_feat_2025).sum()),
        "readmitted_size_outlier_add_gt_1000_or_0": int((~size_ok).sum()),
        "readmitted_abandoned_rejects_no_review_thread": int(abandoned_rej.sum()),
        "rows_passing_the_full_canonical_filter": int(canonical_like.sum()),
        "negatives_passing_the_full_canonical_filter": int(((tr["judgement"] == 0) & canonical_like).sum()),
    }
    print("re-admitted strata:", json.dumps(strata, indent=2))

    # ---- write --------------------------------------------------------------------------
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "split").mkdir(exist_ok=True)
    cols = ["text", "judgement", "group", "row_id"]
    with open(OUT / "split" / "train.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(train_rows)
    ev[cols].to_csv(OUT / "split" / "eval.csv", index=False)
    te[cols].to_csv(OUT / "split" / "test.csv", index=False)
    all_rows = train_rows + ev[cols].to_dict("records") + te[cols].to_dict("records")
    with open(OUT / "data.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(all_rows)

    n_tot = len(all_rows)
    manifest = {
        "cell": f"mathlib_verdict (TRAIN-BIG / EVAL-CANONICAL, variant={variant})",
        "variant": variant,
        "variant_meaning": {
            "bigtrain": "full pre-audit train fold (all strata re-admitted)",
            "regime": "pre-audit train fold restricted to the audit's REGIME filter "
                      "(conv_prefix=='feat' AND year>=2025); hygiene strata still re-admitted",
        }[variant],
        "why": "the canonical n=7,956 / 94.3%-positive slice starved the dense arm (3 class-weighted "
               "seeds: eval mean .5643 spread .126, test mean .467) while class-weighted TF-IDF on the "
               "IDENTICAL split scored .6774/.7856 -- a training-set-size failure, not a dead cell. "
               "User-directed correction 2026-08-08.",
        "design": "train on the FULL pre-audit population minus the canonical eval/test area groups; "
                  "evaluate on the canonical de-confounded rows verbatim (CW stage-0 pattern).",
        "train_source": str(raw_p) + " + " + str(diff_p) + " (status-200 diffs)",
        "eval_test_source": str(CANON) + " (verbatim copies; canonical de-confounded "
                            "accept_reject_clean.parquet rows, area-grouped folds)",
        "text": "diff_noauth (author-stripped diff, no title) -- identical definition on all three splits",
        "group_column": "area (top-level Mathlib/<Area>/ path most touched by the diff)",
        "y_definition": "1=accept(merged) 0=reject(closed-unmerged)",
        "pre_audit_population": {
            "raw_rows": n_raw, "rows_with_status200_diff": n_with_diff,
            "empty_diff_dropped": n_empty, "usable": int(len(raw)),
        },
        "held_out_area_groups": held_areas,
        "assertion_area_grouped_holdout": (
            f"train areas ({len(set(tr['area']))}) INTERSECT held areas {held_areas} = EMPTY; "
            f"train row_ids INTERSECT eval+test row_ids = EMPTY -- PASS"
        ),
        "assertion_provenance": (
            f"all {n_checked} canonical eval+test rows reproduce byte-identically (text and area) "
            f"from accept_reject_dataset.csv.gz + pr_diffs.jsonl + the canonical strip_author/area "
            f"functions -- PASS (so train and eval share one text pipeline)"
        ),
        "split_row_counts": {"train": len(train_rows), "eval": len(ev), "test": len(te)},
        "split_fractions": {"train": len(train_rows) / n_tot, "eval": len(ev) / n_tot,
                            "test": len(te) / n_tot},
        "split_pos_rates": {
            "train": sum(r["judgement"] for r in train_rows) / len(train_rows),
            "eval": float(ev["judgement"].mean()), "test": float(te["judgement"].mean()),
        },
        "split_group_counts": {"train": len(set(r["group"] for r in train_rows)),
                               "eval": ev["group"].nunique(), "test": te["group"].nunique()},
        "readmitted_confound_strata": strata,
        "interpretation_bound": (
            "The train fold re-admits three strata the de-confounding audit removed: the pre-2025 / "
            "non-'feat' regime (port era + change-type confound), size outliers, and 'abandoned' "
            "rejects (closed with zero review threads, i.e. y=0 without a reviewer decision). The "
            "third changes the MEANING of y=0 in training. Because every evaluated row is a canonical "
            "de-confounded row and the areas are disjoint, nothing confounded can leak into the "
            "readout; the cost is a train/eval distribution shift, so the resulting T is a "
            "CONSERVATIVE lower bound on the canonical-row dense ceiling, not a best-effort estimate."
        ),
        "recipe": "Llama-3.1-8B LoRA r16/a32, lr5e-5, batch16, max_len1024, 2 epochs, "
                  "gradient-checkpointing, --class_weight_auto, select-on-eval (dense standard + the "
                  "class-weighting flag this cell's imbalance requires)",
        "trainer_note": "run via methods/dense/run_bigtrain_eval_canonical.py -- a thin harness that "
                        "calls train_reward_model.train() unmodified after relaxing its hard-coded "
                        "80/10/10 on-disk split-ratio assertion (a train-big design is 94.6/2.9/2.5 "
                        "by construction). No shared file is edited.",
    }
    with open(OUT / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(json.dumps({k: v for k, v in manifest.items() if k != "interpretation_bound"}, indent=2))
    print("BUILD_DONE")


if __name__ == "__main__":
    main()
