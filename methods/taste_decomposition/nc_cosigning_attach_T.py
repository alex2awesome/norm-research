#!/usr/bin/env python3
"""V8 co-signing cell: fold the dense arm (T) into the Layer-1 ledger and emit
the final results/nc_cosigning_ledger.json.

FREEZE CHANGE 2 (same-rows T) holds by construction here: the dense arm was
trained on the identical 9,520-unit population with the identical docket-grouped
split, so its eval/test rows are a strict subset of the A/V-scored rows.

Because this cell's eval/test splits carry only ~34 positives each, T is
reported with a DOCKET-LEVEL bootstrap CI and a seed spread beside the point
estimate, and Delta_beyond is quoted against the seed-mean with that CI carried
through. Within-docket T is reported beside pooled T, matching the
grouped-transfer table the Layer-1 script builds for V/A/VA.

Usage:
  python methods/taste_decomposition/nc_cosigning_attach_T.py \
      --preds-dir datasets/notice-and-comment/cosigning/dense_llama/cosign
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
RESULTS = Path(__file__).resolve().parent / "results"


def read_preds(p: Path):
    y, s, g = [], [], []
    with open(p, newline="") as fh:
        for r in csv.DictReader(fh):
            y.append(int(float(r["judgement"])))
            s.append(float(r["prob"]))
            g.append(r.get("group", ""))
    return np.array(y), np.array(s), np.array(g, dtype=object)


def within_group_auc(y, s, g):
    tot, acc, n = 0.0, 0.0, 0
    for u in np.unique(g):
        m = g == u
        if len(np.unique(y[m])) < 2:
            continue
        w = float(y[m].sum() * (len(y[m]) - y[m].sum()))
        acc += w * roc_auc_score(y[m], s[m]); tot += w; n += 1
    return (acc / tot if tot else float("nan")), n


def group_bootstrap_auc(y, s, g, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    uniq = np.unique(g)
    idx_by = {u: np.where(g == u)[0] for u in uniq}
    vals = []
    for _ in range(n_boot):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[u] for u in draw])
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(roc_auc_score(y[idx], s[idx]))
    v = np.array(vals)
    return {"mean": float(v.mean()),
            "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
            "n_boot_used": int(len(v))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds-dir", required=True)
    ap.add_argument("--layer1", default=str(RESULTS / "nc_cosigning_layer1_cosign.json"))
    ap.add_argument("--layer1-nearby", default=str(RESULTS / "nc_cosigning_layer1_nearby.json"))
    ap.add_argument("--out", default=str(RESULTS / "nc_cosigning_ledger.json"))
    args = ap.parse_args()

    D = Path(args.preds_dir)
    led = json.loads(Path(args.layer1).read_text())

    dense = {"recipe": ("dense standard, frozen: Llama-3.1-8B LoRA r16/a32, lr5e-5, "
                        "batch16, max_len1024, 2 epochs, gradient-checkpointing, "
                        "select-on-eval; NO class_weight_auto (not in the frozen recipe)"),
             "same_rows": ("YES by construction -- trained on the identical 9,520-unit "
                           "population and the identical docket-grouped split"),
             "seeds": {}}
    per_split = defaultdict(list)
    pooled_scores = {}
    for run in sorted(D.glob("rm_out_seed*")):
        seed = run.name.replace("rm_out_seed", "")
        rec = {}
        for split in ("eval", "test"):
            p = run / f"preds_{split}.csv"
            if not p.exists():
                continue
            y, s, g = read_preds(p)
            a = float(roc_auc_score(y, s))
            w, nd = within_group_auc(y, s, g)
            rec[split] = {"auc": a, "n": int(len(y)), "n_pos": int(y.sum()),
                          "within_docket_auc": w, "n_mixed_dockets": nd}
            per_split[split].append(a)
            pooled_scores.setdefault(split, {"y": y, "g": g, "s": []})["s"].append(s)
        if rec:
            dense["seeds"][f"seed{seed}"] = rec

    dense["seed_mean"] = {}
    for split, arr in per_split.items():
        dense["seed_mean"][split] = {"mean_auc": float(np.mean(arr)),
                                     "spread": float(max(arr) - min(arr)),
                                     "n_seeds": len(arr)}
    # seed-mean probability ensemble -> the quoted T, with a docket bootstrap CI
    for split, d in pooled_scores.items():
        sm = np.mean(d["s"], axis=0)
        w, nd = within_group_auc(d["y"], sm, d["g"])
        dense["seed_mean"][split].update({
            "auc_of_seedmean_probs": float(roc_auc_score(d["y"], sm)),
            "within_docket_auc_of_seedmean_probs": w,
            "n_mixed_dockets": nd,
            "docket_bootstrap": group_bootstrap_auc(d["y"], sm, d["g"])})

    # Program convention (VAT full-grid note): report per-seed AUCs AND the
    # mean-probability ensemble, and use the ENSEMBLE as the ledger row.
    def _T(split):
        d = dense["seed_mean"].get(split, {})
        return d.get("auc_of_seedmean_probs", d.get("mean_auc"))

    T_eval, T_test = _T("eval"), _T("test")
    L = led["ledger"]
    T_primary = T_eval
    L2 = dict(L)
    L2["T_eval"] = T_eval
    L2["T_test"] = T_test
    L2["T_quoted"] = T_primary
    L2["Delta_total_POOLED_VA"] = (T_primary - L["VA_lin"]) if T_primary is not None else None
    L2["Delta_beyond_POOLED_VA"] = (T_primary - L["VA_nl_mean"]) if T_primary is not None else None

    # ---- STRICT same-rows Delta_beyond (stronger than the pooled convention) --
    # FREEZE CHANGE 2 asks for a same-rows T. The dense arm's eval/test rows are
    # a subset of the A/V-scored population, so instead of differencing T (951 or
    # 952 rows) against VA_nl pooled over all 9,520, restrict the grouped-OOF
    # VA_lin/VA_nl predictions to EXACTLY the dense split's rows and difference
    # there. Every VA prediction used is still out-of-fold. test is the clean
    # leg (eval was consumed by checkpoint selection).
    RES = Path(__file__).resolve().parent / "results"
    pop = {json.loads(l)["doc_id"]: json.loads(l)
           for l in open(REPO / "datasets/notice-and-comment/cosigning/cosign_population.jsonl")}
    ids = np.load(RES / "nc_cosigning_cosign_doc_ids.npy", allow_pickle=True)
    va_nl = np.load(RES / "nc_cosigning_cosign_va_nl_oof_mean3.npy")
    va_lin = np.load(RES / "nc_cosigning_cosign_va_lin_oof.npy")
    yv = np.array([pop[d]["y_cosign"] for d in ids])
    sp = np.array([pop[d]["split"] for d in ids], dtype=object)
    same_rows = {}
    for split, Tv in (("eval", T_eval), ("test", T_test)):
        m = sp == split
        if m.sum() == 0 or len(np.unique(yv[m])) < 2:
            continue
        vn = float(roc_auc_score(yv[m], va_nl[m]))
        vl = float(roc_auc_score(yv[m], va_lin[m]))
        same_rows[split] = {
            "n_rows": int(m.sum()), "n_pos": int(yv[m].sum()),
            "VA_lin_same_rows": vl, "VA_nl_same_rows": vn, "T": Tv,
            "Delta_interact_same_rows": vn - vl,
            "Delta_total_same_rows": (Tv - vl) if Tv is not None else None,
            "Delta_beyond_same_rows": (Tv - vn) if Tv is not None else None}
    # ---- PAIRED docket bootstrap on same-rows Delta_beyond -----------------
    # With ~34 positives per held-out split the point estimate is not usable on
    # its own; the gate must be read off the interval. Rows of split/<s>.csv are
    # written from the population in (docket, doc_id) order, so the dense preds
    # align to the population rows exactly -- asserted via the judgement column.
    for split in list(same_rows):
        sel = sorted([u for u in pop.values() if u["split"] == split],
                     key=lambda u: (u["docket"], u["doc_id"]))
        sel_ids = [u["doc_id"] for u in sel]
        pos_of = {d: i for i, d in enumerate(ids)}
        vn_s = np.array([va_nl[pos_of[d]] for d in sel_ids])
        y_s = np.array([pop[d]["y_cosign"] for d in sel_ids])
        g_s = np.array([pop[d]["docket"] for d in sel_ids], dtype=object)
        probs = []
        for run in sorted(D.glob("rm_out_seed*")):
            p = run / f"preds_{split}.csv"
            if not p.exists():
                continue
            yy, ss, _ = read_preds(p)
            assert np.array_equal(yy, y_s), f"{split}: dense/pop row alignment broken"
            probs.append(ss)
        if not probs:
            continue
        t_s = np.mean(probs, axis=0)
        rng = np.random.default_rng(0)
        uniq = np.unique(g_s)
        idx_by = {u: np.where(g_s == u)[0] for u in uniq}
        dl = []
        for _ in range(2000):
            draw = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([idx_by[u] for u in draw])
            if len(np.unique(y_s[idx])) < 2:
                continue
            dl.append(roc_auc_score(y_s[idx], t_s[idx]) - roc_auc_score(y_s[idx], vn_s[idx]))
        dl = np.array(dl)
        same_rows[split]["T_seedmean_probs"] = float(roc_auc_score(y_s, t_s))
        same_rows[split]["Delta_beyond_seedmean_probs"] = float(
            roc_auc_score(y_s, t_s) - roc_auc_score(y_s, vn_s))
        same_rows[split]["Delta_beyond_paired_docket_bootstrap"] = {
            "mean": float(dl.mean()),
            "ci95": [float(np.percentile(dl, 2.5)), float(np.percentile(dl, 97.5))],
            "p_gt_0": float((dl > 0).mean()),
            "p_gt_gate_02": float((dl > 0.02).mean()),
            "n_boot_used": int(len(dl))}

    L2["same_rows"] = same_rows
    if "test" in same_rows:
        L2["Delta_beyond_SAME_ROWS_test_QUOTED"] = same_rows["test"]["Delta_beyond_same_rows"]
    if "eval" in same_rows:
        L2["Delta_beyond_SAME_ROWS_eval"] = same_rows["eval"]["Delta_beyond_same_rows"]

    out = {
        "cell": "N&C co-signing (V8) -- regulatory field VOTE/REVEALED column",
        "built": "2026-08-08",
        "y_definition": led["y_definition"],
        "y_channels_rejected": {
            "regulations_gov_Duplicate_Comments_field":
                "DEAD: >1 on 6,514 of 11,698,149 comment rows (0.06%)",
            "shipped_minhash_dedup_mapper_as_primary":
                ("partial, directory-varying coverage -- ground truth AMS-NOP-17-0031: "
                 "47,108 docs incl. an 8,258-member byte-identical family, mapper covers "
                 "30,661 with max cluster 1,879; stale .csv sibling over-merges "
                 "(8,501-member cluster for a 1-copy text). Kept only as y_nearby.")},
        "n": led["n"], "pos_rate": led["pos_rate"], "n_pos": led["n_pos"],
        "n_groups": led["n_groups"], "group_column": "docket",
        "reuse": {"a_bank": led["a_bank"], "v_features": led["v_features"],
                  "population_join_coverage": "9,521/9,524 scored doc_ids (99.97%)",
                  "new_judging": "NONE"},
        "split_audit": led["split_audit"],
        "linear": led["linear"], "nonlinear": led["nonlinear"],
        "ledger": L2,
        "delta_interact_bootstrap_docket_PRIMARY": led["delta_interact_bootstrap_docket_PRIMARY"],
        "grouped_transfer": led["grouped_transfer"],
        "nuisance": led["nuisance"],
        "cross_y": led["cross_y"],
        "dense": dense,
    }

    nb = Path(args.layer1_nearby)
    if nb.exists():
        n = json.loads(nb.read_text())
        out["sensitivity_y_nearby"] = {
            "note": ("shipped MinHash near-dup family >=2; PARTIAL, directory-varying "
                     "coverage -- direction check only, never a headline"),
            "n": n["n"], "pos_rate": n["pos_rate"], "ledger": n["ledger"],
            "grouped_transfer": {k: v for k, v in n["grouped_transfer"].items()
                                 if k != "per_docket_VA_nl"}}

    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({"ledger": L2, "dense_seed_mean": dense["seed_mean"]}, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
