#!/usr/bin/env python3
"""V / A / VA_lin / VA_nl readout for the code-review cell on the ENRICHED v3 text,
against the dense v3 model on the SAME rows.

V comes in two blocks, both computed on the v3 rows:
  V_exec  execution-derived features from the PR test-execution fleet
          (verdict category, return code, P2F/F2P gates). This is the SAME
          instrument family the published V .576 used. Its extractor is a docker
          test run, not a text extractor, so it is JOINED, not recomputed -- it is
          invariant to the v2->v3 text enrichment by construction.
  V_text  deterministic text/structure features RECOMPUTED on the v3 text
          (title/description/comment/diff geometry). These are the "code+regex"
          V of the paper's definition and they DO change with enrichment.
  V_all   = V_exec + V_text (the headline V; most generous to V, so a positive
          taste residual measured against it is conservative).

A = Gemma-4-31B scores of the code A-bank criteria on the v3 text (this file
only reads the npz shards written by score_code_abank_v3.py).

Stacks are fit grouped-OOF (GroupKFold by repo) WITHIN eval and, separately,
WITHIN test -- mirroring the published aggregation, which fit the ladder inside
the same row set with GroupKFold(repo). Two linear protocols are reported:
  * layer1  StandardScaler + LogisticRegression(C=1.0), pooled OOF AUC
            (methods/taste_decomposition/layer1_stack.py -- comparable to the
             other cells in the taste-decomposition table)
  * pub     StandardScaler + LogisticRegression(C=0.1), mean-of-fold AUC
            (scripts/pr_vat/run_vat_ladder.py -- the protocol behind .592)
Nonlinear VA_nl uses the frozen Layer-1 HistGB grid, seeds 0/1/2.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(os.environ.get("NR_REPO", "/lfs/skampere3/0/alexspan/norm-research"))
V3 = REPO / "datasets/code-review/dense_standard_v3"
OUT = V3 / "abank_rescore"

GRID = [{"max_leaf_nodes": 15, "learning_rate": 0.06, "max_iter": 400},
        {"max_leaf_nodes": 31, "learning_rate": 0.06, "max_iter": 400}]
N_OUTER, N_INNER = 5, 3

TEST_PATH_RE = re.compile(r"(^|/)(test_|tests?/|spec/|specs?/|__tests__/)|(_test|\.test|_spec|\.spec)\.[^/\s]+",
                          re.IGNORECASE)


# ------------------------------------------------------------------ V ------
def v_text_features(text: str) -> dict:
    t = text
    title = desc = comments = diff = ""
    m = re.search(r"^Diff:\n", t, re.M)
    if m:
        diff, head = t[m.end():], t[:m.start()]
    else:
        head = t
    mc = re.search(r"^Review comments:\n", head, re.M)
    if mc:
        comments, head = head[mc.end():], head[:mc.start()]
    md = re.search(r"^Description:", head, re.M)
    if md:
        desc, head = head[md.end():], head[:md.start()]
    title = head.replace("Title:", "", 1)

    dl = diff.split("\n")
    added = [l for l in dl if l.startswith("+") and not l.startswith("+++")]
    removed = [l for l in dl if l.startswith("-") and not l.startswith("---")]
    files = re.findall(r"^diff --git a/(\S+)", diff, re.M)
    test_files = [f for f in files if TEST_PATH_RE.search(f)]
    cbullets = [l for l in comments.split("\n") if l.startswith("- [")]
    return {
        "vt_len_total": len(t),
        "vt_len_title": len(title.strip()),
        "vt_len_desc": len(desc.strip()),
        "vt_len_comments": len(comments.strip()),
        "vt_len_diff": len(diff),
        "vt_n_comment_bullets": len(cbullets),
        "vt_comment_mean_len": float(np.mean([len(c) for c in cbullets])) if cbullets else 0.0,
        "vt_n_code_suggestions": comments.count("```suggestion"),
        "vt_n_files": len(files),
        "vt_n_test_files": len(test_files),
        "vt_test_file_frac": len(test_files) / len(files) if files else 0.0,
        "vt_n_hunks": diff.count("\n@@"),
        "vt_lines_added": len(added),
        "vt_lines_removed": len(removed),
        "vt_add_del_ratio": len(added) / (len(added) + len(removed) + 1),
        "vt_desc_has_issue_ref": float(bool(re.search(r"#\d+", desc))),
        "vt_desc_n_bullets": desc.count("\n*") + desc.count("\n-"),
        "vt_title_has_tag": float(bool(re.match(r"\s*[\[\(]|^\s*\w+[:(]", title))),
    }


VERDICT_CATS = ["runner_no_output", "env_broken_no_tests", "no_change_all_pass",
                "no_change_still_broken", "env_broken_collection", "baseline_build_failed",
                "unknown_build_tool", "new_passing", "wall_timeout", "no_gradle",
                "fix", "regression"]


def build_v(d: pd.DataFrame) -> tuple[pd.DataFrame, list, list]:
    import glob as _g
    need = set(zip(d["repo"].astype(str), d["pr_number"].astype(str)))
    repos = set(d["repo"].astype(str))
    rows = []
    for f in _g.glob(str(REPO / "datasets/code-review/pr_test_execution/batch_runs/*/verdicts.jsonl")):
        r = f.split("/")[-2]
        if r not in repos:
            continue
        for l in open(f):
            l = l.strip()
            if not l:
                continue
            try:
                dd = json.loads(l)
            except Exception:
                continue
            k = (r, str(dd.get("pr_number")))
            if k in need:
                rows.append({"repo": r, "pr_number": str(dd.get("pr_number")),
                             "verdict": dd.get("verdict"), "rc": dd.get("rc")})
    vd = pd.DataFrame(rows).drop_duplicates(["repo", "pr_number"], keep="first")
    d = d.copy()
    d["pr_number"] = d["pr_number"].astype(str)
    d = d.merge(vd, on=["repo", "pr_number"], how="left")

    d["ve_rc"] = pd.to_numeric(d["rc"], errors="coerce").fillna(-1)
    d["ve_p2f"] = d["verdict"].isin(["regression", "new_failing"]).astype(int)
    d["ve_f2p"] = d["verdict"].isin(["fix", "new_passing"]).astype(int)
    d["ve_has_signal"] = (d["ve_p2f"] | d["ve_f2p"]).astype(int)
    d["ve_env_bad"] = d["verdict"].astype(str).str.startswith(("env_broken", "baseline_build",
                                                              "runner_no", "unknown_build",
                                                              "wall_timeout", "no_gradle")).astype(int)
    for c in VERDICT_CATS:
        d[f"ve_v_{c}"] = (d["verdict"] == c).astype(int)
    v_exec = ["ve_rc", "ve_p2f", "ve_f2p", "ve_has_signal", "ve_env_bad"] + [f"ve_v_{c}" for c in VERDICT_CATS]

    tf = pd.DataFrame([v_text_features(t) for t in d["text"].astype(str)], index=d.index)
    d = pd.concat([d, tf], axis=1)
    v_text = list(tf.columns) + ["n_review_comments"]
    d["n_review_comments"] = pd.to_numeric(d["n_review_comments"], errors="coerce").fillna(0)
    return d, v_exec, v_text


# ------------------------------------------------------------ modelling ----
def clean_cols(M):
    keep, out = [], []
    for j in range(M.shape[1]):
        col = M[:, j].astype(float)
        nonna = col[~np.isnan(col)]
        if len(nonna) == 0:
            continue
        med = np.median(nonna)
        c = np.where(np.isnan(col), med, col)
        vals, counts = np.unique(c, return_counts=True)
        if (len(c) - counts.max()) < 5 or c.std() == 0:
            continue
        keep.append(j)
        out.append(c)
    if not out:
        return np.zeros((M.shape[0], 0)), keep
    return np.column_stack(out), keep


def folds_for(groups):
    gkf = GroupKFold(n_splits=min(N_OUTER, len(np.unique(groups))))
    return list(gkf.split(np.zeros(len(groups)), groups=groups))


def linear_oof(X, y, folds, C=1.0):
    if X.shape[1] == 0:
        return float("nan"), None, float("nan")
    oof = np.zeros(len(y))
    fold_aucs = []
    for tr, te in folds:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
        fold_aucs.append(float(roc_auc_score(y[te], oof[te])) if len(np.unique(y[te])) > 1 else np.nan)
    return float(roc_auc_score(y, oof)), oof, float(np.nanmean(fold_aucs))


def _gbm(p, seed):
    return HistGradientBoostingClassifier(
        max_leaf_nodes=p["max_leaf_nodes"], learning_rate=p["learning_rate"],
        max_iter=p["max_iter"], early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=20, random_state=seed)


def gbm_oof(X, y, groups, folds, seed=0):
    oof = np.zeros(len(y))
    picks, train_aucs, inner_log = [], [], []
    for fi, (tr, te) in enumerate(folds):
        gtr = groups[tr]
        inner = list(GroupKFold(n_splits=min(N_INNER, len(np.unique(gtr)))
                                ).split(np.zeros(len(tr)), groups=gtr))
        scores = []
        for p in GRID:
            aucs = []
            for itr, ite in inner:
                m = _gbm(p, seed)
                m.fit(X[tr][itr], y[tr][itr])
                aucs.append(roc_auc_score(y[tr][ite], m.predict_proba(X[tr][ite])[:, 1]))
            scores.append(float(np.mean(aucs)))
        best = int(np.argmax(scores))
        picks.append(GRID[best]["max_leaf_nodes"])
        inner_log.append({"fold": fi, "inner_auc": dict(zip([str(g["max_leaf_nodes"]) for g in GRID], scores))})
        m = _gbm(GRID[best], seed)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
        train_aucs.append(float(roc_auc_score(y[tr], m.predict_proba(X[tr])[:, 1])))
    return {"auc": float(roc_auc_score(y, oof)), "picks": picks,
            "train_auc_mean": float(np.mean(train_aucs)), "inner": inner_log, "oof": oof}


# ----------------------------------------------------------------- main ----
def main():
    shards = sorted(glob.glob(str(OUT / "scores_shard*.npz")))
    assert shards, "no A score shards found"
    Xs, ids, a_ids, a_names = [], [], None, None
    for s in shards:
        z = np.load(s, allow_pickle=True)
        Xs.append(z["X"])
        ids += [str(x) for x in z["row_ids"]]
        a_ids = [str(x) for x in z["a_ids"]]
        a_names = [str(x) for x in z["a_names"]]
    A_all = np.vstack(Xs)
    print(f"A matrix {A_all.shape}, NA rate {np.isnan(A_all).mean():.4f}")

    frames = []
    for sp in ("eval", "test"):
        d = pd.read_csv(V3 / "split" / f"{sp}.csv")
        d["split"] = sp
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d["row_id"] = d["repo"].astype(str) + "/" + d["pr_number"].astype(str)
    d, v_exec, v_text = build_v(d)

    pos = {r: i for i, r in enumerate(ids)}
    have = d["row_id"].isin(pos)
    if not have.all():
        # partial run (shards still in flight): keep the pipeline runnable, but
        # a partial readout is a DRY RUN, never a result.
        print(f"WARNING partial: {int(have.sum())}/{len(d)} rows scored -- DRY RUN ONLY")
        d = d[have].reset_index(drop=True)
    order = [pos[r] for r in d["row_id"]]
    A_all = A_all[order]
    results_partial = bool(not have.all())

    dense = {}
    for sp in ("eval", "test"):
        p = pd.read_csv(V3 / f"rm_out_seed42/preds_{sp}.csv")
        p["row_id"] = p["repo"].astype(str) + "/" + p["pr_number"].astype(str)
        dense[sp] = dict(zip(p["row_id"], p["prob"]))

    # per-criterion collapse check on the full 11,452 rows
    collapse = []
    for j, aid in enumerate(a_ids):
        col = A_all[:, j]
        nn = col[~np.isnan(col)]
        vals, cnt = (np.unique(nn, return_counts=True) if len(nn) else (np.array([]), np.array([])))
        modal = float(cnt.max() / len(col)) if len(nn) else 1.0
        collapse.append({"aspect_id": aid, "name": a_names[j],
                         "na_rate": float(np.isnan(col).mean()),
                         "modal_frac": modal,
                         "mean": float(np.nanmean(col)) if len(nn) else float("nan"),
                         "collapsed": bool(modal > 0.98 or len(nn) == 0)})
    n_coll = sum(c["collapsed"] for c in collapse)
    print(f"collapse: {n_coll}/{len(collapse)} criteria >98% modal or all-NA")

    results = {"cell": "code review (GitHub PR merge) -- v3 enriched text",
               "judge": "gemma-4-31b-it", "instrument_change": True,
               "PARTIAL_DRY_RUN": results_partial,
               "n_criteria_scored": len(a_ids), "aspect_ids": a_ids,
               "collapse": {"n_collapsed": n_coll, "detail": collapse},
               "v_exec_features": v_exec, "v_text_features": v_text,
               "splits": {}}

    for sp in ("eval", "test"):
        sel = (d["split"] == sp).values
        dd = d[sel]
        y = dd["judgement"].astype(int).values
        groups = dd["repo"].astype(str).values
        folds = folds_for(groups)
        # A matrix mirrors the published coded-bank construction: one `score`
        # column per criterion (NA median-imputed inside clean_cols) PLUS one
        # `applied` indicator per criterion (1 = judge gave a score, 0 = judge
        # answered NA). The coded parquet had exactly this (score, applied) pair.
        A = np.column_stack([A_all[sel], (~np.isnan(A_all[sel])).astype(float)])
        Ve, _ = clean_cols(dd[v_exec].astype(float).values)
        Vt, _ = clean_cols(dd[v_text].astype(float).values)
        Ac, a_keep = clean_cols(A)
        Va = np.column_stack([Ve, Vt]) if Ve.shape[1] and Vt.shape[1] else (Ve if Ve.shape[1] else Vt)
        mats = {"V_exec": Ve, "V_text": Vt, "V_all": Va, "A": Ac,
                "VA": np.column_stack([Va, Ac])}
        dprob = np.array([dense[sp][r] for r in dd["row_id"]])
        t_dense = float(roc_auc_score(y, dprob))

        # descriptive: univariate AUC of each criterion (NA -> median) and of the
        # criterion's abstention indicator, so the report can name what carries signal.
        uni = []
        for j, aid in enumerate(a_ids):
            col = A_all[sel][:, j].astype(float)
            nn = col[~np.isnan(col)]
            if len(nn) == 0:
                continue
            c = np.where(np.isnan(col), np.median(nn), col)
            rec = {"aspect_id": aid, "name": a_names[j],
                   "na_rate": float(np.isnan(col).mean())}
            rec["auc_score"] = float(roc_auc_score(y, c)) if c.std() > 0 else float("nan")
            ind = (~np.isnan(col)).astype(float)
            rec["auc_applied"] = float(roc_auc_score(y, ind)) if ind.std() > 0 else float("nan")
            uni.append(rec)
        uni.sort(key=lambda r: -abs((r["auc_score"] if r["auc_score"] == r["auc_score"] else .5) - .5))

        r = {"n": int(len(y)), "pos_rate": float(y.mean()),
             "n_repos": int(len(np.unique(groups))),
             "n_features": {k: int(v.shape[1]) for k, v in mats.items()},
             "A_matrix": "83 score cols (NA median-imputed) + 83 applied indicators",
             "n_A_kept": int(Ac.shape[1]),
             "T_dense_same_rows": t_dense, "linear_layer1": {}, "linear_published_protocol": {},
             "nonlinear": {}, "per_criterion_univariate": uni}
        print(f"\n=== {sp}: n={len(y)} repos={r['n_repos']} pos={y.mean():.3f} "
              f"dense={t_dense:.4f} | A kept {Ac.shape[1]}/{A.shape[1]}")
        for k, M in mats.items():
            auc, oof, meanf = linear_oof(M, y, folds, C=1.0)
            r["linear_layer1"][k] = auc
            auc2, _, meanf2 = linear_oof(M, y, folds, C=0.1)
            r["linear_published_protocol"][k] = {"pooled_oof": auc2, "mean_of_folds": meanf2}
            print(f"  linear {k:7s}: layer1 OOF {auc:.4f} | pub(C=.1) OOF {auc2:.4f} "
                  f"meanfold {meanf2:.4f}")
        for k in ("V_all", "A", "VA"):
            seeds = [0, 1, 2] if k == "VA" else [0]
            aucs, det = [], []
            for s in seeds:
                g = gbm_oof(mats[k], y, groups, folds, seed=s)
                oof = g.pop("oof")
                if k == "VA":
                    np.save(OUT / f"code_v3_{sp}_va_nl_oof_seed{s}.npy", oof)
                aucs.append(g["auc"])
                det.append(g)
            r["nonlinear"][k] = {"auc_seed0": aucs[0], "auc_seeds": aucs,
                                 "auc_mean": float(np.mean(aucs)), "detail": det}
            print(f"  gbm    {k:7s}: {['%.4f' % a for a in aucs]} mean {np.mean(aucs):.4f} "
                  f"(train {det[0]['train_auc_mean']:.3f})")
        va_lin = r["linear_layer1"]["VA"]
        va_nl = r["nonlinear"]["VA"]["auc_mean"]
        r["delta_total_vs_dense"] = {"lin": t_dense - va_lin, "nl": t_dense - va_nl}
        r["delta_beyond_best_VA"] = t_dense - max(va_lin, va_nl)
        results["splits"][sp] = r

    (OUT / "code_v3_enriched_layer1.json").write_text(json.dumps(results, indent=1))
    print("\nwrote", OUT / "code_v3_enriched_layer1.json")


if __name__ == "__main__":
    main()
