#!/usr/bin/env python3
"""Cell adapter for the Layer-3 articulation-closure campaign on the CODE cell:
GitHub pull-request MERGE outcome, v3 ENRICHED text (Title + Description + inline
review comments + diff), Gemma-4-31B 83-criterion A bank.

Contract mirrors maps_hw_si/cells.py.  Returns

    ids          "<repo>/<pr_number>"  -- stable per-row key
    groups       repository            -- the closure group key for EVERY split/fold
    y            1 = merged, 0 = closed-unmerged   (NEVER shown to a proposer)
    A            RAW Gemma-4-31B criterion scores, 83 cols, NaN = judge abstained
    V            V_exec (17, execution-derived, joined) + V_text (19, recomputed on v3 text)
    dense        per-row dense v3 probability (seed 42 today; seeds 1/2 chained)
    dense_seeds  (n, n_seeds) matrix once seeds 1/2 land
    split        eval / test  (the dense chain's own held-out halves)
    position     within-repo PR-number percentile  (FREEZE ADDENDUM 4, observed covariate)

RECORDED DECISION (T honesty).  Unlike N&C responded and CW community, this cell needs
no "MONITOR must sit inside dense-held-out" intersection: the dense v3 chain trained on
the 47,659 TRAIN rows, whose repositories are disjoint from eval and test by construction
(build_manifest integrity block: train&eval = train&test = eval&test = 0 repo overlap),
and the A bank was scored on eval+test ONLY.  So every row of the closure population is
already dense-held-out and T is honest on all of it.  MONITOR is therefore automatically
T-honest as well as VA-honest, and the nc DECISION-1 two-tier workaround is unnecessary.

RECORDED FLAG (which half is selection-free).  eval is the dense chain's checkpoint
SELECTION split (--selection_split eval); test was never read during training.  Both
halves are reported as replicates and the test half is the selection-free reading.

CPU only.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CLOSURE = HERE.parent
sys.path.insert(0, str(CLOSURE / "maps_hw_si"))
AB = HERE / "abank_rescore"

MIN_REPO_N = 20          # inherited verbatim from abank_rescore/within_repo.py
DENSE_SEEDS = (42, 1, 2)


def _load_A():
    Xs, ids = [], []
    a_ids = a_names = None
    for s in sorted(glob.glob(str(AB / "scores_shard*.npz"))):
        z = np.load(s, allow_pickle=True)
        Xs.append(z["X"])
        ids += [str(x) for x in z["row_ids"]]
        a_ids = [str(x) for x in z["a_ids"]]
        a_names = [str(x) for x in z["a_names"]]
    return np.vstack(Xs), ids, a_ids, a_names


def load(dense_dirs=None):
    A_all, a_row_ids, a_ids, a_names = _load_A()

    V = pd.read_parquet(AB / "V_matrix_v3.parquet")
    vn = json.loads((AB / "V_names.json").read_text())
    v_exec, v_text = vn["v_exec"], vn["v_text"]

    pos = {r: i for i, r in enumerate(a_row_ids)}
    assert V["row_id"].isin(pos).all(), "A shards do not cover the V rows"
    A = A_all[[pos[r] for r in V["row_id"]]]

    d = dict(
        tag="code_v3",
        ids=[str(x) for x in V["row_id"]],
        groups=V["repo"].astype(str).values,
        y=V["judgement"].astype(int).values,
        split=V["split"].astype(str).values,
        pr_number=pd.to_numeric(V["pr_number"], errors="coerce").values.astype(float),
        A=A,
        V=V[v_exec + v_text].astype(float).values,
        a_ids=a_ids, a_names=a_names,
        v_names=v_exec + v_text,
        v_exec=v_exec, v_text=v_text,
    )

    # ---- dense: seed 42 today, seeds 1/2 as they land -----------------------
    dense_dirs = dense_dirs or {s: HERE / (
        "dense_seed42" if s == 42 else f"dense_seed{s}") for s in DENSE_SEEDS}
    cols, have = [], []
    for s in DENSE_SEEDS:
        dd = Path(dense_dirs[s])
        if not (dd / "preds_eval.csv").exists():
            continue
        p = pd.concat([pd.read_csv(dd / f"preds_{sp}.csv") for sp in ("eval", "test")],
                      ignore_index=True)
        p["row_id"] = p["repo"].astype(str) + "/" + p["pr_number"].astype(str)
        m = dict(zip(p["row_id"], p["prob"]))
        assert all(r in m for r in d["ids"]), f"dense seed {s} missing rows"
        cols.append(np.array([m[r] for r in d["ids"]]))
        have.append(s)
    assert cols, "no dense predictions found"
    d["dense_seeds_have"] = have
    d["dense_seed_probs"] = np.column_stack(cols)
    d["dense"] = d["dense_seed_probs"].mean(axis=1)     # ensemble over available seeds
    d["dense_seed42"] = cols[have.index(42)]

    # ---- position (FREEZE ADDENDUM 4): observed covariate, never a feature --
    s = pd.Series(d["pr_number"])
    g = pd.Series(d["groups"])
    d["position"] = {
        "pr_number": d["pr_number"],
        "within_repo_pr_pct": s.groupby(g).rank(pct=True).values.astype(float),
        "within_repo_pr_rank": s.groupby(g).rank(method="average").values.astype(float),
        "repo_size": g.groupby(g).transform("size").values.astype(float),
    }
    return d


# --------------------------------------------------------- within-repo AUC --
def within_repo_auc(y, p, groups, mask=None, min_n=MIN_REPO_N):
    """n-weighted mean of per-repo AUCs.  THE canonical readout for this cell:
    pooled AUC here is composition-dominated and is never quoted as a residual."""
    from sklearn.metrics import roc_auc_score
    y, p, groups = np.asarray(y), np.asarray(p), np.asarray(groups)
    # SHAPE GUARD: a MONITOR-length prediction vector reaching a full-population reader
    # has now been caught twice (round0_code.py, readout_code.py::discount). Assert it by
    # name instead of letting numpy raise a confusing IndexError -- or worse, letting it
    # through silently on a cell where the two lengths happen to coincide.
    assert len(p) == len(y) == len(groups), (
        f"within_repo_auc: prediction vector has length {len(p)} but the population has "
        f"{len(y)} rows -- pass a FULL-LENGTH vector (use _expand on a MONITOR-only one)")
    if mask is not None:
        assert len(mask) == len(y), f"mask length {len(mask)} != population {len(y)}"
        y, p, groups = y[mask], p[mask], groups[mask]
    per, num, tot = [], 0.0, 0
    for r in np.unique(groups):
        m = groups == r
        n = int(m.sum())
        if n < min_n or len(set(y[m])) < 2:
            continue
        a = float(roc_auc_score(y[m], p[m]))
        per.append({"repo": str(r), "n": n, "auc": a})
        num += n * a
        tot += n
    return {"n_repos": len(per), "n_rows": int(tot),
            "nwtd": (num / tot) if tot else float("nan"),
            "median": float(np.median([x["auc"] for x in per])) if per else float("nan"),
            "per_repo": per}


def within_repo_delta(y, p_a, p_b, groups, mask=None, min_n=MIN_REPO_N):
    """Δ = n-weighted within-repo AUC(p_a) − AUC(p_b), with the paired per-repo
    vector, Wilcoxon, and a leave-one-repo-out group jackknife width."""
    from scipy.stats import wilcoxon
    A = within_repo_auc(y, p_a, groups, mask, min_n)
    B = within_repo_auc(y, p_b, groups, mask, min_n)
    assert [x["repo"] for x in A["per_repo"]] == [x["repo"] for x in B["per_repo"]]
    ns = np.array([x["n"] for x in A["per_repo"]], dtype=float)
    da = np.array([x["auc"] for x in A["per_repo"]])
    db = np.array([x["auc"] for x in B["per_repo"]])
    diff = da - db
    out = {"n_repos": A["n_repos"], "n_rows": A["n_rows"],
           "a_nwtd": A["nwtd"], "b_nwtd": B["nwtd"],
           "a_median": A["median"], "b_median": B["median"],
           "delta_nwtd": A["nwtd"] - B["nwtd"],
           "delta_median": float(np.median(diff)) if len(diff) else float("nan"),
           "a_wins_repos": int((diff > 0).sum())}
    if len(diff) > 5:
        out["wilcoxon_p"] = float(wilcoxon(da, db).pvalue)
        jk = [float(((ns[np.arange(len(ns)) != i] * diff[np.arange(len(ns)) != i]).sum()
                     / ns[np.arange(len(ns)) != i].sum())) for i in range(len(ns))]
        jk = np.array(jk)
        n = len(jk)
        out["jackknife_se"] = float(np.sqrt((n - 1) / n * ((jk - jk.mean()) ** 2).sum()))
        out["jackknife_ci95"] = [out["delta_nwtd"] - 1.96 * out["jackknife_se"],
                                 out["delta_nwtd"] + 1.96 * out["jackknife_se"]]
    out["per_repo_delta"] = [{"repo": x["repo"], "n": x["n"], "a": x["auc"], "b": y2["auc"],
                              "d": x["auc"] - y2["auc"]}
                             for x, y2 in zip(A["per_repo"], B["per_repo"])]
    return out


if __name__ == "__main__":
    from sklearn.metrics import roc_auc_score
    d = load()
    print(f"code_v3 n={len(d['y'])} repos={len(set(d['groups']))} pos={d['y'].mean():.4f} "
          f"A={d['A'].shape[1]} V={d['V'].shape[1]} dense_seeds={d['dense_seeds_have']}")
    for sp in ("eval", "test"):
        m = d["split"] == sp
        print(f"  {sp}: n={m.sum()} repos={len(set(d['groups'][m]))} "
              f"pooled_T_seed42={roc_auc_score(d['y'][m], d['dense_seed42'][m]):.4f}")
