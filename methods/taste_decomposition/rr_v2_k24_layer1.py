#!/usr/bin/env python3
"""Layer-1 ledger for rr_v2_k24 -- the RoyalRoad re-matched expansion.

SENSITIVITY DESIGN, NOT THE CELL OF RECORD. rr_v1 (n=1,274) remains canonical.
This build's A-numbers carry a hard flag: the K=50 anchor battery on the expanded
pool FAILED ordering (pos .7157 < neg .7577, pos-vs-neg AUC .445) where the rr_v1
batch PASSED (.658), and the lexical floor rose .524 -> .5759. The expansion filled
the hole but lost certification.

The A/V matrix is assembled from TWO scoring batches -- 1,023 rows carried from the
rr_v1 token-truncated scoring (never re-judged) and 719 new rows scored in the
expansion batch. Both used the identical bank, judge, prompt and token truncation,
and their distributions match (batch means .7232 vs .7270), but the two-batch
construction is recorded here because it is exactly the kind of seam that can
manufacture an artifact.

  python methods/taste_decomposition/rr_v2_k24_layer1.py
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, sklearn
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layer1_gemma_cells as L
import scaleupC_layer1 as SC
import cw_expert_layer1 as X

REPO = Path(__file__).resolve().parents[2]
RESULTS = Path(__file__).resolve().parent / "results"
OLD = REPO / "outputs/va_gemma_banks_cw_expert"
NEW = REPO / "outputs/va_gemma_banks_cw_expert_expanded"
SLUG = "cw_royalroad_rr_v2_k24"


def load_dir(out, name):
    A, V, ids = [], [], []
    si = 0
    while (out / f"{name}_shard{si}.npz").exists():
        z = np.load(out / f"{name}_shard{si}.npz", allow_pickle=True)
        A.append(z["X"]); V.append(z["V"]); ids += [str(s) for s in z["ids"]]
        si += 1
    if not A:
        raise FileNotFoundError(f"{name} under {out}")
    meta = json.loads((out / f"{name}_meta.json").read_text())
    return np.vstack(A), np.vstack(V), ids, meta


def main():
    pop = pd.read_csv(REPO / "datasets/creative-writing/royalroad_stubs/va_expanded/population.csv.gz")
    pop["row_id"] = pop["row_id"].astype(str)
    Ao, Vo, ido, mo = load_dir(OLD, "cw_royalroad_verdict")
    An, Vn, idn, mn = load_dir(NEW, "cw_royalroad_expanded_newrows")
    pa = {d: i for i, d in enumerate(ido)}
    pb = {d: i for i, d in enumerate(idn)}
    rows, src = [], []
    for r in pop.itertuples():
        if r.row_id in pb:
            rows.append((An[pb[r.row_id]], Vn[pb[r.row_id]])); src.append("new")
        elif r.row_id in pa:
            rows.append((Ao[pa[r.row_id]], Vo[pa[r.row_id]])); src.append("carried")
        else:
            rows.append(None); src.append("missing")
    keep = [i for i, x in enumerate(rows) if x is not None]
    pop = pop.iloc[keep].reset_index(drop=True)
    src = [src[i] for i in keep]
    A = np.array([rows[i][0] for i in keep], dtype=float)
    V = np.array([rows[i][1] for i in keep], dtype=float)
    y = pop.judgement.values.astype(int)
    g = pop.row_id.values.astype(object)
    print(f"n={len(y)} pos={y.sum()} | carried {src.count('carried')} new {src.count('new')}")

    VA = np.column_stack([V, A]); mats = {"V": V, "A": A, "VA": VA}
    folds = L.outer_folds(len(y), g, n_splits=5)
    res = {"cell": SLUG, "design_id": "rr_v2_k24", "status": "SENSITIVITY DESIGN",
           "canonical_cell": "rr_v1 (n=1,274) remains the cell of record",
           "n": int(len(y)), "n_pos": int(y.sum()),
           "rows_carried": src.count("carried"), "rows_new": src.count("new"),
           "sklearn_version": sklearn.__version__, "linear": {}, "nonlinear": {}}

    lin = {}
    for k in ("V", "A", "VA"):
        auc, oof, dropped = X.linear_oof_gated(mats[k], y, g, folds)
        res["linear"][k] = auc; lin[k] = oof
        print(f"  linear {k:2s}: {auc:.4f} (collapse-gate dropped/fold {dropped})")
    nl = {}
    for k in ("V", "VA"):
        res["nonlinear"][k] = {}
        for s in L.GBM_SEEDS:
            r = X.gbm_oof_gated(mats[k], y, g, folds, s)
            nl[(k, s)] = r.pop("oof"); res["nonlinear"][k][str(s)] = r
            print(f"  gbm {k:2s} seed {s}: {r['auc']:.4f}")
    va = [res["nonlinear"]["VA"][str(s)]["auc"] for s in L.GBM_SEEDS]
    vv = [res["nonlinear"]["V"][str(s)]["auc"] for s in L.GBM_SEEDS]
    T, Tinfo = SC.dense_T(REPO / "datasets/creative-writing/royalroad_stubs/dense_expanded")
    res["T_dense"] = T; res["T_info"] = Tinfo
    res["seed_spread_range"] = {"VA": float(max(va) - min(va))}
    res["ledger"] = {"V_lin": res["linear"]["V"], "V_nl_mean": float(np.mean(vv)),
                     "A_lin": res["linear"]["A"], "VA_lin": res["linear"]["VA"],
                     "VA_nl_mean": float(np.mean(va)), "VA_nl_seeds": va, "T": T}
    if T is not None:
        res["ledger"]["Delta_beyond"] = T - float(np.mean(va))
        res["ledger"]["Delta_total"] = T - res["linear"]["VA"]

    # is the two-batch seam an artifact channel?
    isnew = np.array([s == "new" for s in src])
    res["batch_seam_audit"] = {
        "batch_predicts_y_auc": round(float(roc_auc_score(y, isnew.astype(float))), 4)
        if len(set(isnew)) > 1 else None,
        "mean_A_carried": round(float(np.nanmean(A[~isnew])), 4),
        "mean_A_new": round(float(np.nanmean(A[isnew])), 4),
        "note": "batch membership should not predict y (both strata are balanced by "
                "construction); the two batch means are reported so any distribution "
                "seam is visible."}
    bat = NEW / "anchor_battery.json"
    res["anchor_battery_EXPANDED"] = json.loads(bat.read_text()).get(
        "cw_royalroad_expanded_newrows") if bat.exists() else None
    res["HARD_FLAG"] = (
        "A-numbers UNCERTIFIED: the K=50 battery on the expanded pool inverted "
        "(pos .7157 < neg .7577, pos-vs-neg AUC .445) where rr_v1 passed at .658, and "
        "the lexical floor rose .524 -> .5759. Quote rr_v1, not this, as the cell.")
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{SLUG}_ledger.json").write_text(json.dumps(res, indent=2, default=str))
    print("\n=== rr_v2_k24 LEDGER (SENSITIVITY) ===")
    for k, v in res["ledger"].items():
        if isinstance(v, float):
            print(f"  {k:14s} {v:+.4f}")
    print(f"  batch seam: A carried {res['batch_seam_audit']['mean_A_carried']} vs "
          f"new {res['batch_seam_audit']['mean_A_new']}, batch-predicts-y "
          f"{res['batch_seam_audit']['batch_predicts_y_auc']}")
    print("wrote", RESULTS / f"{SLUG}_ledger.json")
    print("RR_V2_K24_LAYER1_DONE")


if __name__ == "__main__":
    main()
