#!/usr/bin/env python3
"""JOB 1 -- peer REVEALED topic-stratified robustness check.

QUESTION.  peer revealed (citation percentile) is known to ride a TOPIC FLOOR:
its Track-B spurious map is headed by trend-vocabulary channels at alone-AUC
.64-.71 (`notes/2026-08-06__spurious_maps_batch1.md`, peer_revealed section), and
the Layer-1 note already carries the caveat that citation percentile is
substantially topic-predictable.  If the dense edge Delta_beyond = T - VA_nl is a
TOPIC-COMPOSITION effect -- the dense model is better at telling which SUBFIELD a
paper is in, and subfield predicts citations -- then Delta should collapse once we
compare only papers inside the same topic.

DESIGN (all CPU, all reuse; no new judging, no GPU).
  * population   HONEST = the dense model's own held-out rows (n=478), where T and
                 VA_nl are both out-of-sample.  MONITOR (n=244) reported alongside.
  * T            same-rows dense rescore, `samerows_preds/peer_revealed_dense_preds_slim.csv`
                 (HONEST AUC .8842 -- the number the task names).
  * VA_nl        refit with `maps_batch1/closure_core.fit_block`, two bank states:
                 R0 (V + A_base, the frozen Layer-1 bank) and R2 (plus the A-routed
                 criteria accepted in map rounds 1-2).  R2 is the primary: it is the
                 strongest bank on file, hence the most conservative test of T.
  * topic strata (i)  k-means on BAAI/bge-large-en-v1.5 abstract embeddings, k in
                      {5,10,20}, **fit on the FIT+MINE (train) side only**, HONEST /
                      MONITOR rows assigned to the nearest train-side centroid;
                 (ii) deciles of each MINED trend-family Track-B channel already
                      scored corpus-wide in the maps batch (r1 B07 trend/hype-keyword
                      density; r2 B04 Trend-Aligned Vocabulary; r2 B10 Canonical
                      Ecosystem Naming) and of their standardized mean;
                 (iii) publication year (the conjectured upstream parent of the
                      trend channels).
  * readout      n-weighted within-stratum AUC (`closure_core.stratified_auc`) for T
                 and for VA_nl; Delta_strat = T_strat - VA_strat; group-level
                 bootstrap CI on Delta_strat recomputed stratum-wise per replicate.
  * control      STRATIFICATION-FREE stacked increment (freeze addendum, "does not
                 degenerate as the nuisance set grows"): give the topic information
                 to the bank as a covariate and ask what the dense score still adds.
                 topic_score = grouped-OOF logistic on the top-50 PCs of the abstract
                 embedding (PCA fit on FIT+MINE); then
                   dense increment over topic          = AUC(topic+dense)  - AUC(topic)
                   dense increment over topic+bank     = AUC(topic+bank+dense)
                                                         - AUC(topic+bank)
                   dense increment over topic+trend+bank (adds the 3 mined channels)
                 Small-n strata cannot bite here, so this is the tie-breaker when the
                 k=20 stratification runs out of rows.

VERDICT RULE (declared before running, so the read is not post hoc):
  SURVIVES  if the primary stratified Delta (k=20 topic clusters, R2 bank, HONEST)
            keeps a bootstrap CI excluding 0 AND retains >= 50% of the pooled Delta,
            and the stratification-free dense increment over topic+bank is > 0 with
            a CI excluding 0.
  COLLAPSES if the stratified Delta CI covers 0 or the point estimate falls below
            25% of pooled.
  PARTIAL   anything in between -- reported as a discount on the cell, and the
            closure campaign still runs but carries the size of the discount.

Usage: python job1_topic_strat.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CLOSURE = HERE.parent
MAPS = CLOSURE / "maps_batch1"
sys.path.insert(0, str(MAPS))

import cells as C          # noqa: E402
import closure_core as L   # noqa: E402
import stage1_slice as S1  # noqa: E402

from sklearn.cluster import KMeans                    # noqa: E402
from sklearn.decomposition import PCA                 # noqa: E402
from sklearn.linear_model import LogisticRegression   # noqa: E402
from sklearn.model_selection import GroupKFold        # noqa: E402
from sklearn.pipeline import make_pipeline            # noqa: E402
from sklearn.preprocessing import StandardScaler      # noqa: E402

CELL = "peer_revealed"
EMB_CACHE = HERE / "abstract_emb_bge_large.npz"
KS = (5, 10, 20)
TREND = {  # blind_id -> (round, printable name); the mined trend/timing family
    ("r1", "B07"): "Trend/hype-keyword density",
    ("r2", "B04"): "Trend-Aligned Vocabulary",
    ("r2", "B10"): "Canonical Ecosystem Naming",
}


# ------------------------------------------------------------- embeddings ---
def embed_abstracts(texts):
    """BAAI/bge-large-en-v1.5, CLS pooling, L2-normalised, CPU. Cached on disk."""
    keys = [hashlib.sha256(t.encode()).hexdigest() for t in texts]
    cache = {}
    if EMB_CACHE.exists():
        z = np.load(EMB_CACHE, allow_pickle=True)
        cache = {str(k): v for k, v in zip(z["keys"], z["vecs"])}
    need = [t for t, k in zip(texts, keys) if k not in cache]
    if need:
        print(f"[embed] {len(need)} abstracts to embed (cache holds {len(cache)})", flush=True)
        import torch
        from transformers import AutoModel, AutoTokenizer
        name = "BAAI/bge-large-en-v1.5"
        tok = AutoTokenizer.from_pretrained(name)
        mod = AutoModel.from_pretrained(name).eval()
        with torch.no_grad():
            for i in range(0, len(need), 16):
                enc = tok(need[i:i + 16], padding=True, truncation=True,
                          max_length=512, return_tensors="pt")
                h = mod(**enc).last_hidden_state[:, 0]
                v = torch.nn.functional.normalize(h, dim=-1).numpy()
                for t, vv in zip(need[i:i + 16], v):
                    cache[hashlib.sha256(t.encode()).hexdigest()] = vv
                if (i // 16) % 20 == 0:
                    print(f"  [embed] {i}/{len(need)}", flush=True)
        np.savez_compressed(EMB_CACHE, keys=np.array(list(cache.keys())),
                            vecs=np.array(list(cache.values()), dtype=np.float32))
    return np.array([cache[k] for k in keys], dtype=np.float64)


# ------------------------------------------------------------- stratified ---
def strat_pair(y, t, va, strata, min_n):
    ta, ia = L.stratified_auc(y, t, strata, min_n=min_n)
    vb, ib = L.stratified_auc(y, va, strata, min_n=min_n)
    return {"T_strat": ta, "VA_strat": vb, "Delta_strat": ta - vb,
            "n_strata_used": ia["n_strata_used"], "n_rows_used": ia["n_rows_used"],
            "n_rows_dropped": ia["n_rows_dropped"],
            "coverage": ia["n_rows_used"] / max(1, len(y))}


def strat_boot(y, t, va, strata, groups, min_n, n=1500, seed=0):
    """Group-level paired bootstrap of the STRATIFIED Delta (strata re-read per rep)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in uniq}
    out = []
    for _ in range(n):
        gs = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_g[g] for g in gs])
        ta, _ = L.stratified_auc(y[idx], t[idx], strata[idx], min_n=min_n)
        vb, _ = L.stratified_auc(y[idx], va[idx], strata[idx], min_n=min_n)
        if np.isnan(ta) or np.isnan(vb):
            continue
        out.append(ta - vb)
    out = np.array(out)
    if len(out) == 0:
        return {"lo": None, "hi": None, "p_gt0": None, "n_reps": 0}
    return {"lo": float(np.percentile(out, 2.5)), "hi": float(np.percentile(out, 97.5)),
            "p_gt0": float((out > 0).mean()), "mean": float(out.mean()),
            "n_reps": int(len(out))}


def stack_oof(cols, y, groups, n_splits=5):
    X = np.column_stack(cols)
    folds = list(GroupKFold(n_splits=min(n_splits, len(np.unique(groups)))).split(
        np.zeros(len(y)), groups=groups))
    oof = np.zeros(len(y))
    for tr, te in folds:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return oof


# ------------------------------------------------------------------- main ---
def main():
    d = C.load(CELL)
    sp = json.loads((MAPS / f"{CELL}_splits.json").read_text())
    split = np.array([r["split"] for r in sp["rows"]])
    y, groups, dense = d["y"], d["groups"], d["dense"]
    fitm, monm = split == "fit_mine", split == "monitor"
    held = np.isin(d["dense_split"], ["eval", "test"])
    print(f"n={len(y)}  fit_mine={fitm.sum()}  monitor={monm.sum()}  HONEST={held.sum()}")

    res = {"cell": CELL, "job": "topic-stratified robustness check",
           "n": int(len(y)), "n_HONEST": int(held.sum()), "n_MONITOR": int(monm.sum()),
           "T_HONEST": L.auc(y[held], dense[held]),
           "T_MONITOR": L.auc(y[monm], dense[monm])}

    # ---- bank states: R0 (V+A_base) and R2 (plus A-routed r1,r2 criteria) ----
    banks = {}
    b0, t0 = [d["V"], d["A"]], ["V", "A_base"]
    b2, t2 = S1.current_blocks(d, 3)            # V, A_base, A_round1, A_round2
    for label, (blocks, tags) in {"R0": (b0, t0), "R2": (b2, t2)}.items():
        print(f"[fit] bank {label}: blocks={tags}", flush=True)
        r = L.fit_block(blocks, fitm, monm, y, groups)
        va = np.full(len(y), np.nan)
        va[fitm] = r["oof_nl_fitmine"]
        va[monm] = r["nl_mon"]
        banks[label] = va
        res[f"VA_nl_HONEST_{label}"] = L.auc(y[held], va[held])
        res[f"VA_nl_MONITOR_{label}"] = L.auc(y[monm], va[monm])
        res[f"n_features_{label}"] = r["n_features"]
        res[f"bank_blocks_{label}"] = tags
        print(f"   VA_nl HONEST {label} = {res[f'VA_nl_HONEST_{label}']:.4f}", flush=True)
    res["pooled_Delta_HONEST_R0"] = res["T_HONEST"] - res["VA_nl_HONEST_R0"]
    res["pooled_Delta_HONEST_R2"] = res["T_HONEST"] - res["VA_nl_HONEST_R2"]
    res["pooled_Delta_MONITOR_R2"] = res["T_MONITOR"] - res["VA_nl_MONITOR_R2"]

    # ---- topic clusters, fit on the train side only -------------------------
    E = embed_abstracts([t or "" for t in d["texts"]])
    print(f"[embed] matrix {E.shape}", flush=True)
    clusters = {}
    for k in KS:
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(E[fitm])
        lab = km.predict(E)
        clusters[k] = lab
        sizes_h = np.bincount(lab[held], minlength=k)
        print(f"[kmeans k={k}] HONEST sizes {sorted(sizes_h.tolist(), reverse=True)}")
    res["cluster_sizes_HONEST"] = {str(k): np.bincount(clusters[k][held],
                                                       minlength=k).tolist() for k in KS}

    # ---- mined trend-family channels (reuse the maps-batch corpus scores) ---
    trend_cols, trend_names = {}, {}
    for (rr, bid), nm in TREND.items():
        z = np.load(MAPS / f"{CELL}_{rr}_scores.npz", allow_pickle=True)
        cids = [str(s) for s in z["crit_ids"]]
        col = z["X"][:, cids.index(bid)].astype(float)
        med = np.nanmedian(col)
        col = np.where(np.isnan(col), med, col)
        trend_cols[f"{rr}:{bid}"] = col
        trend_names[f"{rr}:{bid}"] = nm
    Z = np.column_stack([(v - v[fitm].mean()) / (v[fitm].std() + 1e-9)
                         for v in trend_cols.values()])
    trend_cols["trend_family_mean_z"] = Z.mean(axis=1)
    trend_names["trend_family_mean_z"] = "mean z of the 3 mined trend/timing channels"
    res["trend_channel_alone_AUC_HONEST"] = {
        trend_names[k]: L.auc(y[held], v[held]) for k, v in trend_cols.items()}
    print("[trend] alone AUCs HONEST:", json.dumps(res["trend_channel_alone_AUC_HONEST"],
                                                   indent=1, default=float))

    # ---- the stratified table ----------------------------------------------
    years = d["years"]
    strata_defs = {}
    for k in KS:
        strata_defs[f"bge_kmeans_k{k}"] = clusters[k]
    for key, col in trend_cols.items():
        strata_defs[f"decile_{key}"] = L.decile_strata(col, q=10)
        strata_defs[f"quintile_{key}"] = L.decile_strata(col, q=5)
    yr = np.where(np.isfinite(years), years, -1)
    strata_defs["year"] = yr.astype(int)

    table = {}
    for pop, mask in (("HONEST", held), ("MONITOR", monm)):
        table[pop] = {}
        yy, dd, gg = y[mask], dense[mask], groups[mask]
        for bank in ("R0", "R2"):
            vv = banks[bank][mask]
            for sname, s in strata_defs.items():
                ss = s[mask]
                for min_n in (20, 10):
                    row = strat_pair(yy, dd, vv, ss, min_n)
                    row["pooled_T"] = L.auc(yy, dd)
                    row["pooled_VA"] = L.auc(yy, vv)
                    row["pooled_Delta"] = row["pooled_T"] - row["pooled_VA"]
                    row["retained_frac_of_pooled_Delta"] = (
                        row["Delta_strat"] / row["pooled_Delta"]
                        if row["pooled_Delta"] not in (0, None) and
                        not np.isnan(row["Delta_strat"]) else None)
                    key = f"{bank}|{sname}|min_n{min_n}"
                    if pop == "HONEST" and min_n == 20:
                        row["boot_ci"] = strat_boot(yy, dd, vv, ss, gg, min_n)
                    table[pop][key] = row
        print(f"[strat] {pop} done", flush=True)
    res["stratified"] = table

    # ---- stratification-free control: give topic to the bank ---------------
    pca = PCA(n_components=50, random_state=0).fit(E[fitm])
    P = pca.transform(E)
    ctrl = {}
    for pop, mask in (("HONEST", held), ("MONITOR", monm)):
        yy, gg, dd = y[mask], groups[mask], dense[mask]
        topic = stack_oof([P[mask][:, j] for j in range(P.shape[1])], yy, gg)
        tr = [trend_cols[k][mask] for k in trend_cols if k != "trend_family_mean_z"]
        blk = {"n": int(mask.sum()), "AUC_topic_only": L.auc(yy, topic),
               "AUC_dense": L.auc(yy, dd)}
        for bank in ("R0", "R2"):
            vv = banks[bank][mask]
            s_t = topic
            s_tb = stack_oof([topic, vv], yy, gg)
            s_td = stack_oof([topic, dd], yy, gg)
            s_tbd = stack_oof([topic, vv, dd], yy, gg)
            s_ttb = stack_oof([topic, vv] + tr, yy, gg)
            s_ttbd = stack_oof([topic, vv, dd] + tr, yy, gg)
            blk[bank] = {
                "AUC_bank": L.auc(yy, vv),
                "AUC_topic": L.auc(yy, s_t),
                "AUC_topic_bank": L.auc(yy, s_tb),
                "AUC_topic_dense": L.auc(yy, s_td),
                "AUC_topic_bank_dense": L.auc(yy, s_tbd),
                "dense_increment_over_topic": L.auc(yy, s_td) - L.auc(yy, s_t),
                "bank_increment_over_topic": L.auc(yy, s_tb) - L.auc(yy, s_t),
                "dense_increment_over_topic_plus_bank": L.auc(yy, s_tbd) - L.auc(yy, s_tb),
                "ci_dense_increment_over_topic_plus_bank":
                    L.group_boot_ci(yy, s_tbd, s_tb, gg),
                "AUC_topic_trend_bank": L.auc(yy, s_ttb),
                "AUC_topic_trend_bank_dense": L.auc(yy, s_ttbd),
                "dense_increment_over_topic_trend_bank":
                    L.auc(yy, s_ttbd) - L.auc(yy, s_ttb),
                "ci_dense_increment_over_topic_trend_bank":
                    L.group_boot_ci(yy, s_ttbd, s_ttb, gg),
            }
        ctrl[pop] = blk
        print(f"[control] {pop} done", flush=True)
    res["stratification_free_control"] = ctrl

    (HERE / "job1_topic_strat.json").write_text(json.dumps(res, indent=1, default=float))
    print("wrote", HERE / "job1_topic_strat.json")

    # ---- console summary ----------------------------------------------------
    print("\n=== PRIMARY (HONEST, R2 bank, min_n=20) ===")
    for sname in strata_defs:
        r = res["stratified"]["HONEST"][f"R2|{sname}|min_n20"]
        ci = r.get("boot_ci") or {}
        print(f"{sname:34s} T {r['T_strat']:.4f}  VA {r['VA_strat']:.4f}  "
              f"D {r['Delta_strat']:+.4f}  ret {r['retained_frac_of_pooled_Delta']}"
              f"  cov {r['coverage']:.2f}  CI [{ci.get('lo')}, {ci.get('hi')}]")
    print("\n=== stratification-free control (HONEST, R2) ===")
    print(json.dumps(res["stratification_free_control"]["HONEST"]["R2"], indent=1,
                     default=float))


if __name__ == "__main__":
    main()
