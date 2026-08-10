"""Within-repo (repo-centered) comparison of dense v3 vs the VA_nl stack.

Pooled AUC on this cell is known to be dominated by repo composition (v2 note:
eval .573 vs test .704 was composition, not leakage), and eval/test disagree
sharply on the pooled VA_nl. The trustworthy level is within-repo.
"""
import glob, json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
R = "/lfs/skampere3/0/alexspan/norm-research"
D = f"{R}/datasets/code-review/dense_standard_v3"
out = {}
for sp in ("eval", "test"):
    d = pd.read_csv(f"{D}/split/{sp}.csv", usecols=["repo", "pr_number", "judgement"])
    d["row_id"] = d["repo"].astype(str) + "/" + d["pr_number"].astype(str)
    p = pd.read_csv(f"{D}/rm_out_seed42/preds_{sp}.csv")
    p["row_id"] = p["repo"].astype(str) + "/" + p["pr_number"].astype(str)
    d = d.merge(p[["row_id", "prob"]], on="row_id", how="left")
    oof = np.mean([np.load(f"{D}/abank_rescore/code_v3_{sp}_va_nl_oof_seed{s}.npy") for s in (0,1,2)], axis=0)
    d["va_nl"] = oof
    rows, nw_d, nw_v, tot = [], 0.0, 0.0, 0
    for repo, g in d.groupby("repo"):
        if g["judgement"].nunique() < 2 or len(g) < 20:
            continue
        ad = roc_auc_score(g["judgement"], g["prob"])
        av = roc_auc_score(g["judgement"], g["va_nl"])
        rows.append({"repo": repo, "n": len(g), "dense": ad, "va_nl": av})
        nw_d += ad*len(g); nw_v += av*len(g); tot += len(g)
    t = pd.DataFrame(rows)
    from scipy.stats import wilcoxon
    w = wilcoxon(t["dense"], t["va_nl"]) if len(t) > 5 else None
    out[sp] = {"n_repos_scored": len(t), "n_rows": int(tot),
               "dense_within_median": float(t["dense"].median()),
               "va_nl_within_median": float(t["va_nl"].median()),
               "dense_within_nwtd": nw_d/tot, "va_nl_within_nwtd": nw_v/tot,
               "dense_beats_va_nl_repos": int((t["dense"] > t["va_nl"]).sum()),
               "wilcoxon_p": float(w.pvalue) if w is not None else None}
    print(sp, json.dumps(out[sp], indent=1))
json.dump(out, open(f"{D}/abank_rescore/within_repo_dense_vs_vanl.json", "w"), indent=1)
print("wrote within_repo_dense_vs_vanl.json")
