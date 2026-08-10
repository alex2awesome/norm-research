#!/usr/bin/env python3
"""Dataset-first confound audit for the examiner-citation target (patents leg of claim-matching).

Y = was this claim cited/rejected over prior art by the examiner (option3 pos) vs an untargeted
claim in the SAME office action (neg). Before gathering articulated claim-matching metrics we must
know: is Y a metadata leak? If independent-vs-dependent status or claim-number trivially separates
pos/neg, the "articulability of citation" question collapses to a structural feature and any metric
gain must be measured as MARGINAL over structure (per BEST-PRACTICES metadata-confound gauntlet +
content guard).

Prints: base rates, within-app pos/neg structure, univariate + combined structural AUC (the leak
baseline), and 12 eyeballed samples. Run on sk3 (CPU)."""
import json, re, hashlib, collections
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

BASE = "/lfs/skampere3/0/alexspan/norm-research"
SCALE = f"{BASE}/datasets/patents/processed/option3_claims_gemma_scale.jsonl"

DEP_RE = re.compile(r"\bof claim\s+\d+|\baccording to claim\s+\d+|\bas (?:recited|claimed) in claim",
                    re.I)
LIMIT_RE = re.compile(r"\bwherein\b|\bcomprising\b|\bconfigured to\b|;")


def feats(element):
    t = (element or "").strip()
    words = t.split()
    return {
        "is_dependent": int(bool(DEP_RE.search(t))),
        "word_len": len(words),
        "n_limit_markers": len(LIMIT_RE.findall(t)),
        "n_commas": t.count(","),
    }


def hash_fold(app, k=5):
    return int(hashlib.md5(str(app).encode()).hexdigest(), 16) % k


def cv_auc(X, y, groups, k=5):
    oof = np.zeros(len(y))
    for f in range(k):
        te = groups == f; tr = ~te
        if len(set(y[tr])) < 2:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        clf.fit(X[tr], y[tr]); oof[te] = clf.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, oof)


def main():
    rows = []
    with open(SCALE) as f:
        for ln in f:
            r = json.loads(ln)
            fe = feats(r["element"])
            fe.update(label=1 if r["label"] == "pos" else 0, app=str(r["app_id"]),
                      claim_num=str(r["claim_num"]), rej=str(r.get("rejection_type")),
                      elem=r["element"])
            rows.append(fe)
    n = len(rows)
    y = np.array([r["label"] for r in rows])
    print(f"[n] {n} claims, pos rate {y.mean():.3f}", flush=True)

    # within-app structure: how are pos/neg split inside an app?
    byapp = collections.defaultdict(list)
    for r in rows:
        byapp[r["app"]].append(r)
    mixed = sum(1 for v in byapp.values() if len({x["label"] for x in v}) == 2)
    print(f"[within-app] {len(byapp)} apps; {mixed} have BOTH pos&neg "
          f"({mixed/len(byapp):.1%}) — the discriminable set", flush=True)
    # dependent rate by class
    dep = np.array([r["is_dependent"] for r in rows])
    print(f"[dep] dependent-claim rate: pos {dep[y==1].mean():.3f}  neg {dep[y==0].mean():.3f}",
          flush=True)
    # claim_num numeric where possible
    cn = np.array([float(r["claim_num"]) if str(r["claim_num"]).isdigit() else np.nan
                   for r in rows])
    ok = ~np.isnan(cn)
    print(f"[claim#] median claim_num: pos {np.nanmedian(cn[(y==1)&ok]):.0f}  "
          f"neg {np.nanmedian(cn[(y==0)&ok]):.0f}", flush=True)

    # univariate structural AUC (leak check)
    groups = np.array([hash_fold(r["app"]) for r in rows])
    print("\n[structural univariate AUC vs citation]", flush=True)
    for name in ("is_dependent", "word_len", "n_limit_markers", "n_commas"):
        v = np.array([r[name] for r in rows], float)
        a = roc_auc_score(y, v)
        print(f"  {name:16s} AUC={a:.4f}  (|{a-0.5:+.3f}| from chance)", flush=True)
    cn2 = np.nan_to_num(cn, nan=np.nanmedian(cn))
    print(f"  {'claim_num':16s} AUC={roc_auc_score(y, cn2):.4f}", flush=True)

    # combined structural logistic (the trivial baseline to beat)
    X = np.array([[r["is_dependent"], r["word_len"], r["n_limit_markers"], r["n_commas"]]
                  for r in rows], float)
    Xc = np.c_[X, cn2]
    a_struct = cv_auc(Xc, y, groups)
    print(f"\n[STRUCTURAL BASELINE] 5-fold app-CV AUC = {a_struct:.4f}  "
          f"<- articulated metrics must beat THIS, not chance", flush=True)

    # by rejection type
    print("\n[by rejtype] pos rate + n", flush=True)
    rt = collections.Counter(r["rej"] for r in rows)
    for t, c in rt.most_common(6):
        yt = np.array([r["label"] for r in rows if r["rej"] == t])
        print(f"  {t:8s} n={c:6d} pos={yt.mean():.3f}", flush=True)

    print("\n[samples] (label, dep, claim#, first 110 chars)", flush=True)
    for r in rows[:12]:
        print(f"  y={r['label']} dep={r['is_dependent']} c#{r['claim_num']:>3} "
              f"| {r['elem'][:110]}", flush=True)
    print("CITATION_CONFOUND_DONE", flush=True)


if __name__ == "__main__":
    main()
