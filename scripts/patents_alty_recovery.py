#!/usr/bin/env python3
"""Alternative-Y recovery for the patents disclosure metric (user sign-off 2026-07-09 "try all three").

QUESTION: the label-free prior-art disclosure metric M recovers only 2.5% of H(Y) when Y = per-OA
"fell" (examiner targeted this claim). Does the SAME metric recover MORE signal for a differently
constructed outcome? Three candidates, each its own natural unit:
  Y1 rce      : app filed an RCE (transactions RCEX/BRCE/FRCE)     — app-level, ~44%
  Y2 persist  : app got >=2 office actions (labels.n_office_actions)— app-level, ~69%, NO extraction
  Y3 amend    : claim-1 changed a lot filed->granted (pgpub vs granted_v2) — granted-only, continuous

METRIC M (reconstruction-only, NEVER sees Y): patents_va_features.csv disclosure/verifiability
features, row-aligned to app_id via option3_claims_gemma_scale.jsonl (audit_regroup_va proved 0/59937
mismatch). Aggregated to app level (mean+max over the app's cohort claims).

CONFOUND GUARD: for every Y we fit THREE models under identical 5-fold app-hash CV:
  STRUCT  = trivial app descriptors (n_claims, filing_year, mean claim length) — the "free" baseline
  DISC    = the disclosure metric (a_* disclosure aggregates + lexoverlap)
  BOTH    = STRUCT + DISC
Disclosure's MARGINAL recovery = AUC(BOTH) - AUC(STRUCT); if DISC only tracks n_claims it shows here.
Report AUC per model + I(M;Y)/H(Y) (OOF-prob binned) so every Y sits on the 2.5% baseline's scale.

Stages (sk3, CPU):
  build : scan 507M-row transactions once -> alty_cache/app_rce.parquet (app_labels cached by scout)
  run   : all three recovery tables + the claim-level "fell" anchor

SUPERSEDED (dual audit Codex+Fable 2026-07-13) by scripts/patents_event_panel.py +
scripts/patents_y_power.py. Scope limits of THIS script's null (do not quote it as
"low-V however the outcome is defined"):
- sense (b) only — never tested intrinsic text predictability of Y1-Y3, nor construct fit;
- M = stale pre-fix app aggregates (12.2% dup rows row-weighted, position leak, dependent-blob
  units; semantic-delta fix never propagated);
- STRUCT n_claims counts option3 ROWS not unique claims; "claim length" is element wordlen;
- Y1 = EVER-RCE (undated: includes pre-final and post-NOA IDS-consideration RCEs);
  Y2 counts CTAV/CTEQ rows as office actions; Y3 matches claim 1 by NUMBER (renumbering
  conflated with amendment) and conditions on grant;
- linear logistic only, no CI, no examiner/art-unit/CPC controls; MI is unconditional (from
  DISC-only OOF), not disclosure's increment over STRUCT.
"""
import argparse, json, hashlib, os, collections
import numpy as np
import pandas as pd

BASE = "/lfs/skampere3/0/alexspan/norm-research"
PROC = f"{BASE}/datasets/patents/processed"
CACHE = f"{PROC}/alty_cache"
CSV = f"{BASE}/notebooks/data/patents_va_features.csv"
SCALE = f"{PROC}/option3_claims_gemma_scale.jsonl"
TXN = f"{BASE}/datasets/patents/raw/patex/transactions.csv"
OUT = f"{BASE}/outputs/patents_alty/recovery.json"

RCE_CODES = {"RCEX", "BRCE", "FRCE"}
V_LEX = ["v_max_lexoverlap", "v_mean_lexoverlap"]
A_DISC = ["a_n_disclose", "a_any_disclose", "a_frac_disclose", "a_max_disclose_overlap"]


# ---------------- build: cache app->RCE (one 507M-row pass) ----------------
def cohort_apps():
    apps = collections.Counter()
    with open(SCALE) as f:
        for ln in f:
            apps[str(json.loads(ln)["app_id"])] += 1
    return apps


def cmd_build(_):
    apps = cohort_apps()
    cohort_norm = {a.lstrip("0"): a for a in apps}
    has_rce = set()
    seen = set()
    n = 0
    for chunk in pd.read_csv(TXN, chunksize=3_000_000, dtype=str):
        n += len(chunk)
        chunk["an"] = chunk["application_number"].astype(str).str.lstrip("0")
        m = chunk[chunk["an"].isin(cohort_norm)]
        seen.update(m["an"])
        has_rce.update(m[m["event_code"].isin(RCE_CODES)]["an"])
    rows = [{"app_id": cohort_norm[an], "in_txn": True, "rce": an in has_rce} for an in seen]
    df = pd.DataFrame(rows)
    os.makedirs(CACHE, exist_ok=True)
    df.to_parquet(f"{CACHE}/app_rce.parquet")
    print(f"[build] scanned {n} txn rows; {len(df)} cohort apps in txn; "
          f"RCE rate {df['rce'].mean():.3f}", flush=True)
    print("BUILD_DONE", flush=True)


# ---------------- metric M: va features aligned to app_id, aggregated to app ----------------
def load_metric_app():
    rows = pd.read_csv(CSV)
    app_ids, claim_nums = [], []
    with open(SCALE) as f:
        for ln in f:
            r = json.loads(ln)
            app_ids.append(str(r["app_id"])); claim_nums.append(str(r["claim_num"]))
    assert len(app_ids) == len(rows), f"align mismatch {len(app_ids)} vs {len(rows)}"
    rows["app_id"] = app_ids
    rows["claim_num"] = claim_nums
    # per-claim disclosure/lexical features -> app aggregates (mean + max)
    feats = A_DISC + V_LEX + ["v_element_wordlen", "v_n_refs"]
    agg = rows.groupby("app_id").agg(
        **{f"{c}_mean": (c, "mean") for c in feats},
        **{f"{c}_max": (c, "max") for c in feats},
        n_claims=("claim_num", "count"),
        fell_any=("fell", "max"), fell_frac=("fell", "mean"),
    ).reset_index()
    return rows, agg


# ---------------- CV recovery ----------------
def hash_fold(app, k=5):
    return int(hashlib.md5(str(app).encode()).hexdigest(), 16) % k


def entropy(p):
    p = np.asarray(p, float); p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def mi_binned(y, s, bins=10):
    y = np.asarray(y, int); s = np.asarray(s, float)
    edges = np.unique(np.quantile(s, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0, entropy(np.bincount(y) / len(y))
    b = np.clip(np.digitize(s, edges[1:-1]), 0, len(edges) - 2)
    Hy = entropy(np.bincount(y) / len(y))
    hcond = sum((b == bi).mean() * entropy(np.bincount(y[b == bi], minlength=2) / (b == bi).sum())
                for bi in np.unique(b))
    return max(0.0, Hy - hcond), Hy


def cv_auc(df, feat_cols, ycol, k=5):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    X = df[feat_cols].fillna(0.0).values.astype(float)
    y = df[ycol].values.astype(int)
    folds = np.array([hash_fold(a, k) for a in df["app_id"]])
    oof = np.zeros(len(y))
    for f in range(k):
        te = folds == f; tr = ~te
        if len(set(y[tr])) < 2:
            continue
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, C=1.0))
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, oof), oof


def recovery_table(df, ycol, label, results):
    y = df[ycol].values.astype(int)
    base = y.mean()
    struct = ["n_claims", "v_element_wordlen_mean"]
    if "filing_year" in df.columns:
        struct = struct + ["filing_year"]
    disc = [f"{c}_mean" for c in A_DISC + V_LEX] + [f"{c}_max" for c in A_DISC + V_LEX]
    a_struct, _ = cv_auc(df, struct, ycol)
    a_disc, oof_d = cv_auc(df, disc, ycol)
    a_both, oof_b = cv_auc(df, struct + disc, ycol)
    mi, Hy = mi_binned(y, oof_d)
    # best single disclosure feature (univariate)
    from sklearn.metrics import roc_auc_score
    uni = max(disc, key=lambda c: abs(roc_auc_score(y, df[c].fillna(0.0)) - 0.5))
    a_uni = roc_auc_score(y, df[uni].fillna(0.0))
    print(f"\n=== {label}  (n={len(df)} apps, base rate {base:.3f}) ===", flush=True)
    print(f"  STRUCT-only      AUC={a_struct:.4f}", flush=True)
    print(f"  DISCLOSURE-only  AUC={a_disc:.4f}   best-1feat {uni}={a_uni:.4f}", flush=True)
    print(f"  BOTH             AUC={a_both:.4f}   disclosure MARGINAL={a_both - a_struct:+.4f}",
          flush=True)
    print(f"  I(disc;Y)={mi:.4f} bits = {mi / max(Hy,1e-9):.1%} of H(Y)={Hy:.3f}   "
          f"(vs fell baseline 2.5%)", flush=True)
    results[ycol] = {"label": label, "n": len(df), "base_rate": base, "auc_struct": a_struct,
                     "auc_disc": a_disc, "auc_both": a_both, "marginal": a_both - a_struct,
                     "mi_bits": mi, "mi_frac": mi / max(Hy, 1e-9), "best_feat": uni,
                     "best_feat_auc": a_uni}


# ---------------- Y3 amendment (robust filed-side join) ----------------
def norm_pgpub(s):
    s = str(s)
    return s[:4] + s[4:].zfill(7) if len(s) >= 5 else s


def build_amend(app_agg, labels):
    import pyarrow.parquet as pq
    # granted claim-1 by patent_number
    gr = pq.read_table(f"{PROC}/granted_patents_claim1_v2.parquet",
                       columns=["patent_id", "claim_text"]).to_pandas()
    gr["pid"] = gr["patent_id"].astype(str)
    gr = gr.drop_duplicates("pid").set_index("pid")["claim_text"]
    lab = labels[labels["patent_number"].notna()].copy()
    lab["pn"] = lab["patent_number"].astype(str).str.replace(r"\.0$", "", regex=True)
    lab["granted_c1"] = lab["pn"].map(gr)
    # filed claim-1: app_id -> app_num (pgpub_to_appnum) -> pgpub_id -> claim_text (pgpub_claims1)
    p2a = pd.read_parquet(f"{PROC}/pgpub_to_appnum.parquet")
    p2a["an8"] = p2a["app_num"].astype(str).str[-8:].str.lstrip("0")
    lab["an8"] = lab["app_id"].astype(str).str.lstrip("0")
    p2a = p2a.drop_duplicates("an8")
    lab = lab.merge(p2a[["an8", "pgpub_id"]], on="an8", how="left")
    pgc = pq.read_table(f"{PROC}/pgpub_claims1.parquet",
                        columns=["pgpub_id", "claim_text"]).to_pandas()
    pgc["pgn"] = pgc["pgpub_id"].astype(str).map(norm_pgpub)
    pgc = pgc.drop_duplicates("pgn").set_index("pgn")["claim_text"]
    lab["pgn"] = lab["pgpub_id"].map(norm_pgpub)
    lab["filed_c1"] = lab["pgn"].map(pgc)
    jr_g = lab["granted_c1"].notna().mean()
    jr_f = lab["filed_c1"].notna().mean()
    print(f"[Y3] granted-c1 join {jr_g:.1%}, filed-c1 join {jr_f:.1%} of {len(lab)} granted apps",
          flush=True)
    ok = lab[lab["granted_c1"].notna() & lab["filed_c1"].notna()].copy()

    def amend_mag(a, b):
        ta, tb = set(str(a).lower().split()), set(str(b).lower().split())
        if not (ta | tb):
            return 0.0
        return 1.0 - len(ta & tb) / len(ta | tb)  # token Jaccard distance
    ok["amend"] = [amend_mag(a, b) for a, b in zip(ok["filed_c1"], ok["granted_c1"])]
    med = ok["amend"].median()
    ok["Y3_amend_hi"] = (ok["amend"] > med).astype(int)
    print(f"[Y3] amendment magnitude: median {med:.3f}, "
          f"IQR [{ok['amend'].quantile(.25):.3f},{ok['amend'].quantile(.75):.3f}], "
          f"n usable {len(ok)}", flush=True)
    return app_agg.merge(ok[["app_id", "Y3_amend_hi", "amend"]], on="app_id", how="inner")


def cmd_run(_):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows, app = load_metric_app()
    labels = pd.read_parquet(f"{CACHE}/app_labels.parquet")
    rce = pd.read_parquet(f"{CACHE}/app_rce.parquet")
    app = app.merge(labels[["app_id", "n_office_actions", "is_granted", "patent_number",
                            "final_outcome", "filing_year"]], on="app_id", how="left")
    results = {}

    # anchor: claim-level "fell" recovery (the 2.5% baseline, same M, un-aggregated)
    from sklearn.metrics import roc_auc_score
    yf = rows["fell"].astype(int).values
    sf = rows["a_n_disclose"].fillna(0).values
    mi_f, Hy_f = mi_binned(yf, sf)
    print(f"=== ANCHOR: claim-level 'fell' (n={len(rows)} claims) ===", flush=True)
    print(f"  a_n_disclose univariate AUC={roc_auc_score(yf, sf):.4f}  "
          f"I(M;Y)={mi_f:.4f}={mi_f/Hy_f:.1%} of H(Y)  (matches prior .571/2.5%)", flush=True)
    results["fell_claimlevel"] = {"n": len(rows), "auc_uni": float(roc_auc_score(yf, sf)),
                                  "mi_frac": mi_f / Hy_f}

    # Y1 RCE (apps present in txn)
    d1 = app.merge(rce[["app_id", "rce"]], on="app_id", how="inner")
    d1["Y1_rce"] = d1["rce"].astype(int)
    recovery_table(d1, "Y1_rce", "Y1 RCE-filed", results)

    # Y2 persistence
    app["Y2_persist"] = (app["n_office_actions"] >= 2).astype(int)
    recovery_table(app.dropna(subset=["n_office_actions"]), "Y2_persist",
                   "Y2 rejection-persistence (>=2 OA)", results)
    # bonus continuous readout: Spearman(disc, n_office_actions)
    from scipy.stats import spearmanr
    rho, p = spearmanr(app["a_n_disclose_mean"], app["n_office_actions"], nan_policy="omit")
    print(f"  [Y2 cont] Spearman(a_n_disclose_mean, n_office_actions) rho={rho:+.4f} p={p:.1e}",
          flush=True)
    results["Y2_persist"]["spearman_n_oa"] = float(rho)

    # Y3 amendment
    try:
        d3 = build_amend(app, labels)
        recovery_table(d3, "Y3_amend_hi", "Y3 claim-1 amendment magnitude", results)
        from scipy.stats import spearmanr as sp2
        rho3, p3 = sp2(d3["a_n_disclose_mean"], d3["amend"], nan_policy="omit")
        print(f"  [Y3 cont] Spearman(a_n_disclose_mean, amend_magnitude) rho={rho3:+.4f} p={p3:.1e}",
              flush=True)
        results["Y3_amend_hi"]["spearman_amend"] = float(rho3)
    except Exception as e:
        print(f"[Y3] FAILED: {e}", flush=True)
        results["Y3_amend_hi"] = {"error": str(e)}

    json.dump(results, open(OUT, "w"), indent=1)
    print(f"\n[done] -> {OUT}", flush=True)
    print("ALTY_RECOVERY_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    sub.add_parser("run")
    a = ap.parse_args()
    {"build": cmd_build, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    main()
