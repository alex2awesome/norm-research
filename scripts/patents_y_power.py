#!/usr/bin/env python3
"""Y-variable power analysis for patent acceptance prediction (dual-audit implementation, 2026-07-13).

Answers the user's question — "would another y give more power/discrimination (e.g. predict the
patent gets another iteration)?" — in the three senses the audits separated:

  (a) INTRINSIC PREDICTABILITY: linear text baselines (hashing TF-IDF + logistic-SGD) on every
      candidate y, under BOTH an app-grouped random split and an out-of-time split (train<=2016 /
      test>=2017 — the audit measured grant-rate drift .461->.592 inside the "balanced" file, so
      random-split numbers are era-inflated). Exact-duplicate texts deduped (audit: 3,427 dup
      texts, 533 cross-label, straddling random splits).
  (b) DISCLOSURE-METRIC MARGINAL (fixed #87 rerun): dedup option3 rows on (app,claim,element)
      before app-aggregation (12.2% dups row-weighted the old aggregates); STRUCT gets REAL
      unique-claim counts (old n_claims counted rows — Codex finding), filing year, element
      length, plus examiner-leniency + art-unit controls that never existed in the pipeline;
      probes = logistic AND HistGradientBoosting (old run was linear-only); marginal gets an
      app-bootstrap CI (old run had none).
  (c) CONSTRUCT FIT: leniency-alone AUCs quantify how much of each y is examiner lottery;
      y's are the risk-set-conditioned panel variables (Y_first_action_allow, Y_another_round,
      Y_rce_after_final, Y_appeal_after_final, Y_abandon_after_2) from patents_event_panel.py
      rather than the undated ever-RCE / raw n_office_actions booleans of #87.

Reconstruction-only is preserved: M never sees any y; y's enter only as evaluation targets.

Cohorts:
  A  patents_final_outcome_cpc_balanced_with_rejections.csv.gz (579,084 rows; text + judgement)
  N  natural-population sample streamed from patents_dataset.jsonl.gz (disposed, <=2021,
     NO outcome balancing — sense-(a) numbers with real base rates)
  B  option3 21,447-app extraction cohort (M features live here)

Run on sk3 (CPU): python scripts/patents_y_power.py run [--skip-tfidf]
Output: outputs/patents_y_power/results.json + printed tables. Marker: Y_POWER_DONE
"""
import argparse, csv, gzip, hashlib, json, os, sys
import numpy as np
import pandas as pd

csv.field_size_limit(2**31 - 1)
BASE = "/lfs/skampere3/0/alexspan/norm-research"
PAT = f"{BASE}/datasets/patents"
PANEL = f"{PAT}/processed/prosecution_event_panel.parquet"
COHORT_A = f"{PAT}/patents_final_outcome_cpc_balanced_with_rejections.csv.gz"
NATURAL = f"{PAT}/patents_dataset.jsonl.gz"
VA_CSV = f"{BASE}/notebooks/data/patents_va_features.csv"
SCALE = f"{PAT}/processed/option3_claims_gemma_scale.jsonl"
OUT = f"{BASE}/outputs/patents_y_power/results.json"

PANEL_YS = ["Y_first_action_allow", "Y_another_round", "Y_rce_after_final",
            "Y_appeal_after_final", "Y_abandon_after_2"]
V_LEX = ["v_max_lexoverlap", "v_mean_lexoverlap"]
A_DISC = ["a_n_disclose", "a_any_disclose", "a_frac_disclose", "a_max_disclose_overlap"]
NAT_SAMPLE = 300_000
SEED = 13


def hfold(key, k=5):
    return int(hashlib.md5(f"ypower::{key}".encode()).hexdigest(), 16) % k


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y, int); s = np.asarray(s, float)
    ok = ~np.isnan(s)
    if len(set(y[ok])) < 2:
        return float("nan")
    return float(roc_auc_score(y[ok], s[ok]))


# ---------------- panel join ----------------
def load_panel():
    p = pd.read_parquet(PANEL)
    p["an_norm"] = p["application_number"].str.lstrip("0")
    return p.set_index("an_norm")


def load_cohort_a(with_text=False):
    rows = []
    with gzip.open(COHORT_A, "rt") as f:
        for r in csv.DictReader(f):
            rows.append({"an_norm": r["app_id"].lstrip("0"), "y": int(r["judgement"]),
                         "year": int(r["year"]), "cpc": r["cpc_section"],
                         **({"text": r["text"]} if with_text else
                            {"tlen": len(r["text"])})})
    return pd.DataFrame(rows)


# ---------------- 1. leniency probe (sense c) ----------------
def leniency_probe(panel, results):
    a = load_cohort_a(with_text=False)
    d = a.join(panel, on="an_norm", how="inner", rsuffix="_p")
    print(f"\n=== LENIENCY PROBE — cohort A join {len(d):,}/{len(a):,} ===", flush=True)
    res = {"n": len(d)}
    probes = {"exm_loo_grant": "examiner LOO grant rate",
              "au_loo_grant": "art-unit LOO grant rate",
              "auyr_loo_grant": "art-unit x year LOO grant rate",
              "year": "publication year", "tlen": "raw text length"}
    for col, label in probes.items():
        res[col] = auc(d["y"], d[col])
        print(f"  {label:32s} alone: AUC={res[col]:.4f}", flush=True)
    # combined leniency + year (logistic, app-hash CV)
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    feats = ["exm_loo_grant", "au_loo_grant", "auyr_loo_grant", "year"]
    X = d[feats].astype(float).fillna(d[feats].astype(float).median()).values
    y = d["y"].values
    folds = np.array([hfold(k) for k in d["an_norm"]])
    oof = np.zeros(len(y))
    for f in range(5):
        te = folds == f
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        clf.fit(X[~te], y[~te]); oof[te] = clf.predict_proba(X[te])[:, 1]
    res["leniency_plus_year_cv"] = auc(y, oof)
    print(f"  leniency(3) + year combined CV: AUC={res['leniency_plus_year_cv']:.4f}", flush=True)
    # leniency on the panel y's, natural disposed population
    dd = panel[panel["disposed"]]
    for yc in PANEL_YS:
        v = dd[dd[yc].notna()]
        res[f"exm_loo_on_{yc}"] = auc(v[yc].astype(int), v["exm_loo_grant"])
        print(f"  examiner-LOO alone on {yc:24s} (n={len(v):9,}): "
              f"AUC={res[f'exm_loo_on_{yc}']:.4f}", flush=True)
    results["leniency"] = res


# ---------------- 2. sense-(a) text baselines ----------------
def dedup_texts(df):
    h = df["text"].map(lambda t: hashlib.md5(t.encode()).hexdigest())
    df = df.assign(_h=h)
    g = df.groupby("_h")["ybin"].nunique()
    cross = set(g[g > 1].index)
    before = len(df)
    df = df[~df["_h"].isin(cross)].drop_duplicates("_h")
    print(f"    dedup: {before:,} -> {len(df):,} "
          f"(dropped {len(cross)} cross-label text groups + exact dups)", flush=True)
    return df.drop(columns="_h")


def tfidf_eval(df, ycol, tag, results):
    """Linear text baseline under app-grouped and out-of-time splits."""
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.linear_model import SGDClassifier
    d = df[df[ycol].notna()].copy()
    d["ybin"] = d[ycol].astype(int)
    if d["ybin"].nunique() < 2 or len(d) < 2000:
        print(f"  [{tag}:{ycol}] skipped (n={len(d)})", flush=True)
        return
    d = dedup_texts(d)
    vec = HashingVectorizer(n_features=2**20, ngram_range=(1, 2),
                            alternate_sign=False, norm="l2")
    entry = {"n": len(d), "base_rate": float(d["ybin"].mean())}
    splits = {"grouped": np.array([hfold(k) for k in d["an_norm"]]) == 0,
              "out_of_time": d["year"].values >= 2017}
    for sname, te in splits.items():
        tr = ~te
        if d["ybin"][tr].nunique() < 2 or d["ybin"][te].nunique() < 2 or te.sum() < 500:
            entry[sname] = None
            continue
        Xtr = vec.transform(d["text"][tr]); Xte = vec.transform(d["text"][te])
        clf = SGDClassifier(loss="log_loss", alpha=1e-6, max_iter=15,
                            tol=None, random_state=SEED)
        clf.fit(Xtr, d["ybin"][tr])
        s = clf.decision_function(Xte)
        entry[sname] = auc(d["ybin"][te].values, s)
    print(f"  [{tag}:{ycol}] n={entry['n']:,} base={entry['base_rate']:.3f} "
          f"grouped={entry.get('grouped')} out_of_time={entry.get('out_of_time')}", flush=True)
    results.setdefault(f"tfidf_{tag}", {})[ycol] = entry


def sense_a(panel, results):
    print("\n=== SENSE (a): linear text baselines ===", flush=True)
    a = load_cohort_a(with_text=True)
    d = a.join(panel, on="an_norm", how="inner", rsuffix="_p")
    print(f"  cohort A join {len(d):,}", flush=True)
    for ycol in ["y"] + PANEL_YS:
        tfidf_eval(d, ycol, "cohortA", results)
    # natural-population sample (no outcome balancing); JSONL is pgpub-keyed
    p2a = pd.read_parquet(f"{PAT}/processed/pgpub_to_appnum.parquet")
    p2a["pgn"] = p2a["pgpub_id"].astype(str).map(
        lambda s: s[:4] + s[4:].zfill(7) if len(s) >= 5 else s)
    pg2an = dict(zip(p2a["pgn"],
                     p2a["app_num"].astype(str).str[-8:].str.lstrip("0")))
    rng = np.random.default_rng(SEED)
    nat, n_seen = [], 0
    with gzip.open(NATURAL, "rt") as f:
        for ln in f:
            r = json.loads(ln)
            fo = r.get("final_outcome")
            if fo not in ("granted", "abandoned"):
                continue
            try:
                yr = int(str(r.get("date_published", ""))[:4])
            except ValueError:
                continue
            if yr > 2021:
                continue
            pgn = str(r.get("pgpub_id", ""))
            pgn = pgn[:4] + pgn[4:].zfill(7) if len(pgn) >= 5 else pgn
            an = pg2an.get(pgn)
            if not an:
                continue
            n_seen += 1
            text = f"ABSTRACT:\n{r.get('pg_abstract') or ''}\n\nCLAIMS:\n{r.get('pg_claims') or ''}"
            item = {"an_norm": an, "text": text[:40000], "year": yr,
                    "y": 1 if fo == "granted" else 0}
            if len(nat) < NAT_SAMPLE:
                nat.append(item)
            else:
                j = rng.integers(0, n_seen)
                if j < NAT_SAMPLE:
                    nat[j] = item
    dn = pd.DataFrame(nat)
    dn = dn[dn["text"].str.len() > 50]
    dn = dn.join(panel, on="an_norm", how="left", rsuffix="_p")
    print(f"  natural sample {len(dn):,} of {n_seen:,} disposed<=2021 "
          f"(grant rate {dn['y'].mean():.3f})", flush=True)
    for ycol in ["y"] + PANEL_YS:
        tfidf_eval(dn, ycol, "natural", results)


# ---------------- 3. sense-(b): fixed #87 rerun ----------------
def load_metric_app_dedup():
    rows = pd.read_csv(VA_CSV)
    app_ids, claims, elems = [], [], []
    with open(SCALE) as f:
        for ln in f:
            r = json.loads(ln)
            app_ids.append(str(r["app_id"])); claims.append(str(r["claim_num"]))
            elems.append(str(r.get("element", ""))[:400])
    assert len(app_ids) == len(rows)
    rows["app_id"], rows["claim_num"], rows["element"] = app_ids, claims, elems
    before = len(rows)
    rows = rows.drop_duplicates(subset=["app_id", "claim_num", "element"])
    print(f"  option3 dedup {before:,} -> {len(rows):,} rows", flush=True)
    feats = A_DISC + V_LEX + ["v_element_wordlen", "v_n_refs"]
    agg = rows.groupby("app_id").agg(
        **{f"{c}_mean": (c, "mean") for c in feats},
        **{f"{c}_max": (c, "max") for c in feats},
        n_rows=("claim_num", "count"),
        n_unique_claims=("claim_num", "nunique"),
    ).reset_index()
    agg["an_norm"] = agg["app_id"].str.lstrip("0")
    return agg


def cv_oof(X, y, keys, model):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    folds = np.array([hfold(k) for k in keys])
    oof = np.full(len(y), np.nan)
    for f in range(5):
        te = folds == f; tr = ~te
        if len(set(y[tr])) < 2 or te.sum() == 0:
            continue
        if model == "logistic":
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=2000))
        else:
            clf = HistGradientBoostingClassifier(random_state=SEED)
        clf.fit(X[tr], y[tr]); oof[te] = clf.predict_proba(X[te])[:, 1]
    return oof


def boot_ci(y, s_struct, s_both, keys, n_boot=500):
    rng = np.random.default_rng(SEED)
    uk = np.unique(keys)
    idx_by = {k: np.flatnonzero(keys == k) for k in uk}
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(uk, size=len(uk), replace=True)
        idx = np.concatenate([idx_by[k] for k in pick])
        if len(set(y[idx])) < 2:
            continue
        diffs.append(auc(y[idx], s_both[idx]) - auc(y[idx], s_struct[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi)


def sense_b(panel, results):
    print("\n=== SENSE (b): disclosure marginal, fixed M + honest STRUCT ===", flush=True)
    agg = load_metric_app_dedup()
    d = agg.join(panel, on="an_norm", how="inner", rsuffix="_p")
    print(f"  cohort B join {len(d):,}/{len(agg):,}", flush=True)
    d["Y_final_granted"] = np.where(
        d["final_outcome"].isin(["granted", "abandoned"]),
        (d["final_outcome"] == "granted").astype(float), np.nan)
    struct = ["n_unique_claims", "n_rows", "filing_year", "v_element_wordlen_mean",
              "exm_loo_grant", "au_loo_grant", "auyr_loo_grant"]
    disc = [f"{c}_mean" for c in A_DISC + V_LEX] + [f"{c}_max" for c in A_DISC + V_LEX]
    out = {}
    for ycol in ["Y_final_granted"] + PANEL_YS:
        dd = d[d[ycol].notna()].copy()
        if len(dd) < 1000 or dd[ycol].nunique() < 2:
            print(f"  [{ycol}] skipped n={len(dd)}", flush=True)
            continue
        y = dd[ycol].astype(int).values
        keys = dd["an_norm"].values
        med = dd[struct + disc].astype(float).median()
        Xs = dd[struct].astype(float).fillna(med[struct]).values
        Xb = dd[struct + disc].astype(float).fillna(med).values
        row = {"n": len(dd), "base_rate": float(y.mean())}
        for model in ("logistic", "hgb"):
            os_ = cv_oof(Xs, y, keys, model)
            ob_ = cv_oof(Xb, y, keys, model)
            a_s, a_b = auc(y, os_), auc(y, ob_)
            lo, hi = boot_ci(y, os_, ob_, keys)
            row[model] = {"auc_struct": a_s, "auc_both": a_b,
                          "marginal": a_b - a_s, "ci95": [lo, hi]}
            print(f"  [{ycol}] {model:8s} n={len(dd):6,} base={y.mean():.3f} "
                  f"struct={a_s:.4f} both={a_b:.4f} marginal={a_b - a_s:+.4f} "
                  f"CI[{lo:+.4f},{hi:+.4f}]", flush=True)
        out[ycol] = row
    results["disclosure_marginal_v2"] = out


def cmd_run(a):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    panel = load_panel()
    results = {}
    leniency_probe(panel, results)
    sense_b(panel, results)
    if not a.skip_tfidf:
        sense_a(panel, results)
    json.dump(results, open(OUT, "w"), indent=1, default=float)
    print(f"\n[done] -> {OUT}", flush=True)
    print("Y_POWER_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--skip-tfidf", action="store_true")
    a = ap.parse_args()
    cmd_run(a)


if __name__ == "__main__":
    main()
