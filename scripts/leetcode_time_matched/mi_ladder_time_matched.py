"""
Run the MI ladder on the time-matched 1v1 pairs.
For each window in {1, 7, 30}d, per language:
  - bank features (RF / LR) AUC, GroupKFold by problem_slug
  - TF-IDF baseline AUC
  - Age/length baseline AUC

Honest comparison vs the broken upvote label's baseline (0.632 RF).

We use leetcode_balanced_metric_scores.parquet (covers 70K balanced rows).
Pairs are restricted to those where both winner and loser are in the balanced bank.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

BANK_PATH = "/lfs/skampere3/0/alexspan/norm-research/outputs/v2_analysis/leetcode_balanced_metric_scores.parquet"
CPP_FIXED = "/lfs/skampere3/0/alexspan/norm-research/outputs/v2_analysis/leetcode_cpp_metric_scores_fixed.parquet"
LABELS = "/lfs/skampere3/0/alexspan/norm-research/datasets/leetcode_balanced/time_matched_1v1_label.parquet"
PAIRS = "/lfs/skampere3/0/alexspan/norm-research/datasets/leetcode_balanced/time_matched_1v1.parquet"
BAL = "/lfs/skampere3/0/alexspan/norm-research/datasets/leetcode_balanced/_with_ts_tmp.parquet"

OUT = "/lfs/skampere3/0/alexspan/norm-research/outputs/v2_analysis/leetcode_time_matched_mi_ladder.json"


def load_bank_features():
    bank = pd.read_parquet(BANK_PATH)
    # Use *_score columns
    score_cols = [c for c in bank.columns if c.endswith("_score")]
    print(f"Bank: {len(bank)} rows, {len(score_cols)} score columns")
    return bank[["row_id"] + score_cols].set_index("row_id"), score_cols


def load_cpp_fixed_features():
    f = pd.read_parquet(CPP_FIXED)
    score_cols = [c for c in f.columns if c.endswith("_score")]
    return f[["row_id"] + score_cols].set_index("row_id"), score_cols


def eval_features(X, y, groups, name, n_splits=5):
    """RF + LR AUC via GroupKFold."""
    out = {}
    if len(np.unique(y)) < 2:
        return {"rf_auc": np.nan, "lr_auc": np.nan, "n_pairs": len(y)}
    if len(np.unique(groups)) < n_splits:
        n_splits = max(2, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    rf_preds = np.zeros(len(y))
    lr_preds = np.zeros(len(y))
    for tr, te in gkf.split(X, y, groups):
        # RF
        rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0,
                                    class_weight="balanced", max_depth=None)
        rf.fit(X[tr], y[tr])
        rf_preds[te] = rf.predict_proba(X[te])[:, 1]
        # LR
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[te])
        lr = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced")
        lr.fit(Xtr, y[tr])
        lr_preds[te] = lr.predict_proba(Xte)[:, 1]
    try:
        out["rf_auc"] = float(roc_auc_score(y, rf_preds))
        out["lr_auc"] = float(roc_auc_score(y, lr_preds))
    except Exception:
        out["rf_auc"] = np.nan
        out["lr_auc"] = np.nan
    out["n_pairs"] = int(len(y) // 2)
    out["n_pos"] = int(y.sum())
    out["n_neg"] = int((1 - y).sum())
    return out


def main():
    labels = pd.read_parquet(LABELS)
    pairs = pd.read_parquet(PAIRS)
    bal = pd.read_parquet(BAL).reset_index(drop=True)
    bal["balanced_row_id"] = bal.index.astype(int)

    bank_idx, bank_cols = load_bank_features()
    cpp_idx, cpp_cols = load_cpp_fixed_features()

    # Restrict to both-in-balanced
    g = labels.groupby(["window_days", "pair_id"])["balanced_row_id"].apply(lambda x: x.notna().all())
    good = g[g].reset_index().drop(columns=[0]) if 0 in g.reset_index().columns else g[g].reset_index().rename(columns={"balanced_row_id": "ok"})
    # Simpler: just filter
    labels = labels.merge(g.rename("both_in").reset_index(), on=["window_days", "pair_id"])
    labels = labels[labels["both_in"]].copy()
    labels["balanced_row_id"] = labels["balanced_row_id"].astype(int)

    # Add code length + age features for baselines
    print("Joining code length + ts...")
    aux = bal[["balanced_row_id", "code_len", "n_lines", "created_ts"]].copy()
    aux["created_ts"] = pd.to_datetime(aux["created_ts"], errors="coerce", utc=True)
    aux["age_days"] = (pd.Timestamp.utcnow() - aux["created_ts"]).dt.total_seconds() / 86400.0
    labels = labels.merge(aux, on="balanced_row_id", how="left")

    results = {}
    LANGS = ["cpp", "java", "python", "javascript"]
    for w in [1, 7, 30]:
        print(f"\n========== window={w}d ==========")
        results[str(w)] = {}
        sub_all = labels[labels.window_days == w]

        for lang in LANGS + ["ALL"]:
            sub = sub_all if lang == "ALL" else sub_all[sub_all.language == lang]
            sub = sub.sort_values(["pair_id", "label"], ascending=[True, False])
            # ensure exactly 2 rows per pair
            cnt = sub.groupby("pair_id").size()
            keep_pairs = cnt[cnt == 2].index
            sub = sub[sub.pair_id.isin(keep_pairs)].copy()
            if len(sub) < 10:
                print(f"[{w}d / {lang}] too few pairs ({len(sub)//2}), skip")
                results[str(w)][lang] = {"skipped": True, "n_pairs": len(sub) // 2}
                continue

            y = sub["label"].values.astype(int)
            groups = sub["problem_slug"].astype(str).values
            row_ids = sub["balanced_row_id"].values

            # ---- bank features ----
            bank_feats = bank_idx.loc[row_ids].fillna(0).values
            bank_res = eval_features(bank_feats, y, groups, "bank")

            # ---- cpp_fixed features (cpp lang only) ----
            cpp_res = None
            if lang == "cpp":
                # may be partial coverage
                covered = np.isin(row_ids, cpp_idx.index.values)
                if covered.sum() > 20:
                    Xc = cpp_idx.reindex(row_ids).fillna(0).values
                    # only on rows actually covered
                    cpp_res = eval_features(Xc[covered], y[covered], groups[covered], "cpp_fixed")
                    cpp_res["coverage"] = float(covered.mean())

            # ---- TF-IDF baseline ----
            # join code from pairs
            code_df = pairs[["pair_id", "window_days", "solution_id", "code"]].drop_duplicates(["pair_id","window_days","solution_id"])
            sub2 = sub.merge(code_df, on=["pair_id", "window_days", "solution_id"], how="left")
            corpus = sub2["code"].fillna("").astype(str).values
            tfidf = TfidfVectorizer(
                analyzer="char_wb", ngram_range=(3, 5), max_features=20000, min_df=2
            )
            Xt = tfidf.fit_transform(corpus).toarray() if len(corpus) < 5000 else tfidf.fit_transform(corpus)
            # GroupKFold needs sparse-compatible? use it on dense
            if hasattr(Xt, "toarray"):
                Xt = Xt.toarray()
            tfidf_res = eval_features(Xt, y, groups, "tfidf")

            # ---- length+age baseline ----
            base_feats = sub[["code_len", "n_lines", "age_days"]].fillna(0).values
            base_res = eval_features(base_feats, y, groups, "len_age")

            # ---- length only ----
            len_feats = sub[["code_len", "n_lines"]].fillna(0).values
            len_res = eval_features(len_feats, y, groups, "len")

            results[str(w)][lang] = {
                "bank": bank_res,
                "tfidf": tfidf_res,
                "len_age": base_res,
                "len_only": len_res,
            }
            if cpp_res is not None:
                results[str(w)][lang]["cpp_fixed"] = cpp_res

            print(f"[{w}d / {lang:10s}] n_pairs={bank_res['n_pairs']:5d}  "
                  f"bank RF={bank_res['rf_auc']:.3f} LR={bank_res['lr_auc']:.3f}  "
                  f"tfidf RF={tfidf_res['rf_auc']:.3f} LR={tfidf_res['lr_auc']:.3f}  "
                  f"len_age RF={base_res['rf_auc']:.3f}  "
                  f"len_only RF={len_res['rf_auc']:.3f}")
            if cpp_res is not None:
                print(f"        cpp_fixed RF={cpp_res['rf_auc']:.3f} LR={cpp_res['lr_auc']:.3f}  cov={cpp_res['coverage']:.2f}")

    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
