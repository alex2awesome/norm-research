"""Re-run all 1182 codegen programs against the dense_4096tok DIFF TEXT
(not the v2 comment-thread artifact).

Hypothesis: programs that check 'Prisma schema', 'pointer placement',
'Python typing' etc. can't see code in the v2 artifact (review comments
only). Running them on the real diff text should reveal real signal.

Outputs:
  outputs/v2_analysis/cr_codegen_on_diff_scores.parquet
  Per-program diagnostic + AUC ladder comparison (artifact vs diff input).
"""
import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

REPO = Path("/lfs/skampere3/0/alexspan/norm-research")
TASK = "code_review"
SEED = 42
CODEGEN_DIR = REPO / f"runs/validity_full/v2/{TASK}/codegen_claude"
DPS_FILE = REPO / f"runs/validity_full/v2/{TASK}/datapoints.json"
DENSE_FILE = REPO / "datasets/code-review/code_review_dense_4096tok.csv.gz"
T12 = REPO / "outputs/v2_analysis/cr_tier12_features.parquet"
T34 = REPO / "outputs/v2_analysis/cr_tier34_features.parquet"
CG_ARTIFACT = REPO / "outputs/v2_analysis/cr_codegen_scores.parquet"
OUT_SCORES = REPO / "outputs/v2_analysis/cr_codegen_on_diff_scores.parquet"
OUT_DIAG = REPO / "outputs/v2_analysis/cr_codegen_on_diff_diagnostic.parquet"


def load_score_fn(py_path: Path):
    spec = importlib.util.spec_from_file_location(py_path.stem, py_path)
    m = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(m)
    except Exception:
        return None
    fn = getattr(m, "score", None)
    return fn if callable(fn) else None


def main():
    print("Loading v2 datapoints + dense_4096tok join...")
    dps = json.loads(DPS_FILE.read_text())
    dps = [d for d in dps if d.get("judgement") is not None and d.get("text")]

    v2 = pd.DataFrame([{
        "datapoint_id": d["datapoint_id"],
        "y": int(d["judgement"]),
        "title": (re.match(r"PR TITLE: ([^\n]+)", d["text"]) or [None, None])[1],
    } for d in dps]).dropna(subset=["title"])

    dense = pd.read_csv(DENSE_FILE, usecols=["text"])
    dense["title"] = dense["text"].str.extract(
        r"## PR Title\s*(.+?)(?:\n|$)", expand=False)
    j = v2.merge(dense.drop_duplicates("title", keep="first"),
                 on="title", how="left").dropna(subset=["text"])
    print(f"  joined {len(j)} datapoints to dense diff text")
    print(f"  median dense text length: {j['text'].str.len().median():.0f} chars")

    diff_texts = j["text"].tolist()
    dp_ids = j["datapoint_id"].tolist()
    y = j["y"].values.astype(int)

    print("\nLoading programs...")
    files = sorted([f for f in CODEGEN_DIR.iterdir()
                    if f.is_file() and f.suffix == ".py"])
    programs = []
    n_load_fail = 0
    for f in files:
        m = re.match(r"(a\d+)_v(\d+)_(\w+)\.py", f.name)
        if not m:
            continue
        fn = load_score_fn(f)
        if fn is None:
            n_load_fail += 1
            continue
        programs.append((m.group(1), f"v{m.group(2)}_{m.group(3)}", fn))
    print(f"  loaded {len(programs)} programs, {n_load_fail} load-failed")

    print("\nScoring all programs against DIFF TEXT...")
    feat_cols = [f"{a}__{fl}" for a, fl, _ in programs]
    X = np.full((len(diff_texts), len(programs)), 0.5, dtype=np.float32)
    n_exec_err = 0
    for j_idx, (aid, fl, fn) in enumerate(programs):
        if j_idx and j_idx % 100 == 0:
            print(f"  program {j_idx}/{len(programs)}")
        for i, t in enumerate(diff_texts):
            try:
                v = fn(t)
                if isinstance(v, (int, float)) and not (np.isnan(v) or np.isinf(v)):
                    X[i, j_idx] = float(max(0.0, min(1.0, v)))
                else:
                    n_exec_err += 1
            except Exception:
                n_exec_err += 1
    print(f"  exec errors: {n_exec_err}")

    scores_df = pd.DataFrame(X, columns=feat_cols)
    scores_df["datapoint_id"] = dp_ids
    scores_df["y"] = y
    scores_df.to_parquet(OUT_SCORES)
    print(f"  wrote {OUT_SCORES}")

    # ---- Per-program diagnostic ----
    std = X.std(axis=0)
    frac_default = (X == 0.5).mean(axis=0)
    print("\n=== degeneracy on DIFF input ===")
    print(f"  fraction std < 0.05            : {(std < 0.05).mean():.1%}")
    print(f"  fraction std < 0.10            : {(std < 0.10).mean():.1%}")
    print(f"  >=90% scores == 0.5 (fallback) : {(frac_default >= 0.9).sum()}")
    print(f"  median std                     : {np.median(std):.4f}")

    aucs = np.full(len(programs), np.nan)
    for j_idx in range(len(programs)):
        if std[j_idx] == 0:
            continue
        try:
            aucs[j_idx] = roc_auc_score(y, X[:, j_idx])
        except Exception:
            pass
    abs_auc = np.maximum(aucs, 1 - aucs)
    print(f"\n  median |AUC-0.5|              : {np.nanmedian(abs_auc - 0.5):.4f}")
    print(f"  programs with |AUC-0.5| > 0.02 : {(abs_auc > 0.52).sum()}")
    print(f"  programs with |AUC-0.5| > 0.05 : {(abs_auc > 0.55).sum()}")
    print(f"  programs with |AUC-0.5| > 0.10 : {(abs_auc > 0.60).sum()}")
    print(f"  max |AUC-0.5|                 : {np.nanmax(abs_auc - 0.5):.4f}")

    # Compare artifact vs diff for top-K and overall
    art = pd.read_parquet(CG_ARTIFACT)
    art_y = art["y"].values
    art_X = art.drop(columns=["datapoint_id", "y"]).values
    art_std = art_X.std(axis=0)
    art_aucs = np.full(art_X.shape[1], np.nan)
    for j_idx in range(art_X.shape[1]):
        if art_std[j_idx] == 0:
            continue
        try:
            art_aucs[j_idx] = roc_auc_score(art_y, art_X[:, j_idx])
        except Exception:
            pass
    art_abs = np.maximum(art_aucs, 1 - art_aucs)

    print("\n=== artifact vs diff comparison ===")
    print(f"  median |AUC-0.5|: artifact={np.nanmedian(art_abs - 0.5):.4f}  "
          f"diff={np.nanmedian(abs_auc - 0.5):.4f}")
    print(f"  programs |AUC-0.5|>0.05: artifact={(art_abs > 0.55).sum()}  "
          f"diff={(abs_auc > 0.55).sum()}")
    print(f"  median std: artifact={np.median(art_std):.4f}  "
          f"diff={np.median(std):.4f}")

    pd.DataFrame({
        "program": feat_cols,
        "auc_diff": aucs,
        "abs_auc_gap_diff": abs_auc - 0.5,
        "std_diff": std,
        "frac_default_diff": frac_default,
        "auc_artifact": art_aucs[:len(feat_cols)],
        "abs_auc_gap_artifact": art_abs[:len(feat_cols)] - 0.5,
    }).to_parquet(OUT_DIAG)

    # Top 20 programs by |AUC-0.5| on diff
    print("\nTop 20 programs by |AUC-0.5| on DIFF input:")
    order = np.argsort(-abs_auc)
    aspects = json.loads((REPO / f"runs/validity_full/v2/{TASK}/aspects.json").read_text())
    for j_idx in order[:20]:
        aid = feat_cols[j_idx].split("__")[0]
        idx = int(aid[1:])
        try:
            nm = aspects[idx].get("name") or aspects[idx].get("text") or "?"
        except (IndexError, KeyError, TypeError):
            nm = "?"
        art_g = art_abs[j_idx] - 0.5
        print(f"  {feat_cols[j_idx]:<28} AUC={aucs[j_idx]:.3f} std={std[j_idx]:.3f} "
              f"(art Δ={art_g:+.3f})  {nm[:60]}")

    # ---- Ladder: T1+T2+T3 vs +codegen_on_diff ----
    print("\n" + "=" * 78)
    print("VERIFIABILITY LADDER — codegen run on DIFF (not comments)")
    print("=" * 78)
    t12 = pd.read_parquet(T12)
    t34 = pd.read_parquet(T34).drop(columns=["y"], errors="ignore")
    det = t12.merge(t34, on="datapoint_id")
    base_cols = [c for c in t12.columns if c not in ("datapoint_id", "y")]
    t3_cols = [c for c in t34.columns if c.startswith("tier3_")]

    full = det.merge(scores_df, on="datapoint_id", suffixes=("", "_cg"))
    full_y = full["y"].astype(int).values

    def fit_rf(cols, label):
        Xc = full[cols].values
        Xtr, Xte, ytr, yte = train_test_split(
            Xc, full_y, test_size=0.20, stratify=full_y, random_state=SEED)
        rf = RandomForestClassifier(
            n_estimators=500, min_samples_leaf=2,
            class_weight="balanced", n_jobs=-1, random_state=SEED)
        rf.fit(Xtr, ytr)
        p = rf.predict_proba(Xte)[:, 1]
        a = roc_auc_score(yte, p)
        print(f"  {label:<48} feats={len(cols):5d}  RF={a:.3f}")
        return rf, a

    fit_rf(base_cols + t3_cols, "T1+T2+T3 deterministic")
    cg_top17 = list(np.array(feat_cols)[abs_auc > 0.55])
    if cg_top17:
        fit_rf(base_cols + t3_cols + cg_top17,
               f"+codegen(diff)|AUC>0.55 only ({len(cg_top17)})")
    fit_rf(feat_cols, "codegen(diff) ALONE")
    fit_rf(base_cols + t3_cols + feat_cols, "T1+T2+T3 + codegen(diff) ALL")


if __name__ == "__main__":
    main()
