"""Run the 1182 per-aspect Python predict-programs on code_review v2 datapoints.

For each (aspect, flavor): load score(text), call on every datapoint's text,
record the score. Then build feature matrix (rows=datapoints, cols=aspect×flavor)
and train RF.

Combined with the Tier 1+2 features already computed.
"""
import importlib.util
import json
import re
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler

REPO = Path("/lfs/skampere3/0/alexspan/norm-research")
TASK = "code_review"
SEED = 42
CODEGEN_DIR = REPO / f"runs/validity_full/v2/{TASK}/codegen_claude"
DPS_FILE = REPO / f"runs/validity_full/v2/{TASK}/datapoints.json"
TIER12 = REPO / "outputs/v2_analysis/cr_tier12_features.parquet"


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
    print("Loading datapoints...")
    dps = json.loads(DPS_FILE.read_text())
    dps = [d for d in dps if d.get("judgement") is not None and d.get("text")]
    print(f"datapoints: {len(dps)}")

    print("Loading programs...")
    files = sorted([f for f in CODEGEN_DIR.iterdir()
                    if f.is_file() and f.suffix == ".py"])
    print(f"programs to load: {len(files)}")

    programs = []  # (aspect_id, flavor, score_fn)
    n_load_fail = 0
    for f in files:
        m = re.match(r"(a\d+)_v(\d+)_(\w+)\.py", f.name)
        if not m:
            continue
        aspect_id, vnum, flavor = m.group(1), m.group(2), m.group(3)
        fn = load_score_fn(f)
        if fn is None:
            n_load_fail += 1
            continue
        programs.append((aspect_id, f"v{vnum}_{flavor}", fn))
    print(f"loaded: {len(programs)}, load-failed: {n_load_fail}")

    # Build score matrix
    print("Scoring (this may take a few minutes)...")
    feat_cols = [f"{a}__{fl}" for a, fl, _ in programs]
    X_prog = np.full((len(dps), len(programs)), 0.5, dtype=np.float32)
    n_exec_err = 0
    for j, (aspect_id, flavor, fn) in enumerate(programs):
        if j and j % 100 == 0:
            print(f"  program {j}/{len(programs)}")
        for i, d in enumerate(dps):
            try:
                v = fn(d["text"])
                if isinstance(v, (int, float)) and not (np.isnan(v) or np.isinf(v)):
                    X_prog[i, j] = float(max(0.0, min(1.0, v)))
                else:
                    n_exec_err += 1
            except Exception:
                n_exec_err += 1
    print(f"exec errors (left as 0.5): {n_exec_err}")

    dp_ids = [d["datapoint_id"] for d in dps]
    y = np.array([int(d["judgement"]) for d in dps])

    prog_df = pd.DataFrame(X_prog, columns=feat_cols)
    prog_df["datapoint_id"] = dp_ids
    prog_df["y"] = y

    out_p = REPO / "outputs/v2_analysis/cr_codegen_scores.parquet"
    prog_df.to_parquet(out_p)
    print(f"wrote {out_p}")

    # Train RF: codegen alone
    Xtr, Xte, ytr, yte = train_test_split(
        X_prog, y, test_size=0.20, stratify=y, random_state=SEED)
    rf = RandomForestClassifier(
        n_estimators=500, min_samples_leaf=2,
        class_weight="balanced", n_jobs=-1, random_state=SEED)
    rf.fit(Xtr, ytr)
    p = rf.predict_proba(Xte)[:, 1]
    auc_codegen = roc_auc_score(yte, p)
    print(f"\nCODEGEN ALONE ({len(feat_cols)} features): RF AUC = {auc_codegen:.3f}")

    # Combine with Tier 1+2
    print("\nCombining with Tier 1+2 features...")
    t12 = pd.read_parquet(TIER12)
    combo = t12.merge(prog_df.drop(columns=["y"]), on="datapoint_id", how="inner")
    feat_combo = [c for c in combo.columns
                  if c not in ("datapoint_id", "y")]
    Xc = combo[feat_combo].values
    yc = combo["y"].astype(int).values
    print(f"  combined features: {len(feat_combo)}, rows: {len(yc)}")

    Xtr, Xte, ytr, yte = train_test_split(
        Xc, yc, test_size=0.20, stratify=yc, random_state=SEED)
    rf2 = RandomForestClassifier(
        n_estimators=500, min_samples_leaf=2,
        class_weight="balanced", n_jobs=-1, random_state=SEED)
    rf2.fit(Xtr, ytr)
    p = rf2.predict_proba(Xte)[:, 1]
    auc_combo = roc_auc_score(yte, p)
    acc_combo = accuracy_score(yte, (p > 0.5).astype(int))

    print()
    print("=" * 70)
    print("CODE_REVIEW VERIFIABILITY LADDER (so far)")
    print("=" * 70)
    print(f"  Legacy LLM-judge (thin artifact, 394 aspects) :  AUC ~0.518")
    print(f"  Tier 1 (5 metadata)                            :  AUC = 0.601")
    print(f"  Tier 1+2 (34, +diff parsing)                   :  AUC = 0.616")
    print(f"  Codegen alone ({len(feat_cols)} per-aspect Python)        :  AUC = {auc_codegen:.3f}")
    print(f"  Tier 1+2 + Codegen ({len(feat_combo)} feats)              :  AUC = {auc_combo:.3f}  acc = {acc_combo:.1%}")
    print()
    print("Top 15 features by RF importance (combined):")
    imps = sorted(zip(feat_combo, rf2.feature_importances_), key=lambda x: -x[1])[:15]
    for n, i in imps:
        print(f"  {n:<40} {i:.4f}")


if __name__ == "__main__":
    main()
